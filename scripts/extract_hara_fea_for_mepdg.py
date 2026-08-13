# -*- coding: utf-8 -*-
"""
scripts/extract_hara_fea_for_mepdg.py  —  R3-12 cross-spec data producer (v2)
=============================================================================

Produce *real* FEA strain/stress responses for each HARA-optimised final
design, so the ME-PDG cross-spec check reads MEASURED eps_a/eps_z/p_AC instead
of fabricating them. This replaces the SCR-anchored fake-strain logic that
caused the spurious 0/36 ME-PDG result.

v2 change vs the previous draft:
  - Also stores the final-design layer MODULI (E1..E5) — the ME-PDG fatigue
    model needs E_ac, and we must not assume it.
  - Stores all six FEA responses (eps_a, sigma_t, eps_z, and the three p_AC
    mid-depth stresses) so the downstream rutting model has real vertical-stress
    inputs.

PIPELINE
--------
    extract_hara_fea_for_mepdg.py  ->  hara_fea_responses_<ts>.csv  (THIS FILE)
    run_mepdg_cross_spec_check.py  ->  reads that CSV, applies NCHRP 1-37A eqs

Usage
-----
    # final extraction (real ABAQUS verification of the final design; accurate)
    python scripts/extract_hara_fea_for_mepdg.py --escalation

    # fast dry run, surrogate-only (no ABAQUS; safe during ablation)
    python scripts/extract_hara_fea_for_mepdg.py --surrogate-only

    # 3 seeds per section (matches the 12 x 3 = 36 inference matrix)
    python scripts/extract_hara_fea_for_mepdg.py --escalation --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_fea")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLICY_FLEX = "output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip"
POLICY_SEMI = "output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip"
SURROGATE = "output/surrogate_model/surrogate_v3.pt"

SUBGRADE_XLSX_CANDIDATES = [
    "experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx",
    "experiments/ltpp_data/deliverables/ltpp_12_sections_with_subgrade.xlsx",
]

SECTIONS = {
    "04_1034": "flexible", "04_1065": "semi_rigid", "06_2004": "semi_rigid",
    "12_1060": "flexible", "12_4097": "semi_rigid",
    "16_1010": "flexible", "27_1085": "flexible", "27_2023": "semi_rigid",
    "30_7076": "semi_rigid",
    "48_0001": "flexible", "48_1076": "flexible", "48_1109": "semi_rigid",
}

# Per-section JTG climate_zone (MAAT from batch_climate_12sections_summary.csv)
_SECTION_JTG_ZONE = {
    "27_2023": "temperate", "16_1010": "temperate", "27_1085": "temperate",
    "30_7076": "temperate", "04_1065": "temperate",
    "48_1076": "warm",      "06_2004": "warm",
    "12_4097": "warm",      "48_1109": "warm",
    "48_0001": "hot",       "04_1034": "hot",       "12_1060": "hot",
}

MAX_STEPS = 20

# Poisson ratios by pavement type (5 layers: AC_upper, AC_mid, AC_lower, base, subbase)
POISSON_BY_TYPE = {
    "flexible":   [0.25, 0.30, 0.30, 0.35, 0.35],
    "semi_rigid": [0.25, 0.30, 0.30, 0.25, 0.35],
}

# Per-type INITIAL design (top-down 5 layers). CRITICAL: the EnvConfig default
# init uses a semi-rigid base modulus (1500 MPa). For flexible sections that
# value is ABOVE the flexible NumericalGuard upper bound (500 MPa), so every
# action is rejected and the policy freezes at an unrealistic design. We must
# give flexible sections a granular-base init. These match the per-type inits
# used by the LTPP inference matrix.
#   >>> If ltpp_inference.py uses different inits, copy them here verbatim so the
#   >>> ME-PDG check is on the SAME designs the paper reports. <<<
INIT_BY_TYPE = {
    "flexible": {
        "thk": [0.04, 0.06, 0.08, 0.30, 0.25],
        "mod": [14000.0, 11000.0, 9000.0, 350.0, 250.0],   # granular base/subbase
    },
    "semi_rigid": {
        "thk": [0.04, 0.06, 0.08, 0.36, 0.18],
        "mod": [14000.0, 11000.0, 9000.0, 1500.0, 400.0],  # cement-stabilised base
    },
}

# FEA response key aliases — match rl/env.py + surrogate_backend RESPONSE_KEYS,
# but stay tolerant to minor naming drift.
RESP_KEYS = {
    "eps_a_micro":    ["epsilon_a_microstrain", "eps_a_microstrain", "epsilon_a"],
    "eps_z_micro":    ["epsilon_z_microstrain", "eps_z_microstrain", "epsilon_z"],
    "sigma_t_MPa":    ["sigma_t_MPa", "sigma_t"],
    "p_AC_upper_mid": ["p_AC_upper_mid_MPa", "p_AC_upper_mid"],
    "p_AC_mid_mid":   ["p_AC_mid_mid_MPa", "p_AC_mid_mid"],
    "p_AC_lower_mid": ["p_AC_lower_mid_MPa", "p_AC_lower_mid"],
    "deflection_mm":  ["D_FEA_mm", "D_surface_mm", "deflection_mm"],
}


def _first(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None


def _fmt(x: Optional[float]) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "n/a"


def resolve_subgrade_table():
    import pandas as pd
    for cand in SUBGRADE_XLSX_CANDIDATES:
        p = Path(cand)
        if p.exists():
            logger.info(f"Subgrade table: {p}")
            return pd.read_excel(p)
    logger.error("Could not find the section->E_subgrade table. Tried:\n  "
                 + "\n  ".join(SUBGRADE_XLSX_CANDIDATES))
    return None


def get_esub(df, sid: str) -> Optional[float]:
    rows = df[df["section_id"].astype(str) == sid]
    if rows.empty:
        norm = sid.replace("_", "")
        rows = df[df["section_id"].astype(str).str.replace("_", "") == norm]
    if rows.empty:
        return None
    return float(rows.iloc[0]["E_subgrade_MPa"])


def run_one_section(sid: str, ptype: str, esub: float, model,
                    b3_threshold: float, seed: int):
    """Run one deterministic episode, then a clean FEA on the final design."""
    from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig
    from fea import run_fea

    init = INIT_BY_TYPE.get(ptype, INIT_BY_TYPE["flexible"])
    cfg = SurrogateEnvConfig(
        pavement_type=ptype,
        init_thickness_m=list(init["thk"]),
        init_modulus_MPa=list(init["mod"]),
        init_poisson=list(POISSON_BY_TYPE.get(ptype, POISSON_BY_TYPE["flexible"])),
        E_subgrade=esub, nu_subgrade=0.40,
        city="", climate_zone=_SECTION_JTG_ZONE.get(sid, "temperate"),
        road_class="expressway", traffic_level="heavy",
        design_life_years=15, max_episode_steps=MAX_STEPS, max_episodes=1,
        llm_enabled=False, fea_keep_runs=False,
        use_surrogate=True, surrogate_model_path=SURROGATE,
        surrogate_b3_threshold=b3_threshold,
        enable_lcc_eval=True, design_life_years_lcc=20.0,
    )
    env = PavementEnvWithSurrogate(cfg)
    try:
        env.set_fea_output_dir(Path(PROJECT_ROOT))
    except AttributeError:
        pass

    try:
        obs, info = env.reset(seed=seed)
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        final_thk = [float(h) for h in env._design["thickness"]]
        final_mod = [float(m) for m in env._design["modulus"]]
        final_nu = POISSON_BY_TYPE.get(ptype, POISSON_BY_TYPE["flexible"])

        t0 = time.time()
        fea_result = run_fea(
            thickness=final_thk, modulus=final_mod, poisson=final_nu,
            E_subgrade=esub, nu_subgrade=0.40,
            load_pressure=0.7, load_radius=0.1065,
            num_cpus=4, verbose=False,
        )
        elapsed = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:
            pass

    resp = fea_result.get("responses", {}) if isinstance(fea_result, dict) else {}

    row = {
        "section_id": sid, "pavement_type": ptype, "seed": seed,
        "E_subgrade": esub,
        # --- real FEA responses ---
        "eps_a_micro":    _first(resp, RESP_KEYS["eps_a_micro"]),
        "eps_z_micro":    _first(resp, RESP_KEYS["eps_z_micro"]),
        "sigma_t_MPa":    _first(resp, RESP_KEYS["sigma_t_MPa"]),
        "p_AC_upper_mid": _first(resp, RESP_KEYS["p_AC_upper_mid"]),
        "p_AC_mid_mid":   _first(resp, RESP_KEYS["p_AC_mid_mid"]),
        "p_AC_lower_mid": _first(resp, RESP_KEYS["p_AC_lower_mid"]),
        "deflection_mm":  _first(resp, RESP_KEYS["deflection_mm"]),
        # --- final design geometry (cm) ---
        "h1_cm": round(final_thk[0] * 100, 2), "h2_cm": round(final_thk[1] * 100, 2),
        "h3_cm": round(final_thk[2] * 100, 2), "h4_cm": round(final_thk[3] * 100, 2),
        "h5_cm": round(final_thk[4] * 100, 2),
        # --- final design moduli (MPa) — needed by ME-PDG fatigue (E_ac) ---
        "E1_MPa": round(final_mod[0], 0), "E2_MPa": round(final_mod[1], 0),
        "E3_MPa": round(final_mod[2], 0), "E4_MPa": round(final_mod[3], 0),
        "E5_MPa": round(final_mod[4], 0),
        "FEA_elapsed_s": round(elapsed, 1),
        "status": "ok",
    }
    if row["eps_a_micro"] is None or row["eps_z_micro"] is None:
        row["status"] = "fea_missing_keys"
        logger.warning(f"  [{sid}] FEA returned no recognised eps keys; "
                       f"available keys = {sorted(resp.keys())}")
    return row


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--escalation", action="store_true",
                      help="Enable ABAQUS escalation (B3 threshold=1.0). Accurate; final data.")
    mode.add_argument("--surrogate-only", action="store_true",
                      help="Surrogate only (B3 threshold=-1.0). Fast, no ABAQUS; dry run.")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per section (default 1; use 3 to match the matrix).")
    args = parser.parse_args()

    b3_threshold = -1.0 if args.surrogate_only else 1.0
    mode_name = "surrogate_only" if args.surrogate_only else "escalation_enabled"
    seeds = list(range(args.seeds))
    logger.info(f"Mode: {mode_name} (surrogate_b3_threshold={b3_threshold}); seeds={seeds}")

    for label, p in [("flex policy", POLICY_FLEX), ("semi policy", POLICY_SEMI),
                     ("surrogate", SURROGATE)]:
        if not Path(p).exists():
            logger.error(f"Missing {label}: {p}")
            return
    df = resolve_subgrade_table()
    if df is None:
        return

    from stable_baselines3 import PPO
    logger.info("Loading PPO policies...")
    model_flex = PPO.load(POLICY_FLEX)
    model_semi = PPO.load(POLICY_SEMI)

    results: List[dict] = []
    total = len(SECTIONS) * len(seeds)
    idx = 0
    for sid, ptype in SECTIONS.items():
        esub = get_esub(df, sid)
        if esub is None:
            logger.error(f"  [{sid}] not found in subgrade table; skipping.")
            results.append({"section_id": sid, "pavement_type": ptype, "status": "no_subgrade"})
            continue
        model = model_flex if ptype == "flexible" else model_semi
        for seed in seeds:
            idx += 1
            logger.info(f"[{idx}/{total}] {sid} ({ptype}) E_sub={esub:.0f} seed={seed}")
            try:
                row = run_one_section(sid, ptype, esub, model, b3_threshold, seed)
                results.append(row)
                logger.info(f"  [{sid}] eps_a={_fmt(row['eps_a_micro'])} "
                            f"eps_z={_fmt(row['eps_z_micro'])} "
                            f"({row.get('FEA_elapsed_s', 0)}s) [{row['status']}]")
            except Exception as e:
                logger.exception(f"  [{sid}] seed={seed} FAILED: {e}")
                results.append({"section_id": sid, "pavement_type": ptype,
                                "seed": seed, "status": f"error: {e}"})

    out_dir = Path("experiments/ltpp_data/deliverables/mepdg_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"hara_fea_responses_{ts}.csv"
    cols = ["section_id", "pavement_type", "seed", "E_subgrade",
            "eps_a_micro", "eps_z_micro", "sigma_t_MPa",
            "p_AC_upper_mid", "p_AC_mid_mid", "p_AC_lower_mid", "deflection_mm",
            "h1_cm", "h2_cm", "h3_cm", "h4_cm", "h5_cm",
            "E1_MPa", "E2_MPa", "E3_MPa", "E4_MPa", "E5_MPa",
            "FEA_elapsed_s", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info(f"DONE. {ok}/{len(results)} rows OK. CSV: {csv_path}")
    if ok < len(results):
        logger.warning("Some rows are not OK — inspect the 'status' column before "
                       "feeding this CSV to run_mepdg_cross_spec_check.py.")


if __name__ == "__main__":
    main()
