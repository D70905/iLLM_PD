# -*- coding: utf-8 -*-
"""
scripts/run_riohtrack_hara_inference_v2.py
============================================
HARA inference on RIOHTRACK 17 structures (excl. STR4/5 rigid_inverted).

CHANGES from v1:
  [v2-1] Read REAL initial structure from RIOHTRACK xlsx (per-structure)
  [v2-2] Default mode = surrogate_only (no ABAQUS contention with ablation)
  [v2-3] --mode {surrogate_only | fea_verified} CLI switch
  [v2-4] Robust env_surrogate import with field-name fallback
  [v2-5] Map N-layer RIOHTRACK to 5-layer using material category
  [v2-6] Save per-step trajectory (not just final) for inspection

Usage:
    # Safe (no ABAQUS, fast, ~30 min)
    python scripts/run_riohtrack_hara_inference_v2.py --mode surrogate_only

    # Full (B3<1.0 escalates to ABAQUS, ~2 hours, AVOID during ablation)
    python scripts/run_riohtrack_hara_inference_v2.py --mode fea_verified
"""
from __future__ import annotations

import argparse, json, logging, sys, time, os, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("riohtrack_hara_v2")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PPO checkpoint paths (verified in previous discussions)
POLICY_FLEX = Path("output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip")
POLICY_SEMI = Path("output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip")
SURROGATE_MODEL = "output/surrogate_model/surrogate_v3.pt"

# Fallback initial moduli when xlsx doesn't have design moduli
_FALLBACK_MOD = {
    "flexible":   [14000, 11000, 9000, 400,  200],   # last 2 = granular
    "semi_rigid": [14000, 11000, 9000, 1500, 400],   # last 2 = stabilised
}
_DEFAULT_POISSON = [0.25, 0.30, 0.30, 0.25, 0.35]

CITY = "beijing"
MAX_STEPS = 20
SEED = 0


# ============================================================================
# SECTION 1 — Map N-layer RIOHTRACK structure to HARA's 5-layer model
# ============================================================================

def categorize_layer(layer_material: str, layer_category: str = "") -> str:
    """Categorize a layer into AC / stabilised_base / granular / subgrade."""
    s = (str(layer_material) + " " + str(layer_category)).lower()
    if any(k in s for k in ["ac", "asphalt", "sbs", "sma", "sup", "ogfc", "pac"]):
        return "AC"
    if any(k in s for k in ["cement", "stabili", "lcc", "cc", "cs", "ctb",
                             "lean", "semi-rigid"]):
        return "stabilised_base"
    if any(k in s for k in ["graded", "gravel", "crushed", "granular", "gb"]):
        return "granular"
    return "other"


def map_to_5_layer(row: pd.Series, n_layers: int,
                   pavtype: str) -> Tuple[List[float], List[float], List[float]]:
    """Aggregate N-layer RIOHTRACK structure to 5-layer for HARA.

    Returns:
        thickness_m (5,), modulus_MPa (5,), poisson (5,)
    """
    ac_layers, base_layers, sub_layers = [], [], []
    for j in range(1, n_layers + 1):
        mat = row.get(f"L{j}_material", "")
        cat = row.get(f"L{j}_category", "")
        h_cm = float(row.get(f"L{j}_thickness_cm", 0))
        e_mpa = float(row.get(f"L{j}_modulus_MPa_design",
                              row.get(f"L{j}_modulus_MPa", 0)))
        if h_cm <= 0:
            continue

        layer_cat = categorize_layer(mat, cat)
        entry = {"h_cm": h_cm, "E_MPa": e_mpa}
        if layer_cat == "AC":
            ac_layers.append(entry)
        elif layer_cat == "stabilised_base":
            base_layers.append(entry)
        elif layer_cat == "granular":
            sub_layers.append(entry)
        else:
            sub_layers.append(entry)  # default to lower

    # Total AC thickness
    h_ac_total = sum(l["h_cm"] for l in ac_layers) if ac_layers else 18.0
    # AC weighted-average modulus
    e_ac_mean = (sum(l["E_MPa"] * l["h_cm"] for l in ac_layers) / h_ac_total
                 if ac_layers and h_ac_total > 0 else 11000.0)
    e_ac_mean = max(e_ac_mean, 8000.0)

    # AC split: 25/35/40 (upper/mid/lower) — standard HARA partition
    h1 = h_ac_total * 0.25
    h2 = h_ac_total * 0.35
    h3 = h_ac_total * 0.40

    # Base layer (4th HARA layer)
    if base_layers:
        h4 = sum(l["h_cm"] for l in base_layers)
        e4 = (sum(l["E_MPa"] * l["h_cm"] for l in base_layers) / h4
              if h4 > 0 else _FALLBACK_MOD[pavtype][3])
    else:
        h4 = 36.0 if pavtype == "semi_rigid" else 30.0
        e4 = _FALLBACK_MOD[pavtype][3]

    # Subbase / lower granular (5th HARA layer)
    if sub_layers:
        h5 = sum(l["h_cm"] for l in sub_layers)
        e5 = (sum(l["E_MPa"] * l["h_cm"] for l in sub_layers) / h5
              if h5 > 0 else _FALLBACK_MOD[pavtype][4])
    else:
        h5 = 18.0 if pavtype == "semi_rigid" else 25.0
        e5 = _FALLBACK_MOD[pavtype][4]

    # Clip to HARA bounds (consistent with NumericalGuard ranges)
    h_min = [2.0, 3.0, 4.0, 15.0, 10.0]
    h_max = [10.0, 15.0, 25.0, 50.0, 40.0]
    h_list = [h1, h2, h3, h4, h5]
    h_list = [max(min(h, h_max[i]), h_min[i]) for i, h in enumerate(h_list)]
    h_m = [h / 100.0 for h in h_list]

    # AC modulus gradient: upper > mid > lower
    e_list = [
        max(e_ac_mean * 1.20, 10000),  # upper higher modulus
        e_ac_mean,
        max(e_ac_mean * 0.85, 8000),   # lower softer
        e4,
        e5,
    ]

    return h_m, e_list, list(_DEFAULT_POISSON)


# ============================================================================
# SECTION 2 — Robust environment configuration
# ============================================================================

def make_env(section: dict, mode: str):
    """Create env with field-name fallback for SurrogateEnvConfig.

    Returns: env instance, or None if env_surrogate not importable.
    """
    try:
        from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig
    except ImportError as e:
        logger.error(f"Cannot import env_surrogate: {e}")
        return None

    # Build config kwargs with both possible field-name conventions
    common_kwargs = dict(
        pavement_type=section["pavement_type"],
        E_subgrade=section["E_subgrade"],
        nu_subgrade=section["nu_subgrade"],
        city=CITY,
        road_class="expressway",
        traffic_level="heavy",
        design_life_years=15,
        max_episode_steps=MAX_STEPS,
        max_episodes=1,
        llm_enabled=False,
        fea_keep_runs=False,
        fea_verbose=False,
        enable_lcc_eval=True,
        design_life_years_lcc=20.0,
        use_surrogate=True,
        surrogate_model_path=SURROGATE_MODEL,
    )

    # Surrogate B3 threshold based on mode
    if mode == "surrogate_only":
        # Disable FEA escalation: set threshold below physical minimum
        common_kwargs["surrogate_b3_threshold"] = -1.0
    elif mode == "fea_verified":
        common_kwargs["surrogate_b3_threshold"] = 1.0

    # Try field-name variants for initial structure
    init_kwargs_variants = [
        dict(init_thickness_m=section["thickness_m"],
             init_modulus_MPa=section["modulus_MPa"],
             init_poisson=section["poisson"]),
        dict(initial_thickness_m=section["thickness_m"],
             initial_modulus_MPa=section["modulus_MPa"],
             initial_poisson=section["poisson"]),
        dict(thickness_m=section["thickness_m"],
             modulus_MPa=section["modulus_MPa"],
             poisson=section["poisson"]),
    ]

    env = None
    last_err = None
    for ikw in init_kwargs_variants:
        try:
            cfg = SurrogateEnvConfig(**common_kwargs, **ikw)
            env = PavementEnvWithSurrogate(cfg)
            break
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            logger.warning(f"Env construction with {list(ikw.keys())[0]}: {e}")
            last_err = e
            continue

    if env is None:
        logger.error(f"All env config variants failed. Last error: {last_err}")
        return None

    try:
        env.set_fea_output_dir(Path(PROJECT_ROOT))
    except AttributeError:
        pass  # set_fea_output_dir may not exist in older versions

    return env


# ============================================================================
# SECTION 3 — Per-structure inference
# ============================================================================

def run_one(section: dict, policy, mode: str) -> dict:
    """Run one deterministic inference episode."""
    from rl.lifecycle_lcc_intl import lcc_npv_usd
    from rl import metrics as _metrics

    sid = section["structure_id"]
    env = make_env(section, mode)
    if env is None:
        return {"structure_id": sid, "status": "env_failed"}

    t0 = time.time()
    obs, info = env.reset(seed=SEED)
    margins_hist, rewards = [], []
    trajectory = []

    for step in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        m = {k: float(v) for k, v in info.get("margins", {}).items()}
        margins_hist.append(m)
        rewards.append(float(reward))

        # Record trajectory snapshot
        thk = info.get("thickness", None) or info.get("design", {}).get("thickness", None)
        trajectory.append({
            "step": step,
            "reward": float(reward),
            "B1": m.get("B1_asphalt_fatigue", None),
            "B2": m.get("B2_semi_rigid_fatigue", None),
            "B3": m.get("B3_ac_permanent_deformation", None),
            "B4": m.get("B4_subgrade_strain", None),
            "feasible": info.get("feasible", None),
            "thickness_cm": [round(h * 100, 2) for h in thk] if thk else None,
        })
        if terminated or truncated:
            break

    # ── Final metrics ──────────────────────────────────────────
    final_margins = margins_hist[-1] if margins_hist else {}
    dsr = _metrics.compute_dsr(final_margins)
    scr_running = info.get("scr_running", None)
    if scr_running is None:
        is_compliant = _metrics.compute_compliance(final_margins)
        scr_running = (1.0 if is_compliant
                       else sum(1 for v in final_margins.values() if v >= 1.0)
                            / max(len(final_margins), 1))

    # Compute LCC on final design
    try:
        thk_final = [float(h) for h in env._design["thickness"]]
    except (AttributeError, KeyError, TypeError):
        thk_final = section["thickness_m"]

    cny_prices = [1800, 1100, 900,
                  100 if section["pavement_type"] == "flexible" else 320,
                  80  if section["pavement_type"] == "flexible" else 180]
    C_cny = sum(cny_prices[i] * thk_final[i] for i in range(5))
    try:
        lcc = lcc_npv_usd(
            C_construction_usd_per_m2=C_cny / 7.20,
            design_life_years=20.0,
            margin_B1=final_margins.get("B1_asphalt_fatigue", 99),
            margin_B2=final_margins.get("B2_semi_rigid_fatigue", 99),
            discount_rate=0.04,
        )
        npv = round(lcc.get("NPV_total_usd_m2", 0), 2)
    except Exception as e:
        logger.warning(f"[{sid}] LCC failed: {e}")
        npv = None

    try:
        env.close()
    except Exception:
        pass

    return {
        "structure_id": sid,
        "base_type": section["base_type"],
        "pavement_type": section["pavement_type"],
        "n_steps": len(rewards),
        "total_reward": round(sum(rewards), 4),
        "final_dsr": round(dsr, 4),
        "final_scr": round(scr_running, 4),
        "B1": round(final_margins.get("B1_asphalt_fatigue", 0), 2),
        "B2": round(final_margins.get("B2_semi_rigid_fatigue", 0), 2),
        "B3": round(final_margins.get("B3_ac_permanent_deformation", 0), 2),
        "B4": round(final_margins.get("B4_subgrade_strain", 0), 2),
        "init_h1_cm": round(section["thickness_m"][0] * 100, 1),
        "init_h4_cm": round(section["thickness_m"][3] * 100, 1),
        "final_h1_cm": round(thk_final[0] * 100, 1),
        "final_h2_cm": round(thk_final[1] * 100, 1),
        "final_h3_cm": round(thk_final[2] * 100, 1),
        "final_h4_cm": round(thk_final[3] * 100, 1),
        "final_h5_cm": round(thk_final[4] * 100, 1),
        "NPV_usd": npv,
        "C_const_cny": round(C_cny, 1),
        "wall_clock_sec": round(time.time() - t0, 1),
        "trajectory": trajectory,
        "status": "ok",
    }


# ============================================================================
# SECTION 4 — Section loader (uses REAL initial structure from xlsx)
# ============================================================================

def load_structures(xlsx_path: str) -> List[Dict]:
    """Load RIOHTRACK structures with REAL per-section initial design."""
    df = pd.read_excel(xlsx_path, sheet_name="merged_structures_FWD")
    structs = []
    for _, row in df.iterrows():
        sid = str(row["structure_id"])
        bt = str(row.get("base_type", "")).lower()
        if "inverted" in bt or "rigid" in bt:
            logger.info(f"  [{sid}] Skipped (rigid_inverted)")
            continue
        pavtype = "flexible" if "flexible" in bt else "semi_rigid"

        try:
            n_layers = int(row.get("n_layers", 5))
        except (ValueError, TypeError):
            n_layers = 5

        # Map real N-layer structure to 5-layer HARA model
        h5_m, e5_mpa, nu5 = map_to_5_layer(row, n_layers, pavtype)

        # E_subgrade: use FWD-backcalculated or default 80 MPa
        try:
            esub = float(row.get("E_subgrade_FWD_MPa",
                                 row.get("E_subgrade_MPa", 80.0)))
        except (ValueError, TypeError):
            esub = 80.0
        esub = max(20.0, min(esub, 500.0))  # safety clamp

        structs.append({
            "structure_id": sid,
            "base_type": bt,
            "pavement_type": pavtype,
            "thickness_m": h5_m,
            "modulus_MPa": e5_mpa,
            "poisson": nu5,
            "E_subgrade": esub,
            "nu_subgrade": 0.40,
            "n_layers_orig": n_layers,
        })
    return structs


# ============================================================================
# SECTION 5 — Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["surrogate_only", "fea_verified"],
                        default="surrogate_only",
                        help="surrogate_only = no ABAQUS (safe during ablation); "
                             "fea_verified = B3<1.0 triggers ABAQUS (DON'T use during ablation)")
    parser.add_argument("--xlsx",
                        default="data/RIOHTRACK_19_structures.xlsx")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N structures (for testing)")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    sections = load_structures(args.xlsx)
    if args.limit:
        sections = sections[:args.limit]
    logger.info(f"Loaded {len(sections)} structures (excl. rigid_inverted)")
    logger.info(f"Mode: {args.mode}")
    if args.mode == "fea_verified":
        logger.warning("⚠ fea_verified mode will compete with ablation for ABAQUS license!")
        logger.warning("⚠ Recommended: use --mode surrogate_only during ablation")

    # Verify policies exist
    if not POLICY_FLEX.exists():
        logger.error(f"Flex policy not found: {POLICY_FLEX}")
        return
    if not POLICY_SEMI.exists():
        logger.error(f"Semi policy not found: {POLICY_SEMI}")
        return

    policy_flex = PPO.load(POLICY_FLEX)
    policy_semi = PPO.load(POLICY_SEMI)
    logger.info("PPO policies loaded")

    results = []
    for i, sec in enumerate(sections):
        sid = sec["structure_id"]
        policy = policy_flex if sec["pavement_type"] == "flexible" else policy_semi
        logger.info(f"  [{i+1}/{len(sections)}] {sid} "
                    f"({sec['pavement_type']}, init_h_AC={sec['thickness_m'][0]*100:.0f}+"
                    f"{sec['thickness_m'][1]*100:.0f}+{sec['thickness_m'][2]*100:.0f}cm, "
                    f"E_sub={sec['E_subgrade']:.0f} MPa)")
        r = run_one(sec, policy, mode=args.mode)
        results.append(r)
        if r["status"] == "ok":
            logger.info(f"    DSR={r['final_dsr']:.2f} SCR={r['final_scr']:.2f} "
                        f"NPV=${r['NPV_usd']} B3={r['B3']:.2f} "
                        f"final_h_AC={r['final_h1_cm']+r['final_h2_cm']+r['final_h3_cm']:.1f}cm "
                        f"({r['wall_clock_sec']:.0f}s)")
        else:
            logger.error(f"    FAILED: {r.get('error', 'unknown')}")

    # ── Save outputs ───────────────────────────────────────────
    out_dir = Path("experiments/ltpp_data/deliverables/riohtrack")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Summary CSV
    csv_path = out_dir / f"riohtrack_hara_v2_{ts}.csv"
    cols = ["structure_id", "base_type", "pavement_type", "n_steps", "total_reward",
            "final_dsr", "final_scr",
            "B1", "B2", "B3", "B4",
            "init_h1_cm", "init_h4_cm",
            "final_h1_cm", "final_h2_cm", "final_h3_cm", "final_h4_cm", "final_h5_cm",
            "NPV_usd", "C_const_cny", "wall_clock_sec", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    logger.info(f"Summary CSV: {csv_path}")

    # Full trajectory JSON (per-step)
    json_path = out_dir / f"riohtrack_hara_v2_trajectories_{ts}.json"
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "trajectory"} | {
            "trajectory": r.get("trajectory", [])} for r in results], f, indent=2)
    logger.info(f"Trajectory JSON: {json_path}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*70}")
    print(f"RIOHTRACK HARA v2 ({args.mode}): {ok}/{len(results)} ok")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
