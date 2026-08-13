# -*- coding: utf-8 -*-
"""
scripts/run_riohtrack_hara_inference.py
========================================
HARA inference on RIOHTRACK 17 structures (excl. STR4/5 rigid_inverted).

Uses trained PPO policies:
  - A v3 (flexible, 2048ts) for STR18 (GPS-1 granular base)
  - B (semi_rigid, 1024ts) for STR1-3, STR6-17, STR19 (GPS-2 stabilised base)

Each structure: 1 reset + 20 deterministic steps → SCR/DSR/NPV.

Output: CSV to experiments/ltpp_data/deliverables/riohtrack/

Usage:
    python scripts/run_riohtrack_hara_inference.py
"""
from __future__ import annotations

import json, logging, sys, time, os, csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("riohtrack_hara")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PPO checkpoint paths
POLICY_FLEX = Path("output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip")
POLICY_SEMI = Path("output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip")
SURROGATE_MODEL = "output/surrogate_model/surrogate_v3.pt"

_DEFAULT_5L = {"flexible":     [0.04, 0.06, 0.08, 0.30, 0.25],
               "semi_rigid":   [0.04, 0.06, 0.08, 0.36, 0.18]}
_DEFAULT_MOD = {"flexible":    [14000, 11000, 9000, 1500, 400],
                "semi_rigid":  [14000, 11000, 9000, 1500, 400]}
_DEFAULT_POISSON = [0.25, 0.30, 0.30, 0.25, 0.35]

CITY = "beijing"
MAX_STEPS = 20
SEED = 0


def load_structures(xlsx_path: str) -> List[Dict]:
    df = pd.read_excel(xlsx_path, sheet_name="merged_structures_FWD")
    structs = []
    for _, row in df.iterrows():
        sid = str(row["structure_id"])
        bt = str(row.get("base_type", ""))
        if bt == "rigid_inverted":
            continue
        pavtype = "flexible" if bt == "flexible" else "semi_rigid"

        # Use default 5-layer design for HARA initialization
        h5 = list(_DEFAULT_5L[pavtype])
        mod5 = list(_DEFAULT_MOD[pavtype])
        esub = 80.0  # typical RIOHTRACK subgrade

        structs.append({
            "structure_id": sid, "base_type": bt, "pavement_type": pavtype,
            "thickness_m": h5, "modulus_MPa": mod5, "poisson": list(_DEFAULT_POISSON),
            "E_subgrade": esub, "nu_subgrade": 0.40,
        })
    return structs


def run_one(section: dict, policy) -> dict:
    """Run one inference episode."""
    from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig
    from rl.lifecycle_lcc_intl import lcc_npv_usd
    from rl import metrics as _metrics

    sid = section["structure_id"]
    cfg = SurrogateEnvConfig(
        pavement_type=section["pavement_type"],
        init_thickness_m=section["thickness_m"],
        init_modulus_MPa=section["modulus_MPa"],
        init_poisson=section["poisson"],
        E_subgrade=section["E_subgrade"],
        nu_subgrade=section["nu_subgrade"],
        city=CITY, road_class="expressway", traffic_level="heavy",
        design_life_years=15,
        max_episode_steps=MAX_STEPS, max_episodes=1,
        llm_enabled=False, fea_keep_runs=False, fea_verbose=False,
        enable_lcc_eval=True, design_life_years_lcc=20.0,
        use_surrogate=True,
        surrogate_model_path=SURROGATE_MODEL,
    )

    env = PavementEnvWithSurrogate(cfg)
    env.set_fea_output_dir(Path(PROJECT_ROOT))

    t0 = time.time()
    obs, info = env.reset(seed=SEED)
    margins_hist, rewards = [], []

    for step in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        m = {k: float(v) for k, v in info.get("margins", {}).items()}
        margins_hist.append(m)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    # Final metrics
    final_margins = margins_hist[-1] if margins_hist else {}
    dsr = _metrics.compute_dsr(final_margins)
    scr_running = info.get("scr_running", None)
    if scr_running is None:
        # Manual SCR from final margins
        scr_running = sum(1 for v in final_margins.values() if v >= 1.0) / max(len(final_margins), 1)

    # LCC on final design
    thk = [float(h) for h in env._design["thickness"]]
    cny = [1800, 1100, 900,
           100 if section["pavement_type"] == "flexible" else 320,
           80 if section["pavement_type"] == "flexible" else 180]
    C_cny = sum(cny[i] * thk[i] for i in range(5))
    lcc = lcc_npv_usd(C_construction_usd_per_m2=C_cny/7.20,
                      design_life_years=20.0,
                      margin_B1=final_margins.get("B1_asphalt_fatigue", 99),
                      margin_B2=final_margins.get("B2_semi_rigid_fatigue", 99),
                      discount_rate=0.04)

    env.close()
    return {
        "structure_id": sid, "base_type": section["base_type"],
        "pavement_type": section["pavement_type"],
        "n_steps": len(rewards), "total_reward": round(sum(rewards), 4),
        "final_dsr": round(dsr, 4),
        "final_scr": round(scr_running, 4),
        "B1": round(final_margins.get("B1_asphalt_fatigue", 0), 2),
        "B2": round(final_margins.get("B2_semi_rigid_fatigue", 0), 2),
        "B3": round(final_margins.get("B3_ac_permanent_deformation", 0), 2),
        "B4": round(final_margins.get("B4_subgrade_strain", 0), 2),
        "NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
        "C_const_cny": round(C_cny, 1),
        "wall_clock_sec": round(time.time() - t0, 1),
        "status": "ok",
    }


def main():
    from stable_baselines3 import PPO

    xlsx = "data/RIOHTRACK_19_structures.xlsx"
    sections = load_structures(xlsx)
    logger.info(f"Loaded {len(sections)} structures (excl. rigid_inverted)")

    # Load policies
    logger.info(f"Loading flex policy: {POLICY_FLEX}")
    policy_flex = PPO.load(POLICY_FLEX)
    logger.info(f"Loading semi policy: {POLICY_SEMI}")
    policy_semi = PPO.load(POLICY_SEMI)

    results = []
    for i, sec in enumerate(sections):
        sid = sec["structure_id"]
        policy = policy_flex if sec["pavement_type"] == "flexible" else policy_semi
        r = run_one(sec, policy)
        results.append(r)
        if r["status"] == "ok":
            logger.info(f"  [{i+1}/{len(sections)}] {sid}: DSR={r['final_dsr']:.2f} "
                        f"SCR={r['final_scr']:.2f} NPV=${r['NPV_usd']:.1f} "
                        f"B3={r['B3']:.2f} ({r['wall_clock_sec']:.0f}s)")
        else:
            logger.error(f"  [{i+1}/{len(sections)}] {sid}: FAILED")

    out_dir = Path("experiments/ltpp_data/deliverables/riohtrack")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"riohtrack_hara_{ts}.csv"
    cols = ["structure_id","base_type","pavement_type","n_steps","total_reward",
            "final_dsr","final_scr","B1","B2","B3","B4",
            "NPV_usd","C_const_cny","wall_clock_sec","status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"DONE. {ok}/{len(results)} ok. CSV: {csv_path}")


if __name__ == "__main__":
    main()