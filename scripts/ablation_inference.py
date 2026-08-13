# -*- coding: utf-8 -*-
"""
scripts/ablation_inference.py — Ablation variant inference on 12 LTPP sections.

For each ablation variant (full/no-generator/no-rag/no-generator-no-rag/no-language-no-guard/no-guard) × base_type × seed,
loads the trained checkpoint and runs deterministic inference on the appropriate
LTPP sections (flexible on GPS-1, semi_rigid on GPS-2).

Outputs:
  ablation_inference_summary.csv   — per-section results
  ablation_table2.csv              — Table 2: variant × type mean ± sd

Usage:
  python scripts/ablation_inference.py
  python scripts/ablation_inference.py --seeds 0,1,2 --only-variant no-guard
"""
from __future__ import annotations

import argparse, csv, json, logging, os, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ablation_inference")

PROJECT = Path(__file__).resolve().parent.parent
SECTIONS_XLSX = PROJECT / "experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx"
SURROGATE_MODEL = PROJECT / "output/surrogate_model/surrogate_v3.pt"
CHECKPOINTS_ROOT = PROJECT / "output/rl_runs"
OUT_DIR = PROJECT / "experiments/ltpp_data/deliverables/ablation_inference"

VARIANTS = ["full", "no-generator", "no-rag", "no-generator-no-rag", "no-language-no-guard", "no-guard"]
BASE_TYPES = ["flexible", "semi_rigid"]
SEEDS_DEFAULT = [0, 1, 2]
TIMESTEPS = 1000
MAX_STEPS = 20

_DEFAULT_INIT = {
    "semi_rigid": {"thickness": [0.04,0.06,0.08,0.36,0.18], "modulus": [14000,11000,9000,1500,400], "poisson": [0.25,0.30,0.30,0.25,0.35]},
    "flexible":   {"thickness": [0.04,0.06,0.08,0.30,0.25], "modulus": [14000,11000,9000,350,250],   "poisson": [0.25,0.30,0.30,0.40,0.35]},
}

_GPS_FAMILY_FALLBACK = {
    "04_1034":"GPS-1","12_1060":"GPS-1","16_1010":"GPS-1","27_1085":"GPS-1","48_0001":"GPS-1","48_1076":"GPS-1",
    "04_1065":"GPS-2","06_2004":"GPS-2","12_4097":"GPS-2","27_2023":"GPS-2","30_7076":"GPS-2","48_1109":"GPS-2",
}


def load_sections() -> List[Dict[str, Any]]:
    df = pd.read_excel(SECTIONS_XLSX, sheet_name=0)
    sections = []
    for _, row in df.iterrows():
        sid = str(row.get("section_id", row.get("SHRP_ID", ""))).strip()
        esub = float(row.get("E_subgrade_MPa", row.get("E_subgrade", 0)))
        gps = str(row.get("gps_family", _GPS_FAMILY_FALLBACK.get(sid, ""))).strip()
        if gps not in ("GPS-1","GPS-2"):
            gps = _GPS_FAMILY_FALLBACK.get(sid, "")
        if not sid or not gps:
            continue
        sections.append({"section_id": sid, "E_subgrade": esub, "gps_family": gps,
                         "pavement_type": "flexible" if gps == "GPS-1" else "semi_rigid"})
    logger.info(f"Loaded {len(sections)} sections")
    return sections


def find_checkpoint(variant: str, base_type: str, seed: int) -> Path:
    run_dir = CHECKPOINTS_ROOT / f"ablation_{variant}_{base_type}_{TIMESTEPS}ts_seed{seed}"
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")
    finals = list(ckpt_dir.glob("ckpt_final_step_*"))
    if finals:
        zip_path = finals[0] / "ppo_model.zip"
        if zip_path.exists():
            return zip_path
    raise FileNotFoundError(f"No final checkpoint in {ckpt_dir}")


def run_inference(section: dict, policy, seed: int, variant: str) -> dict:
    """Run one episode, return best-compliant design summary."""
    from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig

    pt = section["pavement_type"]
    init = _DEFAULT_INIT[pt]
    cfg = SurrogateEnvConfig(
        pavement_type=pt, E_subgrade=float(section["E_subgrade"]),
        init_thickness_m=list(init["thickness"]),
        init_modulus_MPa=list(init["modulus"]),
        init_poisson=list(init["poisson"]),
        llm_enabled=False, use_surrogate=True,
        surrogate_model_path=str(SURROGATE_MODEL),
        surrogate_b3_threshold=1.0,
        fea_validation_every=9999,  # skip FEA validation for speed
        max_episode_steps=MAX_STEPS,
        enable_lcc_eval=True, design_life_years_lcc=20.0,
    )
    if variant in ("no-guard", "no-language-no-guard"):
        cfg.guard_enabled = False
    env = PavementEnvWithSurrogate(cfg)
    obs, info = env.reset(seed=int(seed))

    # Track all states for best-compliant selection
    states = []
    if info.get("compliant", False):
        cost0 = (info.get("lcc") or {}).get("C_construction_usd_per_m2", 0)
        states.append({"step": 0, "cost": cost0, "dsr": info.get("dsr", 1.0),
                       "scr": info.get("scr_running", 1.0),
                       "h": info.get("design_h_cm", []),
                       "compliant": True})  # step 0 was already compliant-checked

    n_fea_failures = 0
    for t in range(MAX_STEPS):
        try:
            action, _ = policy.predict(obs, deterministic=True)
        except Exception:
            break
        try:
            step_out = env.step(action)
        except Exception:
            break
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
        compliant = bool(info.get("compliant", False))
        if str(info.get("critical", "")) == "FEA_FAILURE" or "FEA_FAILURE" in (info.get("margins") or {}):
            n_fea_failures += 1
        cost = (info.get("lcc") or {}).get("C_construction_usd_per_m2", 0)
        states.append({"step": t + 1, "cost": cost,
                       "dsr": info.get("dsr", 0),
                       "scr": info.get("scr_running", 0),
                       "h": info.get("design_h_cm", []),
                       "compliant": compliant})
    env.close()

    # Best-compliant selection: lowest cost among compliant states
    compliant_states = [s for s in states if s.get("compliant", False)]
    if compliant_states:
        best = min(compliant_states, key=lambda s: s["cost"])
    else:
        best = states[-1] if states else {"cost": 0, "dsr": 0, "scr": 0, "h": [], "step": -1}

    return {
        "section_id": section["section_id"],
        "pavement_type": pt,
        "delivered_dsr": best["dsr"],
        "delivered_cost": best["cost"],
        "delivered_step": best["step"],
        "episode_scr": sum(1 for s in states if s.get("compliant")) / max(len(states), 1),
        "n_states": len(states),  # should be 21 (step 0 + 20 steps)
        "episode_guard_violations": int(info.get("n_guard_violations", 0)) if "info" in locals() else 0,
        "episode_fea_failures": int(n_fea_failures),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation inference on 12 LTPP sections")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--only-variant", type=str, default=None, choices=VARIANTS)
    parser.add_argument("--only-type", type=str, default=None, choices=BASE_TYPES)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    variants = [args.only_variant] if args.only_variant else VARIANTS
    types = [args.only_type] if args.only_type else BASE_TYPES

    sections = load_sections()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(variants) * len(types) * len(seeds) * len([s for s in sections if s["pavement_type"] in types])
    done = 0

    for variant in variants:
        for base_type in types:
            for seed in seeds:
                try:
                    ckpt = find_checkpoint(variant, base_type, seed)
                except FileNotFoundError as e:
                    logger.warning(f"Skip {variant}/{base_type}/seed{seed}: {e}")
                    continue

                logger.info(f"Loading {variant}/{base_type}/seed{seed}: {ckpt}")
                from stable_baselines3 import PPO
                policy = PPO.load(str(ckpt), device="cpu")

                my_sections = [s for s in sections if s["pavement_type"] == base_type]
                for sec in my_sections:
                    done += 1
                    logger.info(f"[{done}/{total}] {variant}/{base_type}/s{seed} {sec['section_id']}")
                    try:
                        r = run_inference(sec, policy, seed, variant)
                        r["variant"] = variant
                        r["base_type"] = base_type
                        r["seed"] = seed
                        all_results.append(r)
                        logger.info(f"  DSR={r['delivered_dsr']:.3f} cost={r['delivered_cost']:.1f}")
                    except Exception as e:
                        logger.error(f"FAILED: {e}")
                        all_results.append({"variant": variant, "base_type": base_type, "seed": seed,
                                            "section_id": sec["section_id"], "status": "failed", "error": str(e)})

    # Save per-section CSV
    if all_results:
        keys = list(all_results[0].keys())
        suffix_parts = []
        if args.only_variant:
            suffix_parts.append(args.only_variant)
        if args.only_type:
            suffix_parts.append(args.only_type)
        if args.seeds != "0,1,2":
            suffix_parts.append("seeds" + args.seeds.replace(",", "-"))
        suffix = ("__" + "__".join(suffix_parts)) if suffix_parts else ""
        per_run_csv = OUT_DIR / f"ablation_inference_summary{suffix}.csv"
        with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        logger.info(f"Saved: {per_run_csv}")

        # Aggregate to Table 2
        ok = [r for r in all_results if r.get("status") != "failed"]
        print("\n" + "=" * 80)
        print("Table 2 — Component ablation (mean ± sd over sections × seeds)")
        print("=" * 80)
        print(f"{'Variant':15s} {'Type':11s} {'DSR':>8s}  {'SCR':>8s}  {'Cost USD/m²':>14s}  {'N':>3s}")
        print("-" * 80)

        table2_rows = []
        for variant in VARIANTS:
            for bt in BASE_TYPES:
                subset = [r for r in ok if r["variant"] == variant and r["base_type"] == bt]
                if len(subset) < 2:
                    continue
                dsr_vals = [r["delivered_dsr"] for r in subset]
                scr_vals = [r["episode_scr"] for r in subset]
                cost_vals = [r["delivered_cost"] for r in subset]
                print(f"{variant:15s} {bt:11s} {np.mean(dsr_vals):7.3f}±{np.std(dsr_vals, ddof=1):.3f}  "
                      f"{np.mean(scr_vals):7.3f}±{np.std(scr_vals, ddof=1):.3f}  "
                      f"{np.mean(cost_vals):7.1f}±{np.std(cost_vals, ddof=1):.1f}  {len(subset):3d}")
                table2_rows.append({
                    "variant": variant, "base_type": bt,
                    "DSR_mean": round(np.mean(dsr_vals), 3), "DSR_sd": round(np.std(dsr_vals, ddof=1), 3),
                    "SCR_mean": round(np.mean(scr_vals), 3), "SCR_sd": round(np.std(scr_vals, ddof=1), 3),
                    "cost_mean": round(np.mean(cost_vals), 1), "cost_sd": round(np.std(cost_vals, ddof=1), 1),
                    "n": len(subset),
                })

        table2_csv = OUT_DIR / f"ablation_table2{suffix}.csv"
        with open(table2_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["variant","base_type","DSR_mean","DSR_sd","SCR_mean","SCR_sd","cost_mean","cost_sd","n"])
            w.writeheader()
            w.writerows(table2_rows)
        logger.info(f"Saved: {table2_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()