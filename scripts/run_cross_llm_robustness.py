# -*- coding: utf-8 -*-
"""
scripts/run_cross_llm_robustness.py — R2-6 Cross-LLM Robustness Test
======================================================================

Tests whether HARA produces consistent final designs when the Generator LLM
backend is changed. Runs the SAME inference scenario through multiple LLMs,
comparing final design parameters, SCR, DSR, and action distributions.

LLM backends:
    1. gpt-4o-mini (ChatFire) — current default Generator
    2. deepseek-reasoner — current Evaluator, tested as Generator
    3. qwen-plus (Alibaba) — alternative Chinese LLM
    4. No LLM (pure PPO baseline) — reference

For each backend, the same 3 representative LTPP sections are evaluated:
    - 16_1010 (Idaho, GPS-1 flexible)
    - 30_7076 (Montana, GPS-2 semi_rigid)
    - 48_1076 (Texas, GPS-1 flexible, baseline candidate)

Each section × backend combination runs 1 deterministic episode (20 steps),
producing a total of 3 sections × 4 backends = 12 inference runs.

Key metrics:
    - Final DSR / SCR / NPV
    - Layer thickness coefficient of variation (CV) across backends
    - Generator confidence distribution per backend
    - Evaluator (fixed to DeepSeek) scores across Generator backends

Usage:
    # Dry-run: print plan
    python scripts/run_cross_llm_robustness.py --dry-run

    # Full run (~2 hours, surrogate-accelerated)
    python scripts/run_cross_llm_robustness.py

    # Single section, single backend (debugging)
    python scripts/run_cross_llm_robustness.py --section 16_1010 --backend gpt

Output:
    experiments/ltpp_data/deliverables/cross_llm/cross_llm_summary_<ts>.csv
"""
from __future__ import annotations

import argparse, csv, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cross_llm")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────
POLICY_FLEX = Path("output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip")
POLICY_SEMI = Path("output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip")
SURROGATE_MODEL = "output/surrogate_model/surrogate_v3.pt"
MAX_STEPS, SEED = 20, 0

# Per-section JTG climate_zone mapping (MAAT from batch_climate_12sections_summary.csv)
_SECTION_JTG_ZONE = {"16_1010": "temperate", "30_7076": "temperate", "48_1076": "warm"}

SECTIONS = [
    {"section_id": "16_1010", "pavement_type": "flexible",   "E_subgrade": 78,
     "init_h": [0.04,0.06,0.08,0.30,0.25], "init_E": [14000,11000,9000,350,250],
     "init_nu": [0.25,0.30,0.30,0.35,0.35]},
    {"section_id": "30_7076", "pavement_type": "semi_rigid", "E_subgrade": 59,
     "init_h": [0.04,0.06,0.08,0.36,0.18], "init_E": [14000,11000,9000,1500,400],
     "init_nu": [0.25,0.30,0.30,0.25,0.35]},
    {"section_id": "48_1076", "pavement_type": "flexible",   "E_subgrade": 115,
     "init_h": [0.04,0.06,0.08,0.30,0.25], "init_E": [14000,11000,9000,350,250],
     "init_nu": [0.25,0.30,0.30,0.35,0.35]},
]

BACKENDS = {
    "gpt":      {"model": "gpt-4o-mini",          "backend": "chatfire"},
    "deepseek": {"model": "deepseek-reasoner",    "backend": "deepseek"},
    "qwen":     {"model": "Qwen2.5-72B-Instruct", "backend": "siliconflow-qwen"},
    "llama":    {"model": "llama3:latest",        "backend": "ollama-llama"},
    "ollama":   {"model": "qwen2.5:7b",           "backend": "ollama"},
    "none":     {"model": "none",                 "backend": "none"},
}


def make_env(section: dict, backend_key: str) -> Optional[Any]:
    """Create PavementEnvWithSurrogate with specified LLM backend."""
    from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig
    from rl.llm_client import get_client
    from rl.generator import Generator, GeneratorConfig
    from rl.rag import RAGStore

    bk = BACKENDS[backend_key]
    llm_enabled = (backend_key != "none")

    cfg = SurrogateEnvConfig(
        pavement_type=section["pavement_type"],
        init_thickness_m=list(section["init_h"]),
        init_modulus_MPa=list(section["init_E"]),
        init_poisson=list(section["init_nu"]),
        E_subgrade=section["E_subgrade"], nu_subgrade=0.40,
        city="", climate_zone=_SECTION_JTG_ZONE.get(section["section_id"], "temperate"),
        road_class="expressway", traffic_level="heavy",
        design_life_years=15, max_episode_steps=MAX_STEPS,
        max_episodes=1, llm_enabled=llm_enabled, fea_keep_runs=False,
        enable_lcc_eval=True, design_life_years_lcc=20.0,
        use_surrogate=True, surrogate_model_path=SURROGATE_MODEL,
        surrogate_b3_threshold=1.0,
    )

    env = PavementEnvWithSurrogate(cfg)
    env.set_fea_output_dir(Path(PROJECT_ROOT))

    # Configure Generator with the target backend (if LLM enabled)
    if llm_enabled and backend_key not in ("none",):
        try:
            client = get_client(bk["backend"])
            gen_cfg = GeneratorConfig(backend=bk["backend"])
            rag = RAGStore(persist_dir="./output/rag_db")
            env.config.generator = Generator(config=gen_cfg, rag=rag, audit=None, fail_fast=True)
            # Get the blend method
            env._generator_blend = getattr(type(env.config.generator), "blend", None)
            logger.info(f"  Generator: {bk['backend']} ({bk['model']})")
        except Exception as e:
            logger.warning(f"  Generator init failed for {backend_key}: {e}")
            env.config.generator = None
            env._generator_blend = None

    return env


def run_one(section: dict, backend: str) -> dict:
    """Run one inference episode."""
    from stable_baselines3 import PPO
    from rl.lifecycle_lcc_intl import lcc_npv_usd
    from rl import metrics as _metrics

    sid = section["section_id"]
    policy_path = POLICY_FLEX if section["pavement_type"] == "flexible" else POLICY_SEMI
    policy = PPO.load(policy_path)

    t0 = time.time()
    env = make_env(section, backend)
    if env is None:
        return {"section_id": sid, "backend": backend, "status": "env_failed"}

    obs, info = env.reset(seed=SEED)
    margins_hist, rewards, gen_confidences, eval_scores = [], [], [], []

    for step in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        m = {k: float(v) for k, v in info.get("margins", {}).items()}
        margins_hist.append(m)
        rewards.append(float(reward))
        gen_confidences.append(info.get("gen_confidence", 0.0))
        eval_scores.append(info.get("eval_score", None))
        if terminated or truncated:
            break

    final_m = margins_hist[-1] if margins_hist else {}
    dsr = _metrics.compute_dsr(final_m)
    scr = info.get("scr_running",
        sum(1 for v in final_m.values() if v >= 1.0) / max(len(final_m), 1))

    thk = [float(h) for h in env._design["thickness"]]
    ptype = section["pavement_type"]
    cny = [1800, 1100, 900, 100 if ptype == "flexible" else 320, 80 if ptype == "flexible" else 180]
    C_cny = sum(cny[i] * thk[i] for i in range(5))
    lcc = lcc_npv_usd(C_construction_usd_per_m2=C_cny/7.20, design_life_years=20.0,
                      margin_B1=final_m.get("B1_asphalt_fatigue", 99),
                      margin_B2=final_m.get("B2_semi_rigid_fatigue", 99),
                      discount_rate=0.04)

    gen_conf = np.mean([c for c in gen_confidences if c is not None]) if gen_confidences else None
    eval_mean = np.mean([s for s in eval_scores if s is not None]) if eval_scores else None

    env.close()
    return {
        "section_id": sid, "pavement_type": ptype, "backend": backend,
        "llm_model": BACKENDS[backend]["model"],
        "n_steps": len(rewards), "total_reward": round(sum(rewards), 4),
        "mean_reward": round(np.mean(rewards), 4),
        "final_dsr": round(dsr, 4), "final_scr": round(scr, 4),
        "B1": round(final_m.get("B1_asphalt_fatigue", 0), 2),
        "B2": round(final_m.get("B2_semi_rigid_fatigue", 0), 2),
        "B3": round(final_m.get("B3_ac_permanent_deformation", 0), 2),
        "B4": round(final_m.get("B4_subgrade_strain", 0), 2),
        "NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
        "h1_cm": round(thk[0]*100, 1), "h2_cm": round(thk[1]*100, 1),
        "h3_cm": round(thk[2]*100, 1), "h4_cm": round(thk[3]*100, 1),
        "h5_cm": round(thk[4]*100, 1),
        "mean_gen_confidence": round(gen_conf, 3) if gen_conf else None,
        "mean_eval_score": round(eval_mean, 1) if eval_mean else None,
        "wall_clock_sec": round(time.time() - t0, 1), "status": "ok",
    }


def compute_robustness_metrics(results: List[dict]) -> dict:
    """Compute cross-LLM coefficient of variation for key metrics."""
    by_section: Dict[str, List[dict]] = {}
    for r in results:
        if r["status"] != "ok": continue
        by_section.setdefault(r["section_id"], []).append(r)

    metrics = {}
    for sid, rows in by_section.items():
        if len(rows) < 2: continue
        metrics[sid] = {"n_backends": len(rows)}
        for key in ["final_dsr", "final_scr", "NPV_usd", "h_total_cm"]:
            vals = []
            for r in rows:
                v = r.get(key, None)
                if v is None:
                    # compute h_total from layers
                    h_k = [r.get(f"h{i}_cm", 0) or 0 for i in range(1,6)]
                    v = sum(h_k)
                vals.append(float(v))
            if vals and np.std(vals) > 0:
                metrics[sid][f"{key}_cv_pct"] = round(100 * np.std(vals) / np.mean(vals), 1)
            else:
                metrics[sid][f"{key}_cv_pct"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="R2-6 Cross-LLM Robustness")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=list(BACKENDS.keys()))
    args = parser.parse_args()

    sections = [s for s in SECTIONS if not args.section or s["section_id"] == args.section]
    backends = [args.backend] if args.backend else list(BACKENDS.keys())

    plan = [(s, b) for s in sections for b in backends]
    logger.info(f"Plan: {len(plan)} runs ({len(sections)} sections × {len(backends)} backends)")
    if args.dry_run:
        for s, b in plan:
            print(f"  {s['section_id']}/{b}")
        return

    results = []
    for i, (s, b) in enumerate(plan):
        logger.info(f"[{i+1}/{len(plan)}] {s['section_id']}/{b}")
        r = run_one(s, b)
        results.append(r)
        if r["status"] == "ok":
            logger.info(f"  DSR={r['final_dsr']:.2f} SCR={r['final_scr']:.2f} "
                        f"NPV=${r['NPV_usd']:.1f} gen_conf={r.get('mean_gen_confidence','N/A')} "
                        f"({r['wall_clock_sec']:.0f}s)")

    robust = compute_robustness_metrics(results)
    logger.info(f"Robustness CV: {json.dumps(robust, indent=2)}")

    out_dir = Path("experiments/ltpp_data/deliverables/cross_llm")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"cross_llm_summary_{ts}.csv"
    cols = ["section_id","pavement_type","backend","llm_model","n_steps",
            "total_reward","mean_reward","final_dsr","final_scr",
            "B1","B2","B3","B4","NPV_usd",
            "h1_cm","h2_cm","h3_cm","h4_cm","h5_cm",
            "mean_gen_confidence","mean_eval_score","wall_clock_sec","status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)

    # Print summary table
    print("\n" + "=" * 85)
    print("CROSS-LLM ROBUSTNESS SUMMARY")
    print("=" * 85)
    print(f"{'Section':<10} {'Backend':<12} {'DSR':<6} {'SCR':<6} {'NPV':<10} "
          f"{'GenConf':<8} {'Eval':<6} {'Status'}")
    print("-" * 85)
    for r in results:
        print(f"{r['section_id']:<10} {r['backend']:<12} {r['final_dsr']!s:<6} "
              f"{r['final_scr']!s:<6} ${r['NPV_usd']!s:<9} "
              f"{r.get('mean_gen_confidence','--')!s:<8} {r.get('mean_eval_score','--')!s:<6} "
              f"{r['status']}")
    print("=" * 85)
    for sid, m in robust.items():
        print(f"  {sid}: DSR CV={m.get('final_dsr_cv_pct', 0):.1f}%  "
              f"SCR CV={m.get('final_scr_cv_pct', 0):.1f}%  "
              f"NPV CV={m.get('NPV_usd_cv_pct', 0):.1f}%  "
              f"(n={m['n_backends']} backends)")
    logger.info(f"DONE. CSV: {csv_path}")


if __name__ == "__main__":
    main()