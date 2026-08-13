# -*- coding: utf-8 -*-
"""
scripts/lhs_sample_for_surrogate.py (v2 — DUAL BASE TYPE)
===========================================================

Latin Hypercube Sampling for Surrogate v3 training.

v2 changes (Phase 2D dual-base):
    - Two sub-samples: N/2 for 'semi_rigid', N/2 for 'flexible'
    - Each uses GuardConfig.from_base_type(...) for correct E ranges
    - 12th input feature `is_semi_rigid` (0 or 1) recorded with each sample
    - Output JSONL format adds `input.pavement_type` and 12-dim vector layout

Strategy: cover full physical design space defined in GuardConfig (per
base type), so the Surrogate works across both pavement structures.

Parallel ABAQUS: 3 concurrent jobs × 2 CPUs each.

Usage:
    conda activate illm_pd
    cd D:\\iLLM_PD_new
    python -m scripts.lhs_sample_for_surrogate --n 1000 --workers 3 --cpus 2

Output:
    output/surrogate_data/lhs_<timestamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fea.runner import run_fea
from rl.guards import GuardConfig, NumericalGuard, GuardViolation

logger = logging.getLogger("lhs_sample_v2")


# ──────────────────────────────────────────────────────────────────────
# Per-base-type sampling configuration
# ──────────────────────────────────────────────────────────────────────

# Poisson ratios DIFFER by base type
POISSON_BY_TYPE = {
    "semi_rigid": [0.25, 0.30, 0.30, 0.25, 0.35],   # cement-base ν=0.25
    "flexible":   [0.25, 0.30, 0.30, 0.40, 0.35],   # granular-base ν=0.40
}
FIXED_NU_SUBGRADE = 0.40


def get_sampling_bounds(base_type: str):
    """11-dim bounds matching GuardConfig.from_base_type(base_type)."""
    g = GuardConfig.from_base_type(base_type)
    lows = list(g.h_min) + list(g.E_min) + [g.E_subgrade_min]
    highs = list(g.h_max) + list(g.E_max) + [g.E_subgrade_max]
    return np.array(lows), np.array(highs)


# ──────────────────────────────────────────────────────────────────────
# Single-sample worker
# ──────────────────────────────────────────────────────────────────────

def _run_single_sample(args_tuple: Tuple[int, np.ndarray, str, int, str]) -> Dict:
    """
    Args (packed for pool.imap):
        sample_idx, design_vector_11d, base_type, num_cpus, base_dir

    Returns dict with input/output/run_time, or skipped record.
    """
    sample_idx, x, base_type, num_cpus, base_dir = args_tuple

    thickness = x[:5].tolist()
    modulus = x[5:10].tolist()
    E_subgrade = float(x[10])
    poisson = POISSON_BY_TYPE[base_type]
    is_semi_rigid = 1.0 if base_type == "semi_rigid" else 0.0

    # Pre-FEA guard check (per-base-type bounds)
    guard = NumericalGuard(base_type=base_type)
    try:
        guard.check_design(
            thickness=np.array(thickness),
            modulus=np.array(modulus),
            E_subgrade=E_subgrade,
        )
    except GuardViolation as gv:
        return {
            "idx": sample_idx,
            "base_type": base_type,
            "skipped": True,
            "reason": "pre_guard_{}".format(gv.code),
        }

    t0 = time.time()
    try:
        result = run_fea(
            thickness=thickness,
            modulus=modulus,
            poisson=poisson,
            E_subgrade=E_subgrade,
            nu_subgrade=FIXED_NU_SUBGRADE,
            load_pressure=0.7,
            load_radius=0.1065,
            base_dir=base_dir,
            num_cpus=num_cpus,
            verbose=False,
        )
        responses = result.get("responses", {})

        try:
            guard.check_fea_result(responses)
        except GuardViolation as gv:
            return {
                "idx": sample_idx,
                "base_type": base_type,
                "skipped": True,
                "reason": "post_guard_{}".format(gv.code),
                "run_time_s": time.time() - t0,
            }

        return {
            "idx": sample_idx,
            "base_type": base_type,
            "skipped": False,
            "input": {
                "thickness_m":  thickness,
                "modulus_MPa":  modulus,
                "poisson":      poisson,
                "E_subgrade":   E_subgrade,
                "nu_subgrade":  FIXED_NU_SUBGRADE,
                "pavement_type": base_type,
                "is_semi_rigid": is_semi_rigid,
            },
            "output": {
                "epsilon_a_microstrain": float(responses.get("epsilon_a_microstrain", float("nan"))),
                "sigma_t_MPa":           float(responses.get("sigma_t_MPa", float("nan"))),
                "epsilon_z_microstrain": float(responses.get("epsilon_z_microstrain", float("nan"))),
                "p_AC_upper_mid_MPa":    float(responses.get("p_AC_upper_mid_MPa", float("nan"))),
                "p_AC_mid_mid_MPa":      float(responses.get("p_AC_mid_mid_MPa", float("nan"))),
                "p_AC_lower_mid_MPa":    float(responses.get("p_AC_lower_mid_MPa", float("nan"))),
            },
            "run_time_s": time.time() - t0,
        }
    except Exception as e:
        return {
            "idx": sample_idx,
            "base_type": base_type,
            "skipped": True,
            "reason": "fea_crash_{}".format(type(e).__name__),
            "error": str(e)[:200],
            "run_time_s": time.time() - t0,
        }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LHS dual-base sampling for surrogate v3 training")
    parser.add_argument("--n",       type=int, default=1000,
                        help="Total number of LHS samples (half per type)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel ABAQUS jobs (default 3)")
    parser.add_argument("--cpus",    type=int, default=2,
                        help="CPUs per ABAQUS job (default 2)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./output/surrogate_data")
    parser.add_argument("--split",   type=str, default="balanced",
                        choices=["balanced", "semi_only", "flex_only"],
                        help="balanced=N/2 each; semi_only/flex_only for ablation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"lhs_dual_{ts}.jsonl"
    status_path = out_dir / f"lhs_dual_{ts}_status.json"

    fea_base = Path("./output/lhs_fea_workspace").resolve()
    fea_base.mkdir(parents=True, exist_ok=True)

    # ── Determine per-type sample counts ─────────────────────────
    if args.split == "balanced":
        n_semi = args.n // 2
        n_flex = args.n - n_semi
    elif args.split == "semi_only":
        n_semi = args.n
        n_flex = 0
    else:
        n_semi = 0
        n_flex = args.n

    logger.info("=" * 70)
    logger.info(f"LHS dual-base sampling: total={args.n}  semi={n_semi}  flex={n_flex}")
    logger.info(f"  workers={args.workers} cpus_per_job={args.cpus} "
                f"(total CPU = {args.workers * args.cpus})")
    logger.info(f"  output: {out_path}")
    logger.info("=" * 70)

    # ── Build sample list per base type ──────────────────────────
    layer_labels = ["h_upper_AC", "h_mid_AC", "h_lower_AC", "h_base", "h_subbase",
                    "E_upper_AC", "E_mid_AC", "E_lower_AC", "E_base", "E_subbase",
                    "E_subgrade"]

    all_pool_args: List[Tuple[int, np.ndarray, str, int, str]] = []
    idx_counter = 0
    for base_type, n_this in [("semi_rigid", n_semi), ("flexible", n_flex)]:
        if n_this == 0:
            continue
        lows, highs = get_sampling_bounds(base_type)
        logger.info(f"\n  [{base_type}] sampling bounds:")
        for lbl, lo, hi in zip(layer_labels, lows, highs):
            logger.info(f"    {lbl:14s}: {lo:8.3f} → {hi:8.3f}")

        sampler = qmc.LatinHypercube(
            d=11, seed=args.seed + (1 if base_type == "flexible" else 0))
        samples_unit = sampler.random(n=n_this)
        samples_phys = qmc.scale(samples_unit, lows, highs)

        for j in range(n_this):
            all_pool_args.append(
                (idx_counter, samples_phys[j], base_type,
                 args.cpus, str(fea_base))
            )
            idx_counter += 1

    # Shuffle order so semi/flex are interleaved (better failure resilience)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_pool_args)
    # Re-index after shuffle so idx is monotonic in order of execution
    all_pool_args = [(i, x, bt, ncpu, bd)
                     for i, (_, x, bt, ncpu, bd) in enumerate(all_pool_args)]

    # ── Estimate ─────────────────────────────────────────────────
    est_per_sample = 50.0
    est_total = args.n * est_per_sample / args.workers
    logger.info(f"\nEstimated total time: {est_total / 3600:.1f} hr "
                f"({args.n} samples / {args.workers} workers × {est_per_sample}s)")
    logger.info(f"Press Ctrl+C to stop; partial results saved.")
    logger.info("")

    t_start = time.time()
    n_ok = {"semi_rigid": 0, "flexible": 0}
    n_skip = {"semi_rigid": 0, "flexible": 0}
    last_status_update = t_start

    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps({
            "_meta": {
                "type": "lhs_surrogate_data_v2_dual_base",
                "version": "v2",
                "n_samples_requested": args.n,
                "n_semi_rigid": n_semi,
                "n_flexible":   n_flex,
                "lhs_seed":     args.seed,
                "lhs_dim":      11,
                "input_labels": layer_labels + ["is_semi_rigid"],
                "fixed_poisson_by_type": POISSON_BY_TYPE,
                "fixed_nu_subgrade": FIXED_NU_SUBGRADE,
                "started_at": datetime.now().isoformat(),
                "workers": args.workers,
                "cpus_per_job": args.cpus,
                "split": args.split,
            }
        }) + "\n")
        fout.flush()

        try:
            with mp.Pool(processes=args.workers) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(_run_single_sample, all_pool_args),
                    start=1
                ):
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()

                    bt = result.get("base_type", "semi_rigid")
                    if result.get("skipped", False):
                        n_skip[bt] += 1
                    else:
                        n_ok[bt] += 1

                    now = time.time()
                    if i % 10 == 0 or (now - last_status_update) > 60:
                        elapsed = now - t_start
                        rate = i / elapsed if elapsed > 0 else 0
                        remaining = (args.n - i) / rate if rate > 0 else 0
                        total_ok = sum(n_ok.values())
                        total_skip = sum(n_skip.values())
                        success_rate = total_ok / i if i > 0 else 0
                        logger.info(
                            f"[{i:4d}/{args.n}] ok={total_ok} "
                            f"(semi={n_ok['semi_rigid']}, flex={n_ok['flexible']}) "
                            f"skip={total_skip} "
                            f"({success_rate:5.1%} good) "
                            f"elapsed={elapsed/60:.1f}min "
                            f"remaining={remaining/60:.1f}min "
                            f"rate={rate*3600:.0f}/hr"
                        )
                        with open(status_path, "w") as fs:
                            json.dump({
                                "progress": i,
                                "total": args.n,
                                "n_ok": n_ok,
                                "n_skip": n_skip,
                                "elapsed_sec": elapsed,
                                "remaining_min": remaining / 60,
                                "rate_per_hr": rate * 3600,
                                "last_update": datetime.now().isoformat(),
                            }, fs, indent=2)
                        last_status_update = now

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Partial results saved.")
            pool.terminate()

    elapsed_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"LHS dual-base sampling COMPLETE.")
    logger.info(f"  Total time:      {elapsed_total/3600:.2f} hr")
    logger.info(f"  semi_rigid: ok={n_ok['semi_rigid']:4d}  skip={n_skip['semi_rigid']}")
    logger.info(f"  flexible:   ok={n_ok['flexible']:4d}  skip={n_skip['flexible']}")
    logger.info(f"  Output:          {out_path}")
    logger.info("=" * 70)
    logger.info(f"Next step: train Surrogate v3 with:")
    logger.info(f"  python -m scripts.train_surrogate_v2 --data {out_path} --out output/surrogate_model/surrogate_v3.pt")


if __name__ == "__main__":
    mp.freeze_support()
    main()
