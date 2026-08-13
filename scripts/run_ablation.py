# -*- coding: utf-8 -*-
"""
scripts/run_ablation.py — Batch ablation training launcher.
=============================================================

Launches all HARA component ablation training runs for Phase 2F.
Each call spawns a subprocess that runs the full training pipeline.

Variants (4 standard + optional pairwise) x base types (2) x seeds (3).

    Full HARA      : Generator + Evaluator + RAG + NumericalGuard  (baseline)
    No Generator   : Generator disabled; Evaluator + Guard retained
    No Evaluator   : Evaluator disabled; Generator + RAG + Guard retained
    No RAG         : Generator runs WITHOUT regulation context
    No Gen + No RAG: pairwise language-subsystem interaction check
    No Language + No Guard: language-guidance x safety-screening interaction check
    Reward-only    : NumericalGuard bypassed; FEA failure -> penalty

Config per run:
    timesteps = 1000, n_steps = 64, lr = 3e-4
    surrogate-accelerated, B3 escalation threshold = 1.0
    seeds = [0, 1, 2]  (n=3 per cell for mean +- std)

Estimated runtime (serial): ~135h total
    flexible:   5 variants x 3 seeds x ~6h = ~90h
    semi_rigid: 5 variants x 3 seeds x ~3h = ~45h

Resume behaviour:
    Before launching each run, the script checks for
    output/rl_runs/{run_name}/checkpoints/ckpt_final_step_*/ppo_model.zip
    and output/rl_runs/{run_name}/training_complete.flag.
    If either exists, the run is skipped (assumed already complete).

Usage (all 30 runs, serial):
    python scripts/run_ablation.py

Usage (specific variant + base type):
    python scripts/run_ablation.py --variant no-generator --pavement flexible

Usage (dry-run):
    python scripts/run_ablation.py --dry-run

Usage (force re-run):
    python scripts/run_ablation.py --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ───────────────────────────────────────────────────────
TIMESTEPS = 1000
SEEDS = [0, 1, 2]   # n=3 per cell — minimum defensible for mean+-std + Wilcoxon

VARIANTS = ["full", "no-generator", "no-rag", "no-generator-no-rag", "no-language-no-guard", "no-guard"]
BASE_TYPES = ["flexible", "semi_rigid"]   # flexible first (slower per run)

SURROGATE_MODEL = "./output/surrogate_model/surrogate_v3.pt"
B3_THRESHOLD = 1.0


def build_command(variant: str, pavement: str, seed: int, run_name: str) -> str:
    """Build the python -m rl.train command for one ablation run."""
    python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # fallback
    cmd = (
        f'"{python_exe}" -m rl.train'
        f" --pavement-type {pavement}"
        f" --timesteps {TIMESTEPS}"
        f" --seed {seed}"
        f" --use-surrogate"
        f" --surrogate-model-path {SURROGATE_MODEL}"
        f" --surrogate-b3-threshold {B3_THRESHOLD}"
        f" --run-name {run_name}"
    )
    if variant != "full":
        cmd += f" --ablation-variant {variant}"
    if variant == "no-language-no-guard":
        # This diagnostic cell removes the language-guidance subsystem entirely.
        # Avoid initializing LLM/RAG backends during training.
        cmd += " --no-llm"
    return cmd


def is_run_complete(run_name: str) -> tuple[bool, str]:
    """
    Check if a run has already completed.

    Looks for:
      - output/rl_runs/{run_name}/training_complete.flag
      - output/rl_runs/{run_name}/checkpoints/ckpt_final_step_*/ppo_model.zip
    """
    run_dir = Path(PROJECT_ROOT) / "output" / "rl_runs" / run_name
    if not run_dir.exists():
        return False, "run directory does not exist"

    # Check training_complete.flag
    flag = run_dir / "training_complete.flag"
    if flag.exists():
        return True, str(flag)

    # Check for final checkpoint
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists():
        for ckpt_dir in ckpt_root.iterdir():
            if ckpt_dir.is_dir() and ckpt_dir.name.startswith("ckpt_final_step_"):
                zip_path = ckpt_dir / "ppo_model.zip"
                if zip_path.exists():
                    return True, str(zip_path)

    return False, "no completion marker found"


def estimate_total_hours(queue: list) -> float:
    """Rough total wall-clock estimate (serial)."""
    h = 0.0
    for vt, pt, sd, rn in queue:
        h += 6.0 if pt == "flexible" else 3.0
    return h


def parse_seeds(s: str) -> list[int]:
    """Parse a comma-separated seed list like '0,1,2' into [0,1,2]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Ablation batch launcher")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing checkpoints; re-run everything")
    parser.add_argument("--variant", type=str, default=None,
                        choices=VARIANTS, help="Run only this variant")
    parser.add_argument("--pavement", type=str, default=None,
                        choices=BASE_TYPES, help="Run only this pavement type")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: [0,1,2])")
    args = parser.parse_args()

    variants = [args.variant] if args.variant else VARIANTS
    pavements = [args.pavement] if args.pavement else BASE_TYPES
    seeds = parse_seeds(args.seeds) if args.seeds else SEEDS

    # Build run queue: (variant, pavement, seed, run_name)
    queue = []
    for vt in variants:
        for pt in pavements:
            for sd in seeds:
                run_name = f"ablation_{vt}_{pt}_{TIMESTEPS}ts_seed{sd}"
                queue.append((vt, pt, sd, run_name))

    # Pre-flight: check which runs are already complete
    pending = []
    skipped = []
    for entry in queue:
        vt, pt, sd, rn = entry
        if not args.force:
            done, reason = is_run_complete(rn)
            if done:
                skipped.append((entry, reason))
                continue
        pending.append(entry)

    total_h = estimate_total_hours(pending)

    print("=" * 70)
    print("Ablation training plan")
    print(f"  Variants:    {variants}")
    print(f"  Pavements:   {pavements}")
    print(f"  Seeds:       {seeds}")
    print(f"  Timesteps:   {TIMESTEPS} per run")
    print(f"  Total cells: {len(queue)}")
    print(f"  Skipped:     {len(skipped)} (already complete)")
    print(f"  Pending:     {len(pending)}")
    print(f"  Estimated:   ~{total_h:.0f}h serial "
          f"(~{total_h/24:.1f}d wall-clock if running 24/7)")
    print("=" * 70)

    if skipped:
        print("\nSkipped runs (already complete):")
        for (vt, pt, sd, rn), reason in skipped:
            print(f"  [{vt}/{pt}/seed{sd}] {rn}")
            print(f"      reason: {reason}")
        print()

    if not pending:
        print("Nothing to do. All requested runs are complete.")
        print("Use --force to re-run.")
        return

    ok_count = 0
    fail_count = 0

    for i, (vt, pt, sd, rn) in enumerate(pending):
        cmd = build_command(vt, pt, sd, rn)
        print(f"\n[{i+1}/{len(pending)}] [{vt}/{pt}/seed{sd}] {rn}")
        print(f"  {cmd}")

        if args.dry_run:
            continue

        t0 = time.time()
        try:
            subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, check=True)
            elapsed = time.time() - t0
            ok_count += 1
            print(f"  -> OK ({elapsed/3600:.2f}h)")
        except subprocess.CalledProcessError as e:
            fail_count += 1
            print(f"  -> FAILED (exit={e.returncode})")
            if args.variant and args.pavement and args.seeds:
                sys.exit(1)
            print("     Continuing with next run...")
            continue
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("Interrupted by user.")
            print(f"  Completed this session: {ok_count}")
            print(f"  Failed this session:    {fail_count}")
            print(f"  Remaining:              {len(pending) - i - 1}")
            print("  To resume: re-run this script (completed runs are skipped)")
            print("=" * 70)
            sys.exit(130)

    if not args.dry_run:
        print("\n" + "=" * 70)
        print("Batch complete.")
        print(f"  Succeeded this session: {ok_count}")
        print(f"  Failed this session:    {fail_count}")
        print(f"  Previously complete:    {len(skipped)}")
        print(f"  Total cells:            {len(queue)}")
        print("=" * 70)


if __name__ == "__main__":
    main()