# -*- coding: utf-8 -*-
"""
scripts/ablation_watchdog.py — Monitor ablation training progress.

Run in a SEPARATE terminal alongside run_ablation.py.
Prints status every 5 minutes; alerts if no run has updated for >2 hours.

Usage:
    python scripts/ablation_watchdog.py
    python scripts/ablation_watchdog.py --interval 300 --stall-threshold 7200
    python scripts/ablation_watchdog.py --once
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_RUNS_DIR = Path(PROJECT_ROOT) / "output" / "rl_runs"

VARIANTS = ["full", "no-generator", "no-evaluator", "no-rag", "no-generator-no-rag", "no-language-no-guard", "reward-only"]
BASE_TYPES = ["flexible", "semi_rigid"]
SEEDS = [0, 1, 2]
TIMESTEPS = 1000


def build_expected_runs() -> list[str]:
    """Build list of all 30 expected ablation run names."""
    runs = []
    for vt in VARIANTS:
        for pt in BASE_TYPES:
            for sd in SEEDS:
                runs.append(f"ablation_{vt}_{pt}_{TIMESTEPS}ts_seed{sd}")
    return runs


def get_run_status(run_name: str) -> dict:
    """Return status dict for one run."""
    run_dir = RL_RUNS_DIR / run_name
    status = {
        "name": run_name,
        "exists": run_dir.exists(),
        "complete": False,
        "last_mtime": None,
        "age_min": None,
        "size_mb": 0.0,
    }
    if not run_dir.exists():
        return status

    flag = run_dir / "training_complete.flag"
    if flag.exists():
        status["complete"] = True

    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists():
        for ckpt_dir in ckpt_root.iterdir():
            if ckpt_dir.is_dir() and ckpt_dir.name.startswith("ckpt_final_step_"):
                if (ckpt_dir / "ppo_model.zip").exists():
                    status["complete"] = True

    latest_mtime = 0.0
    total_size = 0
    try:
        for root, dirs, files in os.walk(run_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp)
                    if st.st_mtime > latest_mtime:
                        latest_mtime = st.st_mtime
                    total_size += st.st_size
                except OSError:
                    continue
    except OSError:
        pass

    if latest_mtime > 0:
        status["last_mtime"] = datetime.fromtimestamp(latest_mtime)
        status["age_min"] = (time.time() - latest_mtime) / 60.0
    status["size_mb"] = total_size / (1024 * 1024)
    return status


def format_status_line(s: dict) -> str:
    """Format one run's status as a line."""
    name = s["name"][:50]
    if s["complete"]:
        symbol = "[DONE]"
        age = f"size={s['size_mb']:.1f}MB"
    elif not s["exists"]:
        symbol = "[----]"
        age = "not started"
    elif s["age_min"] is not None:
        if s["age_min"] < 5:
            symbol = "[RUN ]"
        elif s["age_min"] < 30:
            symbol = "[ACTV]"
        elif s["age_min"] < 120:
            symbol = "[SLOW]"
        else:
            symbol = "[STAL]"  # potentially stalled
        age = f"updated {s['age_min']:.0f}min ago, {s['size_mb']:.1f}MB"
    else:
        symbol = "[????]"
        age = "no files"
    return f"  {symbol} {name:50s}  {age}"


def main():
    parser = argparse.ArgumentParser(description="Ablation training watchdog")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 300 = 5min)")
    parser.add_argument("--stall-threshold", type=int, default=7200,
                        help="Stall alert threshold in seconds (default: 7200 = 2h)")
    parser.add_argument("--once", action="store_true",
                        help="Print status once and exit")
    args = parser.parse_args()

    expected = build_expected_runs()
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 70)
        print(f"Ablation Watchdog — Iteration {iteration} @ {now}")
        print("=" * 70)

        statuses = [get_run_status(rn) for rn in expected]
        complete = [s for s in statuses if s["complete"]]
        in_progress = [s for s in statuses if s["exists"] and not s["complete"]]
        not_started = [s for s in statuses if not s["exists"]]
        stalled = [s for s in in_progress
                   if s["age_min"] is not None and s["age_min"] * 60 > args.stall_threshold]

        print(f"  Complete:    {len(complete):2d}/30")
        print(f"  In progress: {len(in_progress):2d}/30")
        print(f"  Not started: {len(not_started):2d}/30")
        if stalled:
            print(f"  STALLED:     {len(stalled):2d}  <-- ALERT")

        if in_progress:
            print("\n  Active runs:")
            for s in in_progress:
                print(format_status_line(s))

        if stalled:
            print("\n  !!! STALLED RUNS (no update >2h) !!!")
            for s in stalled:
                print(format_status_line(s))

        pct = 100.0 * len(complete) / 30
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"\n  Progress: [{bar}] {pct:.1f}% ({len(complete)}/30)")

        if len(complete) == 30:
            print("\n  ALL 30 CELLS COMPLETE.")
            break

        if args.once:
            break

        print(f"\n  Next check in {args.interval}s. Ctrl-C to stop.")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatchdog stopped by user.")
            break


if __name__ == "__main__":
    main()