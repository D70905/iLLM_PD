"""
Experiment Group 1: Extract process-level metrics from audit chains
=================================================================
Metrics: steps-to-first-compliance, FEA calls to compliance, convergence stability.
Reads existing audit chain JSONL files — NO retraining required.

Usage: python scripts/analysis/extract_process_metrics.py
Output: scripts/analysis/process_metrics_summary.csv
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

ABLATION_DIR = Path(__file__).parent.parent.parent / "output" / "rl_runs"

VARIANTS = {
    "full":     "ablation_full",
    "no_gen":   "ablation_no-generator",
    "no_rag":   "ablation_no-rag",
    "no_guard": "ablation_no-guard",
}
TYPES = ["flexible", "semi_rigid"]
SEEDS = [0, 1, 2]

def load_episode_trajectories(audit_path):
    """Parse audit chain into per-episode step sequences."""
    episodes = defaultdict(list)  # episode_id -> list of step records
    with open(audit_path) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "step":
                ep = r["data"].get("episode", 0)
                episodes[ep].append(r["data"])
    return episodes

def compute_episode_metrics(episode_steps):
    """Compute process metrics for one episode."""
    steps_sorted = sorted(episode_steps, key=lambda s: s["step"])

    # Find first compliance (all B1-B4 >= 1.0, i.e. feasible=True)
    first_compliant = None
    for s in steps_sorted:
        if s.get("feasible", False):
            first_compliant = s["step"]
            break

    # Count compliance violations after first compliance (stability)
    post_compliance_violations = 0
    if first_compliant is not None:
        for s in steps_sorted:
            if s["step"] > first_compliant and not s.get("feasible", False):
                post_compliance_violations += 1

    # Count guard blocks
    guard_blocks = sum(1 for s in steps_sorted if s.get("guard_blocked", False))

    # Count unique designs evaluated (proxy for FEA calls)
    # Each non-guard-blocked step triggers FEA
    fea_calls = sum(1 for s in steps_sorted if not s.get("guard_blocked", False))
    fea_to_first_compliant = None
    if first_compliant is not None:
        fea_to_first_compliant = sum(
            1 for s in steps_sorted
            if s["step"] <= first_compliant and not s.get("guard_blocked", False)
        )

    # Final margins
    final = steps_sorted[-1] if steps_sorted else {}
    final_margins = final.get("margins", {})

    return {
        "n_steps": len(steps_sorted),
        "first_compliant_step": first_compliant,
        "post_compliance_violations": post_compliance_violations,
        "guard_blocks": guard_blocks,
        "fea_calls": fea_calls,
        "fea_to_first_compliant": fea_to_first_compliant,
        "final_feasible": final.get("feasible", False),
        "final_margins": final_margins,
        "reward_mean": np.mean([s.get("reward_total", 0) for s in steps_sorted]),
    }

def main():
    results = []

    for variant_key, variant_prefix in VARIANTS.items():
        for ptype in TYPES:
            for seed in SEEDS:
                dirname = f"{variant_prefix}_{ptype}_1000ts_seed{seed}"
                audit_path = ABLATION_DIR / dirname / "audit" / "audit_chain.jsonl"

                if not audit_path.exists():
                    print(f"  SKIP (missing): {dirname}")
                    continue

                episodes = load_episode_trajectories(audit_path)

                for ep_id, steps in episodes.items():
                    metrics = compute_episode_metrics(steps)
                    results.append({
                        "variant": variant_key,
                        "pavement_type": ptype,
                        "seed": seed,
                        "episode": ep_id,
                        **metrics,
                    })
                print(f"  OK: {dirname} — {len(episodes)} episodes")

    # Write summary CSV
    out_path = Path(__file__).parent / "process_metrics_summary.csv"
    fieldnames = [
        "variant", "pavement_type", "seed", "episode",
        "n_steps", "first_compliant_step", "post_compliance_violations",
        "guard_blocks", "fea_calls", "fea_to_first_compliant",
        "final_feasible", "reward_mean",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Print aggregate summary
    print(f"\n{'='*80}")
    print("AGGREGATE PROCESS METRICS (mean over episodes)")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'Type':<12} {'Steps':>7} {'1stCompl':>10} {'PostViol':>9} {'GuardBlk':>9} {'FEA':>6} {'FEAto1st':>9}")
    print("-" * 75)

    for variant_key in ["full", "no_gen", "no_rag", "no_guard"]:
        for ptype in TYPES:
            subset = [r for r in results if r["variant"] == variant_key and r["pavement_type"] == ptype]
            if not subset:
                continue
            steps = np.mean([r["n_steps"] for r in subset])
            first = np.mean([r["first_compliant_step"] for r in subset if r["first_compliant_step"] is not None])
            violations = np.mean([r["post_compliance_violations"] for r in subset])
            guard = np.mean([r["guard_blocks"] for r in subset])
            fea = np.mean([r["fea_calls"] for r in subset])
            fea1 = np.mean([r["fea_to_first_compliant"] for r in subset if r["fea_to_first_compliant"] is not None])
            print(f"{variant_key:<10} {ptype:<12} {steps:7.1f} {first:10.1f} {violations:9.1f} {guard:9.1f} {fea:6.1f} {fea1:9.1f}")

    print(f"\nSaved: {out_path}")

    # === KEY FINDING: Generator's value in steps-to-compliance ===
    print(f"\n{'='*80}")
    print("GENERATOR VALUE: Steps-to-first-compliance comparison")
    print(f"{'='*80}")
    for ptype in TYPES:
        full_first = [r["first_compliant_step"] for r in results
                      if r["variant"] == "full" and r["pavement_type"] == ptype and r["first_compliant_step"] is not None]
        nogen_first = [r["first_compliant_step"] for r in results
                       if r["variant"] == "no_gen" and r["pavement_type"] == ptype and r["first_compliant_step"] is not None]
        if full_first and nogen_first:
            print(f"  {ptype}: Full={np.mean(full_first):.1f}±{np.std(full_first):.1f}  "
                  f"No-Gen={np.mean(nogen_first):.1f}±{np.std(nogen_first):.1f}  "
                  f"Δ={np.mean(nogen_first)-np.mean(full_first):+.1f} steps")

    return results

if __name__ == "__main__":
    results = main()
