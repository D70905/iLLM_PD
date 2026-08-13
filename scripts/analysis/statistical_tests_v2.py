"""
Statistical tests v2 — uses REAL per-variant per-section costs.
Reads from per-variant inference CSVs (NOT simulated data).

Prerequisites:
  1. Run inference for no-gen / no-rag / no-guard policies on 12 sections
  2. Save outputs as:
     experiments/ltpp_data/deliverables/ltpp_inference_no_gen/ltpp_inference_summary.csv
     experiments/ltpp_data/deliverables/ltpp_inference_no_rag/ltpp_inference_summary.csv
     experiments/ltpp_data/deliverables/ltpp_inference_no_guard/ltpp_inference_summary.csv

Usage: python scripts/analysis/statistical_tests_v2.py
"""

import numpy as np
import csv
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
FULL_INFER = PROJECT_ROOT / "experiments/ltpp_data/deliverables/ltpp_inference/ltpp_inference_summary_20260625_133730.csv"
CORE_RESULTS = PROJECT_ROOT / "experiments/core_results_2048_full.csv"

FLEX_SECTIONS = ["16_1010", "04_1034", "27_1085", "12_1060", "48_1076", "48_0001"]
SEMI_SECTIONS = ["30_7076", "04_1065", "06_2004", "27_2023", "12_4097", "48_1109"]


def extract_per_section_costs(csv_path, sections):
    """Read real per-section costs from inference CSV (mean over seeds)."""
    costs = {}
    try:
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get('section_id') in sections]
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping")
        return None

    for sid in sections:
        sec_rows = [r for r in rows if r['section_id'] == sid]
        if not sec_rows:
            print(f"  WARNING: section {sid} not found in {csv_path}")
            costs[sid] = None
            continue
        # Take mean of seed-0 cost (seeds are deterministic for delivered design)
        cost_vals = [float(r.get('final_C_const_usd_m2', 0)) for r in sec_rows]
        costs[sid] = np.mean(cost_vals)

    return np.array([costs[s] for s in sections if costs[s] is not None])


def bootstrap_ci_paired(diff_array, n_bootstrap=10000):
    """Bootstrap CI for paired differences (same sections, two variants)."""
    n = len(diff_array)
    means = np.array([np.mean(np.random.choice(diff_array, size=n, replace=True))
                      for _ in range(n_bootstrap)])
    return np.percentile(means, [2.5, 97.5])


def paired_permutation_test(x, y, n_perm=10000):
    """Paired permutation (sign-flip) for matched-section design."""
    observed = np.mean(x) - np.mean(y)
    diffs = x - y
    n = len(diffs)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(signs * diffs)
        if abs(perm_diff) >= abs(observed):
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def cohens_d_paired(x, y):
    n = len(x)
    diffs = x - y
    return np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0


def main():
    print("=" * 70)
    print("STATISTICAL TESTS v2 (REAL DATA ONLY)")
    print("=" * 70)

    # Load full-system costs (REAL — from core_results CSV)
    full_flex = extract_per_section_costs(CORE_RESULTS, FLEX_SECTIONS)
    full_semi = extract_per_section_costs(CORE_RESULTS, SEMI_SECTIONS)

    if full_flex is not None:
        print(f"\nFull flexible (n={len(full_flex)}): mean={np.mean(full_flex):.1f}, "
              f"sd={np.std(full_flex, ddof=1):.1f}")
        for s, c in zip(FLEX_SECTIONS, full_flex):
            print(f"  {s}: {c:.1f}")

    # === Try loading No-Gen costs from inference CSV ===
    variants = {
        "No-Gen": "ltpp_inference_no_gen",
        "No-RAG": "ltpp_inference_no_rag",
        "No-Guard": "ltpp_inference_no_guard",
    }

    for var_name, var_dir in variants.items():
        csv_path = PROJECT_ROOT / "experiments/ltpp_data/deliverables" / var_dir / "ltpp_inference_summary.csv"
        print(f"\n--- {var_name} ---")
        flex_costs = extract_per_section_costs(csv_path, FLEX_SECTIONS)
        if flex_costs is not None and full_flex is not None:
            obs, p = paired_permutation_test(full_flex, flex_costs)
            d = cohens_d_paired(full_flex, flex_costs)
            ci = bootstrap_ci_paired(full_flex - flex_costs)
            print(f"  Flexible: mean={np.mean(flex_costs):.1f}, sd={np.std(flex_costs, ddof=1):.1f}")
            print(f"  vs Full: observed diff={obs:+.1f}, p={p:.4f}, Cohen's d={d:.3f}")
            print(f"  Bootstrap 95% CI of diff: [{ci[0]:.2f}, {ci[1]:.2f}]")
            sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
            print(f"  Verdict: {sig} at α=0.05")

    # === Power analysis ===
    print(f"\n--- Power Analysis (n=6, α=0.05) ---")
    from scipy import stats as sp_stats
    for d in [0.2, 0.5, 0.8, 1.0, 1.5]:
        df = 2 * 6 - 2
        t_crit = sp_stats.t.ppf(0.975, df)
        ncp = d * np.sqrt(6 / 2)
        power = 1 - sp_stats.nct.cdf(t_crit, df, ncp) + sp_stats.nct.cdf(-t_crit, df, ncp)
        print(f"  d={d:.1f}: power={power:.1%}")

    print(f"\n{'='*70}")
    print("NEXT STEP: Run inference for no-gen/no-rag/no-guard policies first:")
    print("  cd d:\\iLLM_PD_new")
    print("  conda activate illm_pd")
    print("  python scripts/ltpp_inference.py \\")
    print("    --policy-flex output/rl_runs/ablation_no-generator_flexible_1000ts_seed0/checkpoints \\")
    print("    --policy-semi output/rl_runs/ablation_no-generator_semi_rigid_1000ts_seed0/checkpoints \\")
    print("    --sections experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx \\")
    print("    --surrogate-model-path output/surrogate_model/surrogate_v3.pt \\")
    print("    --seeds 0,1,2 \\")
    print("    --out-dir experiments/ltpp_data/deliverables/ltpp_inference_no_gen")
    print("  (Repeat for no_rag, no_guard)")


if __name__ == "__main__":
    main()
