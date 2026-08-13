"""
Experiment Group 2: Difficulty-stratified ablation analysis
===========================================================
Stratify 12 sections by binding-constraint tightness to test whether
Generator's value is conditional (useful in hard sections, redundant in easy ones).

Uses existing per-section results — NO retraining required.

Usage: python scripts/analysis/difficulty_stratified_ablation.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# Section data from core_results_2048_full.csv
SECTION_DATA = {
    "16_1010": {"type": "flexible", "E_sub": 78,  "AC_cm": 17.3, "SCR": 0.905, "LCC": 56.9},
    "27_1085": {"type": "flexible", "E_sub": 86,  "AC_cm": 17.4, "SCR": 0.905, "LCC": 56.9},
    "48_1076": {"type": "flexible", "E_sub": 115, "AC_cm": 17.4, "SCR": 0.905, "LCC": 57.1},
    "04_1034": {"type": "flexible", "E_sub": 91,  "AC_cm": 17.4, "SCR": 0.905, "LCC": 57.0},
    "48_0001": {"type": "flexible", "E_sub": 700, "AC_cm": 15.7, "SCR": 1.000, "LCC": 54.0},
    "12_1060": {"type": "flexible", "E_sub": 286, "AC_cm": 16.5, "SCR": 0.952, "LCC": 69.9},
    "30_7076": {"type": "semi_rigid", "E_sub": 59,  "AC_cm": 16.8, "SCR": 1.000, "LCC": 65.6},
    "04_1065": {"type": "semi_rigid", "E_sub": 91,  "AC_cm": 16.7, "SCR": 1.000, "LCC": 65.4},
    "27_2023": {"type": "semi_rigid", "E_sub": 131, "AC_cm": 16.8, "SCR": 1.000, "LCC": 65.4},
    "06_2004": {"type": "semi_rigid", "E_sub": 112, "AC_cm": 16.8, "SCR": 1.000, "LCC": 65.4},
    "48_1109": {"type": "semi_rigid", "E_sub": 100, "AC_cm": 16.8, "SCR": 1.000, "LCC": 65.4},
    "12_4097": {"type": "semi_rigid", "E_sub": 286, "AC_cm": 16.8, "SCR": 1.000, "LCC": 65.4},
}

def classify_difficulty(section_id, data):
    """Classify section difficulty based on constraint tightness indicators."""
    scr = data["SCR"]
    lcc = data["LCC"]

    # Flexible: low SCR = harder (more transient non-compliance)
    # Semi-rigid: all SCR=1.0, use LCC as proxy
    if data["type"] == "flexible":
        if scr >= 1.0:
            return "easy"  # always compliant during optimization
        elif scr >= 0.95:
            return "medium"
        else:
            return "hard"  # most transient violations
    else:
        # Semi-rigid: use LCC as proxy (higher LCC = harder to optimize)
        if lcc <= 65.5:
            return "easy"
        else:
            return "medium"

def main():
    # Classify sections
    difficulty_map = {}
    for sid, data in SECTION_DATA.items():
        difficulty_map[sid] = classify_difficulty(sid, data)

    # Load process metrics from Experiment 1
    metrics_path = Path(__file__).parent / "process_metrics_summary.csv"
    if not metrics_path.exists():
        print("ERROR: Run extract_process_metrics.py first!")
        return

    # Read process metrics
    rows = []
    with open(metrics_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Now aggregate by variant × difficulty
    # For actual per-section inference results, use core_results CSV
    # Here we demonstrate the stratification logic using the section metadata
    # The full analysis requires mapping audit chain episodes to sections
    # (which requires the inference scripts output, not just training logs)

    print("="*80)
    print("SECTION DIFFICULTY CLASSIFICATION")
    print("="*80)
    print(f"{'Section':<10} {'Type':<12} {'E_sub':>6} {'SCR':>6} {'LCC':>6} {'Difficulty':>10}")
    print("-"*55)
    for sid, data in SECTION_DATA.items():
        diff = difficulty_map[sid]
        print(f"{sid:<10} {data['type']:<12} {data['E_sub']:>6} {data['SCR']:>6.3f} {data['LCC']:>6.1f} {diff:>10}")

    # Count
    easy = [s for s, d in difficulty_map.items() if d == "easy"]
    medium = [s for s, d in difficulty_map.items() if d == "medium"]
    hard = [s for s, d in difficulty_map.items() if d == "hard"]
    print(f"\nEasy: {len(easy)} sections ({', '.join(easy)})")
    print(f"Medium: {len(medium)} sections ({', '.join(medium)})")
    print(f"Hard: {len(hard)} sections ({', '.join(hard)})")

    print(f"\n{'='*80}")
    print("IMPLICATION FOR GENERATOR VALUE")
    print(f"{'='*80}")
    print("Hypothesis: Generator value is concentrated in 'hard' sections")
    print("(low SCR, multiple binding constraints, difficult optimization landscapes)")
    print()
    print("If true: report ablation stratified by difficulty tier")
    print("If false: Generator is genuinely redundant — reframe as optional explainability module")
    print()
    print("NEXT STEP: Re-run inference on the 12 sections with and without Generator,")
    print("then stratify per-section cost/DSR/SCR by difficulty tier.")
    print("The per-section inference results are in:")
    print("  experiments/ltpp_data/deliverables/ltpp_inference/")

    # Check available inference results
    inference_dir = Path(__file__).parent.parent.parent / "experiments" / "ltpp_data" / "deliverables" / "ltpp_inference"
    if inference_dir.exists():
        csvs = list(inference_dir.glob("*.csv"))
        print(f"\nFound {len(csvs)} inference CSV files:")
        for c in csvs:
            print(f"  {c.name}")

if __name__ == "__main__":
    main()
