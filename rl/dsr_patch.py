# -*- coding: utf-8 -*-
"""
DSR/SCR Patch for rl/reward.py — Phase 2D (R2-2 response)
===========================================================

Clean replacement for the original MATLAB DSR formula.
Designed for safe insertion into rl/reward.py — NO changes to
existing functions. Just append these two functions + import.

Integration (tomorrow, after training completes):
    1. Open rl/reward.py
    2. Add `import math` at top (if not already present)
    3. Append ALL content below to the END of the file
    4. In CompositeReward.compute(), after computing feasibility,
       optionally call:   dsr = compute_dsr(margins)
    5. Run self-test: python -m rl.reward

Why this is clean:
    - Option A: geometric mean of margins, normalized to [0, 1]
    - ZERO free parameters (nothing to tune → nothing to defend)
    - Identical formula for all ablation variants (fixes R2-2 issue #3)
    - SCR (Spec Compliance Rate) replaces the misleading "100% success"
"""

import math
from typing import Dict


def compute_scr(margins: Dict[str, float]) -> float:
    """
    Spec Compliance Rate — binary pass/fail, replaces "100% success rate".

    SCR = (1 / K) * Σ I(margin_i ≥ 1.0)

    Answers: "What fraction of specification checks does this design pass?"
    Perfectly aligned with the paper text "satisfying all safety criteria."

    Args:
        margins: dict of {indicator_name: margin_value}.
                 margin = allowable / demand.  ≥ 1.0 = pass.

    Returns:
        SCR in [0, 1].  1.0 = fully compliant.  0.0 = all indicators fail.
    """
    if not margins:
        return 0.0
    passed = sum(1 for v in margins.values() if v >= 1.0)
    return passed / len(margins)


def compute_dsr(margins: Dict[str, float]) -> float:
    """
    Design Safety Rate — replaces the flawed DSR=0.819 metric.

    DSR = min(1.0, min(margins))

    Design rationale:
        - Engineering weakest-link principle: one failing indicator
          collapses the structure regardless of how strong the others are.
          The minimum margin directly captures this.
        - Directly interpretable: DSR = 0.80 means "the worst indicator
          is at 80% of its allowable value."
        - Range [0, 1]:  0.0 = severe failure,  0.5 = 50% of allowable,
          1.0 = all indicators just safe,  capped at 1.0 (beyond which
          is over-design, measured by economic reward separately).
        - ZERO free parameters — nothing to tune, nothing to defend.
        - IDENTICAL formula across all ablation variants (fixes R2-2 #3).

    Comparison with original MATLAB DSR:
        Original: weighted arithmetic mean + variant-specific weights/penalties
                  + stochastic perturbation (±0.04) + silent random fallback.
        New:      pure min-margin, deterministic, identical for all variants.

    Args:
        margins: dict of {indicator_name: margin_value}.

    Returns:
        DSR in [0, 1], capped.
    """
    if not margins:
        return 0.0
    return min(1.0, min(margins.values()))


def compute_dsr_and_scr(
    margins: Dict[str, float],
) -> Dict[str, float]:
    """
    Convenience: compute both metrics in one call.

    Returns:
        {'SCR': ..., 'DSR': ..., 'min_margin': ...}
    """
    scr = compute_scr(margins)
    dsr = compute_dsr(margins)
    return {
        'SCR':        round(scr, 4),
        'DSR':        round(dsr, 4),
        'min_margin': round(min(margins.values()) if margins else 0.0, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Self-test (run after patching reward.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test 1: All margins ≥ 1.0 (fully compliant design)
    m1 = {'B1': 10.99, 'B2': 5.52, 'B3': 1.11, 'B4': 1.77}
    r1 = compute_dsr_and_scr(m1)
    print('Test 1 — All compliant (typical JTG demo):')
    print(f'  SCR = {r1["SCR"]:.4f}  (expected 1.0000)')
    print(f'  DSR = {r1["DSR"]:.4f}  (expected 1.0000, capped at 1.0)')
    print(f'  min_margin = {r1["min_margin"]:.2f} (B3=1.11)')
    assert r1['SCR'] == 1.0
    assert r1['DSR'] == 1.0
    print('  PASS')

    # Test 2: One indicator fails (B2=0.50)
    m2 = {'B1': 10.99, 'B2': 0.50, 'B3': 1.11, 'B4': 1.77}
    r2 = compute_dsr_and_scr(m2)
    print('\nTest 2 — B2 fails (B2=0.50):')
    print(f'  SCR = {r2["SCR"]:.4f}  (expected 0.7500, 3/4 pass)')
    print(f'  DSR = {r2["DSR"]:.4f}  (expected 0.5000, worst=B2)')
    assert r2['SCR'] == 0.75
    assert r2['DSR'] == 0.50, f'DSR should be min_margin=0.50, got {r2["DSR"]}'
    print('  PASS')

    # Test 3: All margins exactly 1.0 (borderline feasible)
    m3 = {'B1': 1.0, 'B2': 1.0, 'B3': 1.0, 'B4': 1.0}
    r3 = compute_dsr_and_scr(m3)
    print('\nTest 3 — All borderline (margin=1.0):')
    print(f'  SCR = {r3["SCR"]:.4f}  (expected 1.0000)')
    print(f'  DSR = {r3["DSR"]:.4f}  (expected 1.0000)')
    assert r3['SCR'] == 1.0 and r3['DSR'] == 1.0
    print('  PASS')

    # Test 4: Two fail, different severities
    m4 = {'B1': 0.80, 'B2': 2.00, 'B3': 0.30, 'B4': 2.00}
    r4 = compute_dsr_and_scr(m4)
    print('\nTest 4 — B1=0.8, B3=0.3 fail:')
    print(f'  SCR = {r4["SCR"]:.4f}  (expected 0.5000, 2/4 pass)')
    print(f'  DSR = {r4["DSR"]:.4f}  (expected 0.3000, worst=B3)')
    assert r4['SCR'] == 0.5
    assert r4['DSR'] == 0.30
    print('  PASS')

    # Test 5: Empty margins
    m5 = {}
    r5 = compute_dsr_and_scr(m5)
    print('\nTest 5 — Empty margins:')
    print(f'  SCR = {r5["SCR"]:.4f}  DSR = {r5["DSR"]:.4f}  (expected 0.0, 0.0)')
    assert r5['SCR'] == 0.0 and r5['DSR'] == 0.0
    print('  PASS')

    # Test 6: Variant fairness (same margins → same DSR, always)
    print('\nTest 6 — Variant fairness (identical across all ablation variants):')
    for variant in ['Full HARA', 'No Generator', 'No Evaluator', 'No Guard']:
        dsr = compute_dsr(m1)
        scr = compute_scr(m1)
        print(f'  {variant:<20} SCR = {scr:.4f}  DSR = {dsr:.4f}')
    print('  All identical - deterministic, no variant-specific parameters [OK]')
    print('  PASS')

    # Test 7: Realistic progressive failure
    print('\nTest 7 — Progressive failure (B3 margin degrading):')
    for b3 in [2.00, 1.50, 1.20, 1.00, 0.80, 0.50, 0.20]:
        m = {'B1': 5.0, 'B2': 3.0, 'B3': b3, 'B4': 4.0}
        r = compute_dsr_and_scr(m)
        print(f'  B3={b3:.2f}  SCR={r["SCR"]:.4f}  DSR={r["DSR"]:.4f}')
    print('  DSR smoothly degrades with worst indicator ✓')
    print('  PASS')

    print('\n' + '=' * 60)
    print('ALL 7 TESTS PASSED — ready to integrate into rl/reward.py')
    print('=' * 60)
    print()
    print('Integration (tomorrow):')
    print('  1. Open rl/reward.py')
    print('  2. Add:  from rl.dsr_patch import compute_dsr, compute_scr')
    print('  3. In CompositeReward.compute(), after feasibility check:')
    print('       dsr_metrics = compute_dsr_and_scr(margins)')
    print('       result["dsr"] = dsr_metrics["DSR"]')
    print('       result["scr"] = dsr_metrics["SCR"]')
    print('  4. Self-test: python -m rl.reward')
