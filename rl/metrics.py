"""
rl/metrics.py — Clean Design Safety metrics for iLLM-PD (Phase 2D / R2-2 fix)
=============================================================================

Replaces the original MATLAB DSR formulation, which had three issues identified
during R2-2 forensic review (2026-05-21):

  1. Stochastic perturbation injected into DSR post-computation
       (`add_dsr_variation = true` → DSR += uniform(-0.04, +0.04))
  2. Silent failure with random fallback in catch block
       (`catch DSR = 0.3 + rand() * 0.4`)
  3. Variant-specific weights AND penalties in ablation comparison
       (Full p=0, NoLLMParsing p=0.03, NoLLMGuidance p=0.05, ReducedStability p=0.08)

This module provides TWO complementary, clean metrics that any external
auditor can reproduce bit-for-bit given the same margin inputs:

  • SCR (Spec Compliance Rate) — binary, replaces misleading "100% success rate"
       SCR = fraction of trials where all spec margins >= 1.0
       Range: [0, 1].  Aligned exactly with paper text
       "designs satisfying all safety criteria".

  • DSR (Design Safety Reserve) — continuous quality score (Option A: geomean)
       DSR = min(1, geomean(valid margins) / safety_anchor)
       Default safety_anchor = 1.5  (i.e. 50% safety overhead = ideal target)
       Range: [0, 1].  Geometric mean penalizes any single weak margin
       (one near-failing dimension drags the whole score down — desired behavior).

Both metrics:
  - Use IDENTICAL formula across all ablation variants (no variant-specific terms)
  - Have NO random perturbation
  - Return 0.0 (not a random number) on invalid/missing margins
  - Are pure functions of the margins dict — fully deterministic

Margin convention (already established elsewhere in iLLM-PD):
  margin_i = allowable_i / actual_i        (for indicators where lower-actual = better)
  margin_i = actual_i  / threshold_i       (for indicators where higher-actual = better)
  → margin_i >= 1.0  means the i-th criterion is satisfied.
  → margin_i  < 1.0  means the i-th criterion is violated.

Used by:
  - rl/reward.py (eventually — wire after PI approval)
  - Ablation rerun script (replaces MATLAB calculateDSRCompatibleEmbedded)
  - Multi-seed analysis (R1-3, R3-15)
  - LTPP multi-section evaluation (R2-1)
"""

from __future__ import annotations
import math
from typing import Iterable, Mapping, Optional


# --------------------------------------------------------------------------
#  Defaults
# --------------------------------------------------------------------------

SAFETY_ANCHOR_DEFAULT: float = 1.5
"""Margin value at which DSR saturates to 1.0.

Engineering rationale: margin=1.5 corresponds to a 50% safety reserve
above the minimum specification requirement. This is a common target in
infrastructure design.  Lowering this anchor (e.g. to 1.0) would saturate
DSR too quickly and lose discrimination among compliant designs;
raising it (e.g. to 3.0) would make most designs score poorly.

Sensitivity analysis on this anchor is included in the ablation suite
(report `dsr_at_anchor_{1.0, 1.5, 2.0, 2.5}` for each variant).
"""

COMPLIANCE_THRESHOLD: float = 1.0
"""A margin >= this value is considered compliant for SCR computation.

Must equal 1.0 by the margin convention above. Exposed as a parameter
only for future-proofing (e.g. if the paper later defines a stricter
threshold for "high-confidence compliance").
"""


# --------------------------------------------------------------------------
#  Single-trial metrics
# --------------------------------------------------------------------------

def _valid_margins(margins: Mapping[str, Optional[float]]) -> list[float]:
    """Extract finite, positive margin values from a margins dict.

    Returns an empty list if no margins are valid.  This is the single point
    where 'invalid input' is defined — used by both DSR and SCR.
    """
    out: list[float] = []
    for value in margins.values():
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            continue
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            continue
        if v <= 0:
            # A non-positive margin means FEA failure or numerical breakdown.
            # We do NOT silently filter it out — it must register as failure.
            # The convention: caller must already have replaced 'computed' margins
            # by None if the FEA itself failed. A 0 or negative here means the
            # design is genuinely at/below the failure boundary.
            return []  # short-circuit: any non-positive → invalid trial
        out.append(v)
    return out


def compute_dsr(
    margins: Mapping[str, Optional[float]],
    safety_anchor: float = SAFETY_ANCHOR_DEFAULT,
) -> float:
    """Compute DSR (Design Safety Reserve) — Option A: geometric mean.

    Args:
        margins: dict like {'B1_asphalt_fatigue': 1.84, 'B3_...': 1.10, ...}
                 Values that are None or NaN are excluded (e.g. B2 for flexible
                 pavements where no semi-rigid base exists).
        safety_anchor: margin value at which DSR saturates to 1.0.

    Returns:
        DSR in [0.0, 1.0].
        - 0.0 if no valid margins, any margin <= 0, or all margins missing.
        - geomean(margins) / safety_anchor otherwise, clipped at 1.0.

    Formula:
        DSR = min(1, (∏ margin_i)^(1/n) / safety_anchor)

    Worked examples:
        margins = {B1: 35, B2: 5.0, B3: 1.56, B4: 3.64}  →  DSR = 1.000
        margins = {B1: 1.5, B2: 1.2, B3: 1.1, B4: 1.3}   →  DSR ≈ 0.844
        margins = {B1: 1.0, B2: 0.95, B3: 0.9, B4: 1.05} →  DSR ≈ 0.649
        margins = {B1: 0.5, B2: 0.5, B3: 0.5, B4: 0.5}   →  DSR ≈ 0.333
        margins = {B1: None, B3: 1.5, B4: 1.5}            →  DSR = 1.000  (flexible)
        margins = {B1: 0, B2: 5, B3: 5, B4: 5}            →  DSR = 0.000  (failed)
    """
    valid = _valid_margins(margins)
    if not valid:
        return 0.0
    if safety_anchor <= 0:
        raise ValueError(f"safety_anchor must be positive, got {safety_anchor!r}")
    # Geometric mean = exp(mean(log(x))) — numerically stabler than ∏x^(1/n)
    log_sum = sum(math.log(m) for m in valid)
    geomean = math.exp(log_sum / len(valid))
    return min(1.0, geomean / safety_anchor)


def compute_compliance(
    margins: Mapping[str, Optional[float]],
    threshold: float = COMPLIANCE_THRESHOLD,
) -> bool:
    """True iff all valid margins >= threshold.

    Empty/invalid margin set returns False (an unevaluable design is not compliant).
    """
    valid = _valid_margins(margins)
    if not valid:
        return False
    return all(m >= threshold for m in valid)


# --------------------------------------------------------------------------
#  Cross-trial aggregation
# --------------------------------------------------------------------------

def compute_scr(
    margin_history: Iterable[Mapping[str, Optional[float]]],
    threshold: float = COMPLIANCE_THRESHOLD,
) -> float:
    """SCR (Spec Compliance Rate) — fraction of trials with all margins compliant.

    This replaces the original paper's misleading '100% success rate' phrase,
    whose actual MATLAB definition was '(PDE_converged) AND (DSR >= 0.5)' —
    a weaker condition than 'satisfies all safety criteria'.

    Args:
        margin_history: iterable of margins dicts, one per trial.
        threshold: compliance cutoff (default 1.0).

    Returns:
        SCR in [0.0, 1.0].
    """
    trials = list(margin_history)
    if not trials:
        return 0.0
    n_compliant = sum(1 for m in trials if compute_compliance(m, threshold))
    return n_compliant / len(trials)


def report_ablation_metrics(
    trials: Iterable[Mapping],
    margins_key: str = "margins",
    safety_anchor: float = SAFETY_ANCHOR_DEFAULT,
) -> dict:
    """Aggregate trial results into ablation-ready statistics.

    This is the single function the ablation script and multi-seed analysis
    should call.  No variant-specific terms — same formula for all variants.

    Args:
        trials: iterable of dicts; each must contain a `margins` field.
                Other fields (cost, runtime, etc.) ignored here.
        margins_key: key under which the margins dict lives.
        safety_anchor: passed to compute_dsr.

    Returns:
        {
            'n_trials': int,
            'scr': float,                    # 0..1, binary compliance rate
            'dsr_values': list[float],       # raw per-trial DSRs
            'dsr_mean': float,
            'dsr_median': float,
            'dsr_std': float,                # sample std (n-1 denom)
            'dsr_q25': float,                # 25th percentile
            'dsr_q75': float,                # 75th percentile
            'dsr_min': float,
            'dsr_max': float,
        }

    Designed for direct consumption by:
      - Ablation table generator
      - Boxplot/violin plot generator (use dsr_values)
      - Multi-seed CI computation (use dsr_std + n_trials)
    """
    trials_list = list(trials)
    n = len(trials_list)
    if n == 0:
        return {
            "n_trials": 0, "scr": 0.0,
            "dsr_values": [], "dsr_mean": 0.0, "dsr_median": 0.0,
            "dsr_std": 0.0, "dsr_q25": 0.0, "dsr_q75": 0.0,
            "dsr_min": 0.0, "dsr_max": 0.0,
        }

    dsr_values: list[float] = []
    n_compliant = 0
    for t in trials_list:
        margins = t.get(margins_key, {})
        dsr_values.append(compute_dsr(margins, safety_anchor))
        if compute_compliance(margins):
            n_compliant += 1

    # Sort once for quantile computation.
    sorted_dsr = sorted(dsr_values)

    def _quantile(q: float) -> float:
        # Linear interpolation, matches numpy default ('linear' / type-7).
        if n == 1:
            return sorted_dsr[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_dsr[lo]
        frac = pos - lo
        return sorted_dsr[lo] * (1 - frac) + sorted_dsr[hi] * frac

    mean = sum(dsr_values) / n
    if n >= 2:
        var = sum((v - mean) ** 2 for v in dsr_values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    return {
        "n_trials": n,
        "scr": n_compliant / n,
        "dsr_values": dsr_values,
        "dsr_mean": mean,
        "dsr_median": _quantile(0.5),
        "dsr_std": std,
        "dsr_q25": _quantile(0.25),
        "dsr_q75": _quantile(0.75),
        "dsr_min": sorted_dsr[0],
        "dsr_max": sorted_dsr[-1],
    }


# --------------------------------------------------------------------------
#  Self-test (run as `python -m rl.metrics` to verify)
# --------------------------------------------------------------------------

def _selftest() -> None:
    """Lightweight assertion-based unit tests. No external deps."""
    # --- compute_dsr ---
    # Very safe design → DSR = 1
    assert compute_dsr({"B1": 35, "B2": 5, "B3": 1.56, "B4": 3.64}) == 1.0
    # Borderline-safe → DSR around 0.84
    v = compute_dsr({"B1": 1.5, "B2": 1.2, "B3": 1.1, "B4": 1.3})
    assert 0.83 < v < 0.86, f"got {v!r}"
    # Just-failing → DSR < 1
    v = compute_dsr({"B1": 1.0, "B2": 0.95, "B3": 0.9, "B4": 1.05})
    assert 0.63 < v < 0.67, f"got {v!r}"
    # Heavily-failing → DSR ≈ 0.333
    v = compute_dsr({"B1": 0.5, "B2": 0.5, "B3": 0.5, "B4": 0.5})
    assert 0.32 < v < 0.34, f"got {v!r}"
    # B2 missing (flexible pavement) → use only B1/B3/B4
    v_flex = compute_dsr({"B1": 1.5, "B2": None, "B3": 1.5, "B4": 1.5})
    assert v_flex == 1.0, f"got {v_flex!r}"
    # Any margin <= 0 → DSR = 0 (FEA failure)
    assert compute_dsr({"B1": 0, "B2": 5, "B3": 5, "B4": 5}) == 0.0
    assert compute_dsr({"B1": -0.1, "B2": 5, "B3": 5, "B4": 5}) == 0.0
    # NaN/inf → skipped
    assert compute_dsr({"B1": float("nan"), "B3": 1.5, "B4": 1.5}) == 1.0
    # All invalid → 0
    assert compute_dsr({"B1": None}) == 0.0
    assert compute_dsr({}) == 0.0

    # --- compute_compliance / compute_scr ---
    assert compute_compliance({"B1": 1.5, "B3": 1.1}) is True
    assert compute_compliance({"B1": 1.5, "B3": 0.95}) is False
    assert compute_compliance({"B1": None}) is False  # empty/invalid → False
    scr = compute_scr([
        {"B1": 1.5, "B3": 1.1, "B4": 1.3},   # compliant
        {"B1": 0.9, "B3": 1.5, "B4": 1.2},   # B1 fails
        {"B1": 1.1, "B3": 1.0, "B4": 1.0},   # compliant (exactly at threshold)
    ])
    assert abs(scr - 2.0 / 3.0) < 1e-9, f"got {scr!r}"
    assert compute_scr([]) == 0.0

    # --- report_ablation_metrics ---
    report = report_ablation_metrics([
        {"margins": {"B1": 1.5, "B3": 1.1, "B4": 1.3}},
        {"margins": {"B1": 1.5, "B3": 1.5, "B4": 1.5}},
        {"margins": {"B1": 0.9, "B3": 1.5, "B4": 1.2}},
        {"margins": {"B1": 1.1, "B3": 1.0, "B4": 1.0}},
    ])
    assert report["n_trials"] == 4
    assert abs(report["scr"] - 0.75) < 1e-9
    assert len(report["dsr_values"]) == 4
    assert 0 <= report["dsr_min"] <= report["dsr_median"] <= report["dsr_max"] <= 1.0

    # Edge case: empty trials
    empty = report_ablation_metrics([])
    assert empty["n_trials"] == 0
    assert empty["scr"] == 0.0

    print("[rl.metrics] self-test PASSED ✓")
    print(f"  Example: 4 trials, SCR = {report['scr']:.3f}, "
          f"DSR mean = {report['dsr_mean']:.3f} ± {report['dsr_std']:.3f}")


if __name__ == "__main__":
    _selftest()
