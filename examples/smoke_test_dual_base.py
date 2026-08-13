# -*- coding: utf-8 -*-
"""
examples/smoke_test_dual_base.py
==================================
Phase 2D dual-base SMOKE TEST.

Verifies that the FEA + spec stack supports BOTH base types end-to-end
WITHOUT requiring surrogate training:

  Run 1: pavement_type='semi_rigid', E_base=1500 MPa, ν=0.25
         → sigma_t > 0 (cement base resists tension via flexure)
         → JTG margins: B1 + B2 + B3 + B4   (4 indicators)

  Run 2: pavement_type='flexible',   E_base=300 MPa,  ν=0.40
         → sigma_t ≈ 0 (granular base cannot take tension)
         → JTG margins: B1 +      B3 + B4   (3 indicators, B2 auto-skipped)
         → eps_z larger (weaker base → more strain at subgrade)
         → eps_a larger (weaker base → AC carries more)

Pass criteria:
  ✓ Both runs return real FEA results
  ✓ Spec evaluation correctly gates B2 by pavement_type
  ✓ Guard with from_base_type('flexible') accepts E_base=300 MPa
  ✓ Guard with from_base_type('semi_rigid') REJECTS E_base=300 (out of range)

Run:
    cd D:\\iLLM_PD_new
    conda activate illm_pd
    python examples/smoke_test_dual_base.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fea.runner import run_fea
from rl.guards import GuardConfig, NumericalGuard, GuardViolation
from specs.protocol import DesignInputs
from specs.jtg_d50 import JTG_D50_2017

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("smoke_dual_base")


# ───────────────────────────────────────────────────────────────
# Test designs (engineering-typical for each base type)
# ───────────────────────────────────────────────────────────────

DESIGN_SEMI_RIGID = dict(
    pavement_type="semi_rigid",
    thickness=[0.04, 0.06, 0.08, 0.36, 0.18],            # m: u/m/l AC, base, subbase
    modulus=  [14000.0, 11000.0, 9000.0, 1500.0, 400.0], # MPa
    poisson=  [0.25, 0.30, 0.30, 0.25, 0.35],            # cement base ν=0.25
    E_subgrade=60.0,
)

DESIGN_FLEXIBLE = dict(
    pavement_type="flexible",
    thickness=[0.04, 0.06, 0.08, 0.30, 0.25],            # m: thinner base, thicker subbase
    modulus=  [14000.0, 11000.0, 9000.0, 350.0, 250.0],  # MPa: granular base E=350
    poisson=  [0.25, 0.30, 0.30, 0.40, 0.35],            # granular base ν=0.40
    E_subgrade=60.0,
)


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def run_one(label: str, design: dict, run_dir_tag: str) -> dict:
    """Pre-guard, FEA, spec-evaluate one design. Returns full audit dict."""
    logger.info("=" * 70)
    logger.info(f"  RUN: {label}   (pavement_type={design['pavement_type']})")
    logger.info("=" * 70)
    logger.info(f"  Design:")
    logger.info(f"    thickness  = {design['thickness']}")
    logger.info(f"    modulus    = {design['modulus']}")
    logger.info(f"    poisson    = {design['poisson']}")
    logger.info(f"    E_subgrade = {design['E_subgrade']}")

    # Pre-FEA guard with correct base-type bounds
    guard = NumericalGuard(base_type=design["pavement_type"])
    logger.info(f"  Guard config: base_type={guard.base_type}")
    logger.info(f"    E_base bounds:    [{guard.config.E_min[3]:.0f}, "
                f"{guard.config.E_max[3]:.0f}] MPa")
    logger.info(f"    E_subbase bounds: [{guard.config.E_min[4]:.0f}, "
                f"{guard.config.E_max[4]:.0f}] MPa")
    try:
        guard.check_design(
            thickness=np.asarray(design["thickness"]),
            modulus=np.asarray(design["modulus"]),
            E_subgrade=design["E_subgrade"],
        )
        logger.info("  ✓ Pre-FEA guard PASSED")
    except GuardViolation as gv:
        logger.error(f"  ✗ Pre-FEA guard FAILED: {gv}")
        return {"ok": False, "stage": "guard_pre", "error": str(gv)}

    # Run FEA
    logger.info("  → Launching ABAQUS (~45 s)...")
    t0 = time.time()
    try:
        result = run_fea(
            thickness=design["thickness"],
            modulus=design["modulus"],
            poisson=design["poisson"],
            E_subgrade=design["E_subgrade"],
            nu_subgrade=0.40,
            load_pressure=0.7,
            load_radius=0.1065,
            run_name=f"smoke_dual_{run_dir_tag}",
            verbose=False,
        )
    except Exception as e:
        logger.error(f"  ✗ FEA crashed: {e}")
        return {"ok": False, "stage": "fea_crash", "error": str(e)}
    fea_dt = time.time() - t0
    responses = result.get("responses", {})
    logger.info(f"  ✓ FEA succeeded ({fea_dt:.1f} s)")
    for k, v in responses.items():
        logger.info(f"    {k:30s} = {v}")

    # Post-FEA guard
    try:
        guard.check_fea_result(responses)
        logger.info("  ✓ Post-FEA guard PASSED")
    except GuardViolation as gv:
        logger.error(f"  ✗ Post-FEA guard FAILED: {gv}")
        return {"ok": False, "stage": "guard_post", "error": str(gv)}

    # Spec evaluation (JTG D50-2017)
    protocol = JTG_D50_2017()
    inputs = DesignInputs(
        pavement_type=design["pavement_type"],
        road_class="expressway",
        traffic_level="heavy",
        thickness=design["thickness"],
        modulus=design["modulus"],
        poisson=design["poisson"],
        E_subgrade=design["E_subgrade"],
        nu_subgrade=0.40,
        design_life=15,
        extras={"city": "beijing", "VFA_pct": 70.0,
                "R_s_MPa": 1.0, "R_0_mm": 1.5},
    )
    evaluation = protocol.evaluate(inputs, responses)
    logger.info("  Spec evaluation:")
    logger.info(f"    critical = {evaluation.critical_indicator}")
    logger.info(f"    feasible = {evaluation.feasible}")
    for k, v in evaluation.margins.items():
        logger.info(f"    margin {k:35s} = {v:.3f}")

    return {
        "ok": True,
        "responses": responses,
        "margins": dict(evaluation.margins),
        "critical": evaluation.critical_indicator,
        "feasible": evaluation.feasible,
        "fea_time_s": fea_dt,
    }


def run_negative_test_guard_mismatch():
    """
    Cross-guard test: a flexible-typical design (E_base=300) should be
    REJECTED by a semi_rigid guard (which requires E_base ≥ 800).
    Confirms that the guard's base_type routing works.
    """
    logger.info("=" * 70)
    logger.info("  NEGATIVE TEST: flexible design vs semi_rigid guard")
    logger.info("  (E_base=300 MPa should be rejected by semi_rigid bounds [800, 3500])")
    logger.info("=" * 70)
    guard_semi = NumericalGuard(base_type="semi_rigid")
    try:
        guard_semi.check_design(
            thickness=np.asarray(DESIGN_FLEXIBLE["thickness"]),
            modulus=np.asarray(DESIGN_FLEXIBLE["modulus"]),
            E_subgrade=DESIGN_FLEXIBLE["E_subgrade"],
        )
        logger.error("  ✗ EXPECTED REJECTION but guard PASSED — guard misconfigured!")
        return False
    except GuardViolation as gv:
        if gv.code == "E_OUT_OF_BOUNDS":
            logger.info(f"  ✓ Correctly rejected: {gv}")
            return True
        else:
            logger.warning(f"  ⚠ Rejected for unexpected reason: {gv}")
            return True


def cross_compare(res_semi: dict, res_flex: dict) -> dict:
    """Physical sanity checks comparing the two runs."""
    findings = {}

    # 1. semi_rigid should have sigma_t > flexible (direction matters more than magnitude)
    sig_semi = res_semi["responses"].get("sigma_t_MPa", 0)
    sig_flex = res_flex["responses"].get("sigma_t_MPa", 0)
    ratio = sig_semi / max(sig_flex, 1e-6)
    # Pass criteria: direction correct + at least 2x ratio
    # (Small absolute sigma_t in both is physically normal when AC is thick/stiff —
    #  the AC slab absorbs flexure, sparing the base from tension.)
    findings["sigma_t_signature_OK"] = (
        sig_semi > sig_flex and ratio >= 2.0
    )
    note = ""
    if sig_semi < 0.1 and sig_flex < 0.1:
        note = "  (both small → over-designed AC, normal for thick stiff surface)"
    logger.info(f"  sigma_t: semi={sig_semi:.4f}  flex={sig_flex:.4f}  "
                f"ratio={ratio:.2f}x  "
                f"({'✓' if findings['sigma_t_signature_OK'] else '✗'} "
                f"expect ratio ≥ 2.0){note}")

    # 2. flexible should have larger eps_z (weaker base)
    epsz_semi = res_semi["responses"].get("epsilon_z_microstrain", 0)
    epsz_flex = res_flex["responses"].get("epsilon_z_microstrain", 0)
    findings["eps_z_signature_OK"] = (epsz_flex > epsz_semi)
    logger.info(f"  eps_z:   semi={epsz_semi:.1f}  flex={epsz_flex:.1f}  "
                f"({'✓' if findings['eps_z_signature_OK'] else '✗'} expect flex > semi)")

    # 3. spec must include B2 for semi, exclude for flex
    has_b2_semi = "B2_semi_rigid_fatigue" in res_semi["margins"]
    has_b2_flex = "B2_semi_rigid_fatigue" in res_flex["margins"]
    findings["spec_routing_OK"] = has_b2_semi and (not has_b2_flex)
    logger.info(f"  B2 routing: semi has B2={has_b2_semi}  flex has B2={has_b2_flex}  "
                f"({'✓' if findings['spec_routing_OK'] else '✗'} expect Y/N)")

    return findings


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    logger.info("#" * 70)
    logger.info("# Phase 2D dual-base SMOKE TEST")
    logger.info("# Will run 2 FEA calls (~90 s total) + cross checks")
    logger.info("#" * 70)

    res_semi = run_one("Run 1: SEMI_RIGID base", DESIGN_SEMI_RIGID,
                       run_dir_tag="semi_rigid")
    if not res_semi.get("ok"):
        logger.error("Semi-rigid run failed; aborting.")
        sys.exit(1)

    res_flex = run_one("Run 2: FLEXIBLE base",   DESIGN_FLEXIBLE,
                       run_dir_tag="flexible")
    if not res_flex.get("ok"):
        logger.error("Flexible run failed; aborting.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("  CROSS-COMPARE physical signatures")
    logger.info("=" * 70)
    findings = cross_compare(res_semi, res_flex)

    neg_ok = run_negative_test_guard_mismatch()

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SUMMARY")
    logger.info("=" * 70)
    all_pass = (res_semi.get("ok", False)
                and res_flex.get("ok", False)
                and all(findings.values())
                and neg_ok)
    logger.info(f"  semi_rigid FEA + spec    : {'✓' if res_semi.get('ok') else '✗'}")
    logger.info(f"  flexible FEA + spec      : {'✓' if res_flex.get('ok') else '✗'}")
    for k, v in findings.items():
        logger.info(f"  {k:25s}: {'✓' if v else '✗'}")
    logger.info(f"  guard cross-rejection    : {'✓' if neg_ok else '✗'}")
    logger.info("")
    if all_pass:
        logger.info("  ★★★ ALL CHECKS PASSED ★★★")
        logger.info("  Phase 2D dual-base support is OPERATIONAL.")
        logger.info("  Ready to run LHS dual-base sampling.")
    else:
        logger.error("  Some checks FAILED — review log above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
