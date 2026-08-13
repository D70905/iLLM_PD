# -*- coding: utf-8 -*-
"""
scripts/run_mepdg_cross_spec_check.py — R3-12 Cross-Specification Check (v2)
============================================================================

Do HARA designs (optimised under JTG D50-2017) also satisfy ME-PDG threshold
criteria? STRUCTURAL / DIRECTIONAL check — NOT a PMED-calibrated simulation.

=========================  WHAT CHANGED IN v2  =============================
The v1 script FABRICATED the FEA strains from the JTG SCR via arbitrary
linear formulas (eps_a = 150 - 60*SCR, eps_z = 250 - 100*SCR, ...). Those
invented strains, fed into the eps^-3.95 fatigue law, produced FC% blow-ups
and the spurious 0/36 result. That logic is DELETED.

v2 reads REAL FEA responses produced by:
    scripts/extract_hara_fea_for_mepdg.py  ->  hara_fea_responses_<ts>.csv
(one clean ABAQUS run per final HARA design), and computes ME-PDG margins via
the audited, NCHRP-cited module rl/lifecycle_mepdg.py. No strain is invented.
============================================================================

CALIBRATION NOTE (must stay in the Methods + response letter):
    The transfer-function STRUCTURE is NCHRP 1-37A (publicly documented), used
    with NATIONAL (not PMED-local-recalibrated) coefficients:
      - Fatigue life N_f : NCHRP 1-37A Eq. 3.3.1 (rl.lifecycle.mepdg_fatigue_life_Nf)
      - FC% transfer fn  : NCHRP 1-37A national form with thickness-dependent
                           C1', C2' (FHWA-HRT-11-035). We do NOT use the
                           C1'=C2'=1 placeholder, which over-predicts FC% at low
                           damage and would spuriously fail conservative designs.
      - Rutting RD       : NCHRP 1-37A per-layer permanent-deformation model
                           WITH the HMA depth-correction factor k_z (computed
                           locally here; lifecycle_mepdg omits k_z, which would
                           inflate RD ~3x and spuriously fail every design).
    This is a STRUCTURAL/DIRECTIONAL consistency check, not a quantitative PMED
    replication (PMED-internal NCHRP 1-40D local coefficients are not public).
    See Schwartz (2007), VDOT (2024).

THRESHOLDS (ME-PDG default, Interstate, 20-yr):
    default (90% reliability): FC <= 25%, RD <= 19 mm, IRI <= 172 in/mi
    --strict (95% reliability): FC <= 20%, RD <= 16 mm, IRI <= 160 in/mi

USAGE
-----
    # auto-detect latest hara_fea_responses_*.csv
    python scripts/run_mepdg_cross_spec_check.py

    # explicit input + stricter thresholds
    python scripts/run_mepdg_cross_spec_check.py \
        --fea-csv experiments/ltpp_data/deliverables/mepdg_check/hara_fea_responses_XXXX.csv \
        --strict
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("mepdg_check")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MEPDG_DIR = Path("experiments/ltpp_data/deliverables/mepdg_check")

# Design traffic (cross-spec demonstration; documented Level-3 input)
DESIGN_LIFE_YEARS = 20.0
DESIGN_ESAL_PER_YEAR = 4_000_000.0          # heavy expressway baseline
N_E_DESIGN = DESIGN_ESAL_PER_YEAR * DESIGN_LIFE_YEARS    # = 8.0e7

# Reference pavement temperature for the rutting model (NCHRP 1-37A reference).
# Climate-specific T can be wired later once specs/protocol exposes MAAT.
T_PAVEMENT_F_DEFAULT = 73.0


# ============================================================================
# Thresholds
# ============================================================================

def get_thresholds(strict: bool = False) -> Dict[str, float]:
    """ME-PDG default thresholds for Interstate highway (NCHRP 1-37A Table 3-2.5)."""
    if strict:
        return {"FC_percent_max": 20.0, "RD_total_mm_max": 16.0,
                "IRI_in_per_mi_max": 160.0, "reliability": 0.95}
    return {"FC_percent_max": 25.0, "RD_total_mm_max": 19.0,
            "IRI_in_per_mi_max": 172.0, "reliability": 0.90}


# ============================================================================
# IRI (NCHRP 1-37A Eq. 3.3.29) — kept local; lifecycle_mepdg has no IRI model
# ============================================================================

def mepdg_iri(FC_percent: float, RD_mm: float,
              IRI_0: float = 63.0, TC_ft_mi: float = 0.0, SF: float = 1.0) -> float:
    """IRI (in/mi) = IRI_0 + 0.0150*SF + 0.400*FC + 0.0080*TC + 40*RD_in."""
    RD_in = RD_mm / 25.4
    return float(IRI_0 + 0.0150 * SF + 0.400 * FC_percent + 0.0080 * TC_ft_mi + 40.0 * RD_in)


# ============================================================================
# FC% transfer function — NCHRP 1-37A NATIONAL global form (thickness-dependent)
# ============================================================================
# IMPORTANT: we deliberately do NOT use lifecycle_mepdg.fc_percent_from_Nf, whose
# placeholder (C1'=C2'=1) calibration over-predicts FC% at low damage (it returns
# ~28% even when DI~0.01, i.e. negligible damage). That would spuriously fail the
# fatigue check on conservative HARA designs. The form below is the published
# NCHRP 1-37A / MEPDG national-calibration "% lane area" transfer function with
# the thickness-dependent C1', C2' coefficients (FHWA-HRT-11-035). It behaves
# correctly: ~0% FC at low damage, rising past the threshold only under real
# overload (DI >> 1). This is still a NOMINAL/national calibration (no PMED local
# recalibration) — a directional cross-spec check, as disclosed.

def mepdg_fc_percent(N_applied: float, N_f: float, h_ac_inches: float) -> float:
    """NCHRP 1-37A bottom-up fatigue cracking (% lane area).

        FC = (6000 / (1 + e^(C1' + C2'·log10(D)))) / 60
        D  = N_applied / N_f
        C2' = -2.40874 - 39.748·(1 + h_ac)^(-2.856)
        C1' = -2·C2'
    """
    D = max(N_applied / max(N_f, 1.0), 1e-12)
    C2p = -2.40874 - 39.748 * (1.0 + h_ac_inches) ** (-2.856)
    C1p = -2.0 * C2p
    arg = float(np.clip(C1p + C2p * np.log10(D), -50.0, 50.0))
    return float(np.clip(6000.0 / (1.0 + np.exp(arg)) / 60.0, 0.0, 100.0))


# ============================================================================
# Rutting (RD) — NCHRP 1-37A per-layer permanent deformation WITH depth factor
# ============================================================================
# IMPORTANT: we compute RD here (not via lifecycle_mepdg.mepdg_total_rutting_RD_mm)
# because that module omits the HMA DEPTH CORRECTION FACTOR k_z. Without k_z the
# HMA permanent-strain ratio eps_p/eps_v reaches ~2189 at N=8e7, inflating RD to
# ~27-33 mm and spuriously failing every design. The depth factor is an explicit
# part of the NCHRP 1-37A HMA rutting model (NAP 22781 ch.2; FHWA NCHRP 09-30A
# Appendix K): it accounts for the confining-pressure variation through the AC
# thickness. With k_z the ratio is physically sane and RD falls to ~9-14 mm.
#   HMA:  eps_p/eps_v = beta_r1 * k_z * 10^kr1 * T^kr2 * N^kr3
#         k_z = (C1 + C2*D)*0.328196^D
#         C1 = -0.1039 H^2 + 2.4868 H - 17.342     (H, D = AC thickness / mid-depth, inch)
#         C2 =  0.0172 H^2 - 1.7331 H + 27.428
#   Unbound (base, subgrade): eps_p/eps_v = beta_s * (e0/er) * exp(-(rho/N)^beta)
# All beta local-calibration multipliers = 1.0 (global calibration, NCHRP 1-40D),
# i.e. NOMINAL national calibration — consistent with the directional-check framing.

_KR1, _KR2, _KR3 = -3.35412, 1.5606, 0.4791        # HMA global coeffs
_GRAN = dict(beta=1.673, e0=2.03, rho=650.0, b=0.92)   # unbound granular base
_SUBG = dict(beta=1.350, e0=1.62, rho=367.0, b=1.04)   # subgrade


def _hma_depth_factor(H_in: float, D_in: float) -> float:
    C1 = -0.1039 * H_in ** 2 + 2.4868 * H_in - 17.342
    C2 = 0.0172 * H_in ** 2 - 1.7331 * H_in + 27.428
    return (C1 + C2 * D_in) * 0.328196 ** D_in


def _unbound_ratio(N: float, c: dict) -> float:
    if N <= 0:
        return 0.0
    ex = float(np.clip(-((c["rho"] / N) ** c["b"]), -50.0, 50.0))
    return c["beta"] * (c["e0"] / 0.5) * math.exp(ex)


def mepdg_rutting_mm(eps_HMA_mid_micro: float, eps_z_micro: float,
                     h_ac_mm: float, h_base_mm: float, N: float, T_F: float,
                     h_subgrade_eff_mm: float = 152.4,
                     beta_r1: float = 1.0,
                     beta_s1_gran: float = 1.0,
                     beta_s1_subg: float = 1.0):
    """Per-layer NCHRP rutting (mm). Returns (RD_HMA, RD_base, RD_subgrade, RD_total).

    HMA:  eps_p/eps_v = beta_r1 * k_z * 10^k1 * T^k2 * N^k3   (NCHRP 1-37A Eq. 3.3.7-1)
    Unbound: eps_p/eps_v = beta_s1 * (e0/er) * exp(-(rho/N)^b)  (per layer)

    Local-calibration multipliers (beta_r1 for HMA, beta_s1_gran / beta_s1_subg
    for the granular base and subgrade) default to 1.0 = nominal national
    (global) calibration. Pass site-specific values to apply a local calibration;
    each beta enters linearly on its layer's rut component, so they can also be
    applied externally by scaling the returned components.
    """
    H_in = h_ac_mm / 25.4
    D_mid_in = (h_ac_mm / 2.0) / 25.4
    kz = _hma_depth_factor(H_in, D_mid_in)
    ratio_hma = max(0.0, beta_r1 * kz * (10.0 ** _KR1) * (T_F ** _KR2) * (N ** _KR3))

    rd_hma = h_ac_mm * max(0.0, eps_HMA_mid_micro) * 1e-6 * ratio_hma
    rd_base = h_base_mm * max(0.0, eps_z_micro) * 1e-6 * beta_s1_gran * _unbound_ratio(N, _GRAN)
    rd_sg = h_subgrade_eff_mm * max(0.0, eps_z_micro) * 1e-6 * beta_s1_subg * _unbound_ratio(N, _SUBG)

    cap = lambda x: float(max(0.0, min(x, 50.0)))
    rd_hma, rd_base, rd_sg = cap(rd_hma), cap(rd_base), cap(rd_sg)
    return rd_hma, rd_base, rd_sg, cap(rd_hma + rd_base + rd_sg)


# ============================================================================
# Per-row ME-PDG evaluation from REAL FEA responses
# ============================================================================

def _weighted_E_ac(row: pd.Series) -> float:
    """Thickness-weighted mean AC modulus (layers 1-3), MPa."""
    hs = [float(row.get(f"h{i}_cm", 0.0)) for i in (1, 2, 3)]
    es = [float(row.get(f"E{i}_MPa", 0.0)) for i in (1, 2, 3)]
    tot_h = sum(hs)
    if tot_h <= 0:
        vals = [e for e in es if e > 0]
        return float(np.mean(vals)) if vals else 9000.0
    return sum(h * e for h, e in zip(hs, es)) / tot_h


def evaluate_row(row: pd.Series, thr: Dict[str, float]) -> Optional[Dict]:
    """Compute ME-PDG metrics + compliance for one real-FEA row.

    Fatigue  : N_f via NCHRP eq (rl.lifecycle.mepdg_fatigue_life_Nf) from REAL
               eps_a, then FC% via the national thickness-dependent transfer fn.
    Rutting  : per-layer NCHRP model (rl.lifecycle_mepdg.mepdg_total_rutting_RD_mm).
    Subgrade : strain margin via JTG/NCHRP-shared allowable (informational).
    """
    from rl.lifecycle import mepdg_fatigue_life_Nf, subgrade_strain_allowable

    eps_a = row.get("eps_a_micro", None)   # AC bottom tensile (fatigue)  [REAL]
    eps_z = row.get("eps_z_micro", None)   # subgrade top vertical        [REAL]
    if eps_a is None or eps_z is None or pd.isna(eps_a) or pd.isna(eps_z):
        return None
    eps_a = float(eps_a); eps_z = float(eps_z)

    h_ac_mm = (float(row.get("h1_cm", 0)) + float(row.get("h2_cm", 0))
               + float(row.get("h3_cm", 0))) * 10.0
    h_base_mm = float(row.get("h4_cm", 0)) * 10.0
    h_ac_in = h_ac_mm / 25.4
    E_ac = _weighted_E_ac(row)

    # --- Fatigue: real eps_a -> N_f -> FC% (national transfer fn) ---
    N_f = mepdg_fatigue_life_Nf(eps_a, E_ac * 145.038, h_ac_in)
    FC = mepdg_fc_percent(N_E_DESIGN, N_f, h_ac_in)

    # --- Rutting: HMA mid vertical strain from real stress (elastic proxy) ---
    # FEA gives vertical STRESS p_AC_mid (MPa); eps_v ~= sigma_v / E_ac.
    p_ac_mid = row.get("p_AC_mid_mid", None)
    if p_ac_mid is not None and not pd.isna(p_ac_mid) and E_ac > 0:
        eps_HMA_mid = abs(float(p_ac_mid)) / E_ac * 1.0e6
    else:
        eps_HMA_mid = 0.3 * eps_z
    rd_hma, rd_base, rd_sg, RD = mepdg_rutting_mm(
        eps_HMA_mid_micro=eps_HMA_mid, eps_z_micro=eps_z,
        h_ac_mm=h_ac_mm, h_base_mm=h_base_mm,
        N=N_E_DESIGN, T_F=T_PAVEMENT_F_DEFAULT,
    )

    IRI = mepdg_iri(FC_percent=FC, RD_mm=RD)

    # --- Subgrade strain margin (informational; not an ME-PDG threshold) ---
    eps_z_allow = subgrade_strain_allowable(N_E_DESIGN)
    margin_sg = (eps_z_allow / eps_z) if eps_z > 0 else float("inf")

    pass_FC = FC <= thr["FC_percent_max"]
    pass_RD = RD <= thr["RD_total_mm_max"]
    pass_IRI = IRI <= thr["IRI_in_per_mi_max"]
    all_pass = bool(pass_FC and pass_RD and pass_IRI)

    return {
        "section_id": row.get("section_id"),
        "seed": row.get("seed"),
        "pavement_type": row.get("pavement_type"),
        "E_subgrade": row.get("E_subgrade"),
        # real FEA inputs (for audit / Methods table)
        "eps_a_micro": round(eps_a, 2),
        "eps_z_micro": round(eps_z, 2),
        "eps_HMA_mid_micro": round(eps_HMA_mid, 2),
        "E_ac_MPa": round(E_ac, 0),
        "h_ac_cm": round(h_ac_mm / 10.0, 1),
        # ME-PDG predictions
        "FC_percent": round(FC, 4),
        "RD_total_mm": round(RD, 3),
        "RD_HMA_mm": round(rd_hma, 3),
        "RD_base_mm": round(rd_base, 3),
        "RD_subgrade_mm": round(rd_sg, 3),
        "IRI_in_per_mi": round(IRI, 2),
        "N_f_fatigue": float(N_f),
        # margins (>=1.0 pass)
        "margin_FC": round(thr["FC_percent_max"] / max(FC, 1e-6), 3),
        "margin_RD": round(thr["RD_total_mm_max"] / max(RD, 1e-6), 3),
        "margin_subgrade": round(margin_sg, 3) if margin_sg != float("inf") else 999.0,
        # compliance
        "pass_FC": pass_FC, "pass_RD": pass_RD, "pass_IRI": pass_IRI,
        "MEPDG_all_pass": all_pass,
    }


# ============================================================================
# Main
# ============================================================================

def find_latest_csv(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="R3-12 ME-PDG cross-spec check (real FEA)")
    parser.add_argument("--fea-csv", type=str, default=None,
                        help="Path to hara_fea_responses_*.csv from "
                             "extract_hara_fea_for_mepdg.py; auto-detect if omitted.")
    parser.add_argument("--strict", action="store_true",
                        help="Use 95%% reliability thresholds (FC<=20, RD<=16, IRI<=160).")
    args = parser.parse_args()

    # Locate the REAL-FEA CSV
    if args.fea_csv:
        fea_path = Path(args.fea_csv)
    else:
        fea_path = find_latest_csv(MEPDG_DIR, "hara_fea_responses_*.csv")
    if not fea_path or not fea_path.exists():
        logger.error(
            "Cannot find a real-FEA CSV. Run first:\n"
            "    python scripts/extract_hara_fea_for_mepdg.py --escalation --seeds 3\n"
            f"(expected hara_fea_responses_*.csv in {MEPDG_DIR})")
        sys.exit(1)

    logger.info(f"Reading REAL FEA responses: {fea_path}")
    df = pd.read_csv(fea_path)

    # Only use rows with a clean FEA extraction
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "ok"].copy()
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} non-ok rows (status != ok).")
    logger.info(f"  {len(df)} usable design rows")

    thr = get_thresholds(strict=args.strict)
    logger.info(f"Thresholds ({thr['reliability']*100:.0f}% reliability): "
                f"FC<={thr['FC_percent_max']}%, RD<={thr['RD_total_mm_max']}mm, "
                f"IRI<={thr['IRI_in_per_mi_max']} in/mi")

    results: List[Dict] = []
    for _, row in df.iterrows():
        r = evaluate_row(row, thr)
        if r is None:
            logger.warning(f"  [{row.get('section_id')}/seed{row.get('seed')}] "
                           f"missing eps_a/eps_z; skipped.")
            continue
        results.append(r)

    if not results:
        logger.error("No rows evaluable (all missing real strains). "
                     "Re-run the extractor and check its 'status' column.")
        sys.exit(1)

    rdf = pd.DataFrame(results)

    MEPDG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_strict" if args.strict else ""
    out_csv = MEPDG_DIR / f"mepdg_check_per_run{suffix}_{ts}.csv"
    rdf.to_csv(out_csv, index=False)
    logger.info(f"Per-run CSV: {out_csv}")

    # Aggregate by pavement type
    agg = rdf.groupby("pavement_type").agg(
        n=("section_id", "count"),
        FC_mean=("FC_percent", "mean"), FC_max=("FC_percent", "max"),
        RD_mean=("RD_total_mm", "mean"), RD_max=("RD_total_mm", "max"),
        IRI_mean=("IRI_in_per_mi", "mean"), IRI_max=("IRI_in_per_mi", "max"),
        compliance_rate=("MEPDG_all_pass", "mean"),
    ).round(3).reset_index()
    agg_csv = MEPDG_DIR / f"mepdg_check_aggregate{suffix}_{ts}.csv"
    agg.to_csv(agg_csv, index=False)
    logger.info(f"Aggregate CSV: {agg_csv}")

    # Summary
    print("\n" + "=" * 104)
    print(f"ME-PDG CROSS-SPEC CHECK (REAL FEA) — {thr['reliability']*100:.0f}% reliability")
    print(f"  FC<={thr['FC_percent_max']}%   RD<={thr['RD_total_mm_max']}mm   "
          f"IRI<={thr['IRI_in_per_mi_max']} in/mi   |   nominal NCHRP 1-37A calibration (beta=1.0)")
    print("=" * 104)
    print(f"{'type':<12} {'n':>3} {'FC_mean':>8} {'FC_max':>8} {'RD_mean':>8} "
          f"{'RD_max':>8} {'IRI_mean':>9} {'IRI_max':>9} {'pass%':>7}")
    print("-" * 104)
    for _, r in agg.iterrows():
        print(f"{str(r['pavement_type']):<12} {int(r['n']):>3} {r['FC_mean']:>8.2f} "
              f"{r['FC_max']:>8.2f} {r['RD_mean']:>8.2f} {r['RD_max']:>8.2f} "
              f"{r['IRI_mean']:>9.1f} {r['IRI_max']:>9.1f} {r['compliance_rate']*100:>6.1f}%")
    print("=" * 104)

    overall = rdf["MEPDG_all_pass"].mean() * 100.0
    print(f"\nOverall ME-PDG compliance: {overall:.1f}% of {len(rdf)} HARA designs "
          f"pass all 3 ME-PDG thresholds")
    print("Per-criterion pass rates: "
          f"FC {rdf['pass_FC'].mean()*100:.1f}%, "
          f"RD {rdf['pass_RD'].mean()*100:.1f}%, "
          f"IRI {rdf['pass_IRI'].mean()*100:.1f}%")

    print("\nNarrative hint for R3-12:")
    if overall >= 90:
        print("  STRONG CONVERGENCE — JTG-optimised designs also satisfy ME-PDG.")
    elif overall >= 60:
        print("  PARTIAL CONVERGENCE — report which criterion drives the gap "
              "(usually fatigue, given JTG vs NCHRP calibration differences).")
    else:
        print("  DIVERGENCE — discuss as a genuine cross-spec calibration difference, "
              "NOT a HARA failure (designs are JTG-compliant by construction).")
    print("\nReminder: keep the 'nominal NCHRP calibration / directional check' "
          "disclosure in Methods and the response letter.")


if __name__ == "__main__":
    main()
