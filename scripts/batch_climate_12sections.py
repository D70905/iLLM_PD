# -*- coding: utf-8 -*-
"""
scripts/batch_climate_12sections.py
===================================

Batch climate-sensitivity run over all 12 LTPP sections.

WHAT IT COMPUTES (one number per section, TRAFFIC-INDEPENDENT):
    For each section, using its own 5-layer structure + its own subgrade modulus
    + its own real monthly temperatures, it runs the FEA at each month's
    temperature-adjusted AC moduli, gets epsilon_a(T), computes the JTG B.1 AC
    fatigue life N_f per month, and reports:

        ratio = N_f(fixed 23C)  /  N_f_climate_effective
              where N_f_climate_effective = 12 / sum_m (1 / N_f_month)   (harmonic)

    This ratio equals life_fixed / life_climate and the annual traffic CANCELS,
    so NO traffic input is needed (the 2 sections missing ESALs are fine).
        ratio < 1  -> fixed assumption was CONSERVATIVE (cold sections)
        ratio > 1  -> fixed assumption was OPTIMISTIC  (hot sections)

WHY THESE CHOICES (see also the chat):
  * Per-section structure (NOT one uniform structure): keeps the 12 as 12 real
    sections. Default here uses each section's iLLM-PD-designed 5-layer structure
    (natively 5-layer -> no LTPP->5-layer mapping subjectivity). Fill SECTIONS
    below from hara_fea_responses_20260527_114131.csv (h1..h5, E_sub).
  * B.1 AC fatigue for ALL sections: every section has AC; epsilon_a is the most
    temperature-sensitive response. This sidesteps the flexible/semi-rigid
    determination and the semi-rigid R_s data gap. It is a CLIMATE-SENSITIVITY
    indicator, not a claim about the governing design criterion. (Semi-rigid base
    fatigue B.2 is also climate-sensitive; add later if R_s becomes available.)
  * air->pavement offset is EXPLICIT and should be calibrated against measured
    MON_DEFL_TEMP_VALUES before reporting (small for bottom-of-AC monthly means).

HONEST SCOPE: this evaluates FIXED structures under fixed-vs-climate temperature.
It quantifies how wrong the old fixed-temperature assumption was, across climates.
It does NOT show the OPTIMAL design changes across climate — that needs wiring
the climate-coupled FEA into env.py and re-running PPO inference (separate step).

Runs ABAQUS: 12 sections x (12 months + 1 fixed) = 156 FEA. On the ABAQUS box:
    python scripts/batch_climate_12sections.py --abaqus "C:/SIMULIA/Commands/abaqus.bat"
Offline plumbing test (analytical stand-in, NOT real):
    python scripts/batch_climate_12sections.py --mock
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.fea_strain_provider import build_ac_master_curves, make_climate_strain_provider  # noqa: E402
from rl.lifecycle import ac_fatigue_life_Nf1  # noqa: E402

EXTRACTED = os.path.join(os.path.dirname(__file__), "..",
                         "experiments", "ltpp_data", "sdr39", "extracted")

# ──────────────────────────────────────────────────────────────────────
# Per-section config. Fill `structure` (5-layer, metres) from each section's
# iLLM-PD design in hara_fea_responses_20260527_114131.csv (h1..h5 in cm /100).
# If structure is None, DEFAULT_STRUCTURE is used (uniform — option A fallback).
# E_sub from the same CSV. climate_zone is for the summary only.
# ──────────────────────────────────────────────────────────────────────
DEFAULT_STRUCTURE = {
    "thickness": [0.04, 0.06, 0.08, 0.20, 0.20],   # upperAC, midAC, lowerAC, base, subbase (m)
    "modulus":   [14000.0, 11000.0, 9000.0, 350.0, 250.0],  # AC entries overwritten by T
    "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35],
}

# AC base/subbase moduli (NOT temperature-corrected): granular base ~350, subbase ~250
# for flexible; semi-rigid base would be ~1500. For B.1 (AC fatigue) the base modulus
# has negligible effect on epsilon_a, so 350/250 is acceptable as default.
_BASE_MOD = 350.0
_SUBBASE_MOD = 250.0

SECTIONS: List[Dict] = [
    # All structures filled from hara_fea_responses_20260527_114131.csv (iLLM-PD final designs).
    # thickness in METRES (h_cm / 100); AC anchors generic JTG (overwritten per T).
    # Sorted by MAAT (cold→hot) for natural ordering in output.
    {"section_id": "27_2023", "climate_zone": "Wet-Freeze",    "E_sub": 131.2,
     "structure": {"thickness": [0.0200, 0.1475, 0.0674, 0.4447, 0.1618],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "27_1085", "climate_zone": "Wet-Freeze",    "E_sub": 86.3,
     "structure": {"thickness": [0.0200, 0.0930, 0.1116, 0.2972, 0.2963],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "16_1010", "climate_zone": "Dry-Freeze",    "E_sub": 77.8,
     "structure": {"thickness": [0.0200, 0.0931, 0.1127, 0.2979, 0.2998],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "30_7076", "climate_zone": "Dry-Freeze",    "E_sub": 58.9,
     "structure": {"thickness": [0.0200, 0.1485, 0.0671, 0.4452, 0.1627],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "04_1065", "climate_zone": "Dry-NoFreeze",  "E_sub": 91.3,
     "structure": {"thickness": [0.0200, 0.1458, 0.0676, 0.4427, 0.1626],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "48_1076", "climate_zone": "Dry-NoFreeze",  "E_sub": 114.5,
     "structure": {"thickness": [0.0200, 0.0935, 0.1124, 0.2967, 0.2973],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "06_2004", "climate_zone": "Dry-NoFreeze",  "E_sub": 112.5,
     "structure": {"thickness": [0.0200, 0.1469, 0.0674, 0.4439, 0.1621],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "12_4097", "climate_zone": "Wet-NoFreeze",  "E_sub": 286.3,
     "structure": {"thickness": [0.0200, 0.1474, 0.0680, 0.4449, 0.1613],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "48_1109", "climate_zone": "Wet-NoFreeze",  "E_sub": 100.0,
     "structure": {"thickness": [0.0200, 0.1463, 0.0675, 0.4433, 0.1624],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "48_0001", "climate_zone": "Wet-NoFreeze",  "E_sub": 699.9,
     "structure": {"thickness": [0.0200, 0.0750, 0.1012, 0.3000, 0.2866],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "04_1034", "climate_zone": "Dry-NoFreeze",  "E_sub": 91.3,
     "structure": {"thickness": [0.0200, 0.0939, 0.1120, 0.2969, 0.2965],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
    {"section_id": "12_1060", "climate_zone": "Wet-NoFreeze",  "E_sub": 286.3,
     "structure": {"thickness": [0.0200, 0.0885, 0.1093, 0.2961, 0.2923],
                   "modulus":   [14000, 11000, 9000, _BASE_MOD, _SUBBASE_MOD],
                   "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35]}},
]

ANCHORS_20C = (14000.0, 11000.0, 9000.0)   # generic JTG AC moduli @ 20C
EA_DEFAULT = 200000.0                        # master-curve activation energy (literature default)
SURFACE_OFFSET_C = 2.0                       # air->pavement; CALIBRATE vs MON_DEFL_TEMP_VALUES
FIXED_AIR_C = 20.0                           # the fixed assumption (matched to design-loop reference)


def air_to_pavement(air_C: float, offset_C: float = SURFACE_OFFSET_C, slope: float = 1.0) -> float:
    return slope * air_C + offset_C


def load_monthly_air_temps(section_id: str) -> List[float]:
    import pandas as pd
    sdir = os.path.join(EXTRACTED, section_id)
    for name in ("clm_temp_monthly_climatology.csv", "clm_temp_monthly.csv"):
        p = os.path.join(sdir, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            col_t = "MEAN_MON_TEMP_AVG"
            if col_t not in df.columns and "MEAN_TEMP_AVG" in df.columns:
                col_t = "MEAN_TEMP_AVG"
            if col_t in df.columns and "MONTH" in df.columns:
                g = df.groupby("MONTH")[col_t].mean()
                return [float(g.get(m, g.mean())) for m in range(1, 13)]
    raise FileNotFoundError("no monthly temp file for {}".format(section_id))


def _mock_run_fea(thickness, modulus, poisson, E_subgrade, nu_subgrade, verbose=False, **kw):
    h_ac = sum(thickness[0:3])
    E_ac = sum(t * m for t, m in zip(thickness[0:3], modulus[0:3])) / max(h_ac, 1e-9)
    eps_a = 120.0 * (10000.0 / max(E_ac, 1e-6)) ** 0.40
    return {"responses": {"epsilon_a_microstrain": eps_a,
                          "epsilon_z_microstrain": 100.0, "sigma_t_MPa": 0.02,
                          "p_AC_upper_mid_MPa": 0.7, "p_AC_mid_mid_MPa": 0.62,
                          "p_AC_lower_mid_MPa": 0.26}}


def run_one(section: Dict, monthly_air: List[float], run_fea_fn) -> Dict:
    structure = section.get("structure") or DEFAULT_STRUCTURE
    base_design = {
        "thickness": list(structure["thickness"]),
        "modulus": list(structure["modulus"]),
        "poisson": list(structure["poisson"]),
        "E_subgrade": float(section["E_sub"]),
        "nu_subgrade": 0.40,
    }
    h_ac_mm = sum(base_design["thickness"][0:3]) * 1000.0
    curves = build_ac_master_curves(ANCHORS_20C, freq_hz=10.0, Ea_J_per_mol=EA_DEFAULT)
    provider = make_climate_strain_provider(base_design, curves, run_fea_fn=run_fea_fn, verbose=False)

    eps_list, Nf_list = [], []
    for T_air in monthly_air:
        out = provider(air_to_pavement(T_air))
        eps = out["eps_a_microstrain"]
        Nf = ac_fatigue_life_Nf1(eps, out["E_ac_equiv_MPa"], h_ac_mm,
                                 VFA_pct=70.0, beta=1.65, k_a=1.0, k_T1=1.0)
        eps_list.append(eps)
        Nf_list.append(Nf)

    # Climate-effective N_f (harmonic mean over months) — traffic-independent
    inv = sum(1.0 / nf for nf in Nf_list if nf > 0)
    Nf_climate_eff = (len(Nf_list) / inv) if inv > 0 else float("inf")

    # Fixed baseline
    out_fixed = provider(air_to_pavement(FIXED_AIR_C))
    Nf_fixed = ac_fatigue_life_Nf1(out_fixed["eps_a_microstrain"], out_fixed["E_ac_equiv_MPa"],
                                   h_ac_mm, VFA_pct=70.0, beta=1.65, k_a=1.0, k_T1=1.0)

    ratio = (Nf_fixed / Nf_climate_eff) if Nf_climate_eff not in (0, float("inf")) else float("inf")
    return {
        "section_id": section["section_id"],
        "climate_zone": section["climate_zone"],
        "E_sub": section["E_sub"],
        "MAAT": sum(monthly_air) / len(monthly_air),
        "eps_min": min(eps_list), "eps_max": max(eps_list),
        "eps_swing": (max(eps_list) / min(eps_list)) if min(eps_list) > 0 else float("inf"),
        "eps_fixed": out_fixed["eps_a_microstrain"],
        "Nf_fixed": Nf_fixed, "Nf_climate_eff": Nf_climate_eff,
        "fixed_over_climate": ratio,
        "verdict": "conservative" if ratio < 1 else "optimistic",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="analytical stand-in (TEST ONLY)")
    ap.add_argument("--abaqus", default=None, help="path to abaqus.bat (real FEA)")
    args = ap.parse_args()

    run_fea_fn = None
    if args.mock:
        run_fea_fn = _mock_run_fea
        print(">>> MOCK FEA — plumbing test only, numbers not real <<<\n")
    elif args.abaqus:
        # Wrap fea.runner.run_fea to pass the abaqus command path
        from fea.runner import run_fea as _rf

        def run_fea_fn(**kw):  # noqa
            kw.setdefault("abaqus_command", args.abaqus)
            return _rf(**kw)

    results = []
    for sec in SECTIONS:
        sid = sec["section_id"]
        try:
            monthly = load_monthly_air_temps(sid)
        except Exception as e:
            print("  {}: SKIP ({})".format(sid, e))
            continue
        print("  running {} ...".format(sid))
        results.append(run_one(sec, monthly, run_fea_fn))

    # ── Summary table (sorted by MAAT, cold -> hot) ────────────────────
    results.sort(key=lambda r: r["MAAT"])
    print("\n" + "=" * 104)
    print("12-SECTION CLIMATE SENSITIVITY — JTG B.1 AC fatigue (traffic-independent ratio)")
    print("=" * 104)
    hdr = ("{:>9} {:>13} {:>7} {:>7} {:>14} {:>9} {:>13} {:>13} {:>10} {:>12}"
           .format("Section", "Climate", "MAAT", "E_sub", "eps_a(ue)min-max",
                   "swing", "Nf_fixed", "Nf_climate", "fix/clim", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print("{:>9} {:>13} {:>7.1f} {:>7.0f} {:>6.1f}-{:<7.1f} {:>9.2f}x {:>13.3e} {:>13.3e} {:>9.2f}x {:>12}"
              .format(r["section_id"], r["climate_zone"], r["MAAT"], r["E_sub"],
                      r["eps_min"], r["eps_max"], r["eps_swing"],
                      r["Nf_fixed"], r["Nf_climate_eff"], r["fixed_over_climate"], r["verdict"]))

    if results:
        ratios = [r["fixed_over_climate"] for r in results if r["fixed_over_climate"] != float("inf")]
        print("-" * len(hdr))
        print("fixed/climate ratio across sections: min={:.2f}x  max={:.2f}x"
              .format(min(ratios), max(ratios)))
        print("  <1 = fixed-temperature assumption was CONSERVATIVE (cold);"
              " >1 = OPTIMISTIC (hot).")
        print("  This quantifies, across 12 real sections / climate zones, how much the old")
        print("  fixed-temperature setup misestimated AC fatigue. (Anchors generic; Ea default;")
        print("  air->pavement offset should be calibrated vs MON_DEFL_TEMP_VALUES.)")

    out_csv = os.path.join(os.path.dirname(__file__), "..", "experiments",
                           "batch_climate_12sections_summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "climate_zone", "MAAT_C", "E_sub_MPa",
                    "eps_a_min_ue", "eps_a_max_ue", "eps_swing", "eps_a_fixed_ue",
                    "Nf_fixed", "Nf_climate_eff", "fixed_over_climate", "verdict"])
        for r in results:
            w.writerow([r["section_id"], r["climate_zone"], round(r["MAAT"], 2),
                        round(r["E_sub"], 1), round(r["eps_min"], 2), round(r["eps_max"], 2),
                        round(r["eps_swing"], 3), round(r["eps_fixed"], 2),
                        "{:.4e}".format(r["Nf_fixed"]), "{:.4e}".format(r["Nf_climate_eff"]),
                        round(r["fixed_over_climate"], 3), r["verdict"]])
    print("\nSaved summary to {}".format(out_csv))


if __name__ == "__main__":
    main()
