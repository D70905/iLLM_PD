# -*- coding: utf-8 -*-
"""
scripts/run_section_climate_loop.py
===================================

FIRST REAL closed loop for ONE LTPP section:

    real monthly temperature  ->  per-layer AC modulus E(T)
                              ->  ABAQUS FEA  ->  epsilon_a(T)
                              ->  JTG/ME-PDG fatigue life N_f per month
                              ->  Miner damage summed over the year
                              ->  predicted fatigue life

and compares it against the OLD fixed-modulus assumption (one temperature for
all months), giving the first reportable "fixed vs climate" fatigue-life number.

This runs ABAQUS (one FEA per month = 12 runs for the climate case + 1 for the
fixed case). Run it on your ABAQUS machine:

    python scripts/run_section_climate_loop.py

To test the plumbing WITHOUT ABAQUS, run with --mock (uses a clearly-labelled
analytical stand-in for FEA, NOT for reported results):

    python scripts/run_section_climate_loop.py --mock

HONEST SCOPE / CAVEATS (read before using any number):
  * Temperature input: monthly MEAN AIR temperatures from CLM_VWS_TEMP_MONTH.
    For FATIGUE (bottom of AC, monthly mean) air temp ~ AC-bottom temp is an
    acceptable first approximation. air_to_pavement() below applies a small,
    EXPLICIT, CALIBRATABLE offset; calibrate it against measured pavement temps
    in MON_DEFL_TEMP_VALUES (LAYER_TEMPERATURE_1/2/3) before reporting. For
    RUTTING (surface, summer peak) this offset+a depth gradient matter much more.
  * AC anchor moduli (14000/11000/9000 @ 20 C) are GENERIC JTG values, identical
    across sections — LTPP has no lab |E*| for these sections and FWD-backcalc is
    a different quantity. So this isolates the CLIMATE effect on response; it is
    not section-specific material. State this in the paper.
  * This evaluates a FIXED structure under fixed-vs-climate temperature. Whether
    the OPTIMAL design changes across climates is a separate step (wire the
    climate-coupled FEA into env.py and re-run PPO inference per section).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.dynamic_modulus import DynamicModulusMasterCurve  # noqa: E402
from rl.fea_strain_provider import build_ac_master_curves, make_climate_strain_provider  # noqa: E402
from rl.lifecycle import ac_fatigue_life_Nf1  # noqa: E402
from rl.lifecycle_climate import miner_damage  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Air -> pavement temperature (EXPLICIT, CALIBRATABLE — do not treat as truth)
# ──────────────────────────────────────────────────────────────────────
def air_to_pavement(air_temp_C: float,
                    surface_offset_C: float = 2.0,
                    slope: float = 1.0) -> float:
    """
    First-order monthly-mean air -> AC-bottom temperature.
    T_pav = slope * T_air + surface_offset_C.

    DEFAULTS ARE A PLACEHOLDER. Calibrate (surface_offset_C, slope) against the
    section's measured pavement temperatures (MON_DEFL_TEMP_VALUES) before any
    reported result. For bottom-of-AC monthly means the offset is small.
    """
    return slope * air_temp_C + surface_offset_C


# ──────────────────────────────────────────────────────────────────────
# Load real monthly air temperatures (12 values) for a section
# ──────────────────────────────────────────────────────────────────────
def load_monthly_air_temps(section_dir: str) -> List[float]:
    """
    Read 12 monthly-mean air temps from the extracted climatology.
    Tries clm_temp_monthly_climatology.csv, then clm_temp_monthly.csv
    (grouping by MONTH). Returns Jan..Dec.
    """
    import pandas as pd

    clim = os.path.join(section_dir, "clm_temp_monthly_climatology.csv")
    raw = os.path.join(section_dir, "clm_temp_monthly.csv")

    if os.path.exists(clim):
        df = pd.read_csv(clim)
    elif os.path.exists(raw):
        df = pd.read_csv(raw)
    else:
        raise FileNotFoundError(
            "No monthly temperature file in {} (expected clm_temp_monthly"
            "[_climatology].csv)".format(section_dir))

    col_t = "MEAN_MON_TEMP_AVG"
    col_m = "MONTH"
    # Fallback: our extraction script uses MEAN_TEMP_AVG in climatology files
    if col_t not in df.columns:
        if "MEAN_TEMP_AVG" in df.columns:
            col_t = "MEAN_TEMP_AVG"
    if col_t not in df.columns or col_m not in df.columns:
        raise KeyError("expected columns MEAN_MON_TEMP_AVG (or MEAN_TEMP_AVG) "
                       "and MONTH in monthly temp file; got {}"
                       .format(list(df.columns)))
    monthly = df.groupby(col_m)[col_t].mean()
    return [float(monthly.get(m, monthly.mean())) for m in range(1, 13)]


# ──────────────────────────────────────────────────────────────────────
# Mock FEA (TEST ONLY — clearly not real)
# ──────────────────────────────────────────────────────────────────────
def _mock_run_fea(thickness, modulus, poisson, E_subgrade, nu_subgrade, verbose=False, **kw):
    """
    Analytical stand-in for ABAQUS to test the pipeline offline. eps_a falls as
    AC stiffens (physically correct direction). NOT for reported results.
    """
    h_ac = sum(thickness[0:3])
    E_ac = sum(t * m for t, m in zip(thickness[0:3], modulus[0:3])) / max(h_ac, 1e-9)
    eps_a = 120.0 * (10000.0 / max(E_ac, 1e-6)) ** 0.40           # microstrain
    eps_z = 200.0 * (100.0 / max(E_subgrade, 1e-6)) ** 0.35
    sigma_t = 0.05 * (E_ac / 10000.0) ** 0.5
    return {"responses": {
        "epsilon_a_microstrain": eps_a,
        "epsilon_z_microstrain": eps_z,
        "sigma_t_MPa": sigma_t,
        "p_AC_upper_mid_MPa": 0.70, "p_AC_mid_mid_MPa": 0.62, "p_AC_lower_mid_MPa": 0.26,
    }}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def run_section(
    section_id: str,
    base_design: Dict,
    monthly_air_temps_C: Sequence[float],
    annual_traffic: float,
    *,
    ac_anchors_20C_MPa: Sequence[float] = (14000.0, 11000.0, 9000.0),
    Ea_J_per_mol: float = 200000.0,
    fixed_air_temp_C: float = 23.0,
    surface_offset_C: float = 2.0,
    run_fea_fn: Optional[callable] = None,
    out_csv: Optional[str] = None,
    verbose_fea: bool = False,
) -> Dict:
    h_ac_mm = sum(base_design["thickness"][0:3]) * 1000.0
    curves = build_ac_master_curves(ac_anchors_20C_MPa, freq_hz=10.0, Ea_J_per_mol=Ea_J_per_mol)
    provider = make_climate_strain_provider(base_design, curves,
                                            run_fea_fn=run_fea_fn, verbose=verbose_fea)

    n_month = annual_traffic / 12.0
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    rows = []
    n_list, N_list = [], []
    print("=" * 86)
    print("SECTION {} — climate-resolved fatigue (REAL FEA per month)".format(section_id))
    print("=" * 86)
    print("AC anchors @20C (generic JTG): {} MPa | E_sub={:.0f} MPa | h_AC={:.0f} mm"
          .format(list(ac_anchors_20C_MPa), base_design["E_subgrade"], h_ac_mm))
    print("\n{:>4} {:>7} {:>7} {:>10} {:>10} {:>12}".format(
        "Mon", "Tair", "Tpav", "E_AC(MPa)", "eps_a(ue)", "Nf"))
    print("-" * 60)

    for m, T_air in zip(months, monthly_air_temps_C):
        T_pav = air_to_pavement(T_air, surface_offset_C=surface_offset_C)
        out = provider(T_pav)
        eps_a = out["eps_a_microstrain"]
        E_eq = out["E_ac_equiv_MPa"]
        Nf = ac_fatigue_life_Nf1(eps_a, E_eq, h_ac_mm,
                                 VFA_pct=70.0, beta=1.65, k_a=1.0, k_T1=1.0)
        rows.append({"month": m, "T_air": T_air, "T_pav": T_pav,
                     "E_ac_equiv_MPa": E_eq, "eps_a_microstrain": eps_a, "Nf": Nf})
        n_list.append(n_month)
        N_list.append(Nf)
        print("{:>4} {:>7.1f} {:>7.1f} {:>10.0f} {:>10.1f} {:>12.3e}"
              .format(m, T_air, T_pav, E_eq, eps_a, Nf))

    D_annual = miner_damage(n_list, N_list)
    life_climate = (1.0 / D_annual) if D_annual > 0 else float("inf")

    # Fixed-modulus baseline (old assumption): one temperature for all months
    T_pav_fixed = air_to_pavement(fixed_air_temp_C, surface_offset_C=surface_offset_C)
    out_fixed = provider(T_pav_fixed)
    Nf_fixed = ac_fatigue_life_Nf1(out_fixed["eps_a_microstrain"],
                                   out_fixed["E_ac_equiv_MPa"], h_ac_mm,
                                   VFA_pct=70.0, beta=1.65, k_a=1.0, k_T1=1.0)
    D_fixed = annual_traffic / Nf_fixed if Nf_fixed > 0 else float("inf")
    life_fixed = (1.0 / D_fixed) if D_fixed > 0 else float("inf")

    print("\n" + "-" * 60)
    print("FIXED modulus ({:.0f}C):  E_AC={:.0f} MPa  eps_a={:.1f} ue  Nf={:.3e}"
          .format(fixed_air_temp_C, out_fixed["E_ac_equiv_MPa"],
                  out_fixed["eps_a_microstrain"], Nf_fixed))
    print("  -> fixed-modulus predicted fatigue life = {:.2f} years".format(life_fixed))
    print("CLIMATE-resolved (Miner over 12 months):")
    print("  -> climate predicted fatigue life       = {:.2f} years".format(life_climate))
    ratio = (life_fixed / life_climate) if life_climate not in (0, float("inf")) else float("inf")
    print("  -> fixed / climate                       = {:.2f}x".format(ratio))
    print("     (>1: the old fixed assumption was optimistic about fatigue life)")
    print("=" * 86)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["section", "month", "T_air_C", "T_pav_C",
                        "E_ac_equiv_MPa", "eps_a_microstrain", "Nf"])
            for r in rows:
                w.writerow([section_id, r["month"], r["T_air"], r["T_pav"],
                            round(r["E_ac_equiv_MPa"], 1), round(r["eps_a_microstrain"], 2),
                            "{:.4e}".format(r["Nf"])])
            w.writerow([])
            w.writerow(["summary", "life_fixed_yr", round(life_fixed, 3),
                        "life_climate_yr", round(life_climate, 3),
                        "fixed_over_climate", round(ratio, 3)])
        print("Saved per-month results to {}".format(out_csv))

    return {"section_id": section_id, "rows": rows,
            "life_fixed_years": life_fixed, "life_climate_years": life_climate,
            "fixed_over_climate": ratio, "D_annual": D_annual}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="16_1010")
    ap.add_argument("--mock", action="store_true",
                    help="use analytical stand-in instead of ABAQUS (TEST ONLY)")
    ap.add_argument("--annual_traffic", type=float, default=2.0e6)
    ap.add_argument("--esub", type=float, default=78.0, help="subgrade modulus MPa")
    ap.add_argument("--abaqus", type=str, default="abaqus",
                    help="ABAQUS command (default 'abaqus'; use full path if not on PATH)")
    ap.add_argument("--base-dir", type=str, default=None,
                    help="FEA project root (output/runs go here)")
    args = ap.parse_args()

    # ── Base design (EDIT to the section's real structure) ─────────────
    # thickness in METRES (fea.runner convention): [upperAC, midAC, lowerAC, base, subbase]
    # Using iLLM-PD initial design for flexible pavement (16_1010 is GPS-1 flexible)
    base_design = {
        "thickness": [0.04, 0.06, 0.08, 0.20, 0.20],
        "modulus":   [14000.0, 11000.0, 9000.0, 350.0, 250.0],  # AC entries overwritten by T
        "poisson":   [0.35, 0.35, 0.35, 0.35, 0.35],
        "E_subgrade": args.esub,
        "nu_subgrade": 0.40,
    }

    # ── Monthly air temps ──────────────────────────────────────────────
    base = os.path.join(os.path.dirname(__file__), "..",
                        "experiments", "ltpp_data", "sdr39", "extracted", args.section)
    try:
        monthly = load_monthly_air_temps(base)
        print("Loaded real monthly air temps for {}: {}".format(
            args.section, [round(t, 1) for t in monthly]))
    except Exception as e:
        # Fallback example profile (Idaho-like) if files not found / in --mock dev
        monthly = [-6.8, -4.0, 2.0, 7.0, 12.0, 16.0, 20.3, 19.0, 14.0, 7.0, 0.0, -6.0]
        print("WARNING: could not load monthly temps ({}); using example profile.".format(e))

    # ── FEA function (mock or real ABAQUS) ────────────────────────────
    if args.mock:
        run_fea_fn = _mock_run_fea
        print(">>> RUNNING WITH MOCK FEA — numbers are for plumbing test only <<<\n")
    else:
        from fea.runner import run_fea as _real_run_fea
        # Create wrapper that forwards abaqus_command + base_dir
        _abaqus_cmd = args.abaqus
        _base_dir = args.base_dir
        def run_fea_fn(**kwargs):
            kwargs.setdefault("abaqus_command", _abaqus_cmd)
            if _base_dir:
                kwargs.setdefault("base_dir", _base_dir)
            return _real_run_fea(**kwargs)
        print(">>> RUNNING REAL ABAQUS FEA (cmd={}) <<<".format(_abaqus_cmd))

    out_csv = os.path.join(os.path.dirname(__file__), "..",
                           "experiments", "{}_climate_fatigue.csv".format(args.section))
    run_section(args.section, base_design, monthly,
                annual_traffic=args.annual_traffic,
                fixed_air_temp_C=23.0, run_fea_fn=run_fea_fn, out_csv=out_csv)
