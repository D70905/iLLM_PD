# -*- coding: utf-8 -*-
"""
scripts/run_aashto1993_baseline.py — AASHTO 1993 design as baseline.
=====================================================================

Implements the AASHTO Guide for Design of Pavement Structures (1993)
flexible pavement SN equation, solves for required structural number,
distributes thickness across a 5-layer model, and evaluates each
design under the same JTG D50-2017 + LCC pipeline as HARA and As-built.

AASHTO 1993 equation:
    log10(W18) = ZR*S0 + 9.36*log10(SN+1) - 0.20
               + log10(dPSI/(4.2-1.5)) / (0.40 + 1094/(SN+1)^5.19)
               + 2.32*log10(MR) - 8.07

where:
    W18  = predicted 18-kip ESAL (from LTPP TRF_ESAL, KESAL × 1000)
    ZR   = -1.282 (R=90%% interstate)
    S0   = 0.45 (flexible pavement)
    dPSI = 1.7 (Po=4.2, Pt=2.5)
    MR   = subgrade resilient modulus (psi)
    SN   = structural number (solved numerically)

Layer distribution:
    SN = a1*D1 + a2*D2*m2 + a3*D3*m3
    a1=0.44, a2=0.14(GPS-1)/0.20(GPS-2), a3=0.11
    m2=m3=1.0 (good drainage)
    D1=AC total → split 25:35:40 across 3 sublayers
    D2=base, D3=subbase, minimum 15cm/10cm

Usage:
    python scripts/run_aashto1993_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("aashto1993")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea import run_fea
from specs import get_protocol, DesignInputs
from rl.lifecycle_lcc_intl import lcc_npv_usd
from rl import metrics as _metrics

_GPS_FAMILY = {
    "04_1034": "GPS-1", "12_1060": "GPS-1", "16_1010": "GPS-1",
    "27_1085": "GPS-1", "48_0001": "GPS-1", "48_1076": "GPS-1",
    "04_1065": "GPS-2", "06_2004": "GPS-2", "12_4097": "GPS-2",
    "27_2023": "GPS-2", "30_7076": "GPS-2", "48_1109": "GPS-2",
}

# ESAL data extracted from LTPP TRF_ESAL tables (KESAL = thousands of ESAL)
_KESAL = {
    "04_1034": 57, "04_1065": 583, "06_2004": 120,
    "12_1060": 81, "12_4097": 200,
    "16_1010": 123, "27_1085": 10, "27_2023": 574,
    "30_7076": 94, "48_0001": 133, "48_1076": 164, "48_1109": 81,
}

# AASHTO 1993 parameters
_ZR = -1.282      # R=90%
_S0 = 0.45        # flexible pavement
_DPSI = 1.7       # Po=4.2 - Pt=2.5

# Layer coefficients
_A1 = 0.44        # asphalt concrete
_A2_GPS1 = 0.14   # granular base
_A2_GPS2 = 0.20   # stabilised base
_A3 = 0.11        # granular subbase
_M2 = 1.0         # base drainage
_M3 = 1.0         # subbase drainage

# Minimum thicknesses (m)
_H_AC_MIN = 0.10       # 10 cm AC minimum
_H_BASE_MIN = 0.15     # 15 cm base minimum
_H_SUBBASE_MIN = 0.10  # 10 cm subbase minimum

_DEFAULT_POISSON = [0.25, 0.30, 0.30, 0.25, 0.35]
_AC_SPLIT = [0.25, 0.35, 0.40]   # upper:mid:lower AC


def aashto_sn_equation(SN: float, W18: float, MR_psi: float) -> float:
    """Evaluate AASHTO 1993 equation; return residual (0 = solved)."""
    import math
    dPSI_term = math.log10(_DPSI / (4.2 - 1.5))
    dPSI_denom = 0.40 + 1094.0 / ((SN + 1.0) ** 5.19)
    rhs = (_ZR * _S0
           + 9.36 * math.log10(SN + 1.0) - 0.20
           + dPSI_term / dPSI_denom
           + 2.32 * math.log10(MR_psi) - 8.07)
    return rhs - math.log10(W18)


def solve_SN(W18: float, MR_psi: float) -> float:
    """Bisection solve for required structural number."""
    lo, hi = 0.1, 15.0
    f_lo = aashto_sn_equation(lo, W18, MR_psi)
    f_hi = aashto_sn_equation(hi, W18, MR_psi)

    if f_lo * f_hi > 0:
        # W18 is beyond the equation's range; return boundary
        if abs(f_lo) < abs(f_hi):
            return lo
        return hi

    for _ in range(60):
        mid = (lo + hi) / 2.0
        f_mid = aashto_sn_equation(mid, W18, MR_psi)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def design_thickness(SN: float, gps: str) -> List[float]:
    """Distribute SN into 5-layer thicknesses (m)."""
    a2 = _A2_GPS2 if gps == "GPS-2" else _A2_GPS1
    # subbase contribution
    D3 = max(_H_SUBBASE_MIN, 0.10)
    SN3 = _A3 * D3 / 0.0254 * _M3   # m → inch for SN
    # base contribution
    D2 = max(_H_BASE_MIN, 0.15)
    SN2 = a2 * D2 / 0.0254 * _M2
    # remaining SN → AC
    SN1_needed = SN - SN2 - SN3
    if SN1_needed < 0:
        SN1_needed = max(0.5, SN * 0.6)
    D1 = max(_H_AC_MIN, SN1_needed * 0.0254 / _A1)   # inch → m
    # Split AC into 3 sublayers
    return [D1 * r for r in _AC_SPLIT] + [D2, D3]


def run_one_aashto(section: Dict[str, Any]) -> Dict[str, Any]:
    """Run AASHTO 1993 design + FEA evaluation for one section."""
    sid = section["section_id"]
    pavtype = section["pavement_type"]
    gps = section["gps_family"]
    esub = section["E_subgrade"]
    kesal = section.get("KESAL", 100)
    W18 = kesal * 1000.0           # KESAL → ESAL
    MR_psi = esub * 145.038        # MPa → psi

    # Solve SN
    SN_req = solve_SN(W18, MR_psi)
    thickness_5 = design_thickness(SN_req, gps)
    moduli = [14000, 11000, 9000,
              1500 if gps == "GPS-1" else 2000,
              400 if gps == "GPS-1" else 600]

    logger.info(f"[{sid}] {gps} KESAL={kesal} W18={W18:.0f} "
                f"MR={MR_psi:.0f}psi → SN={SN_req:.2f} → "
                f"AC={sum(thickness_5[:3])*100:.1f}cm "
                f"base={thickness_5[3]*100:.1f}cm "
                f"subbase={thickness_5[4]*100:.1f}cm")

    t0 = time.time()
    try:
        result = run_fea(thickness=thickness_5, modulus=moduli,
                         poisson=_DEFAULT_POISSON, E_subgrade=esub,
                         nu_subgrade=0.40, load_pressure=0.7,
                         load_radius=0.1065, num_cpus=4, verbose=False)
        fea_responses = result.get("responses", {})

        inputs = DesignInputs(
            pavement_type=pavtype, road_class="expressway",
            traffic_level="heavy", thickness=thickness_5,
            modulus=moduli, poisson=_DEFAULT_POISSON,
            E_subgrade=esub, nu_subgrade=0.40, design_life=15,
            extras={"city": "beijing", "VFA_pct": 70.0,
                    "R_s_MPa": 1.0, "R_0_mm": 1.5},
        )
        evaluation = get_protocol("JTG_D50_2017").evaluate(inputs, fea_responses)
        margins = {k: float(v) for k, v in evaluation.margins.items()}
        dsr = _metrics.compute_dsr(margins)
        scr = (sum(1 for v in margins.values() if v >= 1.0) / max(len(margins), 1)
               if _metrics.compute_compliance(margins) else
               sum(1 for v in margins.values() if v >= 1.0) / len(margins))
        if _metrics.compute_compliance(margins):
            scr = 1.0

        cny_per_m3 = [1800, 1100, 900,
                      100 if pavtype == "flexible" else 320,
                      80 if pavtype == "flexible" else 180]
        C_const_cny = sum(cny_per_m3[i] * thickness_5[i] for i in range(5))
        lcc = lcc_npv_usd(C_construction_usd_per_m2=C_const_cny / 7.20,
                          design_life_years=20.0,
                          margin_B1=margins.get("B1_asphalt_fatigue", 99),
                          margin_B2=margins.get("B2_semi_rigid_fatigue", 99),
                          discount_rate=0.04)

        return {
            "section_id": sid, "state_name": section["state_name"],
            "climate_zone": section["climate_zone"], "gps_family": gps,
            "pavement_type": pavtype, "E_subgrade": esub,
            "KESAL": kesal, "SN_req": round(SN_req, 2),
            "h_ac_cm": round(sum(thickness_5[:3]) * 100, 1),
            "h_base_cm": round(thickness_5[3] * 100, 1),
            "h_subbase_cm": round(thickness_5[4] * 100, 1),
            "DSR": round(dsr, 4), "SCR": round(scr, 4),
            "B1": round(margins.get("B1_asphalt_fatigue", 0), 2),
            "B2": round(margins.get("B2_semi_rigid_fatigue", 0), 2),
            "B3": round(margins.get("B3_ac_permanent_deformation", 0), 2),
            "B4": round(margins.get("B4_subgrade_strain", 0), 2),
            "NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
            "C_const_cny": round(C_const_cny, 1),
            "wall_clock_sec": round(time.time() - t0, 1),
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"[{sid}] FAILED: {e}")
        return {"section_id": sid, "status": "failed",
                "error": str(e), "wall_clock_sec": round(time.time() - t0, 1)}


def main():
    xlsx = "experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx"
    df = pd.read_excel(xlsx)

    sections = []
    for _, row in df.iterrows():
        sid = str(row["section_id"]).strip()
        gps = _GPS_FAMILY.get(sid, "GPS-1")
        sections.append({
            "section_id": sid,
            "state_name": str(row.get("state_name", "")),
            "climate_zone": str(row.get("climate_zone", "")),
            "gps_family": gps,
            "pavement_type": "flexible" if gps == "GPS-1" else "semi_rigid",
            "E_subgrade": float(row["E_subgrade_MPa"]),
            "KESAL": _KESAL.get(sid, 100),
        })

    out_dir = Path("experiments/ltpp_data/deliverables/ltpp_aashto1993")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded {len(sections)} sections")

    results = []
    ok = fail = 0
    for i, sec in enumerate(sections):
        r = run_one_aashto(sec)
        results.append(r)
        if r["status"] == "ok":
            ok += 1
            logger.info(f"  [{i+1}/12] {sec['section_id']}: "
                        f"SN={r['SN_req']} DSR={r['DSR']:.2f} SCR={r['SCR']:.2f} "
                        f"NPV=${r['NPV_usd']:.1f} ({r['wall_clock_sec']:.0f}s)")
        else:
            fail += 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"aashto1993_summary_{ts}.csv"
    cols = ["section_id", "state_name", "climate_zone", "gps_family",
            "pavement_type", "E_subgrade", "KESAL", "SN_req",
            "h_ac_cm", "h_base_cm", "h_subbase_cm",
            "DSR", "SCR", "B1", "B2", "B3", "B4",
            "NPV_usd", "C_const_cny", "wall_clock_sec", "status"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    logger.info(f"DONE. ok={ok} failed={fail}")
    logger.info(f"Summary: {csv_path}")

    print("\n" + "=" * 85)
    print("AASHTO 1993 BASELINE SUMMARY")
    print("=" * 85)
    print(f"{'Section':<12} {'GPS':<6} {'ESAL':<10} {'SN':<6} "
          f"{'AC':<7} {'Base':<7} {'Sub':<7} {'DSR':<6} {'SCR':<6} {'NPV':<10}")
    print("-" * 85)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['section_id']:<12} {r['gps_family']:<6} "
                  f"{r['KESAL']:<10} {r['SN_req']:<6.2f} "
                  f"{r['h_ac_cm']:<7.1f} {r['h_base_cm']:<7.1f} "
                  f"{r['h_subbase_cm']:<7.1f} {r['DSR']:<6.2f} "
                  f"{r['SCR']:<6.2f} ${r['NPV_usd']:<9.2f}")
        else:
            print(f"{r['section_id']:<12} FAILED")
    print("=" * 85)


if __name__ == "__main__":
    main()