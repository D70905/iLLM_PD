# -*- coding: utf-8 -*-
"""
scripts/run_asbuilt_baseline.py — As-built baseline against LTPP measured layers.
================================================================================

Reads the 12 LTPP sections' actual built layer thicknesses from the master xlsx,
runs ONE FEA evaluation per section (no PPO optimisation), and reports SCR, DSR,
and LCC NPV in USD/m2 alongside the corresponding HARA-optimised results from
the 36-run inference matrix.

Layer mapping strategy:
    LTPP layer numbering (L1=surface) is material-code-agnostic. We take the
    ACTUAL measured thicknesses, sum them, and distribute across our 6-layer
    model proportionally to the DEFAULT design for that pavement type.
    This preserves:
      - total asphalt thickness (from LTPP)
      - total base+subbase thickness (from LTPP)
      - the relative sub-layer ratios from the default flexible/semi-rigid design

Usage:
    conda activate illm_pd
    python scripts/run_asbuilt_baseline.py
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
logger = logging.getLogger("asbuilt_baseline")

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea import run_fea
from specs import get_protocol, DesignInputs
from rl.lifecycle_lcc_intl import lcc_npv_usd
from rl import metrics as _metrics

# GPS family fallback (same as ltpp_inference.py)
_GPS_FAMILY: Dict[str, str] = {
    "04_1034": "GPS-1", "12_1060": "GPS-1", "16_1010": "GPS-1",
    "27_1085": "GPS-1", "48_0001": "GPS-1", "48_1076": "GPS-1",
    "04_1065": "GPS-2", "06_2004": "GPS-2", "12_4097": "GPS-2",
    "27_2023": "GPS-2", "30_7076": "GPS-2", "48_1109": "GPS-2",
}

# Default 5-layer proportions for each pavement type
_DEFAULT_PROPORTIONS = {
    "flexible": {
        "thickness": [0.04, 0.06, 0.08, 0.30, 0.25],  # m
        "modulus":   [14000, 11000, 9000, 1500, 400],   # MPa
    },
    "semi_rigid": {
        "thickness": [0.04, 0.06, 0.08, 0.36, 0.18],
        "modulus":   [14000, 11000, 9000, 1500, 400],
    },
}
_DEFAULT_POISSON = [0.25, 0.30, 0.30, 0.25, 0.35]


def load_sections(xlsx_path: str) -> List[Dict[str, Any]]:
    """Load 12 LTPP sections, extracting actual layer thicknesses + material codes."""
    df = pd.read_excel(xlsx_path)
    sections = []
    for _, row in df.iterrows():
        sid = str(row["section_id"]).strip()
        gps = _GPS_FAMILY.get(sid, "GPS-1")
        pavtype = "flexible" if gps == "GPS-1" else "semi_rigid"

        # Collect (thickness_m, material_code) for all non-NaN, positive layers
        ltp_layers = []
        for j in range(1, 9):
            h_col = f"h_layer{j}_m"
            m_col = f"matl_layer{j}"
            if h_col not in df.columns:
                continue
            h = row[h_col]
            if pd.notna(h) and h > 0.001:
                mat = row.get(m_col, 0)
                try: mat = int(float(mat))
                except: mat = 0
                ltp_layers.append({"h_m": float(h), "mat_code": mat})

        if len(ltp_layers) == 0:
            logger.warning(f"{sid}: no valid LTPP layers found, skipping")
            continue

        sections.append({
            "section_id":   sid,
            "state_name":   str(row.get("state_name", "")),
            "climate_zone": str(row.get("climate_zone", "")),
            "gps_family":   gps,
            "pavement_type": pavtype,
            "subgrade_bin": str(row.get("subgrade_bin", "")),
            "E_subgrade":   float(row["E_subgrade_MPa"]),
            "is_baseline":  sid in ("48_0001", "48_1076"),
            "ltp_layers":   ltp_layers,
        })
    return sections


# LTPP material code → layer type mapping
_AC_CODES   = {23, 24, 26, 27, 41, 42, 43, 44}  # asphalt concrete
_BASE_CODES = {28, 32, 37, 46, 47, 48}           # stabilised / treated base
_UNBOUND_CODES = {1, 2, 3, 4, 5, 6}              # unbound granular / soil-agg
_SURFACE_CODES = {71, 72, 78}                     # thin surface treatments


def map_to_5_layer(ltp_layers: List[Dict], pavtype: str) -> List[float]:
    """
    Map arbitrary LTPP layers to our 5-layer structure using material codes.

    Strategy:
      - AC layers (mat 23-44) → summed → split across 3 AC sublayers (upper/mid/lower)
      - Stabilised base layers (mat 28-48) → base (layer 3)
      - Unbound granular layers (mat 1-6) → subbase (layer 4) if below base,
        otherwise treated as base for flexible pavements
      - Thin surface treatments (71-78) → added to upper AC

    Default AC split ratio: upper:mid:lower = 25:35:40 (reflecting typical
    Chinese expressway design, thinner upper AC, thicker lower AC).
    """
    h_ac_total = 0.0      # all asphalt concrete layers
    h_base = 0.0          # stabilised base
    h_subbase = 0.0        # unbound granular below base
    h_surface = 0.0        # thin surface treatments

    # Two-pass: first classify, then resolve
    classified = []
    for layer in ltp_layers:
        code = layer["mat_code"]
        h = layer["h_m"]
        if code in _SURFACE_CODES:
            classified.append(("surface", h))
        elif code in _AC_CODES:
            classified.append(("ac", h))
        elif code in _BASE_CODES:
            classified.append(("base", h))
        elif code in _UNBOUND_CODES or code == 0:
            classified.append(("unbound", h))
        else:
            # Unknown material code: guess by position
            classified.append(("unbound", h))

    for kind, h in classified:
        if kind == "surface":
            h_surface += h
        elif kind == "ac":
            h_ac_total += h
        elif kind == "base":
            h_base += h
        elif kind == "unbound":
            h_subbase += h

    # Merge surface into AC (it's thin asphalt)
    h_ac_total += h_surface

    # If no AC identified, use the topmost layer as AC
    if h_ac_total < 0.01 and len(ltp_layers) > 0:
        h_ac_total = ltp_layers[0]["h_m"]
        # Shift remaining layers down
        if len(ltp_layers) > 1:
            h_base = sum(l["h_m"] for l in ltp_layers[1:])
            h_subbase = 0.0

    # If no base identified, split subbase
    if h_base < 0.01 and h_subbase > 0:
        h_base = h_subbase * 0.6
        h_subbase = h_subbase * 0.4

    # If no subbase, borrow from base
    if h_subbase < 0.01 and h_base > 0:
        h_subbase = h_base * 0.3
        h_base = h_base * 0.7

    # Split AC into 3 sublayers
    r = [0.25, 0.35, 0.40]  # upper:mid:lower
    h_upper  = h_ac_total * r[0]
    h_mid    = h_ac_total * r[1]
    h_lower  = h_ac_total * r[2]

    return [h_upper, h_mid, h_lower, h_base, h_subbase]


def run_one_asbuilt(section: Dict[str, Any]) -> Dict[str, Any]:
    """Run single FEA evaluation for as-built section."""
    sid = section["section_id"]
    pavtype = section["pavement_type"]
    gps = section["gps_family"]
    esub = section["E_subgrade"]
    ltp_layers = section["ltp_layers"]

    # Map LTPP layers (list of {h_m, mat_code}) to 5-layer structure
    thickness_5 = map_to_5_layer(ltp_layers, pavtype)
    ltp_total_h = sum(l["h_m"] for l in ltp_layers)
    modulus_5 = _DEFAULT_PROPORTIONS[pavtype]["modulus"]

    logger.info(f"[{sid}] {gps} {pavtype}: LTPP {len(ltp_layers)} layers "
                f"(mat codes: {[l['mat_code'] for l in ltp_layers]}, "
                f"{sum(l['h_m'] for l in ltp_layers)*100:.1f} cm total) -> "
                f"5-layer AC={sum(thickness_5[:3])*100:.1f}cm "
                f"base={thickness_5[3]*100:.1f}cm "
                f"subbase={thickness_5[4]*100:.1f}cm")

    t0 = time.time()
    design_inputs = None
    try:
        # ── Run FEA ──────────────────────────────────────────
        result = run_fea(
            thickness=thickness_5,
            modulus=modulus_5,
            poisson=_DEFAULT_POISSON,
            E_subgrade=esub,
            nu_subgrade=0.40,
            load_pressure=0.7,
            load_radius=0.1065,
            num_cpus=4,
            verbose=False,
        )
        fea_responses = result.get("responses", {})

        # ── Evaluate JTG D50-2017 ────────────────────────────
        inputs = DesignInputs(
            pavement_type=pavtype,
            road_class="expressway",
            traffic_level="heavy",
            thickness=thickness_5,
            modulus=modulus_5,
            poisson=_DEFAULT_POISSON,
            E_subgrade=esub,
            nu_subgrade=0.40,
            design_life=15,
            extras={
                "city": "beijing",
                "VFA_pct": 70.0, "R_s_MPa": 1.0, "R_0_mm": 1.5,
            },
        )
        protocol = get_protocol("JTG_D50_2017")
        evaluation = protocol.evaluate(inputs, fea_responses)

        # ── DSR / SCR ────────────────────────────────────────
        margins = {k: float(v) for k, v in evaluation.margins.items()}
        dsr = _metrics.compute_dsr(margins)
        is_compliant = _metrics.compute_compliance(margins)
        scr = 1.0 if is_compliant else (
            sum(1 for v in margins.values() if v >= 1.0) / len(margins))

        # ── LCC ───────────────────────────────────────────────
        cny_per_m3 = [1800, 1100, 900,
                      100 if pavtype == "flexible" else 320,
                      80 if pavtype == "flexible" else 180]
        C_const_cny = sum(cny_per_m3[i] * thickness_5[i] for i in range(5))
        C_const_usd = C_const_cny / 7.20
        margin_B1 = margins.get("B1_asphalt_fatigue", 99.0)
        margin_B2 = margins.get("B2_semi_rigid_fatigue", 99.0)
        lcc = lcc_npv_usd(
            C_construction_usd_per_m2=C_const_usd,
            design_life_years=20.0,
            margin_B1=margin_B1,
            margin_B2=margin_B2,
            discount_rate=0.04,
        )

        elapsed = time.time() - t0
        return {
            "section_id": sid,
            "state_name": section["state_name"],
            "climate_zone": section["climate_zone"],
            "gps_family": gps,
            "pavement_type": pavtype,
            "subgrade_bin": section["subgrade_bin"],
            "E_subgrade": esub,
            "is_baseline": section["is_baseline"],
            "ltp_n_layers": len(ltp_layers),
            "ltp_total_h_cm": round(ltp_total_h * 100, 1),
            "mapped_total_h_cm": round(sum(thickness_5) * 100, 1),
            "h1_cm": round(thickness_5[0] * 100, 1),
            "h2_cm": round(thickness_5[1] * 100, 1),
            "h3_cm": round(thickness_5[2] * 100, 1),
            "h4_cm": round(thickness_5[3] * 100, 1),
            "h5_cm": round(thickness_5[4] * 100, 1),
            "feasible": evaluation.feasible,
            "critical": evaluation.critical_indicator,
            "B1": round(margins.get("B1_asphalt_fatigue", 0), 2),
            "B2": round(margins.get("B2_semi_rigid_fatigue", 0), 2),
            "B3": round(margins.get("B3_ac_permanent_deformation", 0), 2),
            "B4": round(margins.get("B4_subgrade_strain", 0), 2),
            "DSR": round(dsr, 4),
            "SCR": round(scr, 4),
            "compliant": bool(is_compliant),
            "C_const_cny": round(C_const_cny, 1),
            "C_const_usd": round(C_const_usd, 2),
            "NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
            "C_maint_NPV_usd": round(lcc.get("C_maintenance_NPV_usd_m2", 0), 2),
            "n_maint_events": lcc.get("n_events", 0),
            "wall_clock_sec": round(elapsed, 1),
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"[{sid}] FAILED: {e}")
        return {
            "section_id": sid, "status": "failed",
            "error": str(e),
            "wall_clock_sec": round(time.time() - t0, 1),
        }


def main():
    xlsx = "experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx"
    out_dir = Path("experiments/ltpp_data/deliverables/ltpp_asbuilt")
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = load_sections(xlsx)
    logger.info(f"Loaded {len(sections)} sections from {xlsx}")

    results = []
    ok, fail = 0, 0
    t_start = time.time()

    for i, sec in enumerate(sections):
        r = run_one_asbuilt(sec)
        results.append(r)
        if r["status"] == "ok":
            ok += 1
            logger.info(f"  [{i+1}/{len(sections)}] {sec['section_id']}: "
                        f"DSR={r['DSR']:.2f} SCR={r['SCR']:.2f} "
                        f"NPV=${r['NPV_usd']:.1f}/m2 "
                        f"B3={r['B3']:.2f} ({r['wall_clock_sec']:.0f}s)")
        else:
            fail += 1

    # Save summary CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"asbuilt_summary_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["section_id", "state_name", "climate_zone", "gps_family",
                "pavement_type", "subgrade_bin", "E_subgrade", "is_baseline",
                "ltp_n_layers", "ltp_total_h_cm", "mapped_total_h_cm",
                "h1_cm", "h2_cm", "h3_cm", "h4_cm", "h5_cm",
                "feasible", "critical",
                "B1", "B2", "B3", "B4",
                "DSR", "SCR", "compliant",
                "C_const_cny", "C_const_usd",
                "NPV_usd", "C_maint_NPV_usd", "n_maint_events",
                "wall_clock_sec", "status"]
        for c in cols:
            f.write(c + ",")
        f.write("\n")
        for r in results:
            for c in cols:
                f.write(str(r.get(c, "")) + ",")
            f.write("\n")

    elapsed = time.time() - t_start
    logger.info(f"DONE. ok={ok} failed={fail} total={elapsed/60:.1f} min")
    logger.info(f"Summary: {csv_path}")

    # Print comparison-ready summary table
    print("\n" + "=" * 80)
    print("AS-BUILT BASELINE SUMMARY")
    print("=" * 80)
    print(f"{'Section':<12} {'GPS':<6} {'E_sub':<8} {'H_tot':<8} "
          f"{'DSR':<6} {'SCR':<6} {'B3':<6} {'NPV_usd':<10} {'Status'}")
    print("-" * 80)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['section_id']:<12} {r['gps_family']:<6} "
                  f"{r['E_subgrade']:<8.0f} {r['mapped_total_h_cm']:<8.1f} "
                  f"{r['DSR']:<6.2f} {r['SCR']:<6.2f} {r['B3']:<6.2f} "
                  f"${r['NPV_usd']:<9.2f} {r['status']}")
        else:
            print(f"{r['section_id']:<12} {'--':<6} {'--':<8} {'--':<8} "
                  f"{'--':<6} {'--':<6} {'--':<6} {'--':<10} FAILED")
    print("=" * 80)


if __name__ == "__main__":
    main()