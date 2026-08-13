# -*- coding: utf-8 -*-
"""
scripts/run_riohtrack_asbuilt.py — RIOHTRACK 19-structure As-built baseline.
================================================================================

Reads the RIOHTRACK merged_structures_FWD sheet, runs ONE FEA evaluation per
structure (no PPO optimisation), and reports SCR, DSR, LCC NPV.

Excludes STR4 and STR5 (rigid_inverted: LCC/CC base not covered by HARA).

Usage:
    python scripts/run_riohtrack_asbuilt.py
"""
from __future__ import annotations

import logging, sys, time, os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("riohtrack_asbuilt")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea import run_fea
from specs import get_protocol, DesignInputs
from rl.lifecycle_lcc_intl import lcc_npv_usd
from rl import metrics as _metrics


def map_to_5_layer(row, n_layers: int) -> tuple:
    """Map RIOHTRACK N layers (3-8) to our 5-layer structure using material categories."""
    ac_thicks, base_thicks, sub_thicks = [], [], []
    for j in range(1, n_layers + 1):
        cat = str(row.get(f"L{j}_category", ""))
        h = float(row.get(f"L{j}_thickness_cm", 0))
        if not h: continue
        if "AC" in cat: ac_thicks.append(h)
        elif "cement" in cat or "stabilised" in cat: base_thicks.append(h)
        else: sub_thicks.append(h)

    h_ac = sum(ac_thicks) if ac_thicks else 10.0
    h_base = sum(base_thicks) if base_thicks else 15.0
    h_sub = sum(sub_thicks) if sub_thicks else 10.0

    r = [0.25, 0.35, 0.40]
    h5 = [h_ac * r[0], h_ac * r[1], h_ac * r[2],
          max(h_base, 15.0), max(h_sub, 10.0)]
    return [h / 100.0 for h in h5], [round(h, 1) for h in h5]  # m, cm


def run_one(sid: str, row) -> dict:
    """Run FEA evaluation for one RIOHTRACK structure."""
    n = int(row["n_layers"])
    base_type = str(row.get("base_type", "semi_rigid"))
    pavtype = "flexible" if base_type == "flexible" else "semi_rigid"

    thickness_5_m, h5_cm = map_to_5_layer(row, n)
    # Use design moduli from the xlsx; fall back to typical values
    E_ac = float(row.get("L1_modulus_MPa_design", 12000))
    E_base = float(row.get(f"L{n}_modulus_MPa_design", 1500))
    mod_5 = [max(E_ac, 8000), max(E_ac*0.85, 6000), max(E_ac*0.7, 4000),
             max(E_base, 1500) if pavtype == "semi_rigid" else 400,
             400 if pavtype == "semi_rigid" else 200]
    E_sub = 80.0   # typical RIOHTRACK subgrade
    poisson_5 = [0.25, 0.30, 0.30, 0.25, 0.35]

    t0 = time.time()
    try:
        result = run_fea(thickness=thickness_5_m, modulus=mod_5,
                         poisson=poisson_5, E_subgrade=E_sub,
                         nu_subgrade=0.40, load_pressure=0.7,
                         load_radius=0.1065, num_cpus=4, verbose=False)
        fea_responses = result.get("responses", {})

        inputs = DesignInputs(pavement_type=pavtype, road_class="expressway",
            traffic_level="heavy", thickness=thickness_5_m, modulus=mod_5,
            poisson=poisson_5, E_subgrade=E_sub, nu_subgrade=0.40,
            design_life=15, extras={"city": "beijing", "VFA_pct": 70.0,
                "R_s_MPa": 1.0, "R_0_mm": 1.5})
        evaluation = get_protocol("JTG_D50_2017").evaluate(inputs, fea_responses)
        margins = {k: float(v) for k, v in evaluation.margins.items()}
        dsr = _metrics.compute_dsr(margins)
        scr = (sum(1 for v in margins.values() if v >= 1.0) / max(len(margins), 1)
               if not _metrics.compute_compliance(margins)
               else 1.0)

        cny = [1800, 1100, 900, 100 if pavtype == "flexible" else 320,
               80 if pavtype == "flexible" else 180]
        C_cny = sum(cny[i] * thickness_5_m[i] for i in range(5))
        lcc = lcc_npv_usd(C_construction_usd_per_m2=C_cny / 7.20,
                          design_life_years=20.0,
                          margin_B1=margins.get("B1_asphalt_fatigue", 99),
                          margin_B2=margins.get("B2_semi_rigid_fatigue", 99),
                          discount_rate=0.04)

        return {
            "structure_id": sid, "base_type": base_type, "pavtype": pavtype,
            "n_layers_orig": n, "h_total_cm": round(sum(thickness_5_m)*100, 1),
            "h1_cm": h5_cm[0], "h2_cm": h5_cm[1], "h3_cm": h5_cm[2],
            "h4_cm": h5_cm[3], "h5_cm": h5_cm[4],
            "DSR": round(dsr, 4), "SCR": round(scr, 4),
            "B1": round(margins.get("B1_asphalt_fatigue", 0), 2),
            "B3": round(margins.get("B3_ac_permanent_deformation", 0), 2),
            "B4": round(margins.get("B4_subgrade_strain", 0), 2),
            "NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
            "C_const_cny": round(C_cny, 1),
            "wall_clock_sec": round(time.time() - t0, 1), "status": "ok",
        }
    except Exception as e:
        logger.error(f"[{sid}] FAILED: {e}")
        return {"structure_id": sid, "status": "failed", "error": str(e)}


def main():
    xlsx = "data/RIOHTRACK_19_structures.xlsx"
    df = pd.read_excel(xlsx, sheet_name="merged_structures_FWD")

    sections = []
    for _, row in df.iterrows():
        sid = str(row["structure_id"])
        bt = str(row.get("base_type", ""))
        if bt == "rigid_inverted":
            logger.info(f"[{sid}] Skipped (rigid_inverted, outside HARA scope)")
            continue
        sections.append((sid, row))

    logger.info(f"Loaded {len(sections)} sections (excl. rigid_inverted)")

    results = []
    for i, (sid, row) in enumerate(sections):
        r = run_one(sid, row)
        results.append(r)
        if r["status"] == "ok":
            logger.info(f"  [{i+1}/{len(sections)}] {sid}: DSR={r['DSR']:.2f} "
                        f"SCR={r['SCR']:.2f} NPV=${r['NPV_usd']:.1f} "
                        f"B1={r['B1']:.2f} B3={r['B3']:.2f} ({r['wall_clock_sec']:.0f}s)")

    out_dir = Path("experiments/ltpp_data/deliverables/riohtrack")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"riohtrack_asbuilt_{ts}.csv"
    cols = ["structure_id","base_type","pavtype","n_layers_orig","h_total_cm",
            "h1_cm","h2_cm","h3_cm","h4_cm","h5_cm",
            "DSR","SCR","B1","B3","B4",
            "NPV_usd","C_const_cny","wall_clock_sec","status"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*70}\nRIOHTRACK AS-BUILT: {ok}/{len(results)} ok\n{csv_path}\n{'='*70}")


if __name__ == "__main__":
    main()