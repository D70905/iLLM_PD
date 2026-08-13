# -*- coding: utf-8 -*-
"""
candidate_scan.py — Scan all SDR39 state databases for LTPP candidate sections.

Inputs:
  D:\\iLLM_PD_new\\experiments\\ltpp_data\\sdr39\\states\\SDR39_*\\*_Primary_Data.accdb

Outputs:
  D:\\iLLM_PD_new\\experiments\\ltpp_data\\candidates_raw.csv     (all sections)
  D:\\iLLM_PD_new\\experiments\\ltpp_data\\candidates_scored.csv  (filtered + scored)

Each candidate row contains:
    state_code, shrp_id, state_name
    experiment_no             ← raw EXPERIMENT_SECTION value
    inferred_gps_family       ← 'GPS-1' / 'GPS-2' / 'OTHER' / 'UNKNOWN'
    base_material_code        ← INV_LAYER MATERIAL_TYPE for the base layer
    base_is_semi_rigid        ← True if material code suggests stabilized base
    n_ac_layers, n_inv_layers
    total_ac_thickness_in
    construction_year
    n_iri_timepoints, iri_years_observed
    n_fwd_tests
    mean_ann_temp_C, total_ann_precip_mm, freeze_index_C_days
    inferred_climate_zone     ← WF / WNF / DF / DNF
    completeness_score        ← 0-100, see RANKING below
    addresses_matrix_cell     ← climate × subgrade hint

Filtering produces candidates_scored.csv ranked by completeness_score.
You then manually pick 12 sections (6 GPS-1 + 6 GPS-2) covering the 4×3 matrix.

Run:
    cd D:\\iLLM_PD_new\\experiments\\ltpp_data\\scripts
    python candidate_scan.py

Expected runtime: 3-8 minutes (one-time bulk pyodbc reads per state).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyodbc


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # experiments/ltpp_data/
SDR_DIR      = PROJECT_ROOT / "sdr39" / "states"
OUT_DIR      = PROJECT_ROOT                          # experiments/ltpp_data/
RAW_CSV      = OUT_DIR / "candidates_raw.csv"
SCORED_CSV   = OUT_DIR / "candidates_scored.csv"

STATE_NAMES = {
    "04": "Arizona",      "06": "California",   "12": "Florida",
    "16": "Idaho",        "26": "Michigan",     "27": "Minnesota",
    "30": "Montana",      "32": "Nevada",       "37": "North Carolina",
    "48": "Texas",        "55": "Wisconsin",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("candidate_scan")


# ────────────────────────────────────────────────────────────────────
# Material code → "is it a stabilized (semi-rigid) base"?
#
# LTPP MATERIAL_TYPE codes (per IMS Tech Brief, simplified):
#   Asphalt-bound:     1-19   (HMA, dense graded, SMA, etc.)
#   Cement-stabilized: 320-339 in some encodings; 31-39 in others
#   Lime-stabilized:   340-349 / 41-49
#   Soil-cement:       301, 302 / 31, 32
#   Unbound aggregate: 200-299 / 21-29
#   Granular:          20-29 in compact encodings
#
# We use a permissive heuristic. If you find your data uses different codes,
# tune _BASE_SEMI_RIGID_CODES below after inspecting candidates_raw.csv.
# ────────────────────────────────────────────────────────────────────

_BASE_SEMI_RIGID_CODES = {
    # 2-digit compact form
    31, 32, 33, 34, 35, 36, 37, 38, 39,
    41, 42, 43, 44, 45,
    # 3-digit longer form
    301, 302, 303, 304, 311, 321, 322, 331, 332, 333, 334, 335, 339,
    341, 342, 343, 344, 345, 349,
    321,  # cement-treated
}
_BASE_UNBOUND_CODES = {
    # Permissive — granular / crushed / gravel / sand bases
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    200, 201, 202, 203, 204, 211, 212, 213, 214, 215,
    221, 222, 223, 224, 225, 226, 227,
}


def _is_semi_rigid_base(material_code) -> Optional[bool]:
    """True if cement/lime/asphalt-stabilized; False if unbound; None if unknown."""
    try:
        c = int(material_code)
    except (TypeError, ValueError):
        return None
    if c in _BASE_SEMI_RIGID_CODES:
        return True
    if c in _BASE_UNBOUND_CODES:
        return False
    return None


# ────────────────────────────────────────────────────────────────────
# Per-state bulk loader
# ────────────────────────────────────────────────────────────────────

def find_primary_accdb(state_dir: Path) -> Optional[Path]:
    """Locate the *_Primary_Data.accdb file recursively."""
    if not state_dir.exists():
        return None
    for p in state_dir.rglob("*_Primary_Data.accdb"):
        return p
    for p in state_dir.rglob("*Primary*.accdb"):
        return p
    return None


def connect_accdb(path: Path) -> pyodbc.Connection:
    conn_str = (r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                r"DBQ=" + str(path) + r";")
    return pyodbc.connect(conn_str)


def safe_read(conn, table: str, where: str = "",
               params=None) -> Optional[pd.DataFrame]:
    """Read a whole table (or filtered), return None if missing."""
    try:
        sql = "SELECT * FROM {}".format(table)
        if where:
            sql += " WHERE " + where
        df = pd.read_sql(sql, conn, params=params or [])
        return df
    except Exception:
        return None


def scan_one_state(accdb_path: Path, state_code: str) -> pd.DataFrame:
    """Scan one state .accdb, return per-section summary DataFrame."""
    conn = connect_accdb(accdb_path)
    state_name = STATE_NAMES.get(state_code, "?")
    log.info(f"  bulk-reading tables ...")

    # ── Bulk reads filtered by STATE_CODE where possible
    df_exp  = safe_read(conn, "EXPERIMENT_SECTION",
                          "STATE_CODE = ?", [int(state_code)])
    df_inv  = safe_read(conn, "INV_LAYER",
                          "STATE_CODE = ?", [int(state_code)])
    df_iri  = safe_read(conn, "ANALYSIS_IRI",
                          "STATE_CODE = ?", [int(state_code)])
    df_fwd  = safe_read(conn, "MON_DEFL_DROP_DATA",
                          "STATE_CODE = ?", [int(state_code)])
    df_link = safe_read(conn, "CLM_SITE_VWS_LINK",
                          "STATE_CODE = ?", [int(state_code)])
    df_temp = safe_read(conn, "CLM_VWS_TEMP_ANNUAL")    # full table, small-ish
    df_prec = safe_read(conn, "CLM_VWS_PRECIP_ANNUAL")

    if df_exp is None or df_exp.empty:
        log.warning(f"  no EXPERIMENT_SECTION rows; state skipped")
        conn.close()
        return pd.DataFrame()

    # ── Identify unique sections
    sections = (df_exp[["STATE_CODE", "SHRP_ID"]].drop_duplicates()
                  .reset_index(drop=True))
    log.info(f"  {len(sections)} unique sections found")

    rows: List[dict] = []
    for _, sec in sections.iterrows():
        shrp_id = sec["SHRP_ID"]
        row = {
            "state_code": str(int(state_code)).zfill(2),
            "shrp_id": str(shrp_id).strip(),
            "state_name": state_name,
        }

        # ── EXPERIMENT_SECTION  →  experiment_no, n_construction_events
        sub_exp = df_exp[df_exp["SHRP_ID"] == shrp_id]
        if not sub_exp.empty:
            # Find experiment number column (varies by version)
            exp_col = None
            for col in ("EXPERIMENT_NO", "EXPERIMENT_ID", "GPS_TYPE",
                         "EXPERIMENT_CODE", "EXP_TYPE"):
                if col in sub_exp.columns:
                    exp_col = col; break
            if exp_col:
                vals = sub_exp[exp_col].dropna().astype(str).unique()
                row["experiment_no"] = ",".join(sorted(vals))
            else:
                row["experiment_no"] = ""
            row["n_construction_events"] = int(len(sub_exp))
        else:
            row["experiment_no"] = ""; row["n_construction_events"] = 0

        # Infer GPS family from experiment_no
        exp_str = str(row["experiment_no"]).upper()
        gps_family = "OTHER"
        if "GPS-1" in exp_str or exp_str.strip() in ("1", "01"):
            gps_family = "GPS-1"
        elif "GPS-2" in exp_str or exp_str.strip() in ("2", "02"):
            gps_family = "GPS-2"
        elif exp_str.startswith("SPS"):
            gps_family = "SPS"
        elif exp_str == "":
            gps_family = "UNKNOWN"
        row["inferred_gps_family"] = gps_family

        # ── INV_LAYER  →  layers, base material, AC thickness
        sub_inv = df_inv[df_inv["SHRP_ID"] == shrp_id] if df_inv is not None else None
        if sub_inv is not None and not sub_inv.empty:
            # Take the most recent CONSTRUCTION_NO
            if "CONSTRUCTION_NO" in sub_inv.columns:
                latest_cn = sub_inv["CONSTRUCTION_NO"].max()
                sub_inv = sub_inv[sub_inv["CONSTRUCTION_NO"] == latest_cn]
            row["n_inv_layers"] = int(len(sub_inv))

            # AC layers: LAYER_TYPE = 'A' (asphalt) per LTPP spec
            ac_layers = sub_inv[sub_inv.get("LAYER_TYPE", "") == "A"] \
                if "LAYER_TYPE" in sub_inv.columns else pd.DataFrame()
            row["n_ac_layers"] = int(len(ac_layers))

            # Total AC thickness (sum MEAN_THICKNESS, in inches)
            if not ac_layers.empty and "MEAN_THICKNESS" in ac_layers.columns:
                row["total_ac_thickness_in"] = float(
                    pd.to_numeric(ac_layers["MEAN_THICKNESS"],
                                   errors="coerce").sum(skipna=True))
            else:
                row["total_ac_thickness_in"] = np.nan

            # Base layer: LAYER_TYPE = 'B' (the structural base, usually 1 row)
            base_layers = sub_inv[sub_inv.get("LAYER_TYPE", "") == "B"] \
                if "LAYER_TYPE" in sub_inv.columns else pd.DataFrame()
            if not base_layers.empty:
                # Take the top base (highest LAYER_NO if numbered from bottom up)
                base = base_layers.iloc[-1]
                mat_code = base.get("MATERIAL_TYPE", None)
                row["base_material_code"] = (int(mat_code)
                                              if pd.notna(mat_code) else np.nan)
                semi = _is_semi_rigid_base(mat_code)
                row["base_is_semi_rigid"] = (
                    "yes" if semi is True else
                    "no"  if semi is False else "unknown")
            else:
                row["base_material_code"] = np.nan
                row["base_is_semi_rigid"] = "unknown"
        else:
            row.update({"n_inv_layers": 0, "n_ac_layers": 0,
                          "total_ac_thickness_in": np.nan,
                          "base_material_code": np.nan,
                          "base_is_semi_rigid": "unknown"})

        # ── ANALYSIS_IRI  →  n_timepoints, years_observed
        sub_iri = df_iri[df_iri["SHRP_ID"] == shrp_id] if df_iri is not None else None
        if sub_iri is not None and not sub_iri.empty:
            row["n_iri_timepoints"] = int(len(sub_iri))
            # Find date column
            date_col = None
            for c in ("VISIT_DATE", "SURVEY_DATE", "DATE_TESTED",
                        "MEASUREMENT_DATE"):
                if c in sub_iri.columns:
                    date_col = c; break
            if date_col:
                dates = pd.to_datetime(sub_iri[date_col], errors="coerce").dropna()
                if len(dates) >= 2:
                    row["iri_years_observed"] = float(
                        (dates.max() - dates.min()).days / 365.25)
                    row["iri_first_date"] = str(dates.min().date())
                    row["iri_last_date"]  = str(dates.max().date())
                else:
                    row["iri_years_observed"] = np.nan
                    row["iri_first_date"] = ""; row["iri_last_date"] = ""
            else:
                row["iri_years_observed"] = np.nan
                row["iri_first_date"] = ""; row["iri_last_date"] = ""
        else:
            row["n_iri_timepoints"] = 0; row["iri_years_observed"] = np.nan
            row["iri_first_date"] = ""; row["iri_last_date"] = ""

        # ── FWD count
        sub_fwd = df_fwd[df_fwd["SHRP_ID"] == shrp_id] if df_fwd is not None else None
        row["n_fwd_tests"] = int(len(sub_fwd)) if sub_fwd is not None else 0

        # ── Climate via VWS link
        mean_temp = total_prec = freeze_idx = np.nan
        if df_link is not None and df_temp is not None and df_prec is not None:
            link = df_link[df_link["SHRP_ID"] == shrp_id]
            if not link.empty and "VWS_ID" in link.columns:
                vws_ids = link["VWS_ID"].dropna().unique()
                if len(vws_ids) > 0:
                    # Mean annual temperature across years
                    if "VWS_ID" in df_temp.columns:
                        t_sub = df_temp[df_temp["VWS_ID"].isin(vws_ids)]
                        for col in ("MEAN_ANN_TEMP_AVG", "MEAN_ANN_TEMP",
                                     "MEAN_ANNUAL_TEMP"):
                            if col in t_sub.columns:
                                vals = pd.to_numeric(t_sub[col], errors="coerce")
                                if vals.notna().any():
                                    mean_temp = float(vals.mean(skipna=True))
                                    break
                        for col in ("FREEZE_INDEX_AVG", "FREEZE_INDEX",
                                     "FREEZING_INDEX_DAYS"):
                            if col in t_sub.columns:
                                vals = pd.to_numeric(t_sub[col], errors="coerce")
                                if vals.notna().any():
                                    freeze_idx = float(vals.mean(skipna=True))
                                    break
                    if "VWS_ID" in df_prec.columns:
                        p_sub = df_prec[df_prec["VWS_ID"].isin(vws_ids)]
                        for col in ("TOTAL_ANN_PRECIP_AVG", "TOTAL_ANN_PRECIP",
                                     "ANN_PRECIPITATION"):
                            if col in p_sub.columns:
                                vals = pd.to_numeric(p_sub[col], errors="coerce")
                                if vals.notna().any():
                                    total_prec = float(vals.mean(skipna=True))
                                    break

        row["mean_ann_temp_C"]     = mean_temp
        row["total_ann_precip_mm"] = total_prec
        row["freeze_index_C_days"] = freeze_idx

        # ── Climate zone inference (LTPP convention)
        #   Wet  if total_ann_precip > 508 mm/yr (20 in/yr)
        #   Freeze if freeze_index > 0 (or temp < threshold as fallback)
        zone = "UNKNOWN"
        wet = (not np.isnan(total_prec)) and (total_prec > 508)
        dry = (not np.isnan(total_prec)) and (total_prec <= 508)
        if not np.isnan(freeze_idx):
            frozen = (freeze_idx > 50.0)   # >50 C-days freeze index
        elif not np.isnan(mean_temp):
            frozen = (mean_temp < 7.0)     # rough fallback
        else:
            frozen = None
        if wet and frozen is True:  zone = "WF"
        if wet and frozen is False: zone = "WNF"
        if dry and frozen is True:  zone = "DF"
        if dry and frozen is False: zone = "DNF"
        row["inferred_climate_zone"] = zone

        rows.append(row)

    conn.close()
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Scoring (rank sections for selection)
# ────────────────────────────────────────────────────────────────────

def score_candidate(r: pd.Series) -> float:
    """Heuristic completeness/quality score 0-100."""
    s = 0.0
    # Has at least 2 AC layers (project needs multi-layer AC)
    if r.get("n_ac_layers", 0) >= 2:  s += 20
    elif r.get("n_ac_layers", 0) >= 1: s += 10
    # IRI quality
    n_iri = r.get("n_iri_timepoints", 0) or 0
    if n_iri >= 20: s += 25
    elif n_iri >= 10: s += 20
    elif n_iri >= 5:  s += 10
    yrs = r.get("iri_years_observed", 0) or 0
    if yrs >= 15: s += 20
    elif yrs >= 10: s += 15
    elif yrs >= 5:  s += 8
    # FWD coverage
    n_fwd = r.get("n_fwd_tests", 0) or 0
    if n_fwd >= 500:  s += 15
    elif n_fwd >= 100: s += 10
    elif n_fwd >= 20:  s += 5
    # Climate zone known
    if r.get("inferred_climate_zone", "UNKNOWN") != "UNKNOWN": s += 10
    # GPS family known
    if r.get("inferred_gps_family", "UNKNOWN") in ("GPS-1", "GPS-2"): s += 10
    return float(s)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("LTPP candidate scan: 11 states × full Primary_Data.accdb")
    log.info("=" * 70)
    log.info(f"SDR dir : {SDR_DIR}")
    log.info(f"Out dir : {OUT_DIR}")

    if not SDR_DIR.exists():
        log.error(f"SDR directory not found: {SDR_DIR}")
        sys.exit(1)

    all_dfs: List[pd.DataFrame] = []
    for state_dir in sorted(SDR_DIR.iterdir()):
        if not state_dir.is_dir() or not state_dir.name.startswith("SDR39_"):
            continue
        state_abbr = state_dir.name.replace("SDR39_", "")
        # Map abbr → FIPS code
        abbr_to_code = {
            "AZ": "04", "CA": "06", "FL": "12", "ID": "16", "MI": "26",
            "MN": "27", "MT": "30", "NC": "37", "NV": "32", "TX": "48",
            "WI": "55",
        }
        state_code = abbr_to_code.get(state_abbr)
        if not state_code:
            log.warning(f"  unknown state abbreviation: {state_abbr}")
            continue

        primary = find_primary_accdb(state_dir)
        if not primary:
            log.warning(f"State {state_code} ({state_abbr}): no Primary_Data.accdb")
            continue

        log.info(f"State {state_code} ({STATE_NAMES.get(state_code, state_abbr)}): "
                  f"{primary.name}")
        try:
            df = scan_one_state(primary, state_code)
        except Exception as e:
            log.error(f"  scan failed: {e}")
            continue
        if not df.empty:
            log.info(f"  → {len(df)} sections recorded")
            all_dfs.append(df)

    if not all_dfs:
        log.error("No data scanned. Check SDR directory structure.")
        sys.exit(2)

    big = pd.concat(all_dfs, ignore_index=True)

    # Sort and write raw
    big = big.sort_values(by=["state_code", "shrp_id"]).reset_index(drop=True)
    big.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    log.info("=" * 70)
    log.info(f"Wrote RAW: {RAW_CSV}  ({len(big)} sections total)")

    # Score
    big["completeness_score"] = big.apply(score_candidate, axis=1)

    # Filter & rank
    filt = big.copy()
    filt = filt[filt["inferred_gps_family"].isin(["GPS-1", "GPS-2"])]
    filt = filt[filt["n_iri_timepoints"] >= 5]
    filt = filt[filt["n_ac_layers"] >= 2]
    filt = filt.sort_values(
        by=["inferred_gps_family", "inferred_climate_zone",
            "completeness_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    filt.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Wrote SCORED: {SCORED_CSV}  ({len(filt)} qualified sections)")

    # Summary per (gps_family, climate_zone)
    log.info("=" * 70)
    log.info("Summary of qualified candidates by GPS family × climate zone:")
    summary = (filt.groupby(["inferred_gps_family", "inferred_climate_zone"])
                  .size().reset_index(name="n_sections"))
    for _, r in summary.iterrows():
        log.info(f"  {r['inferred_gps_family']:6s}  {r['inferred_climate_zone']:8s}  "
                  f"n={r['n_sections']}")

    log.info("=" * 70)
    log.info("DONE. Inspect candidates_scored.csv and pick 12 sections")
    log.info("  (6 GPS-1 + 6 GPS-2, covering 4 climate × 3 subgrade matrix).")


if __name__ == "__main__":
    main()
