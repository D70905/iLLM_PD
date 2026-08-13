"""
extract_fwd_temperature_modulus_pairs.py
=========================================
Extract (temperature, modulus) pairs from LTPP FWD data for master curve calibration.

Data sources (SDR39 Access DB):
  - MON_DEFL_TEMP_VALUES: LAYER_TEMPERATURE_1 (pavement temp at FWD test)
  - BAKCAL_MODULUS_SECTION_LAYER: AVG_MODULUS per BC_LAYER_NO (FWD back-calculated E)

Strategy:
  - MON_DEFL_TEMP_VALUES keyed by (STATE_CODE, SHRP_ID, TEST_DATE)
  - BAKCAL_MODULUS_SECTION_LAYER keyed by (STATE_CODE, SHRP_ID, FWD_PASS)
  - Join via MON_DEFL_DROP_DATA: (STATE_CODE, SHRP_ID, TEST_DATE) -> available passes
  - Match each temperature record to the closest modulus measurement on the same date

Output: CSV with (section_id, test_date, T_C, E_MPa, layer_no) for calibration
"""

import os, sys
import numpy as np
import pandas as pd
import pyodbc
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATE_ABBREVS = {
    "04": "AZ", "06": "CA", "12": "FL", "16": "ID",
    "27": "MN", "30": "MT", "48": "TX",
}
STATES_DIR = Path("D:/iLLM_PD_new/experiments/ltpp_data/sdr39/states")


def find_primary_db(state_code: str):
    abbrev = STATE_ABBREVS.get(state_code, "")
    for pattern in ("*.accdb", "*.mdb"):
        for p in STATES_DIR.rglob(pattern):
            if "Skeleton_Database" in p.parts:
                continue
            if abbrev and abbrev in p.name.upper() and "PRIMARY_DATA" in p.name.upper():
                return p
    return None


def extract_t_e_pairs(state_code: str, shrp_id: str) -> pd.DataFrame:
    """Extract (temperature, modulus) pairs for one section."""
    db_path = find_primary_db(state_code)
    if db_path is None:
        return pd.DataFrame()

    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
    conn = pyodbc.connect(conn_str)

    # 1. Get temperature data
    temp_query = """
        SELECT TEST_DATE, LAYER_TEMPERATURE_1 as T1, LAYER_TEMPERATURE_2 as T2,
               LAYER_TEMPERATURE_3 as T3, SURFACE_TEMPERATURE as Tsurf
        FROM [MON_DEFL_TEMP_VALUES]
        WHERE STATE_CODE = ? AND SHRP_ID = ?
        ORDER BY TEST_DATE
    """
    temp_df = pd.read_sql(temp_query, conn, params=[int(state_code), shrp_id])
    if temp_df.empty:
        conn.close()
        return pd.DataFrame()

    # 2. Get modulus data per FWD_PASS
    mod_query = """
        SELECT FWD_PASS, BC_LAYER_NO, AVG_MODULUS
        FROM [BAKCAL_MODULUS_SECTION_LAYER]
        WHERE STATE_CODE = ? AND SHRP_ID = ?
        ORDER BY FWD_PASS, BC_LAYER_NO
    """
    mod_df = pd.read_sql(mod_query, conn, params=[int(state_code), shrp_id])
    if mod_df.empty:
        conn.close()
        return pd.DataFrame()

    # 3. Get FWD drop data to link TEST_DATE and FWD_PASS
    # Each unique test_date can have multiple measurements
    fwd_query = """
        SELECT DISTINCT TEST_DATE
        FROM [MON_DEFL_DROP_DATA]
        WHERE STATE_CODE = ? AND SHRP_ID = ?
        ORDER BY TEST_DATE
    """
    fwd_df = pd.read_sql(fwd_query, conn, params=[int(state_code), shrp_id])
    conn.close()

    if fwd_df.empty:
        return pd.DataFrame()

    # 4. Merge: for each temperature measurement date,
    #    find modulus from nearest available FWD_PASS
    temp_df["TEST_DATE"] = pd.to_datetime(temp_df["TEST_DATE"])
    fwd_df["TEST_DATE"] = pd.to_datetime(fwd_df["TEST_DATE"])

    # Assign sequential pass numbers to FWD test dates (crude but functional)
    fwd_dates = sorted(fwd_df["TEST_DATE"].unique())

    records = []
    for _, trow in temp_df.iterrows():
        tdate = trow["TEST_DATE"]
        t1 = trow["T1"]  # AC layer temperature
        if pd.isna(t1):
            continue

        # Find closest FWD date
        closest_date = min(fwd_dates, key=lambda d: abs((d - tdate).days))
        # Get corresponding modulus
        # Estimate which FWD_PASS this corresponds to (by date index)
        pass_idx = fwd_dates.index(closest_date)
        # FWD_PASS values in modulus table are sequential
        mod_for_pass = mod_df[mod_df["FWD_PASS"] == pass_idx + 1]
        if mod_for_pass.empty:
            # Try the range around this pass
            mod_for_pass = mod_df[
                (mod_df["FWD_PASS"] >= max(1, pass_idx))
                & (mod_df["FWD_PASS"] <= pass_idx + 2)
            ]

        for _, mrow in mod_for_pass.iterrows():
            records.append({
                "test_date": tdate,
                "T_pav_C": float(t1),
                "T_mid_C": float(trow["T2"]) if not pd.isna(trow["T2"]) else None,
                "T_deep_C": float(trow["T3"]) if not pd.isna(trow["T3"]) else None,
                "layer_no": int(mrow["BC_LAYER_NO"]),
                "E_MPa": float(mrow["AVG_MODULUS"]),
            })

    return pd.DataFrame(records)


def main():
    config_path = Path(__file__).parent.parent / \
        "experiments/ltpp_data/scripts/section_selection_final12.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_pairs = []

    for sec in config["sections"]:
        sid = sec["section_id"]
        state = sec["state_code"]
        shrp = sec["shrp_id"]
        cz = sec["climate_zone"]

        print(f"  {sid} ({cz})...", end=" ")
        df = extract_t_e_pairs(state, shrp)
        if df.empty:
            print(f"NO DATA")
            continue

        df["section_id"] = sid
        df["climate_zone"] = cz

        # Only AC layer (BC_LAYER_NO = 1 in LTPP backcalc is typically surface AC)
        ac = df[df["layer_no"] == 1]
        print(f"{len(df)} total pairs, {len(ac)} AC layer (T range: "
              f"{ac['T_pav_C'].min():.0f}-{ac['T_pav_C'].max():.0f}C, "
              f"E range: {ac['E_MPa'].min():.0f}-{ac['E_MPa'].max():.0f} MPa)")

        all_pairs.append(df)

    if all_pairs:
        combined = pd.concat(all_pairs, ignore_index=True)
        out_path = Path(__file__).parent.parent / \
            "experiments/ltpp_data/sdr39/extracted/fwd_T_E_pairs_all_sections.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        print(f"Total (T,E) pairs: {len(combined)}")

        # Summary
        for sid in combined["section_id"].unique():
            sdf = combined[(combined["section_id"] == sid) & (combined["layer_no"] == 1)]
            if len(sdf) == 0:
                continue
            print(f"  {sid}: {len(sdf)} AC pairs, "
                  f"T=[{sdf['T_pav_C'].min():.0f}, {sdf['T_pav_C'].max():.0f}]C, "
                  f"E=[{sdf['E_MPa'].min():.0f}, {sdf['E_MPa'].max():.0f}] MPa")


if __name__ == "__main__":
    main()