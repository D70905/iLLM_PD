# -*- coding: utf-8 -*-
"""
Extract MON_DIS_AC_REV (fatigue cracking) and MON_RUT_DEPTH_POINT (rutting)
from SDR39 Access databases for the 12 LTPP sections.
"""
import os, sys, pyodbc, csv, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rut_crack_extract")

# 12 sections → state codes
SECTIONS = {
    "04_1034": "AZ", "04_1065": "AZ",
    "06_2004": "CA",
    "12_1060": "FL", "12_4097": "FL",
    "16_1010": "ID",
    "27_1085": "MN", "27_2023": "MN",
    "30_7076": "MT",
    "48_0001": "TX", "48_1076": "TX", "48_1109": "TX",
}

SDR39_DIR = Path("experiments/ltpp_data/sdr39/states")


def extract_table(state: str, table_name: str, section_id: str, out_dir: Path):
    """Extract a table for one section and save as CSV."""
    db_path = SDR39_DIR / f"SDR39_{state}" / f"SDR_39_{state}_Primary_Data.accdb"
    if not db_path.exists():
        logger.warning(f"[{section_id}] DB not found: {db_path}")
        return False

    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        state_code = int(section_id.split("_")[0])
        shrp_id = int(section_id.split("_")[1])

        # Try with state_code + shrp_id filter
        sql = f"SELECT * FROM [{table_name}] WHERE STATE_CODE = {state_code} AND SHRP_ID = {shrp_id}"
        try:
            cursor.execute(sql)
        except Exception:
            # Some tables use different column naming
            sql = f"SELECT * FROM [{table_name}] WHERE STATE_CODE = {state_code}"
            cursor.execute(sql)

        rows = cursor.fetchall()
        if not rows:
            logger.info(f"[{section_id}] {table_name}: 0 rows")
            conn.close()
            return False

        cols = [d[0] for d in cursor.description]
        out_path = out_dir / f"{table_name.lower()}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in rows:
                w.writerow([str(v) if v is not None else "" for v in row])
        logger.info(f"[{section_id}] {table_name}: {len(rows)} rows → {out_path}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[{section_id}] {table_name}: {e}")
        return False


def main():
    tables_to_extract = ["MON_DIS_AC_REV", "MON_RUT_DEPTH_POINT", "MON_DIS_AC_CRACK_INDEX"]

    for sid, state in SECTIONS.items():
        out_dir = Path(f"experiments/ltpp_data/sdr39/extracted/{sid}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for tbl in tables_to_extract:
            extract_table(state, tbl, sid, out_dir)

    # Also check for ANALYSIS_DIS_AC and ANALYSIS_RUTTING (computed distress)
    for sid, state in SECTIONS.items():
        out_dir = Path(f"experiments/ltpp_data/sdr39/extracted/{sid}")
        for tbl in ["ANALYSIS_DIS_AC", "ANALYSIS_RUTTING"]:
            extract_table(state, tbl, sid, out_dir)


if __name__ == "__main__":
    main()