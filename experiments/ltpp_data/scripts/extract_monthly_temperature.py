"""
extract_monthly_temperature.py
==============================
从 LTPP SDR 39 Access 数据库提取 12 个路段的 逐月温度数据。

数据源：CLM_VWS_TEMP_MONTH（通过 CLM_SITE_VWS_LINK 关联）
输出：每个路段一个 clm_temp_monthly.csv + 一份汇总表

用法：
    cd <PROJECT_ROOT>\experiments\ltpp_data\scripts
    python extract_monthly_temperature.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyodbc
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_monthly_temp")

# 州代码 → 州缩写 映射（用于查找数据库文件）
STATE_ABBREVS = {
    "04": "AZ", "06": "CA", "12": "FL", "16": "ID",
    "27": "MN", "30": "MT", "48": "TX",
}

# SDR 目录
STATES_DIR = Path("D:/iLLM_PD_new/experiments/ltpp_data/sdr39/states")
OUTPUT_DIR = Path("D:/iLLM_PD_new/experiments/ltpp_data/sdr39/extracted")


def find_primary_db(state_code: str) -> Optional[Path]:
    """查找某个州的 Primary_Data.accdb 文件。"""
    abbrev = STATE_ABBREVS.get(state_code, "")
    for pattern in ("*.accdb", "*.mdb"):
        for p in STATES_DIR.rglob(pattern):
            if "Skeleton_Database" in p.parts:
                continue
            if abbrev and abbrev in p.name.upper() and "PRIMARY_DATA" in p.name.upper():
                return p
    return None


def extract_monthly_temp(state_code: str, shrp_id: str, output_dir: Path) -> Optional[pd.DataFrame]:
    """提取一个路段的逐月温度数据。"""
    db_path = find_primary_db(state_code)
    if db_path is None:
        logger.error(f"  State {state_code}: Primary_Data.accdb NOT FOUND")
        return None

    logger.info(f"  State {state_code} SHRP {shrp_id}: connecting to {db_path.name}")

    try:
        conn_str = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            f"DBQ={db_path};"
        )
        conn = pyodbc.connect(conn_str)
    except Exception as e:
        logger.error(f"  Cannot open {db_path}: {e}")
        return None

    try:
        # 查询 CLM_SITE_VWS_LINK 获取 VWS_ID
        cursor = conn.cursor()
        cursor.execute(
            "SELECT VWS_ID FROM [CLM_SITE_VWS_LINK] WHERE STATE_CODE = ? AND SHRP_ID = ?",
            [int(state_code), shrp_id],
        )
        link_rows = cursor.fetchall()
        if not link_rows:
            logger.warning(f"  No VWS_ID found for STATE_CODE={state_code}, SHRP_ID={shrp_id}")
            return None

        vws_ids = [row[0] for row in link_rows]
        logger.info(f"  Found {len(vws_ids)} VWS_ID(s): {vws_ids}")

        # 提取逐月温度
        all_rows = []
        for vws_id in vws_ids:
            cursor.execute(
                "SELECT * FROM [CLM_VWS_TEMP_MONTH] WHERE VWS_ID = ? ORDER BY YEAR, MONTH",
                [vws_id],
            )
            cols = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                all_rows.append(dict(zip(cols, row)))

        if not all_rows:
            logger.warning(f"  No monthly temp data for any VWS_ID")
            return None

        df = pd.DataFrame(all_rows)
        # 按年月排序
        df = df.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)
        logger.info(f"  Extracted {len(df)} monthly records "
                    f"({df['YEAR'].min()}-{df['YEAR'].max()})")

        return df

    finally:
        conn.close()


def compute_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """计算多年平均的逐月气候统计（用于温度修正）。"""
    monthly = df.groupby("MONTH").agg(
        MEAN_TEMP_AVG=("MEAN_MON_TEMP_AVG", "mean"),
        MEAN_TEMP_STD=("MEAN_MON_TEMP_AVG", "std"),
        MAX_TEMP_AVG=("MAX_MON_TEMP_AVG", "mean"),
        MIN_TEMP_AVG=("MIN_MON_TEMP_AVG", "mean"),
        N_YEARS=("YEAR", "nunique"),
    ).reset_index()
    # 添加年平均
    annual_mean = df["MEAN_MON_TEMP_AVG"].mean()
    monthly["ANNUAL_MEAN_TEMP"] = annual_mean
    return monthly


def main():
    # 加载配置
    config_path = Path(__file__).parent / "section_selection_final12.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sections = config["sections"]
    logger.info(f"Extracting monthly temperature for {len(sections)} sections")

    # 汇总数据
    all_monthly_summaries = []

    for sec in sections:
        section_id = sec["section_id"]
        state_code = sec["state_code"]
        shrp_id = sec["shrp_id"]
        state_name = sec["state_name"]
        climate_zone = sec["climate_zone"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Section {section_id} ({state_name}, {climate_zone})")

        section_dir = OUTPUT_DIR / section_id
        section_dir.mkdir(parents=True, exist_ok=True)

        df = extract_monthly_temp(state_code, shrp_id, section_dir)

        if df is None or df.empty:
            logger.warning(f"  SKIPPED: no monthly temp data for {section_id}")
            continue

        # 保存原始数据
        csv_path = section_dir / "clm_temp_monthly.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")

        # 计算逐月气候统计
        monthly_clim = compute_monthly_climatology(df)
        clim_path = section_dir / "clm_temp_monthly_climatology.csv"
        monthly_clim.to_csv(clim_path, index=False)
        logger.info(f"  Saved climatology: {clim_path}")

        # 汇总信息
        summary = {
            "section_id": section_id,
            "state_name": state_name,
            "climate_zone": climate_zone,
            "n_records": len(df),
            "year_min": int(df["YEAR"].min()),
            "year_max": int(df["YEAR"].max()),
            "annual_mean_temp_C": round(float(df["MEAN_MON_TEMP_AVG"].mean()), 1),
            "annual_max_temp_C": round(float(df["MAX_MON_TEMP"].max()), 1),
            "annual_min_temp_C": round(float(df["MIN_MON_TEMP"].min()), 1),
            "n_years_with_data": int(df["YEAR"].nunique()),
        }
        # 逐月均值（1-12月）
        for month in range(1, 13):
            m_data = df[df["MONTH"] == month]["MEAN_MON_TEMP_AVG"]
            summary[f"T_month_{month:02d}_C"] = round(float(m_data.mean()), 1)

        all_monthly_summaries.append(summary)

    # 保存汇总表
    if all_monthly_summaries:
        summary_df = pd.DataFrame(all_monthly_summaries)
        summary_path = OUTPUT_DIR / "ltpp_12_sections_monthly_temp_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary saved: {summary_path}")
        logger.info(f"  {len(all_monthly_summaries)} sections extracted")

        # 打印概览
        print("\n" + "="*80)
        print("MONTHLY TEMPERATURE CLIMATOLOGY (multi-year average)")
        print("="*80)
        cols = ["section_id", "climate_zone", "annual_mean_temp_C",
                "annual_max_temp_C", "annual_min_temp_C", "n_years_with_data"]
        print(summary_df[cols].to_string(index=False))

        print("\n" + "="*80)
        print("MONTHLY MEAN TEMPERATURES (°C) — rows=sections, columns=months")
        print("="*80)
        month_cols = [f"T_month_{m:02d}_C" for m in range(1, 13)]
        print(summary_df[["section_id"] + month_cols].to_string(index=False))
    else:
        logger.error("No sections extracted!")


if __name__ == "__main__":
    main()