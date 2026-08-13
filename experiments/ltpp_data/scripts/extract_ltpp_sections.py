"""
extract_ltpp_sections.py
========================
从 LTPP SDR 39 (Microsoft Access 数据库) 自动提取 12 个路段的数据。

工作流：
1. 读 section_selection.yaml 拿到 12 段的 (state_code, shrp_id)
2. 对每个州（10 个左右），连接对应的 .mdb/.accdb 文件
3. 对每段，从每个 tables_to_extract 里 SELECT 数据
4. 整理成每段一个子文件夹 + 一份 master xlsx

用法：
    cd D:\\iLLM_PD_new\\experiments\\ltpp_data\\scripts
    python extract_ltpp_sections.py --config section_selection.yaml

输出：
    extracted/                          (每段一个子目录)
        27_1023/
            INV_LAYER.csv
            TST_AC01.csv
            MON_DEFL_DROP_DATA.csv
            MON_PROFILE_IRI.csv
            TRF_ESAL_COMPUTED.csv
            CLM_VWS_TEMP_ANNUAL.csv
            CLM_VWS_PRECIP_ANNUAL.csv
            section_metadata.json
        26_xxxx/
            ...
    ltpp_12_sections.xlsx               (master 表)
    extraction_log.txt                  (日志，方便审查)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
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
logger = logging.getLogger("extract_ltpp")


# ============================================================
# 数据类
# ============================================================

@dataclass
class SectionSpec:
    section_id: str
    state_code: str
    shrp_id: str
    state_name: str
    climate_zone: str
    expected_subgrade_bin: str
    addresses_reviewer_comments: str
    is_baseline: bool


@dataclass
class ExtractedSection:
    spec: SectionSpec
    data_paths: dict           # {table_name: Path of CSV}
    metadata: dict             # 抽取过程的统计信息（行数、首测/末测日期等）


# ============================================================
# 工具函数
# ============================================================

def find_sdr_for_state(states_dir: Path, state_code: str) -> Optional[Path]:
    """在 states_dir 递归查找该州的 Primary_Data.accdb。

    LTPP SDR 39 by-State 的实际结构是：
        states/
          SDR39_<STATE_ABBREV>/                          (顶层子目录)
            SDR_39_<STATE_ABBREV>_Primary_Data.accdb     (核心数据库 ← 用这个)
            SDR_39_<STATE_ABBREV>_LTAS_Tables.accdb      (LTAS 流量数据库)
            Data_User_Documents/
            Skeleton_Database/

    优先返回 Primary_Data.accdb；找不到就返回任意 .accdb 文件。
    """
    state_abbrevs = {
    "04": "AZ", "06": "CA", "12": "FL", "16": "ID", "26": "MI", "27": "MN",
    "30": "MT", "32": "NV", "37": "NC", "48": "TX", "55": "WI",
    "56": "WY",
    }
    abbrev = state_abbrevs.get(state_code, "")

    # 递归扫描所有 .accdb / .mdb（含子目录），但跳过 Skeleton_Database
    all_dbs = []
    for pattern in ("*.accdb", "*.mdb"):
        for p in states_dir.rglob(pattern):
            if "Skeleton_Database" in p.parts:
                continue
            all_dbs.append(p)

    if not all_dbs:
        return None

    # 按州缩写过滤
    state_specific = [p for p in all_dbs if abbrev and abbrev in p.name.upper()]
    if not state_specific:
        return None

    # 优先 Primary_Data
    primary = [p for p in state_specific if "PRIMARY_DATA" in p.name.upper()]
    if primary:
        return primary[0]

    # 退而求其次：返回任意匹配（避免 LTAS）
    non_ltas = [p for p in state_specific if "LTAS" not in p.name.upper()]
    if non_ltas:
        return non_ltas[0]

    return state_specific[0]


def connect_access(mdb_path: Path) -> pyodbc.Connection:
    """打开一个 Access 数据库连接。"""
    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={mdb_path};"
    )
    return pyodbc.connect(conn_str)


def _get_table_columns(conn: pyodbc.Connection, table: str) -> Optional[set]:
    """获取一张表的字段名集合（大写）；表不存在则返回 None。"""
    cursor = conn.cursor()
    tables = {row.table_name.upper()
              for row in cursor.tables(tableType="TABLE")}
    if table.upper() not in tables:
        return None
    col_info = cursor.columns(table=table)
    return {c.column_name.upper() for c in col_info}


def _read_sql_silent(sql: str, conn, params: list) -> pd.DataFrame:
    """pd.read_sql 包装：屏蔽 pandas-pyodbc 的 UserWarning。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.read_sql(sql, conn, params=params)


def safe_query(conn: pyodbc.Connection, table: str, state_code: str,
               shrp_id: str, query_mode: str = "direct",
               fields: Optional[list] = None) -> pd.DataFrame:
    """容错查询。

    query_mode:
      - "direct"        (默认): WHERE STATE_CODE=? AND SHRP_ID=?
      - "via_vws_link"  通过 CLM_SITE_VWS_LINK JOIN VWS_ID，用于气候表
    """
    try:
        actual_cols = _get_table_columns(conn, table)
        if actual_cols is None:
            logger.warning(f"  Table {table} NOT FOUND in this database")
            return pd.DataFrame()

        # ── 直接模式 ──
        if query_mode == "direct":
            if "STATE_CODE" not in actual_cols or "SHRP_ID" not in actual_cols:
                logger.warning(f"  Table {table} missing STATE_CODE/SHRP_ID; "
                                f"available cols (first 15): "
                                f"{sorted(actual_cols)[:15]}")
                return pd.DataFrame()
            sql = f"SELECT * FROM [{table}] WHERE STATE_CODE = ? AND SHRP_ID = ?"
            return _read_sql_silent(sql, conn, params=[int(state_code), shrp_id])

        # ── 通过 VWS_ID 链接气候表 ──
        if query_mode == "via_vws_link":
            link_cols = _get_table_columns(conn, "CLM_SITE_VWS_LINK")
            if link_cols is None:
                logger.warning("  CLM_SITE_VWS_LINK NOT FOUND; cannot join")
                return pd.DataFrame()
            # 找 VWS_ID 字段（候选 3 个常见名）
            vws_id_field = None
            for cand in ["VWS_ID", "VWS_NUMBER", "STATION_ID"]:
                if cand in link_cols and cand in actual_cols:
                    vws_id_field = cand
                    break
            if vws_id_field is None:
                logger.warning(f"  Cannot find common VWS_ID field; "
                                f"link cols={sorted(link_cols)[:10]}, "
                                f"target cols={sorted(actual_cols)[:10]}")
                return pd.DataFrame()
            sql = (
                f"SELECT t.* FROM [{table}] AS t "
                f"INNER JOIN [CLM_SITE_VWS_LINK] AS l "
                f"ON t.{vws_id_field} = l.{vws_id_field} "
                f"WHERE l.STATE_CODE = ? AND l.SHRP_ID = ?"
            )
            return _read_sql_silent(sql, conn, params=[int(state_code), shrp_id])

        logger.warning(f"  Unknown query_mode '{query_mode}' for {table}")
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"  Query failed for {table} (mode={query_mode}): {e}")
        return pd.DataFrame()


# ============================================================
# 单段提取
# ============================================================

def extract_one_section(
    spec: SectionSpec,
    sdr_path: Path,
    output_root: Path,
    tables_cfg: dict,
) -> Optional[ExtractedSection]:
    """对一个段做完整抽取。"""
    section_dir = output_root / spec.section_id
    section_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Section {spec.section_id}: connecting to {sdr_path.name}")
    try:
        conn = connect_access(sdr_path)
    except Exception as e:
        logger.error(f"  Cannot open {sdr_path}: {e}")
        return None

    data_paths = {}
    metadata = {
        "section_id": spec.section_id,
        "state_code": spec.state_code,
        "shrp_id": spec.shrp_id,
        "state_name": spec.state_name,
        "climate_zone": spec.climate_zone,
        "expected_subgrade_bin": spec.expected_subgrade_bin,
        "addresses_reviewer_comments": spec.addresses_reviewer_comments,
        "is_baseline": spec.is_baseline,
        "sdr_source_file": str(sdr_path.name),
        "extraction_date": datetime.now().isoformat(),
        "table_row_counts": {},
    }

    for logical_name, cfg in tables_cfg.items():
        candidates = [cfg["sdr_table"]]
        if "fallback_table" in cfg:
            candidates.append(cfg["fallback_table"])
        query_mode = cfg.get("query_mode", "direct")

        df = pd.DataFrame()
        used_table = None
        for tbl in candidates:
            df = safe_query(conn, tbl, spec.state_code, spec.shrp_id,
                            query_mode=query_mode)
            if not df.empty:
                used_table = tbl
                break

        if df.empty:
            logger.warning(f"  {logical_name}: 0 rows (table tried: {candidates}, "
                           f"mode={query_mode})")
            metadata["table_row_counts"][logical_name] = 0
            continue

        csv_path = section_dir / f"{logical_name}.csv"
        df.to_csv(csv_path, index=False)
        data_paths[logical_name] = csv_path
        metadata["table_row_counts"][logical_name] = len(df)
        logger.info(f"  {logical_name}: {len(df)} rows ({used_table}) "
                    f"→ {csv_path.name}")

    conn.close()

    # 保存 metadata
    with open(section_dir / "section_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return ExtractedSection(spec=spec, data_paths=data_paths, metadata=metadata)


# ============================================================
# Master 表汇总
# ============================================================

def aggregate_to_master(extracted: list[ExtractedSection],
                       config: dict,
                       output_xlsx: Path):
    """把每段的数据汇总成 1 行写进 master xlsx。"""
    rows = []
    for sec in extracted:
        if sec is None:
            continue
        row = {
            "section_id": sec.spec.section_id,
            "state_code": sec.spec.state_code,
            "shrp_id": sec.spec.shrp_id,
            "state_name": sec.spec.state_name,
            "climate_zone": sec.spec.climate_zone,
            "expected_subgrade_bin": sec.spec.expected_subgrade_bin,
            "addresses_reviewer_comments": sec.spec.addresses_reviewer_comments,
            "is_baseline": sec.spec.is_baseline,
            "sdr_version": config["sdr"]["sdr_version"],
        }

        # 从 INV_LAYER 提取结构（LTPP 单位是 inch；字段名 MEAN_THICKNESS）
        if "inv_layer" in sec.data_paths:
            inv = pd.read_csv(sec.data_paths["inv_layer"])
            inv = inv.sort_values("LAYER_NO") if "LAYER_NO" in inv.columns else inv
            # 兼容多种厚度字段名
            inv_cols_upper = {c.upper(): c for c in inv.columns}
            thick_col = None
            for cand in ("MEAN_THICKNESS", "REPR_THICKNESS", "THICKNESS_MEAN", "THICKNESS"):
                if cand in inv_cols_upper:
                    thick_col = inv_cols_upper[cand]
                    break
            matl_col = inv_cols_upper.get("MATERIAL_TYPE") or inv_cols_upper.get("MATL_CODE")
            desc_col = inv_cols_upper.get("DESCRIPTION") or inv_cols_upper.get("LAYER_TYPE")
            for i, layer in enumerate(inv.itertuples()):
                col_h = f"h_layer{i+1}_m"
                col_m = f"matl_layer{i+1}"
                if thick_col is not None:
                    thick = getattr(layer, thick_col, None)
                    if thick is not None and pd.notna(thick):
                        # LTPP 厚度单位是 inch → 转 m
                        row[col_h] = float(thick) * 0.0254
                if matl_col is not None:
                    row[col_m] = getattr(layer, matl_col, None)
                elif desc_col is not None:
                    row[col_m] = getattr(layer, desc_col, None)
        else:
            row["inv_layer_missing"] = True

        # IRI 时序统计（ANALYSIS_IRI 表的字段名可能为 VISIT_DATE / MRI / IRI）
        if "mon_profile_iri" in sec.data_paths:
            iri = pd.read_csv(sec.data_paths["mon_profile_iri"])
            cols_upper = {c.upper(): c for c in iri.columns}
            date_col = None
            for cand in ("VISIT_DATE", "PROFILE_DATE", "TEST_DATE",
                         "SURVEY_DATE", "MEASUREMENT_DATE", "DATE"):
                if cand in cols_upper:
                    date_col = cols_upper[cand]
                    break
            iri_col = None
            for cand in ("MRI", "MEAN_IRI", "IRI", "MRI_M_PER_KM"):
                if cand in cols_upper:
                    iri_col = cols_upper[cand]
                    break
            if date_col is not None:
                try:
                    iri[date_col] = pd.to_datetime(iri[date_col], errors="coerce")
                    iri = iri.dropna(subset=[date_col]).sort_values(date_col)
                    if len(iri) > 0:
                        years = (iri[date_col].max() - iri[date_col].min()).days / 365.25
                        row["iri_n_years_observed"] = round(float(years), 1)
                        row["iri_n_timepoints"] = int(len(iri))
                        row["iri_first_date"] = str(iri[date_col].min().date())
                        row["iri_last_date"] = str(iri[date_col].max().date())
                        if iri_col is not None:
                            row["iri_first_value"] = float(iri[iri_col].iloc[0])
                            row["iri_last_value"] = float(iri[iri_col].iloc[-1])
                            row["iri_col_used"] = iri_col
                except Exception as e:
                    row["iri_parse_error"] = str(e)
            else:
                row["iri_no_date_col"] = "; ".join(iri.columns[:10])

        # FWD 测试次数
        if "mon_defl_drop_data" in sec.data_paths:
            fwd = pd.read_csv(sec.data_paths["mon_defl_drop_data"])
            if "TEST_DATE" in fwd.columns:
                row["fwd_n_tests"] = int(fwd["TEST_DATE"].nunique())
            else:
                row["fwd_n_tests"] = len(fwd)

        # 累计 ESAL
        if "trf_esal" in sec.data_paths:
            esal = pd.read_csv(sec.data_paths["trf_esal"])
            if "ANL_KESAL_LTPP_LN" in esal.columns:
                row["traffic_ESAL_M"] = float(esal["ANL_KESAL_LTPP_LN"].sum()) / 1000.0
            elif "AADTT_AT_TIME" in esal.columns:
                row["traffic_AADTT_avg"] = float(esal["AADTT_AT_TIME"].mean())

        # 气候平均值（CLM 表通过 VWS_LINK 查到，可能字段名也漂移）
        if "clm_temp" in sec.data_paths:
            t = pd.read_csv(sec.data_paths["clm_temp"])
            t_cols = {c.upper(): c for c in t.columns}
            for cand in ("MEAN_ANN_TEMP_AVG", "MEAN_TEMP", "MEAN_ANNUAL_TEMP", "AVG_TEMP"):
                if cand in t_cols:
                    row["mean_ann_temp_C_avg"] = float(t[t_cols[cand]].mean())
                    break
        if "clm_precip" in sec.data_paths:
            p = pd.read_csv(sec.data_paths["clm_precip"])
            p_cols = {c.upper(): c for c in p.columns}
            for cand in ("TOTAL_ANN_PRECIP", "TOTAL_PRECIP", "ANNUAL_PRECIP", "PRECIP_TOTAL"):
                if cand in p_cols:
                    row["total_ann_precip_mm_avg"] = float(p[p_cols[cand]].mean())
                    break

        # 通车日期
        if "experiment_section" in sec.data_paths:
            exp = pd.read_csv(sec.data_paths["experiment_section"])
            if "ASSIGN_DATE" in exp.columns and len(exp) > 0:
                try:
                    row["construction_date"] = str(pd.to_datetime(exp["ASSIGN_DATE"].iloc[0]).date())
                except Exception:
                    row["construction_date"] = str(exp["ASSIGN_DATE"].iloc[0])

        # 质检标记
        qc = config.get("quality_checks", {})
        flags = []
        if row.get("iri_n_years_observed", 0) < qc.get("iri_min_n_years", 10):
            flags.append(f"iri_n_years<{qc['iri_min_n_years']}")
        if row.get("iri_n_timepoints", 0) < qc.get("iri_min_n_timepoints", 5):
            flags.append(f"iri_n_timepoints<{qc['iri_min_n_timepoints']}")
        if row.get("fwd_n_tests", 0) < qc.get("fwd_min_n_tests", 3):
            flags.append(f"fwd_n_tests<{qc['fwd_min_n_tests']}")
        row["qc_flags"] = "; ".join(flags) if flags else "PASS"

        rows.append(row)

    master_df = pd.DataFrame(rows)

    # 排序：baseline 段放最后，其它按 climate + subgrade
    if "is_baseline" in master_df.columns:
        master_df = master_df.sort_values(
            ["is_baseline", "climate_zone", "expected_subgrade_bin"]
        ).reset_index(drop=True)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_excel(output_xlsx, index=False)
    logger.info(f"Master xlsx written: {output_xlsx}")
    logger.info(f"   total sections in master: {len(master_df)}")
    return master_df


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to section_selection.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate config + list SDR files, no extraction")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    states_dir = Path(config["sdr"]["states_dir"])
    output_root = Path(config["sdr"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    sections = [SectionSpec(**{k: v for k, v in s.items()
                               if k in SectionSpec.__annotations__})
                for s in config["sections"]]
    tables_cfg = config["tables_to_extract"]

    logger.info(f"Loaded {len(sections)} section specs")
    logger.info(f"SDR directory: {states_dir}")
    logger.info(f"Output directory: {output_root}")

    # SDR 文件清单
    available_states = {}
    for s in sections:
        if s.state_code not in available_states:
            mdb = find_sdr_for_state(states_dir, s.state_code)
            if mdb is None:
                logger.error(f"  No SDR file found for state {s.state_code} ({s.state_name})")
            else:
                logger.info(f"  State {s.state_code} ({s.state_name}): {mdb.name}")
            available_states[s.state_code] = mdb

    if args.dry_run:
        logger.info("\n[Dry-run mode] Validation complete; nothing extracted.")
        return

    # 逐段提取
    extracted = []
    for spec in sections:
        sdr_path = available_states.get(spec.state_code)
        if sdr_path is None:
            logger.warning(f"Skipping {spec.section_id}: no SDR for state {spec.state_code}")
            continue
        if spec.shrp_id.upper() in ("XXXX", "REPLACE_WITH_REAL_TX_SHRPID"):
            logger.warning(f"Skipping {spec.section_id}: placeholder SHRP_ID, edit yaml first")
            continue

        result = extract_one_section(spec, sdr_path, output_root, tables_cfg)
        if result is not None:
            extracted.append(result)

    # 汇总到 master
    if extracted:
        master_xlsx = Path(config["output"]["master_xlsx"])
        aggregate_to_master(extracted, config, master_xlsx)
    else:
        logger.warning("No sections extracted; master xlsx not created.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
