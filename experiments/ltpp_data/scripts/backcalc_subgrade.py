# -*- coding: utf-8 -*-
"""
backcalc_subgrade.py — Estimate E_subgrade per section from FWD deflection.

Reads each section's mon_defl_drop_data.csv (output by extract_ltpp_sections.py)
and computes subgrade modulus using the AASHTO 1993 method:

    E_R [MPa] = 0.24 * P [N] / (d_r [mm] * r [mm])

Using sensor 7 (PEAK_DEFL_7) at offset 60 in (1524 mm) — far enough from
load center that response is dominated by subgrade alone (Boussinesq).

Sensor convention (LTPP IMS spec):
    PEAK_DEFL_1..7 in micrometers (μm)
    Sensor offsets [in]:   0, 8, 12, 18, 24, 36, 60
    Sensor offsets [mm]:   0, 203, 305, 457, 610, 914, 1524

Load convention:
    DROP_HEIGHT = 3 → nominal 9000 lbf ≈ 40 kN (most common standard load)

Outputs:
  D:/iLLM_PD_new/experiments/ltpp_data/subgrade_classification.csv
  D:/iLLM_PD_new/experiments/ltpp_data/subgrade_classification.png  (bar chart)
  Console summary table + bin counts

Run:
    cd D:\\iLLM_PD_new\\experiments\\ltpp_data\\scripts
    python backcalc_subgrade.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────

PROJECT_ROOT     = Path(__file__).resolve().parents[1]   # ltpp_data/
EXTRACTED_DIR    = PROJECT_ROOT / "sdr39" / "extracted"
OUT_CSV          = PROJECT_ROOT / "subgrade_classification.csv"
OUT_PLOT         = PROJECT_ROOT / "subgrade_classification.png"
MASTER_XLSX_IN   = PROJECT_ROOT / "ltpp_12_sections.xlsx"
MASTER_XLSX_OUT  = PROJECT_ROOT / "ltpp_12_sections_with_subgrade.xlsx"

# FWD parameters
SENSOR_OFFSET_MM = 1524.0     # sensor 7 (60 in)
TARGET_LOAD_N    = 40000.0    # ~9000 lbf
STANDARD_DROP_HEIGHTS = (3,)  # filter to drops 3 (9 kip standard);
                              # set to None to skip filtering and use all drops

# Bin boundaries (per yaml's quality_checks)
SOFT_MAX_MPA   = 50.0
MEDIUM_MAX_MPA = 100.0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backcalc")


# ── Core formula ──────────────────────────────────────────────────────

def compute_E_subgrade(df_fwd: pd.DataFrame,
                        sensor_col: str = "PEAK_DEFL_7",
                        load_filter=STANDARD_DROP_HEIGHTS,
                       ) -> Tuple[Optional[float], dict]:
    """
    AASHTO 1993 backcalc on outermost sensor.

    Returns (E_subgrade_MPa, info_dict). info_dict has n_drops, d_r_mean_μm, etc.
    """
    info: dict = {"n_total_rows": int(len(df_fwd))}

    if df_fwd.empty:
        info["status"] = "empty_df"
        return None, info

    # Filter to standard drop height
    sub = df_fwd
    if load_filter is not None and "DROP_HEIGHT" in df_fwd.columns:
        before = len(sub)
        sub = sub[sub["DROP_HEIGHT"].isin(load_filter)]
        info["n_after_drop_filter"] = int(len(sub))
        info["n_filtered_out"] = int(before - len(sub))
        if len(sub) == 0:
            info["status"] = "no_rows_after_drop_filter"
            return None, info

    if sensor_col not in sub.columns:
        info["status"] = f"missing_col_{sensor_col}"
        return None, info

    d_r = pd.to_numeric(sub[sensor_col], errors="coerce").dropna()
    d_r = d_r[d_r > 0]  # negative/zero deflections are garbage
    if len(d_r) == 0:
        info["status"] = "no_valid_deflections"
        return None, info

    # Robust statistics — use median to dampen outliers
    d_r_median = float(d_r.median())
    d_r_mean = float(d_r.mean())
    d_r_p25 = float(d_r.quantile(0.25))
    d_r_p75 = float(d_r.quantile(0.75))

    # Convert μm → mm
    d_r_mm = d_r_median / 1000.0

    # E_R [MPa] = 0.24 × P [N] / (d_r [mm] × r [mm])
    E_R = 0.24 * TARGET_LOAD_N / (d_r_mm * SENSOR_OFFSET_MM)

    info.update({
        "n_drops_used":   int(len(d_r)),
        "d_r_median_um":  d_r_median,
        "d_r_mean_um":    d_r_mean,
        "d_r_p25_um":     d_r_p25,
        "d_r_p75_um":     d_r_p75,
        "E_R_MPa":        E_R,
        "status":         "ok",
    })
    return E_R, info


def bin_subgrade(E_subgrade_MPa: Optional[float]) -> str:
    if E_subgrade_MPa is None or np.isnan(E_subgrade_MPa):
        return "Unknown"
    if E_subgrade_MPa < SOFT_MAX_MPA:   return "Soft"
    if E_subgrade_MPa < MEDIUM_MAX_MPA: return "Medium"
    return "Stiff"


# ── Section iteration ────────────────────────────────────────────────

def process_section(section_dir: Path) -> dict:
    fwd_csv = section_dir / "mon_defl_drop_data.csv"
    row: dict = {"section_id": section_dir.name}

    if not fwd_csv.exists():
        row["status"] = "no_fwd_file"
        row["E_subgrade_MPa"] = np.nan
        row["subgrade_bin"] = "Unknown"
        return row

    try:
        df = pd.read_csv(fwd_csv)
    except Exception as e:
        row["status"] = f"read_error: {e}"
        row["E_subgrade_MPa"] = np.nan
        row["subgrade_bin"] = "Unknown"
        return row

    E, info = compute_E_subgrade(df)
    row.update(info)
    row["E_subgrade_MPa"] = float(E) if E is not None else np.nan
    row["subgrade_bin"] = bin_subgrade(E)
    return row


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("FWD-based E_subgrade backcalculation (AASHTO 1993, sensor 7)")
    log.info("=" * 70)
    log.info(f"Extracted dir : {EXTRACTED_DIR}")
    log.info(f"Sensor        : PEAK_DEFL_7 at {SENSOR_OFFSET_MM:.0f} mm offset")
    log.info(f"Load          : {TARGET_LOAD_N/1000:.0f} kN  (DROP_HEIGHT in {STANDARD_DROP_HEIGHTS})")
    log.info(f"Bin bounds    : Soft<{SOFT_MAX_MPA:.0f}  Medium<{MEDIUM_MAX_MPA:.0f}  Stiff≥{MEDIUM_MAX_MPA:.0f}  [MPa]")
    log.info("-" * 70)

    if not EXTRACTED_DIR.exists():
        log.error(f"Directory not found: {EXTRACTED_DIR}")
        sys.exit(1)

    rows = []
    section_dirs = sorted([p for p in EXTRACTED_DIR.iterdir() if p.is_dir()])
    log.info(f"Found {len(section_dirs)} section directories")
    log.info("")
    for sd in section_dirs:
        row = process_section(sd)
        rows.append(row)
        E = row.get("E_subgrade_MPa", np.nan)
        bin_ = row.get("subgrade_bin", "?")
        n_drops = row.get("n_drops_used", "?")
        d_med = row.get("d_r_median_um", float("nan"))
        if not np.isnan(E):
            log.info(
                f"  {sd.name:<10s}  E_sub = {E:>7.1f} MPa   "
                f"bin = {bin_:<7s}   "
                f"(d_7 median = {d_med:>5.0f} μm, n_drops = {n_drops})"
            )
        else:
            log.warning(f"  {sd.name:<10s}  FAILED: {row.get('status', '?')}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log.info("")
    log.info("=" * 70)
    log.info(f"Wrote: {OUT_CSV}")

    # Summary
    log.info("")
    log.info("Bin distribution:")
    for b, n in df["subgrade_bin"].value_counts().items():
        log.info(f"  {b:<8s}  n={n}")

    # Merge into master xlsx
    if MASTER_XLSX_IN.exists():
        try:
            master = pd.read_excel(MASTER_XLSX_IN)
            keep = df[["section_id", "E_subgrade_MPa", "subgrade_bin"]].copy()
            merged = master.merge(keep, on="section_id", how="left")
            merged.to_excel(MASTER_XLSX_OUT, index=False)
            log.info(f"Wrote merged master: {MASTER_XLSX_OUT}")
        except Exception as e:
            log.warning(f"Failed to merge with master xlsx: {e}")
    else:
        log.warning(f"Master xlsx not found at {MASTER_XLSX_IN}, skipping merge")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = df.dropna(subset=["E_subgrade_MPa"]).sort_values("E_subgrade_MPa")
        if len(good) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = {"Soft": "#d62728", "Medium": "#ff7f0e",
                      "Stiff": "#2ca02c", "Unknown": "#7f7f7f"}
            bar_colors = [colors[b] for b in good["subgrade_bin"]]
            ax.bar(good["section_id"], good["E_subgrade_MPa"], color=bar_colors)
            ax.axhline(SOFT_MAX_MPA,   ls="--", c="k", lw=0.8, label=f"Soft / Medium @ {SOFT_MAX_MPA:.0f}")
            ax.axhline(MEDIUM_MAX_MPA, ls="--", c="k", lw=0.8, label=f"Medium / Stiff @ {MEDIUM_MAX_MPA:.0f}")
            ax.set_ylabel("E_subgrade  [MPa]")
            ax.set_title("FWD-Backcalculated Subgrade Modulus per LTPP Section")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_PLOT, dpi=150)
            log.info(f"Wrote plot: {OUT_PLOT}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    log.info("=" * 70)
    log.info("DONE.")


if __name__ == "__main__":
    main()
