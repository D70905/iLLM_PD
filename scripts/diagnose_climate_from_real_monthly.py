"""
diagnose_climate_from_real_monthly.py
======================================
使用 真实逐月温度数据（CLM_VWS_TEMP_MONTH, 从 SDR39 Access DB 提取）
对 12 个 LTPP 路段做气候多样性诊断。

输入: sdr39/extracted/<section_id>/clm_temp_monthly.csv
输出: 逐段 |E*| 季节变化 + 跨段对比
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.dynamic_modulus import DynamicModulusMasterCurve

# ── 12 段列表 ──────────────────────────────────────────
SECTIONS = [
    ("16_1010", "flexible",   "Dry-Freeze",    14000.0),
    ("27_1085", "flexible",   "Dry-Freeze",    14000.0),
    ("27_2023", "semi_rigid", "Dry-Freeze",    14000.0),
    ("30_7076", "semi_rigid", "Dry-Freeze",    14000.0),
    ("04_1065", "semi_rigid", "Wet-NoFreeze",  14000.0),
    ("48_1076", "flexible",   "Wet-NoFreeze",  14000.0),
    ("06_2004", "semi_rigid", "Wet-NoFreeze",  14000.0),
    ("12_4097", "semi_rigid", "Wet-NoFreeze",  14000.0),
    ("48_1109", "semi_rigid", "Wet-NoFreeze",  14000.0),
    ("48_0001", "flexible",   "Wet-NoFreeze",  14000.0),
    ("04_1034", "flexible",   "Wet-NoFreeze",  14000.0),
    ("12_1060", "flexible",   "Wet-NoFreeze",  14000.0),
]

BASE = os.path.join(os.path.dirname(__file__), "..",
                    "experiments", "ltpp_data", "sdr39", "extracted")


def main():
    print("=" * 90)
    print("REAL MONTHLY TEMPERATURE DATA -> DYNAMIC MODULUS |E*| DIAGNOSIS")
    print("Data: CLM_VWS_TEMP_MONTH (SDR39 Access DB, extracted 2026-06-02)")
    print("Anchor: 14,000 MPa @ 20C (typical AC upper layer, JTG D50-2017)")
    print("=" * 90)

    results = []
    for sid, ptype, cz, E_ref in SECTIONS:
        fpath = os.path.join(BASE, sid, "clm_temp_monthly.csv")
        if not os.path.exists(fpath):
            print(f"  {sid}: NO DATA")
            continue

        df = pd.read_csv(fpath)
        temps = df["MEAN_MON_TEMP_AVG"].values

        mc = DynamicModulusMasterCurve(E_ref_MPa=E_ref, T_ref_C=20.0)

        # 冬 (Jan) / 夏 (Jul) 平均温度
        jan_t = df[df["MONTH"] == 1]["MEAN_MON_TEMP_AVG"].mean()
        jul_t = df[df["MONTH"] == 7]["MEAN_MON_TEMP_AVG"].mean()
        E_jan = mc.modulus_MPa(jan_t)
        E_jul = mc.modulus_MPa(jul_t)
        seasonal_ratio = E_jan / E_jul

        # 全年统计
        maat = df["MEAN_MON_TEMP_AVG"].mean()
        E_maat = mc.modulus_MPa(maat)
        t_min = df["MEAN_MON_TEMP_AVG"].min()
        t_max = df["MEAN_MON_TEMP_AVG"].max()
        full_range = mc.modulus_MPa(t_min) / mc.modulus_MPa(t_max)

        results.append({
            "sid": sid, "type": ptype, "climate": cz,
            "maat": maat, "E_maat": E_maat,
            "jan_t": jan_t, "jul_t": jul_t,
            "E_jan": E_jan, "E_jul": E_jul,
            "seasonal_ratio": seasonal_ratio,
            "full_range": full_range,
            "t_min": t_min, "t_max": t_max,
        })

    # ── 详细表 ──────────────────────────────────────────
    header = (
        f"{'Section':>9s}  {'Type':>8s}  {'Climate':>15s}  "
        f"{'MAAT':>6s}  {'|E*|(MAAT)':>10s}  "
        f"{'JanT':>6s}  {'JulT':>6s}  "
        f"{'|E*|_Jan':>10s}  {'|E*|_Jul':>10s}  "
        f"{'Win/Sum':>8s}  {'FullRng':>8s}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        print(
            f"{r['sid']:>9s}  {r['type']:>8s}  {r['climate']:>15s}  "
            f"{r['maat']:6.1f}  {r['E_maat']:10.0f}  "
            f"{r['jan_t']:6.1f}  {r['jul_t']:6.1f}  "
            f"{r['E_jan']:10.0f}  {r['E_jul']:10.0f}  "
            f"{r['seasonal_ratio']:8.2f}x  {r['full_range']:8.2f}x"
        )

    # ── 逐月模量曲线（代表性段） ──────────────────────
    print(f"\n{'=' * 90}")
    print("MONTHLY |E*| CURVES — 4 representative climate zones")
    print("=" * 90)

    reps = ["16_1010", "04_1065", "48_0001", "12_1060"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for sid in reps:
        fpath = os.path.join(BASE, sid, "clm_temp_monthly.csv")
        df = pd.read_csv(fpath)
        mc = DynamicModulusMasterCurve(E_ref_MPa=14000.0, T_ref_C=20.0)

        # 多年逐月平均
        monthly_avg = df.groupby("MONTH")["MEAN_MON_TEMP_AVG"].mean()
        monthly_E = [mc.modulus_MPa(monthly_avg[m]) for m in range(1, 13)]

        maat = df["MEAN_MON_TEMP_AVG"].mean()
        print(f"\n  {sid} (MAAT={maat:.1f}C):")
        print(f"  Month:  " + "  ".join(f"{m:>4s}" for m in months))
        print(f"  Temp(C):" + "  ".join(f"{monthly_avg[m]:4.0f}" for m in range(1, 13)))
        print(f"  |E*|(MPa):" + "  ".join(f"{E:4.0f}" for E in monthly_E))

    # ── 汇总 ──────────────────────────────────────────
    E_maats = [r["E_maat"] for r in results]
    ratios = [r["seasonal_ratio"] for r in results]
    fulls = [r["full_range"] for r in results]

    E_beijing = DynamicModulusMasterCurve(E_ref_MPa=14000.0, T_ref_C=20.0).modulus_MPa(23.0)

    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)
    print(f"  Cross-section |E*| range (MAAT level): {min(E_maats):.0f} ~ {max(E_maats):.0f} MPa = {max(E_maats)/min(E_maats):.2f}x")
    print(f"  Intra-section seasonal ratio (mean):   {np.mean(ratios):.2f}x (min={min(ratios):.2f}, max={max(ratios):.2f})")
    print(f"  Intra-section full T range (mean):     {np.mean(fulls):.2f}x (min={min(fulls):.2f}, max={max(fulls):.2f})")
    print()
    print(f"  Old Beijing-fixed modulus:             {E_beijing:.0f} MPa (23C, ONE value for ALL sections)")
    print(f"  Real |E*| across 12 sections (MAAT):   {min(E_maats):.0f} ~ {max(E_maats):.0f} MPa")
    print(f"  + seasonal swing per section:          up to {max(fulls):.1f}x")
    print()
    print(f"  >>> The Access DB CLM_VWS_TEMP_MONTH table had the real data all along.")
    print(f"  >>> 12 sections x avg 46 years = ~6,800 real monthly temperature records.")
    print(f"  >>> Can be cited as 'LTPP SDR39, CLM_VWS_TEMP_MONTH (weather station)'.")


if __name__ == "__main__":
    main()