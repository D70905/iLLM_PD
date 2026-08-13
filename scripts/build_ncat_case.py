# -*- coding: utf-8 -*-
"""
build_ncat_case.py
将 NCAT Test Track (2015-2021 Cracking Group) 的实测数据组装成 iLLM-PD 可用的案例。

输入(两个 Excel，放在 DATA_DIR 下，或用命令行第1个参数指定目录):
  - Dynamic_Modulus_Mastercurves.xlsx   (各段沥青层 |E*| 主曲线)
  - Field_Data.xlsx                      (各段实测 车辙/开裂/IRI vs ESAL)

输出(写到 OUT_DIR):
  - ncat_cases.json        每段一个 EnvConfig 兼容案例(真实结构+实测E*)+元数据
  - ncat_sections_row.csv  LTPP 段文件格式的行(供"模式A:让agent设计"用)
  - 终端打印一张汇总表

字段对应 env.py 的 EnvConfig:
  init_thickness_m / init_modulus_MPa / init_poisson / E_subgrade / nu_subgrade
"""
import sys, os, json
import pandas as pd
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "."
MC_XLSX  = os.path.join(DATA_DIR, "Dynamic Modulus_Mastercurves.xlsx")
FD_XLSX  = os.path.join(DATA_DIR, "Field Data.xlsx")

KSI_TO_MPA = 6.894757
IN_TO_M    = 0.0254

# ── 设计参考条件:从主曲线取哪一点作为 FEA 单一弹性模量 ────────────────
#    主曲线网格点为 4/20/40°C × 0.1/1/10 Hz。默认 20°C/10Hz(常用等效设计条件)。
#    注意:模板里 AC 模量 14000/11000/9000 MPa 偏高,接近 NCAT 的 4°C/10Hz;
#    若 harness 的参考温度更低,把 REF_TEMP_C 改成 4。脚本同时导出多条件值备用。
REF_TEMP_C = 20.0
REF_FREQ_HZ = 10.0
EXPORT_CONDITIONS = [(4.0, 10.0), (20.0, 10.0), (40.0, 10.0)]
AGING_PRIMARY = "RH"   # 设计模量默认用 RH(reheated plant mix,接近建成态);CA 作敏感性

# ── 地基模量:务必先核对 NCAT rep18-04 第2章后再用!────────────────────
#    需求文档转写为 "Base 10,000 ksi / Subgrade 30,000 ksi" —— 物理上不可能
#    (路基不可能比基层硬;30,000 ksi≈混凝土)。下面是合理的临时值(很可能单位
#    应为 psi 且基层/路基被写反),仅占位,核实后修改这两个数。
VERIFY_FOUNDATION = True
# rep18-04/rep21-03 已定性确认: 基层=级配碎石(crushed granite), 路基=A-4 土。
# 精确 Mr 不在这两份 findings 报告里(在 NCAT 13-02 结构表征报告/FWD 反算库)。
# 下列为合理取值, 建议对路基做敏感性扫描(如 80/114/150 MPa)。
BASE_E_MPA     = 207.0   # 级配碎石基层 ≈30,000 psi
NU_BASE        = 0.40
SUBGRADE_E_MPA = 114.0   # A-4 路基 MEPDG 默认 ≈16,500 psi  [可扫描]
NU_SUBGRADE    = 0.45

# ── 公共结构(7段相同):1.5"面 + 2.25"中 + 2.25"沥青基 = 6"AC;6"集料基层 ──
T_SURF = 1.50 * IN_TO_M   # 0.0381 m
T_INTM = 2.25 * IN_TO_M   # 0.0572 m
T_ACBS = 2.25 * IN_TO_M   # 0.0572 m
T_AGGB = 6.00 * IN_TO_M   # 0.1524 m (集料基层, 整层作 layer[3], 过 Guard≥15cm)
T_USG  = 0.10             # 0.10 m 上层路基有限层(layer[4], 取路基模量), 其下为路基半空间

# 气候/位置/交通(来自需求文档 NCAT 回复)
CLIMATE_ZONE = "warm"          # Opelika, AL 年均温≈17.6°C → JTG warm(15-20°C)
MAAT_C       = 17.6
APPLIED_ESAL_18KIP = 20_000_000  # 受控卡车队累计(10M:2015-2017 + 10M:2018-2021)
LOCATION = "NCAT Test Track, 1600 Lee Road 151, Opelika, AL 36804"
REFERENCES = ["NCAT rep18-04 (Ch.2)", "NCAT rep21-03 (Ch.2)"]

# 段 → (主曲线sheet前缀, 现场数据列名)
SECTIONS = [
    ("N1", "N1-1", "N01"), ("N2", "N2-1", "N02"), ("N5", "N5-1", "N05"),
    ("N8", "N8-1", "N08"), ("S5", "S5-1", "S05"), ("S6", "S6-1", "S06"),
    ("S13", "S13-1", "S13"),
]

def read_estar_grid(path, sheet):
    """返回 {(T_C, f_Hz): modulus_ksi},取'Average Measured'平均表;失败回退拟合表。"""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    grid = {}
    # 主:平均表(col0=Temperature, col2 含 'Avg')
    for r in range(raw.shape[0]):
        c0 = str(raw.iat[r, 0]) if pd.notna(raw.iat[r, 0]) else ""
        c2 = str(raw.iat[r, 2]) if (raw.shape[1] > 2 and pd.notna(raw.iat[r, 2])) else ""
        if c0.startswith("Temperature") and "Avg" in c2:
            rr = r + 1
            while rr < raw.shape[0]:
                t, f, m = raw.iat[rr, 0], raw.iat[rr, 1], raw.iat[rr, 2]
                try:
                    grid[(round(float(t), 3), round(float(f), 3))] = float(m)
                except (ValueError, TypeError):
                    break
                rr += 1
            if grid:
                return grid
    # 回退:拟合表(col12=Temperature, col13=Hz, col14=Ksi)
    for r in range(raw.shape[0]):
        c12 = str(raw.iat[r, 12]) if (raw.shape[1] > 14 and pd.notna(raw.iat[r, 12])) else ""
        if c12.strip() == "Temperature":
            rr = r + 1
            while rr < raw.shape[0]:
                t, f, m = raw.iat[rr, 12], raw.iat[rr, 13], raw.iat[rr, 14]
                try:
                    tt = float(t)
                except (ValueError, TypeError):
                    if str(t).strip() in ("C", "Condition"):  # 跳过单位行
                        rr += 1; continue
                    break
                try:
                    grid[(round(tt, 3), round(float(f), 3))] = float(m)
                except (ValueError, TypeError):
                    pass
                rr += 1
            if grid:
                return grid
    return grid

def estar_MPa(grid, T, f):
    key = (round(float(T), 3), round(float(f), 3))
    if key not in grid:
        return None
    return round(grid[key] * KSI_TO_MPA, 1)

def field_finals(path):
    """每段取最后一个非空测点 → {col: (esal, value)}。"""
    out = {}
    for sh in ["Rutting", "Cracking", "IRI"]:
        raw = pd.read_excel(path, sheet_name=sh, header=None)
        hr = next(r for r in range(raw.shape[0])
                  if any("ESAL" in str(x) for x in raw.iloc[r].tolist()))
        hdr = raw.iloc[hr].tolist()
        col = {str(v): c for c, v in enumerate(hdr)}
        esal_c = col["ESAL"]
        body = raw.iloc[hr + 1:]
        esal = pd.to_numeric(body.iloc[:, esal_c], errors="coerce")
        res = {}
        for _, _, fcol in SECTIONS:
            if fcol not in col:
                res[fcol] = None; continue
            vals = pd.to_numeric(body.iloc[:, col[fcol]], errors="coerce")
            m = vals.notna() & esal.notna()
            if not m.any():
                res[fcol] = None; continue
            i = esal[m].idxmax()
            res[fcol] = (float(esal[i]), float(vals[i]))
        out[sh] = res
    return out

def main():
    fin = field_finals(FD_XLSX)
    cases = []
    rows = []
    print("\n=== NCAT Cracking Group: 抽取汇总 (设计条件 %g°C/%gHz, aging=%s) ===\n"
          % (REF_TEMP_C, REF_FREQ_HZ, AGING_PRIMARY))
    hdr = f"{'段':<5}{'面层E*':>9}{'基层AC E*':>10}{'末车辙mm':>9}{'末开裂%':>8}{'末IRI':>8}{'@ESAL':>13}"
    print(hdr); print("-" * len(hdr))
    for key, mc_prefix, fcol in SECTIONS:
        sheet_surf = f"{mc_prefix} {AGING_PRIMARY}"
        grid_surf = read_estar_grid(MC_XLSX, sheet_surf)
        grid_base = read_estar_grid(MC_XLSX, f"CG Base {AGING_PRIMARY}")
        E_surf = estar_MPa(grid_surf, REF_TEMP_C, REF_FREQ_HZ)
        E_base_ac = estar_MPa(grid_base, REF_TEMP_C, REF_FREQ_HZ)
        # 多条件 E*(备用,便于对齐 harness 参考温度)
        Egrid_surf = {f"{int(t)}C_{int(f)}Hz": estar_MPa(grid_surf, t, f) for t, f in EXPORT_CONDITIONS}
        Egrid_base = {f"{int(t)}C_{int(f)}Hz": estar_MPa(grid_base, t, f) for t, f in EXPORT_CONDITIONS}

        rut = fin["Rutting"].get(fcol); crk = fin["Cracking"].get(fcol); iri = fin["IRI"].get(fcol)
        rut_mm = round(rut[1] * 25.4, 2) if rut else None
        esal_at = rut[0] if rut else (iri[0] if iri else None)
        perf_complete = bool(rut and esal_at and esal_at >= 19.5e6)

        # 5 结构层 + 路基半空间:[面, 中, 沥青基, 集料基层(整层0.1524), 上层路基(0.10)]
        # 集料基层整层作 layer[3](≥15cm 过 NumericalGuard; 车辙模型 h_base 也取它);
        # layer[4] 为上层路基有限层(取路基模量), 其下为路基半空间。
        init_thickness = [round(T_SURF,4), round(T_INTM,4), round(T_ACBS,4),
                          round(T_AGGB,4), round(T_USG,4)]
        init_modulus   = [E_surf, E_base_ac, E_base_ac, BASE_E_MPA, SUBGRADE_E_MPA]
        init_poisson   = [0.25, 0.30, 0.30, NU_BASE, NU_SUBGRADE]

        case = {
            "section_id": f"NCAT_CG_{key}",
            # —— EnvConfig 字段(直接喂给 env.py) ——
            "envconfig": {
                "init_thickness_m": init_thickness,
                "init_modulus_MPa": init_modulus,   # 注:若 env 启用气候模量改写,需绕过才能真正用上实测E*
                "init_poisson":     init_poisson,
                "E_subgrade":       SUBGRADE_E_MPA,
                "nu_subgrade":      NU_SUBGRADE,
            },
            # —— 上下文/元数据(非 EnvConfig) ——
            "meta": {
                "gps_family": "GPS-1 (flexible, AC on unbound aggregate base)",
                "climate_zone": CLIMATE_ZONE, "MAAT_C": MAAT_C,
                "applied_ESAL_18kip": APPLIED_ESAL_18KIP,
                "aging_primary": AGING_PRIMARY,
                "estar_design_condition": f"{int(REF_TEMP_C)}C/{int(REF_FREQ_HZ)}Hz",
                "estar_surface_MPa_byCond": Egrid_surf,
                "estar_ACbase_MPa_byCond":  Egrid_base,
                "layer_mapping_assumption":
                    "已据 rep18-04/rep21-03 确认: 面层(1.5in)=分段 'Nx-1' 主曲线; 中面层+沥青基层(各2.25in)=共用 HiMA 19mm 混合料='CG Base' 主曲线; 集料基层(6in)拆成2等分子层(弹性等价)。",
                "foundation_VERIFY": (
                    "基层=级配碎石/路基=A-4(rep18-04/21-03 已定性确认); 精确Mr见NCAT 13-02; 建议路基Mr敏感性扫描"
                    if VERIFY_FOUNDATION else "verified"),
                "measured_final": {
                    "rut_mm": rut_mm,
                    "topdown_cracking_pct_lane": round(crk[1]*100,2) if crk else None,
                    "IRI_in_per_mile": round(iri[1],2) if iri else None,
                    "ESAL_at_final": esal_at,
                    "performance_complete_to_20M": perf_complete,
                },
                "location": LOCATION, "references": REFERENCES,
            },
        }
        cases.append(case)
        rows.append({
            "section_id": f"NCAT_CG_{key}", "state_code": "01", "state_name": "Alabama",
            "climate_zone": CLIMATE_ZONE, "E_subgrade_MPa": SUBGRADE_E_MPA,
            "subgrade_bin": "NCAT_CG", "gps_family": "GPS-1",
        })
        print(f"{key:<5}{(str(E_surf)):>9}{(str(E_base_ac)):>10}"
              f"{(str(rut_mm)):>9}{(str(round(crk[1],3)) if crk else 'NA'):>8}"
              f"{(str(round(iri[1],1)) if iri else 'NA'):>8}{(f'{esal_at:,.0f}' if esal_at else 'NA'):>13}")

    payload = {
        "dataset": "NCAT Test Track 2015-2021 Cracking Group (7 sections, ONE common structure)",
        "structure_note": "1.5in surface + 2.25in intermediate + 2.25in AC base (=6in AC) over 6in aggregate base over subgrade",
        "design_condition": f"{int(REF_TEMP_C)}C/{int(REF_FREQ_HZ)}Hz, aging={AGING_PRIMARY}",
        "RECONCILE_BEFORE_USE": [
            "1) 地基模量:基层=级配碎石/路基=A-4 已确认;精确Mr查NCAT 13-02或做敏感性扫描(80/114/150 MPa)",
            "2) 路面温度:NCAT只有气温,B3车辙需用 air→pavement 偏移估温并披露",
            "3) 轴载:实测为 20M×18kip ESAL;JTG用BZZ-100标准轴,需换算累计轴次",
            "4) 开裂:实测全为top-down;B1为bottom-up疲劳,不可用此开裂验证B1",
        ],
        "cases": cases,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ncat_cases.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "ncat_sections_row.csv"), index=False, encoding="utf-8-sig")
    print("\n已写出: ncat_cases.json , ncat_sections_row.csv")
    if VERIFY_FOUNDATION:
        print("⚠  地基模量为临时值(Base=%.0f / Subgrade=%.0f MPa),核对 rep18-04 Ch.2 后再跑 FEA。"
              % (BASE_E_MPA, SUBGRADE_E_MPA))

if __name__ == "__main__":
    main()
