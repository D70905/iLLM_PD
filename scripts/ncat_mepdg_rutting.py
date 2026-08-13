# -*- coding: utf-8 -*-
"""
ncat_mepdg_rutting.py — NCAT 的 ME-PDG 车辙"预测 vs 实测"验证 (MVP, 已对齐真实API)。

对 NCAT Cracking Group 7 段建成态结构: 用真实 E*(参考温度) 跑一次前向 FEA, 取
AC 中深竖向应变与路基应变, 调用你已验证的 NCHRP 1-37A 车辙函数 mepdg_rutting_mm,
得到总车辙及"车辙-ESAL"轨迹, 与实测末态车辙对比。回应 R2-1 / R3-9。

与论文中 ME-PDG 交叉校核保持一致的约定(来自 run_mepdg_cross_spec_check.py):
  · eps_HMA_mid = |p_AC_mid_mid| / E_ac · 1e6   (竖向应力/层加权AC模量, 单轴近似)
  · eps_z 直接取 FEA 的 epsilon_z_microstrain
  · 参考温度 T_PAVEMENT_F = 73°F (设置E*与车辙模型温度项用同一温度)
唯一差别(正是NCAT验证的意义): N 用【实测累计 ESAL】, 而非80M设计轴次 → 预测 vs 实测。

用法(本地, illm_pd 环境, 从项目根运行):
    cd /d D:\\iLLM_PD_new
    set PYTHONPATH=.
    python scripts\\ncat_mepdg_rutting.py experiments\\ncat_data\\ncat_cases.json ^
           experiments\\ncat_data\\Dynamic Modulus_Mastercurves.xlsx
可选第三参数: βr1 (HMA局部标定系数, 默认1.0国家标定; 例如填 Guo&Timm 的NCAT标定值)。
"""
import json, sys, os, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_estar

# ── 与论文 ME-PDG 校核一致的参考条件 ──
T_REF_C       = 22.8     # ≈ 73°F: 设置 AC 的 E*(T)
T_PAVEMENT_F  = 73.0     # 车辙模型温度项(与上同一温度); [可调] 阿拉巴马有效车辙温度更高
FREQ_HZ       = 10.0     # 卡车速度 ~10 Hz
AGING         = "RH"     # 建成态(reheated); CA 可做敏感性

def _ngrid(nfinal):
    pts = [1e6,2e6,4e6,6e6,8e6,10e6,12e6,14e6,16e6,18e6,20e6]
    g = [n for n in pts if n <= nfinal*1.0001]
    return g or [nfinal]

def load_fns():
    """真实依赖: 你的 FEA 与已验证的 1-37A 车辙函数。"""
    from fea import run_fea                                   # fea/runner.py
    from run_mepdg_cross_spec_check import mepdg_rutting_mm   # 已验证 NCHRP 1-37A
    return run_fea, mepdg_rutting_mm

def run_fea_at_temperature(run_fea, thickness_m, modulus_MPa, poisson, E_sub, nu_sub):
    """单次轴对称 FEA(5层+路基半空间), 返回 responses 字典。"""
    res = run_fea(thickness=list(thickness_m), modulus=list(modulus_MPa),
                  poisson=list(poisson), E_subgrade=float(E_sub), nu_subgrade=float(nu_sub),
                  load_pressure=0.7, load_radius=0.1065, num_cpus=4,
                  base_dir=os.getcwd(), verbose=False)
    return res.get("responses", {}) if isinstance(res, dict) else {}

def _weighted_E_ac(h3, e3):
    tot = sum(h3)
    return (sum(h*e for h,e in zip(h3,e3))/tot) if tot > 0 else 9000.0

def evaluate_section(case, xlsx, run_fea, mepdg_rutting_mm, beta_r1=1.0,
                     beta_s1_gran=1.0, beta_s1_subg=1.0,
                     T_pavement_F=73.0, T_ref_C=22.8):
    sid = case["section_id"]; sec = sid.replace("NCAT_CG_", "")
    ec  = case["envconfig"]
    thk = list(ec["init_thickness_m"]); nu = list(ec["init_poisson"]); modL = list(ec["init_modulus_MPa"])
    Esub = float(ec["E_subgrade"]); nusub = float(ec["nu_subgrade"])
    nfinal = float(case["meta"]["measured_final"]["ESAL_at_final"])
    meas   = float(case["meta"]["measured_final"]["rut_mm"])
    if len(thk) != 5:
        raise ValueError(f"{sid}: 需要5层结构, 实际 {len(thk)} 层。")
    if thk[3] < 0.15:
        raise ValueError(f"{sid}: 基层 {thk[3]*100:.1f}cm < 15cm — 请用补齐后的5层结构"
                         f"(集料基层=0.1524m 作 layer[3], 上层路基作 layer[4])重生成 ncat_cases.json。")
    # 真实 E*(T_ref): 面层=段自身主曲线; 中面层+沥青基层=共用 CG Base
    E_surf = mc_estar.estar_MPa(xlsx, sec, AGING, "surface", T_ref_C, FREQ_HZ)
    E_cg   = mc_estar.estar_MPa(xlsx, sec, AGING, "base",    T_ref_C, FREQ_HZ)
    modulus = [E_surf, E_cg, E_cg, modL[3], modL[4]]
    resp = run_fea_at_temperature(run_fea, thk, modulus, nu, Esub, nusub)
    eps_z    = resp.get("epsilon_z_microstrain")
    p_ac_mid = resp.get("p_AC_mid_mid_MPa")
    if eps_z is None or p_ac_mid is None:
        raise ValueError(f"{sid}: FEA 缺键 epsilon_z/p_AC_mid_mid (得到: {sorted(resp.keys())})")
    eps_z = float(eps_z)
    E_ac  = _weighted_E_ac([thk[0],thk[1],thk[2]], [E_surf,E_cg,E_cg])   # MPa
    # ── 修复一: 沥青层竖向应变 ──
    #   旧: σ_v/E 单轴近似(忽略侧向围压, 系统性偏大);
    #   新: 直接用 FEA 输出的中层 AC 竖向弹性应变(考虑三向受压)。两者都打印对照。
    eps_sigmaE = abs(float(p_ac_mid)) / E_ac * 1.0e6                     # 旧近似(microstrain)
    eps_fea    = resp.get("eps_AC_mid_mid_microstrain")                  # 新: FEA 直接竖向应变
    if eps_fea is not None:
        eps_HMA_mid = float(eps_fea); strain_src = "FEA"
    else:
        eps_HMA_mid = eps_sigmaE;     strain_src = "σ/E(回退)"
    h_ac_mm = sum(thk[:3]) * 1000.0
    h_base_mm = thk[3] * 1000.0

    def rut(N):
        # βr1 / βs1 现在作为参数传入 mepdg_rutting_mm(已解除内部硬编码), 不再外部二次相乘
        rh, rb, rs, _ = mepdg_rutting_mm(eps_HMA_mid, eps_z, h_ac_mm, h_base_mm, N, T_pavement_F,
                                         beta_r1=beta_r1, beta_s1_gran=beta_s1_gran,
                                         beta_s1_subg=beta_s1_subg)
        return rh, rb, rs, rh + rb + rs

    traj = []
    for N in _ngrid(nfinal):
        rh, rb, rs, tot = rut(N)
        traj.append({"ESAL": int(N), "rd_hma": round(rh,3), "rd_base": round(rb,3),
                     "rd_sg": round(rs,3), "rd_total": round(tot,3)})
    rh, rb, rs, tot = rut(nfinal)
    # 由实测反解的 βr1(派生量; 仅作合理性锚, 不能自证为"验证")
    beta_implied = ((meas - rb - rs) / rh * beta_r1) if rh > 0 else float("nan")
    return {"section": sid, "strain_src": strain_src,
            "eps_HMA_mid": round(eps_HMA_mid,1), "eps_sigmaE": round(eps_sigmaE,1),
            "eps_z": round(eps_z,1), "E_ac_MPa": round(E_ac,0), "N": nfinal, "meas_rut_mm": meas,
            "rd_hma": round(rh,2), "rd_base": round(rb,2), "rd_sg": round(rs,2),
            "pred_total_mm": round(tot,2), "err_mm": round(tot - meas,2),
            "beta_r1_implied": round(beta_implied,2),
            "complete_to_20M": case["meta"]["measured_final"]["performance_complete_to_20M"],
            "traj": traj}

def main():
    cases_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/ncat_data/ncat_cases.json"
    xlsx       = sys.argv[2] if len(sys.argv) > 2 else "experiments/ncat_data/Dynamic Modulus_Mastercurves.xlsx"
    beta_r1    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    T_F        = float(sys.argv[4]) if len(sys.argv) > 4 else T_PAVEMENT_F
    beta_s1_g  = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    beta_s1_s  = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    T_ref      = (T_F - 32.0) / 1.8 if len(sys.argv) > 4 else T_REF_C
    CASES = json.load(open(cases_path, encoding="utf-8"))
    run_fea, mepdg_rutting_mm = load_fns()

    print(f"\n参考: T={T_F}°F, βr1={beta_r1}, βs1(粒料/路基)={beta_s1_g}/{beta_s1_s}, "
          f"AC的E*取@%.1f°C/%.0fHz" % (T_ref, FREQ_HZ))
    print("(βr1/βs1 默认=1.0 国家标定; 传入其它值即应用局部标定。沥青应变优先用 FEA 直接输出, 缺则回退 σ/E)\n")
    hdr = (f"{'段':<13}{'应变源':>7}{'epsHMA':>8}{'(σ/E)':>8}{'epsZ':>7}"
           f"{'预测rut':>8}{'实测rut':>8}{'误差':>7}{'隐含βr1':>8}")
    print(hdr); print("-"*len(hdr.encode('gbk',errors='ignore')))
    rows = []
    for c in CASES["cases"]:
        try:
            r = evaluate_section(c, xlsx, run_fea, mepdg_rutting_mm, beta_r1=beta_r1,
                                 beta_s1_gran=beta_s1_g, beta_s1_subg=beta_s1_s,
                                 T_pavement_F=T_F, T_ref_C=T_ref)
            note = "" if r["complete_to_20M"] else " (~16M)"
            print(f"{r['section']:<13}{r['strain_src']:>7}{r['eps_HMA_mid']:>8.0f}{r['eps_sigmaE']:>8.0f}"
                  f"{r['eps_z']:>7.0f}{r['pred_total_mm']:>8.2f}{r['meas_rut_mm']:>8.2f}"
                  f"{r['err_mm']:>7.2f}{r['beta_r1_implied']:>8.2f}{note}")
            rows.append(r)
            with open(f"ncat_ruttraj_{r['section']}.csv","w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=["ESAL","rd_hma","rd_base","rd_sg","rd_total"]); w.writeheader(); w.writerows(r["traj"])
        except Exception as e:
            print(f"{c['section_id']:<13} ERR: {e}")
    if rows:
        keys=["section","strain_src","E_ac_MPa","eps_HMA_mid","eps_sigmaE","eps_z","N",
              "rd_hma","rd_base","rd_sg","pred_total_mm","meas_rut_mm","err_mm",
              "beta_r1_implied","complete_to_20M"]
        with open("ncat_mepdg_rut_summary.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        print("\n已写出: ncat_mepdg_rut_summary.csv + 每段 ncat_ruttraj_*.csv")
    print("\n说明:")
    print(" · 'epsHMA' 现在优先用 FEA 直接竖向弹性应变;'(σ/E)' 列是旧的单轴近似, 两者对比即修复一的效果。")
    print(" · βr1/βs1 为参数(默认1.0); 若要做本批段的局部标定, 用你自己反算的系数传入, 不要套用其它周期的值。")
    print(" · 提醒: 即使应变改对, 1-37A 在 N=全程、单温度下的一次性评估仍可能偏高;")
    print("   真正定量对齐需 (a) 用本批实测反算 βr1/βs1 + 训练/验证拆分, 或 (b) 逐月温度增量累积。")

if __name__ == "__main__":
    main()
