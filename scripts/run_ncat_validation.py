# -*- coding: utf-8 -*-
"""
run_ncat_validation.py — 模式B: 对NCAT公共结构做前向FEA评估, 对比预测 vs 实测。
读取 build_ncat_case.py 产出的 ncat_cases.json。不重设计, 只评估建成态结构。

Usage:
    cd /d <PROJECT_ROOT>
    python scripts/run_ncat_validation.py experiments/ncat_data/ncat_cases.json
"""
import json, sys, os
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig

CASES_PATH = sys.argv[1] if len(sys.argv) > 1 else "experiments/ncat_data/ncat_cases.json"
CASES = json.load(open(CASES_PATH, encoding="utf-8"))


def evaluate_asbuilt(case):
    """灌入NCAT真实结构和实测E*, 零动作评估建成态B1-B4."""
    ec = case["envconfig"]
    sid = case["section_id"]

    cfg = SurrogateEnvConfig(
        protocol_name="JTG_D50_2017",
        pavement_type="flexible",
        init_thickness_m=list(ec["init_thickness_m"]),
        init_modulus_MPa=list(ec["init_modulus_MPa"]),
        init_poisson=list(ec["init_poisson"]),
        E_subgrade=float(ec["E_subgrade"]),
        nu_subgrade=float(ec["nu_subgrade"]),
        city="",
        climate_zone="warm",          # Opelika AL, MAAT ~17.6°C
        road_class="expressway",
        traffic_level="heavy",
        design_life_years=15,
        max_episode_steps=1,          # 只评估建成态, 不迭代
        max_episodes=1,
        llm_enabled=False,            # 验证不需要LLM
        fea_keep_runs=False,
        enable_lcc_eval=False,        # NCAT验证暂不看LCC
        design_life_years_lcc=20.0,
        use_surrogate=False,          # 强制全量FEA, 不走代理模型
        surrogate_model_path="output/surrogate_model/surrogate_v3.pt",
        surrogate_b3_threshold=1.0,
    )

    env = PavementEnvWithSurrogate(cfg)
    obs, info = env.reset(seed=0)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float64)
    obs, rew, term, trunc, info = env.step(zero_action)

    # 从 info 提取 margin
    margins = info.get("margins", {})
    dsr = info.get("dsr", 0.0)
    compliant = info.get("compliant", False)

    # 从 evaluation (如果有)
    ev = info.get("evaluation", {})
    if isinstance(ev, dict):
        margins = ev.get("margins", margins)

    return {
        "section_id": sid,
        "B1": float(margins.get("B1_asphalt_fatigue", 0)),
        "B2": float(margins.get("B2_semi_rigid_fatigue", float("inf"))),
        "B3": float(margins.get("B3_ac_permanent_deformation", 0)),
        "B4": float(margins.get("B4_subgrade_strain", 0)),
        "DSR": float(dsr),
        "compliant": bool(compliant),
        "design_h_cm": info.get("design_h_cm", []),
        "design_E_MPa": info.get("design_E_MPa", []),
    }


# ── 主流程 ──
print(f"{'段':<13} {'B1':>8} {'B3':>8} {'B4':>8} {'DSR':>6} {'合规':>5} {'实测rut':>9}")
print("-" * 60)

for c in CASES["cases"]:
    r = evaluate_asbuilt(c)
    mf = c["meta"]["measured_final"]
    print(f"{r['section_id']:<13} {r['B1']:>8.2f} {r['B3']:>8.2f} {r['B4']:>8.2f} "
          f"{r['DSR']:>6.3f} {'Y' if r['compliant'] else 'N':>5} {mf['rut_mm']:>9.2f}")

print()
print("注意: 地基模量为临时值(Base=207/Subgrade=103 MPa), 核对 rep18-04 后更新 ncat_cases.json 重跑。")
print("预测 rut 需从 FEA 响应 + JTG B3 公式反算, 此处仅展示 margin。")
