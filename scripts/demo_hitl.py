"""
scripts/demo_hitl.py — HITL 完整流程演示（非交互模式，自检验证）
===========================================================
演示: 设计完成 → explain_design_full → HITL 审核 → 裁决
覆盖 3 种场景: 自动通过 / 数值核验失败触发 / 设计不合规触发
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.hitl import _check_triggers, _fmt_margins, _fmt_structure


# ── 合成数据 ──
class FakeEval:
    def __init__(self, margins, feasible=True, critical="B3_ac_permanent_deformation"):
        self.margins = margins
        self.feasible = feasible
        self.critical_indicator = critical
        self.responses = {
            "epsilon_a_microstrain": 46.4,
            "epsilon_z_microstrain": 96.3,
            "sigma_t_MPa": 0.01,
        }

class FakeResult:
    def __init__(self, trustworthy, report=""):
        self.trustworthy = trustworthy
        self.report = report or "（简化自检报告）"


def demo():
    design = {
        "thickness_cm": [4.0, 6.0, 8.0, 36.0, 18.0],
        "modulus_MPa": [14000, 11000, 9000, 1500, 400],
    }

    print("=" * 64)
    print("  HITL 完整流程演示（非交互）")
    print("=" * 64)

    # ─── 场景 1: 安全通过 ───
    print("\n" + "─" * 64)
    print("场景 1: 全合规 + 数值核验通过")
    print("─" * 64)

    e1 = FakeEval({"B1_asphalt_fatigue": 2.51, "B3_ac_permanent_deformation": 1.84,
                   "B4_subgrade_strain": 2.11})
    r1 = FakeResult(trustworthy=True,
                    report="[数值核验] 8✓/0✗/0? | [附录核验] Appendix B.3 存在")
    triggered, reasons = _check_triggers(r1, e1)

    print(f"  触发: {triggered}")
    print(f"  Margins: { {k:round(v,2) for k,v in e1.margins.items()} }")
    if not triggered:
        print("  → 自动通过 ✓ (不需工程师介入)")
    else:
        print(f"  → 需要裁决: {'; '.join(reasons)}")

    # ─── 场景 2: LLM 编造数字 ───
    print("\n" + "─" * 64)
    print("场景 2: LLM 编造了数字（trustworthy=False）")
    print("─" * 64)

    e2 = FakeEval({"B1_asphalt_fatigue": 2.51, "B3_ac_permanent_deformation": 1.84,
                   "B4_subgrade_strain": 2.11})
    r2 = FakeResult(trustworthy=False,
                    report="[数值核验] 4✓/2✗/2? | B1 margin 解释称2.50 实际32.81 ✗")
    triggered, reasons = _check_triggers(r2, e2)

    print(f"  触发: {triggered}")
    if triggered:
        for reason in reasons:
            print(f"  → {reason}")
        print("  → 呈现给工程师裁决 (A/R/M)")

    # ─── 场景 3: 设计不合规 ───
    print("\n" + "─" * 64)
    print("场景 3: B3 margin=0.85 < 1.0（设计不合规）")
    print("─" * 64)

    e3 = FakeEval({"B1_asphalt_fatigue": 1.20, "B3_ac_permanent_deformation": 0.85,
                   "B4_subgrade_strain": 1.50})
    r3 = FakeResult(trustworthy=True,
                    report="[数值核验] 8✓/0✗/0? | [附录核验] Appendix B.3 存在")
    triggered, reasons = _check_triggers(r3, e3)

    print(f"  触发: {triggered}")
    if triggered:
        for reason in reasons:
            print(f"  → {reason}")
        print("  → 呈现给工程师裁决 (A/R/M)")

    # ─── 场景 4: 双重触发 ───
    print("\n" + "─" * 64)
    print("场景 4: 编造 + 不合规 同时触发")
    print("─" * 64)

    e4 = FakeEval({"B1_asphalt_fatigue": 0.92, "B3_ac_permanent_deformation": 1.10,
                   "B4_subgrade_strain": 2.11})
    r4 = FakeResult(trustworthy=False,
                    report="[数值核验] 3✓/3✗/2? | B1=1.50(MOCK) 实际0.92 ✗")
    triggered, reasons = _check_triggers(r4, e4)

    print(f"  触发: {triggered}, 理由数: {len(reasons)}")
    for reason in reasons:
        print(f"  → {reason}")

    # ── 汇总 ──
    print("\n" + "=" * 64)
    print("HITL 流程验证完成")
    print("=" * 64)
    print("""
触发条件（全确定性）:
  1. trustworthy == False  → LLM 编造了数字，需人工审核
  2. margin < 1.0          → 设计不合规，需工程师裁决

不触发条件:
  - 全部 margin >= 1.0 且 trustworthy == True  → 自动通过

交互流程（当触发时）:
  1. 呈现设计结构 + margins + 双核验报告
  2. 工程师选择: [A]ccept / [R]eject / [M]odify
  3. 记录裁决到审计链

集成到推理循环:
  for each episode:
      env.reset() / env.step() ...
      evaluation = env._last_evaluation
      design = env._design_to_dict()
      result = explain_design_full(evaluation, design)
      decision = review_design(result, evaluation, design)
      if decision.action == 'reject':
          continue  # 重新优化
      elif decision.action == 'modify':
          design = decision.modified_design
      # else: accept → 输出设计
""")


if __name__ == "__main__":
    demo()