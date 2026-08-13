"""
rl/hitl.py — Human-in-the-Loop 设计审核
========================================
在推理/部署阶段，当设计完成时触发双核验。若出现：
  (a) 数值核验失败（LLM 编造了数字），或
  (b) 任一 margin < 1.0（设计不合规）
则暂停、呈现设计状态和核验报告，等待工程师裁决。

触发条件是确定性的（不依赖 LLM 评分或经验阈值）。

用法:
    from rl.hitl import review_design

    decision = review_design(result, design, evaluation)
    # decision: 'accept' | 'reject' | 'modify'
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HITLDecision:
    action: str               # "accept" | "reject" | "modify"
    reason: str               # 工程师输入的简短理由
    modified_design: Optional[Dict] = None  # 若 modify，工程师给的新参数


def _fmt_margins(margins: Dict) -> str:
    lines = []
    labels = {
        "B1_asphalt_fatigue": "B1 沥青层疲劳",
        "B2_semi_rigid_fatigue": "B2 半刚性基层疲劳",
        "B3_ac_permanent_deformation": "B3 沥青永久变形(车辙)",
        "B4_subgrade_strain": "B4 路基顶面压应变",
    }
    for k, v in margins.items():
        label = labels.get(k, k)
        flag = " ✓" if v >= 1.0 else " ✗ 不合规!"
        lines.append(f"  {label}: {v:.2f}{flag}")
    return "\n".join(lines)


def _fmt_structure(design: Dict) -> str:
    h = design.get("thickness_cm", design.get("thickness", []))
    e = design.get("modulus_MPa", design.get("modulus", []))
    layers = ["上面层 AC", "中面层 AC", "下面层 AC", "基层", "底基层"]
    lines = []
    for i in range(min(len(h), len(layers))):
        hi = h[i] * 100 if isinstance(h[i], float) and h[i] < 1 else h[i]
        ei = e[i] if i < len(e) else "?"
        lines.append(f"  {layers[i]}: h={hi:.1f} cm, E={ei:.0f} MPa")
    return "\n".join(lines)


def _check_triggers(result, evaluation) -> tuple:
    """检查是否触发 HITL。返回 (triggered: bool, reasons: list[str])。"""
    reasons = []

    if not result.trustworthy:
        reasons.append("数值核验失败：LLM 解释中有数字与真实计算结果不符")

    margins = getattr(evaluation, "margins", {}) or {}
    for k, v in margins.items():
        if v is not None and v < 1.0:
            reasons.append(f"{k} = {v:.2f} < 1.0，设计不合规")

    return len(reasons) > 0, reasons


def _present(report: str, design: Dict, evaluation, reasons: list):
    """在控制台呈现设计审核界面。"""
    print()
    print("=" * 64)
    print("  ⚠️  设计审核 — 需要工程师裁决")
    print("=" * 64)
    print(f"触发原因: {'; '.join(reasons)}")
    print()

    # 结构
    print("【设计结构】")
    print(_fmt_structure(design))
    print()

    # Margins
    margins = getattr(evaluation, "margins", {}) or {}
    print("【合规 Margins (≥1.0 = 通过)】")
    print(_fmt_margins(margins))
    print()

    # 核验报告
    print(report)
    print()

    print("【裁决选项】")
    print("  [A]ccept   — 接受当前设计，继续")
    print("  [R]eject   — 拒绝当前设计，返回重优化")
    print("  [M]odify   — 手动修改设计参数后继续")

    return input("请输入选择 (A/R/M): ").strip().upper()


def review_design(result, evaluation, design: Dict) -> HITLDecision:
    """
    对设计结果进行人工审核。

    Args:
        result: explain_unified.FullExplainResult
        evaluation: specs.protocol.DesignEvaluation
        design: {'thickness_cm': [...], 'modulus_MPa': [...]}

    Returns:
        HITLDecision
    """
    triggered, reasons = _check_triggers(result, evaluation)

    if not triggered:
        return HITLDecision(action="accept", reason="自动通过：数值核验通过且所有 margin ≥ 1.0")

    # 需要人工裁决
    choice = _present(result.report, design, evaluation, reasons)

    if choice == "A":
        return HITLDecision(action="accept", reason="工程师审核通过")
    elif choice == "R":
        reason = input("拒绝理由（可选）: ").strip() or "工程师判定不合规"
        return HITLDecision(action="reject", reason=reason)
    elif choice == "M":
        reason = input("修改说明: ").strip()
        print("输入修改后的设计参数（JSON 格式，保持当前值可以不填）:")
        print("  例: {\"thickness_cm\": [4.0, 6.0, 8.0, 36.0, 18.0], \"modulus_MPa\": [14000, 11000, 9000, 1500, 400]}")
        try:
            new = json.loads(input("> "))
            return HITLDecision(action="modify", reason=reason, modified_design=new)
        except Exception:
            print("JSON 解析失败，按 accept 处理")
            return HITLDecision(action="accept", reason=f"工程师修改（解析失败）: {reason}")
    else:
        print(f"无效输入 '{choice}'，默认 accept")
        return HITLDecision(action="accept", reason="工程师未明确选择，默认通过")


# ── 自检：用合成数据验证触发逻辑 ──
def _selftest():
    """离线验证触发逻辑（不需真实 LLM/FEA）。"""
    from dataclasses import dataclass as dc

    @dc
    class FakeResult:
        trustworthy: bool

    @dc
    class FakeEval:
        margins: Dict

    print("=" * 60)
    print("HITL 触发逻辑自检")

    # 案例 1: trustworthy=True + 全部合规 → 不应触发
    r1 = FakeResult(trustworthy=True)
    e1 = FakeEval(margins={"B1": 2.0, "B3": 1.5, "B4": 3.0})
    triggered, reasons = _check_triggers(r1, e1)
    assert not triggered, f"案例1不应触发，实际触发: {reasons}"
    print("  [✓] 案例1: 全合规 + 数值核验通过 → 不触发 ✓")

    # 案例 2: trustworthy=False → 应触发
    r2 = FakeResult(trustworthy=False)
    e2 = FakeEval(margins={"B1": 2.0, "B3": 1.5, "B4": 3.0})
    triggered, reasons = _check_triggers(r2, e2)
    assert triggered, "案例2应触发（数值核验失败）"
    print(f"  [✓] 案例2: 数值核验失败 → 触发: {reasons[0]}")

    # 案例 3: margin < 1.0 → 应触发
    r3 = FakeResult(trustworthy=True)
    e3 = FakeEval(margins={"B1": 2.0, "B3": 0.85, "B4": 3.0})
    triggered, reasons = _check_triggers(r3, e3)
    assert triggered, "案例3应触发（margin<1.0）"
    print(f"  [✓] 案例3: B3=0.85<1.0 → 触发: {reasons[0]}")

    # 案例 4: 两者都触发
    r4 = FakeResult(trustworthy=False)
    e4 = FakeEval(margins={"B1": 0.9, "B3": 1.5})
    triggered, reasons = _check_triggers(r4, e4)
    assert triggered and len(reasons) == 2, f"案例4应有2个触发理由，实际{len(reasons)}"
    print(f"  [✓] 案例4: 双重触发 → {len(reasons)} 个理由")

    print("\n[HITL] 触发逻辑自检全部通过 ✓")
    print("  - 触发条件1: trustworthy==False（确定性）")
    print("  - 触发条件2: margin < 1.0（确定性，规范规定）")


if __name__ == "__main__":
    _selftest()