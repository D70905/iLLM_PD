"""
rl/explain_unified.py — 统一设计解释入口（双核验）
================================================
合并 design_explainer（数值断言核验）和 spec_verifier（附录存在性检查），
一次调用出解释 + 双核验报告。

用法:
    from rl.explain_unified import explain_design_full

    result = explain_design_full(evaluation, design)
    print(result.report)        # 带双核验标注的完整报告
    print(result.trustworthy)   # True = 数值核验全通过
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rl.design_explainer import explain_design, ExplainResult
from rl.spec_verifier import SpecConstants, verify_spec_claims


# ── 懒加载规范常量（只加载一次） ──
_spec_consts: Optional[SpecConstants] = None


def _get_spec_consts() -> SpecConstants:
    global _spec_consts
    if _spec_consts is None:
        json_path = os.path.join(os.path.dirname(__file__), "..",
                                 "specs", "data", "jtg_d50.json")
        _spec_consts = SpecConstants(json_path)
    return _spec_consts


@dataclass
class FullExplainResult:
    """统一解释结果：数值核验 + 规范附录核验。"""
    explanation: str                # 原始解释文本
    annotated: str                  # 带双核验标注的解释
    trustworthy: bool               # 数值核验全部通过
    # 数值核验
    n_numeric_ok: int
    n_numeric_mismatch: int
    n_numeric_unmatched: int
    attempts: int
    # 规范核验（仅附录存在性）
    spec_ok: int
    spec_missing: int               # 引用了不存在的附录
    referenced_appendixes: List[str]
    # 组合报告
    report: str


def explain_design_full(
    evaluation,
    design: Dict,
    backend: str = "deepseek",
    model: Optional[str] = "deepseek-chat",
    max_retries: int = 2,
) -> FullExplainResult:
    """
    统一入口：LLM 生成设计解释 → 数值核验 + 规范附录核验 → 组合报告。

    Args:
        evaluation: specs.protocol.DesignEvaluation（含 margins/feasible/critical/responses）
        design: {'thickness_cm': [...], 'modulus_MPa': [...]}
        backend / model: LLM 配置
        max_retries: 数值不符时打回重写的最大次数

    Returns:
        FullExplainResult，含双核验标注的完整报告
    """
    # ── 第 1 层：数值断言核验 ──
    explain_result: ExplainResult = explain_design(
        evaluation, design,
        backend=backend, model=model, max_retries=max_retries,
    )

    # ── 第 2 层：规范附录核验（仅附录存在性，不做常数池匹配） ──
    consts = _get_spec_consts()
    raw_checks = verify_spec_claims(explain_result.explanation, consts)

    # 只保留"附录存在性"和"附录缺失"两类；常数匹配的全部过滤掉
    spec_checks = [c for c in raw_checks
                   if c["status"] in ("spec_ok", "spec_ref_missing")
                   and "附录" in c.get("claim", "") or "appendix" in c.get("claim", "").lower()
                   or "Appendix" in c.get("claim", "")]

    # 也保留 spec_ok 中明确引用附录的
    spec_checks_filtered = []
    for c in raw_checks:
        if c["status"] == "spec_ref_missing":
            spec_checks_filtered.append(c)
        elif c["status"] == "spec_ok" and ("附录" in c.get("note", "") or
                                           "appendix" in c.get("note", "").lower()):
            spec_checks_filtered.append(c)

    spec_ok = sum(1 for c in spec_checks_filtered if c["status"] == "spec_ok")
    spec_missing = sum(1 for c in spec_checks_filtered if c["status"] == "spec_ref_missing")
    appendixes = []
    for c in spec_checks_filtered:
        import re
        for m in re.finditer(r"附录\s*([B-G])[\.．]?(\d*)|[Aa]ppendix\s*([B-G])[\.]?(\d*)",
                             c.get("claim", "")):
            letter = (m.group(1) or m.group(3) or "").upper()
            num = m.group(2) or m.group(4) or ""
            ref = f"Appendix {letter}.{num}" if num else f"Appendix {letter}"
            if ref not in appendixes:
                appendixes.append(ref)

    # ── 组合报告 ──
    report_lines = [explain_result.annotated]

    if spec_checks_filtered:
        report_lines.append("─" * 40)
        report_lines.append("【规范附录核验】")
        sym = {"spec_ok": "[✓]", "spec_ref_missing": "[?]"}
        for c in spec_checks_filtered:
            s = sym.get(c["status"], "[?]")
            report_lines.append(f"  {s} {c['note'][:100]}")

    report_lines.append("─" * 40)
    status = "✓ 可交付" if explain_result.trustworthy else "✗ 含数值不符，需人工审核"
    report_lines.append(f"核验结论: {status} | "
                        f"数值 {explain_result.n_verified}✓/{explain_result.n_mismatch}✗/"
                        f"{explain_result.n_unmatched}? | "
                        f"附录 {spec_ok}✓/{spec_missing}?")

    return FullExplainResult(
        explanation=explain_result.explanation,
        annotated=explain_result.annotated,
        trustworthy=explain_result.trustworthy,
        n_numeric_ok=explain_result.n_verified,
        n_numeric_mismatch=explain_result.n_mismatch,
        n_numeric_unmatched=explain_result.n_unmatched,
        attempts=explain_result.attempts,
        spec_ok=spec_ok,
        spec_missing=spec_missing,
        referenced_appendixes=appendixes,
        report="\n".join(report_lines),
    )


# ── 自检 ──
def _selftest():
    """用合成数据验证统一入口不崩、双核验都执行。"""
    print("=" * 60)
    print("explain_unified 自检")
    print("=" * 60)

    # 合成 evaluation + design（复现 design_explainer 自检的场景）
    class FakeEval:
        feasible = True
        critical_indicator = "B3_ac_permanent_deformation"
        margins = {
            "B1_asphalt_fatigue": 32.81,
            "B2_semi_rigid_fatigue": 5.04,
            "B3_ac_permanent_deformation": 1.84,
            "B4_subgrade_strain": 3.59,
        }
        responses = {
            "epsilon_a_microstrain": 46.4,
            "epsilon_z_microstrain": 96.3,
            "sigma_t_MPa": 0.01,
        }

    design = {
        "thickness_cm": [4.0, 6.0, 8.0, 36.0, 18.0],
        "modulus_MPa": [14000, 11000, 9000, 1500, 400],
    }

    # 用 mock LLM 返回诚实解释（含附录引用）
    def mock_llm(sys_p, usr_p):
        return (
            "本设计 B3 沥青永久变形 margin 为 1.84，是最临界的指标。"
            "B1 沥青疲劳 margin 高达 32.81，非常安全。"
            "路基顶面竖向压应变 96.3 微应变，远低于容许值。"
            "上面层厚度 4.0cm。参见 JTG D50-2017 附录 B.3 和附录 B.1。"
        )

    # 注入 mock
    original_explain = None
    try:
        import rl.design_explainer as de
        original_explain = de.explain_design
        # 用 wrapper 替换
        def mock_explain(evaluation, design, **kw):
            kw["verify_fn"] = mock_llm
            return original_explain(evaluation, design, **kw)
        de.explain_design = mock_explain

        result = explain_design_full(FakeEval(), design, max_retries=0)

        print(f"\ntrustworthy: {result.trustworthy}")
        print(f"数值: {result.n_numeric_ok}✓/{result.n_numeric_mismatch}✗/"
              f"{result.n_numeric_unmatched}?")
        print(f"附录: {result.spec_ok}✓/{result.spec_missing}?")
        print(f"引用附录: {result.referenced_appendixes}")
        print(f"\n完整报告:\n{result.report}")

        # 断言
        assert result.trustworthy, "诚实解释应该 trustworthy"
        assert result.n_numeric_ok >= 3, f"应该有 >=3 个数值核验通过，实际 {result.n_numeric_ok}"
        assert result.spec_ok >= 1, f"应该至少命中 1 个附录，实际 {result.spec_ok}"
        print("\n[explain_unified] 自检通过 ✓")

    finally:
        if original_explain is not None:
            import rl.design_explainer as de
            de.explain_design = original_explain


if __name__ == "__main__":
    _selftest()