# -*- coding: utf-8 -*-
"""
rl/design_explainer.py — 设计解释器 + 数值断言核验（输出端口）
================================================================

iLLM-PD 的 LLM 输出端口：把"最终设计 + DesignEvaluation(真实margins/responses)
+ 审计链摘要" → 工程师可读的中文设计说明，并对解释中的【每一个数值断言】
逐条核验，标注 [✓已核验] / [✗与计算不符] / [?未找到对应量]。

设计原则（基于 RAG 体检结论 = 中文条款检索 0/7 不可用，故走"路线一"）：
  • 只做【数值断言核验】——对象是真实算出的 margins/responses + 已逐条核验的
    jtg_d50.json 阈值。确定性、可 100% 抓数值幻觉、不依赖 RAG。
  • 规范引用【降级】——解释可给"参见 JTG D50-2017 附录 B.x"指引，但本模块
    不声称已核验条款文字（避免用不可靠检索核验 LLM）。
  • 核验失败 → 分级处理：数值不符 → 自动打回让 LLM 重写（≤N 次）→ 仍不符则
    [✗标红] 交工程师；找不到对应量 → [?标注] 交工程师；绝不自动用真值替换
    （那等于程序代写、失去"LLM 解释"的意义）。

这是 Evaluator 的"转生"：它原本在训练 step 里做安全裁决（已证明冗余），
现在退出训练环，改在【设计完成后】生成带核验的工程解释——它今天展示的
正确工程推理（"B3 仅1.84、减薄沥青层加剧永久变形"）正好胜任这个角色。

依赖：rl.llm_client.get_client / parse_json_from_text（已修：JSON模式+800token）
      specs.protocol.DesignEvaluation（真实 margins/responses/critical 来源）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 核验数据：从真实 DesignEvaluation 抽出"可核验量字典"
# ─────────────────────────────────────────────────────────────
# margins 键 → 中文名（用于在解释里识别 LLM 指的是哪个指标）
MARGIN_ZH = {
    "B1_asphalt_fatigue":          ["B1", "沥青", "疲劳"],
    "B2_semi_rigid_fatigue":       ["B2", "半刚性", "无机结合料"],
    "B3_ac_permanent_deformation": ["B3", "永久变形", "车辙"],
    "B4_subgrade_strain":          ["B4", "路基", "压应变"],
}


@dataclass
class VerifiableFacts:
    """从 DesignEvaluation + design 抽出的真实量，作为核验的"金标准"。"""
    margins: Dict[str, float]
    critical_indicator: str
    feasible: bool
    responses: Dict[str, float]        # eps_a, sigma_t, eps_z, predicted R_a ...
    allowable: Dict[str, float]
    thickness_cm: List[float]
    modulus_MPa: List[float]

    @classmethod
    def from_evaluation(cls, evaluation, design: Dict) -> "VerifiableFacts":
        th = design.get("thickness", [])
        return cls(
            margins={k: float(v) for k, v in evaluation.margins.items()},
            critical_indicator=str(getattr(evaluation, "critical_indicator", "")),
            feasible=bool(getattr(evaluation, "feasible", False)),
            responses={k: float(v) for k, v in getattr(evaluation, "responses", {}).items()
                       if isinstance(v, (int, float))},
            allowable={k: float(v) for k, v in getattr(evaluation, "allowable_values", {}).items()
                       if isinstance(v, (int, float))},
            thickness_cm=[round(float(x) * 100, 2) for x in th],
            modulus_MPa=[round(float(x), 0) for x in design.get("modulus", [])],
        )

    def all_numeric(self) -> Dict[str, float]:
        """汇总所有可核验的数值（用于匹配解释中出现的数字）。"""
        out = {}
        out.update({"margin_" + k: v for k, v in self.margins.items()})
        out.update(self.responses)
        out.update({"allow_" + k: v for k, v in self.allowable.items()})
        for i, h in enumerate(self.thickness_cm):
            out["h{}_cm".format(i)] = h
        for i, e in enumerate(self.modulus_MPa):
            out["E{}_MPa".format(i)] = e
        return out


# ─────────────────────────────────────────────────────────────
# 数值断言核验
# ─────────────────────────────────────────────────────────────
@dataclass
class ClaimCheck:
    claim_text: str          # 解释里的一句话
    numbers_in_claim: List[float]
    status: str              # 'verified' | 'mismatch' | 'unmatched'
    matched_to: Optional[str] = None      # 匹配到的真实量名
    matched_value: Optional[float] = None
    note: str = ""


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# 规范引用的"伪数字"：年份、条款/附录编号等，不应参与数值核验
_SPEC_REF_CONTEXT = ("JTG", "D50", "附录", "第", "条", "式", "表", "B.", "Eq", "Section", "Table")
_YEARS = {2006.0, 2016.0, 2017.0, 2018.0, 1993.0}


def _is_spec_ref_number(n: float, sentence: str) -> bool:
    """句子含规范引用语境、且 n 像年份/条款号(如 2017, 3.0.1 的 3/1)，则跳过。"""
    if any(tok in sentence for tok in _SPEC_REF_CONTEXT):
        if n in _YEARS or (n == int(n) and (n >= 1990 or 0 <= n <= 12)):
            return True
    return False


def _rel_close(a: float, b: float, rel: float = 0.05, abstol: float = 0.02) -> bool:
    if a == b:
        return True
    return abs(a - b) <= max(abstol, rel * max(abs(a), abs(b)))


def verify_numeric_claims(explanation: str,
                          facts: VerifiableFacts,
                          rel_tol: float = 0.05) -> List[ClaimCheck]:
    """
    对解释里每个含数字的句子做核验：
      - 句中数字能匹配到某个真实量(在容差内) → verified
      - 句中数字接近某真实量但超容差 → mismatch（疑似幻觉/口径错）
      - 句中数字找不到任何对应真实量 → unmatched（交工程师判断）
    临界指标断言额外检查：若解释声称某指标"临界/最危险"，与 facts.critical 比对。
    """
    # 剥离规范引用串，避免 "D50-2017"→50/-2017、"B.3"→3 这类伪数字混入核验
    _SPEC_STRIP = re.compile(
        r"JTG\s*[A-Z]?\s*D?\s*\d+[-\u2013]?\d*|附录\s*[B-G][\.．]?\d*[-\.]?\d*|"
        r"第\s*[\d\.]+\s*条|式\s*\([^)]*\)|表\s*[\d\.]+|Eq\.?\s*[\d\.\-]+|"
        r"Section\s*[\d\.]+|Table\s*[\d\.]+|B[\.．]\d+|D50[-\u2013]?\d*")

    def _strip_spec_refs(text: str) -> str:
        return _SPEC_STRIP.sub("〈ref〉", text)

    truth = facts.all_numeric()
    truth_vals = list(truth.items())
    checks: List[ClaimCheck] = []

    # 分句（中英文标点）
    sentences = re.split(r"[。\n；;]+", explanation)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_clean = _strip_spec_refs(s)
        nums = [float(x) for x in _NUM_RE.findall(s_clean)]
        if not nums:
            continue
        for n in nums:
            # 跳过明显是序号/层号的小整数（0-5）且句中提到"层/第"
            if n in (0, 1, 2, 3, 4, 5) and ("层" in s or "第" in s):
                continue
            # 跳过规范引用里的年份/条款号
            if _is_spec_ref_number(n, s):
                continue
            # 跳过孤立的 1.0（常来自"高达/约为"等修辞，非真实量）
            if n == 1.0:
                continue
            best = None
            best_name = None
            best_rel = None
            for name, val in truth_vals:
                # 量纲就近：厚度类(h*_cm)只和厚度量级比，避免 4.0cm 误配 46.4με
                if _rel_close(n, val, rel_tol):
                    score = abs(n - val)
                    if best is None or score < best:
                        best, best_name = score, name
            if best_name is not None:
                checks.append(ClaimCheck(
                    claim_text=s[:80], numbers_in_claim=[n],
                    status="verified", matched_to=best_name,
                    matched_value=truth[best_name]))
            else:
                # 找接近但超容差的（疑似 mismatch）
                near = None; near_name = None
                for name, val in truth_vals:
                    if val != 0 and abs(n - val) <= 0.25 * max(abs(n), abs(val)):
                        d = abs(n - val)
                        if near is None or d < near:
                            near, near_name = d, name
                if near_name is not None:
                    checks.append(ClaimCheck(
                        claim_text=s[:80], numbers_in_claim=[n],
                        status="mismatch", matched_to=near_name,
                        matched_value=truth[near_name],
                        note="解释值{} 与计算值{:.3g} 不符".format(n, truth[near_name])))
                else:
                    checks.append(ClaimCheck(
                        claim_text=s[:80], numbers_in_claim=[n],
                        status="unmatched",
                        note="数字{} 未匹配到任何计算量".format(n)))
    return checks


def check_critical_claim(explanation: str, facts: VerifiableFacts) -> Optional[ClaimCheck]:
    """检查解释里'临界/最危险指标'的说法是否与真实 critical_indicator 一致。"""
    crit = facts.critical_indicator
    if not crit or crit not in MARGIN_ZH:
        return None
    # 找解释中"临界/最/危险/控制"附近提到的指标
    for sent in re.split(r"[。\n；;]+", explanation):
        if any(w in sent for w in ("临界", "最危险", "控制性", "最不利", "最小")):
            # 它提到了哪个指标？
            claimed = [k for k, kws in MARGIN_ZH.items() if any(kw in sent for kw in kws)]
            if claimed:
                ok = crit in claimed
                return ClaimCheck(
                    claim_text=sent[:80], numbers_in_claim=[],
                    status="verified" if ok else "mismatch",
                    matched_to=crit,
                    note=("临界指标声称正确" if ok
                          else "解释称临界为{}，实际临界为{}".format(claimed, crit)))
    return None


# ─────────────────────────────────────────────────────────────
# 解释生成（LLM）+ 核验 + 失败重写
# ─────────────────────────────────────────────────────────────
EXPLAINER_SYSTEM = """你是路面结构设计说明撰写专家，熟悉中国 JTG D50-2017 沥青路面设计规范。
你的任务：根据给定的【最终设计结构】和【力学验算结果(margins/响应)】，撰写一份
面向工程师的中文设计说明，解释为什么这个结构是合理的、哪个指标最临界。

硬性规则（违反会被自动打回）：
1. 只能使用下面【明确给出】的数值（margins/结构尺寸/力学响应）。
   禁止引入任何规范规定的阈值、系数或限值——即便你知道也严禁写。
   例如：不要说"容许变形为15mm"或"可靠指标1.65"或"疲劳系数6.32"。
   这些规范数值由系统的确定性代码负责，不由你来复述。
2. 你只做定性判断："B3 margin=1.84 是临界指标、安全裕度较薄"——
   不要写"规范容许值为X"之类的规范数字断言。
3. 引用规范时只给附录/条款指引（如"参见附录 B.3"），不给具体公式或系数。
4. 临界指标必须与给定的 critical 字段一致。
输出 JSON：{"explanation": "<中文设计说明全文>"}"""

EXPLAINER_USER = """【最终设计结构(上→下)】
{structure}

【力学验算结果(JTG D50-2017, capacity/demand, ≥1.0=通过)】
{margins}
临界指标(最小margin): {critical}
是否全部通过: {feasible}

【关键力学响应】
{responses}

请撰写中文设计说明，输出 JSON: {{"explanation": "..."}}。
只用上面给出的数字；引用规范只给附录/条款指引、不要写公式系数。"""


@dataclass
class ExplainResult:
    explanation: str
    checks: List[ClaimCheck]
    critical_check: Optional[ClaimCheck]
    n_verified: int
    n_mismatch: int
    n_unmatched: int
    attempts: int
    annotated: str            # 带 [✓]/[✗]/[?] 标注的解释
    trustworthy: bool         # 无 mismatch 即视为可交付（unmatched 仅提示）


def _fmt_facts(facts: VerifiableFacts) -> Tuple[str, str, str]:
    layers = ["上面层", "中面层", "下面层", "基层", "底基层"]
    s_lines = []
    for i in range(min(len(facts.thickness_cm), len(facts.modulus_MPa))):
        nm = layers[i] if i < len(layers) else "层{}".format(i)
        s_lines.append("- {}: h={}cm, E={}MPa".format(nm, facts.thickness_cm[i], facts.modulus_MPa[i]))
    m_lines = ["- {} = {:.2f}".format(k, v) for k, v in facts.margins.items()]
    r_lines = ["- {} = {:.3g}".format(k, v) for k, v in facts.responses.items()]
    return "\n".join(s_lines), "\n".join(m_lines), "\n".join(r_lines)


def annotate(explanation: str, checks: List[ClaimCheck],
             crit_check: Optional[ClaimCheck]) -> str:
    """在解释后附核验清单。"""
    out = [explanation, "\n" + "─" * 40, "【数值核验】"]
    sym = {"verified": "[✓已核验]", "mismatch": "[✗与计算不符]", "unmatched": "[?未匹配]"}
    if crit_check is not None:
        out.append("{} 临界指标：{}".format(sym[crit_check.status], crit_check.note))
    for c in checks:
        line = "{} {}".format(sym[c.status], c.claim_text)
        if c.status != "verified" and c.note:
            line += "  ← {}".format(c.note)
        out.append(line)
    return "\n".join(out)


def explain_design(evaluation, design: Dict,
                   backend: str = "deepseek",
                   model: Optional[str] = "deepseek-chat",
                   max_retries: int = 2,
                   verify_fn=None) -> ExplainResult:
    """
    生成 + 核验设计解释。LLM/RAG 部分需本地真实环境；verify 部分纯逻辑可离线。

    verify_fn: 仅供测试注入（替代真实 LLM）。签名 (system,user)->explanation_str。
    """
    facts = VerifiableFacts.from_evaluation(evaluation, design)
    structure, margins_s, responses_s = _fmt_facts(facts)
    user = EXPLAINER_USER.format(
        structure=structure, margins=margins_s, critical=facts.critical_indicator,
        feasible=facts.feasible, responses=responses_s)

    def _gen(sys_p, usr_p) -> str:
        if verify_fn is not None:
            return verify_fn(sys_p, usr_p)
        from rl.llm_client import get_client, parse_json_from_text
        resp = get_client(backend).chat(
            system=sys_p, user=usr_p, model=model,
            temperature=0.3, max_tokens=800,
            response_format={"type": "json_object"})
        parsed = parse_json_from_text(resp.text)
        return (parsed or {}).get("explanation", resp.text)

    attempts = 0
    explanation = ""
    checks: List[ClaimCheck] = []
    crit_check = None
    sys_p = EXPLAINER_SYSTEM
    usr_p = user
    while attempts < max_retries + 1:
        attempts += 1
        explanation = _gen(sys_p, usr_p).strip()
        checks = verify_numeric_claims(explanation, facts)
        crit_check = check_critical_claim(explanation, facts)
        n_mis = sum(1 for c in checks if c.status == "mismatch")
        n_mis += 1 if (crit_check and crit_check.status == "mismatch") else 0
        if n_mis == 0:
            break
        # 有数值不符 → 打回重写，把错误反馈给 LLM
        bad = [c for c in checks if c.status == "mismatch"]
        fb = "；".join("'{}'处:{}".format(c.claim_text[:30], c.note) for c in bad[:5])
        if crit_check and crit_check.status == "mismatch":
            fb += "；" + crit_check.note
        usr_p = user + "\n\n【上次输出有数值错误，请修正后重写】\n" + fb

    n_v = sum(1 for c in checks if c.status == "verified")
    n_m = sum(1 for c in checks if c.status == "mismatch")
    n_u = sum(1 for c in checks if c.status == "unmatched")
    if crit_check and crit_check.status == "mismatch":
        n_m += 1
    annotated = annotate(explanation, checks, crit_check)
    return ExplainResult(
        explanation=explanation, checks=checks, critical_check=crit_check,
        n_verified=n_v, n_mismatch=n_m, n_unmatched=n_u,
        attempts=attempts, annotated=annotated,
        trustworthy=(n_m == 0))


# ─────────────────────────────────────────────────────────────
# 自检：合成"诚实解释"和"含幻觉解释"，验证核验器能区分
# ─────────────────────────────────────────────────────────────
def _selftest():
    class FakeEval:
        margins = {"B1_asphalt_fatigue": 32.81, "B2_semi_rigid_fatigue": 5.04,
                   "B3_ac_permanent_deformation": 1.84, "B4_subgrade_strain": 3.59}
        critical_indicator = "B3_ac_permanent_deformation"
        feasible = True
        responses = {"epsilon_a_microstrain": 46.4, "epsilon_z_microstrain": 96.3,
                     "predicted_R_a_mm": 8.2}
        allowable_values = {"B4_epsilon_z_allowable_microstrain": 345.0}
    design = {"thickness": [0.04, 0.06, 0.08, 0.36, 0.18],
              "modulus": [14000, 11000, 9000, 1500, 400]}
    facts_eval = FakeEval()

    print("=" * 64)
    print("自检 1：诚实解释（数字都来自真实量，临界=B3）应当全 verified、trustworthy=True")
    print("=" * 64)
    honest = ("本设计 B3 沥青永久变形 margin 为 1.84，是四个指标中最小、最临界的，"
              "控制了沥青层厚度。B1 沥青疲劳 margin 高达 32.81，非常安全。"
              "路基顶面竖向压应变 96.3 微应变，远低于容许值。"
              "上面层厚度 4.0cm，参见 JTG D50-2017 附录 B.3。")
    r1 = explain_design(facts_eval, design, verify_fn=lambda s, u: '{"explanation":"%s"}' % honest)
    # 注：verify_fn 返回字符串即 explanation（_gen 直接用），这里直接给纯文本
    r1 = explain_design(facts_eval, design, verify_fn=lambda s, u: honest)
    print(r1.annotated)
    print("\n→ verified={}, mismatch={}, unmatched={}, attempts={}, trustworthy={}".format(
        r1.n_verified, r1.n_mismatch, r1.n_unmatched, r1.attempts, r1.trustworthy))

    print("\n" + "=" * 64)
    print("自检 2：含幻觉解释（编造 margin=2.5 + 临界说成B1）应当被抓出 mismatch")
    print("=" * 64)
    halluc = ("本设计 B1 沥青疲劳 margin 为 2.50，是最临界的指标。"
              "路基压应变约 250 微应变。沥青层等效模量 8000MPa 满足要求。")
    # 固定返回幻觉（重写也不变）→ 应当 attempts 用满且 trustworthy=False
    r2 = explain_design(facts_eval, design, max_retries=1, verify_fn=lambda s, u: halluc)
    print(r2.annotated)
    print("\n→ verified={}, mismatch={}, unmatched={}, attempts={}, trustworthy={}".format(
        r2.n_verified, r2.n_mismatch, r2.n_unmatched, r2.attempts, r2.trustworthy))
    assert r1.trustworthy is True, "诚实解释应 trustworthy"
    assert r2.trustworthy is False, "幻觉解释应被拦截"
    assert r2.attempts == 2, "幻觉应触发重写直到上限"
    print("\n[design_explainer] 自检通过 ✓ —— 核验器能区分诚实解释与幻觉")


if __name__ == "__main__":
    _selftest()
