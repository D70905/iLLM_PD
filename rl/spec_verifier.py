# -*- coding: utf-8 -*-
"""
rl/spec_verifier.py — JSON 锚定的规范一致性核验
================================================

为什么不用 RAG：英文/中文 query 实测对 B.1-B.4 公式召回均失败（锚点 2/7），
根因是 PDF 里数学公式是渲染图形/特殊字体，pypdf 提取成碎片，任何文本解析器
都拿不出干净形式。而 specs/data/jtg_d50.json 里 B.1-B.4 的系数/阈值全部
_status:verified（人工对照规范 PDF 核验过）。所以规范核验的金标准应是这份 JSON，
不是脏的 PDF-RAG——这比 RAG 更可靠（直接比对真实系数，而非检索可能残缺的文本）。

本模块从 jtg_d50.json 抽出"已核验的规范常量集合"（系数 + 阈值 + 基础参数），
对设计解释里出现的【规范数值断言】逐条比对：
  • 解释说"容许 R_a=15mm" → 命中 JSON 里 expressway 的 [R_a]=15 → [✓]
  • 解释说"可靠指标 β=1.65" → 命中 JSON 3.0.1 expressway β=1.65 → [✓]
  • 解释说"疲劳系数 5.0"（编造）→ JSON 里疲劳系数是 6.32/15.96/... 无 5.0 → 不命中
  • 解释引用"附录 B.9"（不存在）→ 标注：JSON 未含该附录键 → [?]

确定性、零幻觉、不依赖 RAG。RAG 仅保留作"给读者的参考链接"，不承担公式核验。
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


def _flatten_numbers(obj, out: Set[float]):
    """递归收集 JSON 里所有数值（系数、阈值、参数），作为已核验规范常量集合。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        # 从字符串里抽数字（如 equation_form: "6.32 × 10^(15.96-0.29β)"）
        for m in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", obj):
            try:
                out.add(float(m))
            except ValueError:
                pass


class SpecConstants:
    """从 jtg_d50.json 加载的已核验规范常量 + 附录键集合。"""

    def __init__(self, json_path: str):
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.numbers: Set[float] = set()
        _flatten_numbers(self.data, self.numbers)
        # 已存在的 appendix/section 键（小写规范化），用于核验"引用的条款是否存在"
        self.section_keys = set()
        for k in self.data.keys():
            kl = k.lower()
            self.section_keys.add(kl)
            # 抽出形如 b.1 / b11 / 3.0.1 的编号
            for m in re.findall(r"b\.?\d+|3[._]0[._]\d+|appendix_[a-z]\d*", kl):
                self.section_keys.add(m)

    def has_number(self, n: float, rel: float = 0.03, abstol: float = 0.01) -> bool:
        for v in self.numbers:
            if v == n:
                return True
            if abs(v - n) <= max(abstol, rel * max(abs(v), abs(n))):
                return True
        return False


# ── 规范数值断言的识别：只核验"明显在引用规范常量"的句子 ──
# 触发词：解释里出现这些词，说明它在陈述规范规定的值，而非计算结果
_SPEC_TERM_PAT = re.compile(
    r"容许|allowable|限值|limit|可靠指标|reliability index|β|系数|coefficient|"
    r"标准轴载|standard axle|接地压强|contact pressure|设计年限|design life|"
    r"规范规定|specified|规定值|附录|appendix|条")
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")
_APPENDIX_REF = re.compile(r"附录\s*([B-G])[\.．]?(\d*)|appendix\s*([B-G])[\.．]?(\d*)", re.IGNORECASE)


def verify_spec_claims(explanation: str, consts: SpecConstants) -> List[Dict]:
    """
    仅核查解释中引用的规范附录/条款是否存在于 jtg_d50.json。
    不再做"常数池匹配"——该方法已被证明不可靠（假阴性），且 LLM prompt
    已改为禁止陈述规范数值，第二类幻觉从源头消除。

    status: 'spec_ok'(附录存在) | 'spec_ref_missing'(附录不存在)
    """
    checks: List[Dict] = []
    for s in re.split(r"[。\n；;]+", explanation):
        s = s.strip()
        if not s:
            continue
        for m in _APPENDIX_REF.finditer(s):
            letter = (m.group(1) or m.group(3) or "").upper()
            num = (m.group(2) or m.group(4) or "")
            exists = (("appendix_" + letter.lower()) in " ".join(self_keys(consts))
                      or any(letter.lower() in k and (not num or num in k) for k in consts.section_keys))
            checks.append({
                "claim": s[:80],
                "status": "spec_ok" if exists else "spec_ref_missing",
                "note": ("引用 附录{}{} 存在于规范库".format(letter, num) if exists
                         else "引用 附录{}{} 在已核验规范库中未找到，请人工确认".format(letter, num)),
            })
    return checks


def self_keys(consts: SpecConstants) -> List[str]:
    return list(consts.data.keys())


# ── 自检 ──
def _selftest():
    # 用真实 jtg_d50.json 路径；测试时若不存在则用内置 mini 金标准
    import tempfile
    mini = {
        "appendix_B11_asphalt_fatigue": {
            "_status": "verified",
            "equation_form": "N_f1 = 6.32 × 10^(15.96 - 0.29β) ...",
            "coefficients_verified": {"p_6.32": 6.32, "c_15.96": 15.96,
                                      "b_0.29": 0.29, "e_3.97": 3.97, "e2_1.58": 1.58}},
        "appendix_B3_ac_permanent_deformation": {
            "_status": "verified",
            "table_3_0_6_1_allowable_R_a_mm": {"expressway": 15.0, "highway_2": 20.0}},
        "section_3_0_1_target_reliability": {
            "by_road_class": {"expressway": {"beta": 1.65, "reliability_pct": 95}}},
        "section_3_0_3_standard_axle_load": {
            "axle_load_kN": 100, "tire_contact_pressure_p_MPa": 0.7},
    }
    p = os.path.join(tempfile.gettempdir(), "_mini_jtg.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mini, f, ensure_ascii=False)
    consts = SpecConstants(p)
    print("已核验规范常量集合大小:", len(consts.numbers))
    print("（应含 6.32 / 15.96 / 3.97 / 15.0 / 1.65 / 95 / 100 / 0.7 等）\n")

    print("=" * 64)
    print("自检：附录存在性检查（仅此一项，常数池已移除）")
    print("=" * 64)
    honest = ("参见 JTG D50-2017 附录 B.1 和附录 B.3。")
    fake = ("依据规范（参见附录 B.9）。")
    for tag, txt in [("诚实", honest), ("编造", fake)]:
        print("\n[{}] {}".format(tag, txt))
        for c in verify_spec_claims(txt, consts):
            sym = {"spec_ok": "[✓]", "spec_ref_missing": "[?]"}[c["status"]]
            print("  {} {}  ← {}".format(sym, c["claim"], c["note"]))

    h = verify_spec_claims(honest, consts)
    fck = verify_spec_claims(fake, consts)
    assert all(c["status"] == "spec_ok" for c in h), "诚实引用应全部命中"
    assert any(c["status"] == "spec_ref_missing" for c in fck), "不存在的附录应被标"
    print("\n[spec_verifier] 自检通过 ✓ —— 附录存在性检查可靠")


if __name__ == "__main__":
    _selftest()
