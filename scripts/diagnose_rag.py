# -*- coding: utf-8 -*-
"""
scripts/diagnose_rag.py — RAG 库体检
=====================================

在写"输出解释器的规范核验"之前，先回答三个问题：
  1) chunk 干不干净？  —— 中文 PDF 经 pypdf 提取常有乱码/表格错位/公式丢失
  2) 关键条款在不在库里、检索得到吗？ —— B.1-B.4 四个指标 + 可靠度/轴载/设计年限
  3) 检索回来的内容，和已核验的 jtg_d50.json 阈值对得上吗？

只读 RAG（retrieve）+ 打印，不修改任何东西。

运行（务必带镜像，否则 BGE 模型下载超时）：
    set HF_ENDPOINT=https://hf-mirror.com           # Windows
    python scripts/diagnose_rag.py

输出会落一份 rag_health_report.txt，便于贴回来一起判读。
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPORT = []


def log(s=""):
    print(s)
    REPORT.append(s)


# ─────────────────────────────────────────────────────────────
# 关键查询：覆盖 JTG D50-2017 的 B.1-B.4 + 基础参数
# 每条给 query + 期望在召回里出现的"锚点关键词"（用于粗判检索是否命中正确条款）
# ─────────────────────────────────────────────────────────────
PROBES = [
    {"name": "B.1 沥青层疲劳",
     "query": "沥青混合料层疲劳开裂寿命 Nf 容许 应力应变",
     "anchors": ["疲劳", "沥青"], "expect_num": ["6.32", "15.96", "3.97"]},
    {"name": "B.2 半刚性基层疲劳",
     "query": "无机结合料稳定层 半刚性基层 疲劳 弯拉应力",
     "anchors": ["无机结合料", "疲劳"], "expect_num": []},
    {"name": "B.3 沥青层永久变形(车辙)",
     "query": "沥青混合料层 永久变形 车辙 容许变形量",
     "anchors": ["永久变形", "车辙"], "expect_num": []},
    {"name": "B.4 路基顶面竖向压应变",
     "query": "路基顶面 竖向 压应变 容许",
     "anchors": ["路基", "压应变"], "expect_num": []},
    {"name": "3.0.1 目标可靠度",
     "query": "目标可靠指标 可靠度 高速公路 一级公路",
     "anchors": ["可靠"], "expect_num": ["95", "1.65", "90"]},
    {"name": "3.0.3 标准轴载 BZZ-100",
     "query": "标准轴载 BZZ-100 轮胎接地压强 0.7 MPa",
     "anchors": ["轴载", "BZZ"], "expect_num": ["100", "0.7"]},
    {"name": "3.0.2 设计使用年限",
     "query": "设计使用年限 高速公路 15 年",
     "anchors": ["设计", "年限"], "expect_num": ["15"]},
]


def garbled_ratio(text: str) -> float:
    """粗判一段中文文本的'可读性'：可打印中文/英数字占比越低=越可能是乱码。"""
    if not text:
        return 1.0
    good = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:        # CJK
            good += 1
        elif ch.isalnum() or ch in " ，。、；：（）()%.-/×·°":
            good += 1
    return 1.0 - good / max(len(text), 1)


def main():
    log("=" * 72)
    log("RAG 库体检")
    log("=" * 72)

    try:
        from rl.rag import RAGStore
    except Exception as e:
        log("[FATAL] 无法 import RAGStore: {}".format(e))
        _save(); return

    store = RAGStore()
    try:
        n = store.count()
    except Exception as e:
        log("[FATAL] store.count() 失败（多半是 BGE 模型没下载，记得 set HF_ENDPOINT=https://hf-mirror.com）：{}".format(e))
        _save(); return

    log("\n库中 chunk 总数: {}".format(n))
    if n == 0:
        log("[STOP] 库是空的，需要先 ingest JTG/NCHRP PDF。")
        _save(); return

    # ── 体检 1：抽样看 chunk 干不干净 ──
    log("\n" + "-" * 72)
    log("体检 1：随机抽样 8 个 chunk，看中文提取质量（乱码率）")
    log("-" * 72)
    sample_q = ["沥青 路面 设计", "疲劳 应力", "路基 模量", "厚度 基层",
                "可靠 指标", "轴载 标准", "温度 修正", "材料 参数"]
    seen = set()
    garbled_scores = []
    for q in sample_q:
        try:
            ps = store.retrieve(q, top_k=2)
        except Exception as e:
            log("  retrieve('{}') 失败: {}".format(q, e)); continue
        for p in ps:
            if p.source in seen:
                continue
            seen.add(p.source)
            g = garbled_ratio(p.text)
            garbled_scores.append(g)
            flag = "  <<< 可能乱码/低质" if g > 0.25 else ""
            log("\n  [{}] 乱码率={:.0%}{}".format(p.source, g, flag))
            log("    {}".format(p.text[:160].replace("\n", " ")))
            if len(garbled_scores) >= 8:
                break
        if len(garbled_scores) >= 8:
            break
    if garbled_scores:
        avg = sum(garbled_scores) / len(garbled_scores)
        bad = sum(1 for g in garbled_scores if g > 0.25)
        log("\n  → 抽样平均乱码率 {:.0%}，{}/{} 个 chunk 偏脏".format(avg, bad, len(garbled_scores)))
        log("    （>25% 视为偏脏；偏脏多说明 PDF 提取质量差，规范核验要降级处理）")

    # ── 体检 2+3：关键条款召回 + 与已核验 JSON 对照 ──
    log("\n" + "-" * 72)
    log("体检 2+3：B.1-B.4 + 基础参数的检索召回 & 数值对照")
    log("-" * 72)
    hit_anchor = 0
    hit_num = 0
    num_total = 0
    for pr in PROBES:
        log("\n■ {}".format(pr["name"]))
        log("  query: {}".format(pr["query"]))
        try:
            ps = store.retrieve(pr["query"], top_k=3)
        except Exception as e:
            log("  retrieve 失败: {}".format(e)); continue
        if not ps:
            log("  ✗ 无召回")
            continue
        joined = " ".join(p.text for p in ps)
        # 锚点关键词命中
        a_hit = [a for a in pr["anchors"] if a in joined]
        anchor_ok = len(a_hit) == len(pr["anchors"])
        hit_anchor += 1 if anchor_ok else 0
        log("  top1 source={} score={:.2f}".format(ps[0].source, ps[0].score))
        log("  锚点命中 {}/{}: {} {}".format(
            len(a_hit), len(pr["anchors"]), a_hit, "✓" if anchor_ok else "✗ 检索可能跑偏"))
        # 数值对照（期望出现的已核验阈值/系数）
        for num in pr["expect_num"]:
            num_total += 1
            present = num in joined
            hit_num += 1 if present else 0
            log("    数值[{}] 在召回中: {}".format(num, "✓" if present else "✗ 未出现"))
        # 展示 top1 片段
        log("    top1 片段: {}".format(ps[0].text[:140].replace("\n", " ")))

    # ── 汇总判读 ──
    log("\n" + "=" * 72)
    log("体检汇总")
    log("=" * 72)
    log("  关键条款锚点命中: {}/{}".format(hit_anchor, len(PROBES)))
    if num_total:
        log("  已核验数值召回:   {}/{}".format(hit_num, num_total))
    log("")
    log("判读指引：")
    log("  • 锚点命中率高(≥5/7) + 抽样乱码率低(<25%) + 数值多数能召回")
    log("    → RAG 库可用，输出解释器可做'规范条款核验'（检索+比对）。")
    log("  • 锚点命中差 或 乱码率高")
    log("    → 库不可靠。规范核验降级为：只做数值断言核验(对真实margins/JSON)，")
    log("      规范引用仅作'参考链接'不做强校验，避免'用脏检索核验LLM'的双重不可靠。")
    log("  • 不论哪种，数值断言核验那一半（对 jtg_d50.json + 真实 margins）都成立、且最硬。")
    log("=" * 72)
    _save()


def _save():
    out = os.path.join(os.path.dirname(__file__), "..", "rag_health_report.txt")
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT))
        print("\n[报告已保存] {}".format(os.path.abspath(out)))
    except Exception as e:
        print("保存报告失败: {}".format(e))


if __name__ == "__main__":
    main()
