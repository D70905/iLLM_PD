# -*- coding: utf-8 -*-
"""
scripts/clean_rag_supplement.py — 删除未经人工核验的补充 chunk
=============================================================
fix_rag_supplement.py 用 .get(key, 默认值) 的方式从 JSON 取条款，
但多个 key 实际不存在 → 落入硬编码默认公式（等于 LLM 凭印象写的），
未经对照规范 PDF 核验即入库。这些 chunk 作为"规范核验的标准答案"
是不可信的，必须先清除，避免"用编造的库核验会编造的 LLM"。

本脚本只删除 id 形如 json_suppl_* 的 chunk，不动原始 1504 条 PDF chunk。

用法:
    set HF_ENDPOINT=https://hf-mirror.com
    python scripts/clean_rag_supplement.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.rag import RAGStore


def main():
    store = RAGStore()
    if not store._try_init():
        print("ERROR: RAG init failed (记得 set HF_ENDPOINT=https://hf-mirror.com)")
        return

    before = store._collection.count()
    print("当前 chunk 总数: {}".format(before))

    # 补充 chunk 的 id 规律： json_suppl_00 .. json_suppl_NN
    suppl_ids = ["json_suppl_{:02d}".format(i) for i in range(50)]
    try:
        existing = store._collection.get(ids=suppl_ids)
        found = [i for i in (existing.get("ids") or [])]
    except Exception as e:
        print("查询补充 chunk 失败: {}".format(e))
        found = []

    # 兜底：也按 metadata origin=json_supplement 找
    try:
        by_meta = store._collection.get(where={"origin": "json_supplement"})
        for i in (by_meta.get("ids") or []):
            if i not in found:
                found.append(i)
    except Exception:
        pass

    if not found:
        print("没有发现 json_suppl_* / origin=json_supplement 的 chunk。")
        print("（若你之前没跑过 fix_rag_supplement，或 id 命名不同，请告诉我。）")
        return

    print("发现 {} 条未核验补充 chunk，将删除: {}".format(len(found), found))
    store._collection.delete(ids=found)
    after = store._collection.count()
    print("删除完成。chunk 总数: {} → {}".format(before, after))
    print("RAG 已恢复到仅含 PDF 提取的原始状态。")


if __name__ == "__main__":
    main()
