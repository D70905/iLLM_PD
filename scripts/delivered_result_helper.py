"""
delivered_result_helper.py — 从 env info 中提取交付设计 (兼容旧字段)
=============================================================
env.py v3 新增 delivered_design/delivered_dsr/delivered_cost_cny 等字段，
表示 episode 内成本最低的完全合规态。旧脚本读的是 design_h_cm/dsr（末状态，
可能不合规）。用本模块的 get_delivered() 替换旧读取逻辑，自动优先使用新字段。

用法:
    from scripts.delivered_result_helper import get_delivered

    result = get_delivered(info)
    print(result["dsr"])        # delivered_dsr 或 last-step dsr
    print(result["cost_cny"])   # delivered_cost_cny 或从末状态估
    print(result["h_cm"])       # 5 层厚度 (cm)
"""

from typing import Dict, List, Optional


def get_delivered(info: Dict) -> Dict:
    """
    从 env step 返回的 info 中提取"最终交付设计"。
    若 info 包含 delivered_* 字段则使用；否则回退到末状态字段。
    """
    dd = info.get("delivered_design")

    if dd is not None:
        # ── 有 delivered_* — 使用最优合规态 ──
        h_m = dd.get("thickness")
        e_MPa = dd.get("modulus")
        h_cm = [round(float(x) * 100, 1) for x in h_m] if h_m is not None else info.get("design_h_cm")
        e_MPa = [round(float(x), 0) for x in e_MPa] if e_MPa is not None else info.get("design_E_MPa")
        return {
            "source": "delivered",
            "h_cm": h_cm,
            "E_MPa": e_MPa,
            "dsr": info.get("delivered_dsr", info.get("dsr")),
            "cost_cny": info.get("delivered_cost_cny"),
            "margins": info.get("delivered_margins", info.get("margins")),
            "scr_running": info.get("scr_running"),     # 轨迹 SCR，不分来源
            "compliant": True,                           # delivered 一定是合规的
            "lcc": info.get("lcc"),
            "evaluation": info.get("evaluation"),
        }
    else:
        # ── 无 delivered_* — 回退到末状态（兼容旧版 env.py 或全 episode 无合规态）
        return {
            "source": "last_step",
            "h_cm": info.get("design_h_cm"),
            "E_MPa": info.get("design_E_MPa"),
            "dsr": info.get("dsr"),
            "cost_cny": None,
            "margins": info.get("margins"),
            "scr_running": info.get("scr_running"),
            "compliant": info.get("compliant", False),
            "lcc": info.get("lcc"),
            "evaluation": info.get("evaluation"),
        }


def patch_inference_summary(summaries: List[Dict], last_info_per_section: Dict[str, Dict]) -> List[Dict]:
    """
    给已有推理汇总表补上 delivered_* 字段。
    调用此函数后再写 CSV / 出图，确保报告的是最优合规态。

    summaries: 旧格式的汇总列表（每段一行 dict）
    last_info_per_section: {section_id: 该段最后一个 info dict}
    """
    for s in summaries:
        sid = s.get("section_id", "")
        info = last_info_per_section.get(sid, {})
        d = get_delivered(info)
        s["delivered_dsr"] = d["dsr"]
        s["delivered_cost_cny"] = d.get("cost_cny")
        s["delivered_h_cm"] = d.get("h_cm")
        s["delivered_E_MPa"] = d.get("E_MPa")
        s["delivered_margins"] = d.get("margins")
        s["delivered_source"] = d["source"]
    return summaries