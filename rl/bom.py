"""
rl/bom.py  —  材料工程量表 (仅供 LCA 碳计算用; 成本直接复用 reward._material_cost)
================================================================================
设计(厚度 米 + 路面类型) -> 各层体积(m³)/吨位(t)。碳 = 吨位 × 碳因子。
注意: 厚度单位 = 米 (与 env/surrogate/guards/protocol 一致)。
放置位置: rl/bom.py
"""
from __future__ import annotations
from typing import Dict, List, Sequence

# 仅影响“碳”的密度 (t/m³); 成本用 reward.py 的 CNY/m³ 价表, 不经此处。
DENSITY = {
    "AC_surface": 2.40, "AC_binder": 2.40, "AC_base": 2.40,
    "granular_base": 2.20, "subbase": 2.10, "cement_stabilized": 2.30,
}


def layer_materials(pavement_type: str = "flexible") -> List[str]:
    """5 个结构层(top-down)的材料键, 与 env 的层序一致。"""
    if (pavement_type or "").lower() == "flexible":
        return ["AC_surface", "AC_binder", "AC_base", "granular_base", "subbase"]
    return ["AC_surface", "AC_binder", "AC_base", "cement_stabilized", "subbase"]  # semi_rigid


def make_bom(thickness_m: Sequence[float], pavement_type: str = "flexible",
             area_m2: float = 1.0) -> Dict[str, dict]:
    """厚度(米)5 维 -> {material: {'volume_m3':, 'tonnage_t':}} (默认每 m²)。"""
    mats = layer_materials(pavement_type)
    bom: Dict[str, dict] = {}
    for i, m in enumerate(mats):
        vol = float(thickness_m[i]) * area_m2
        ton = vol * DENSITY[m]
        d = bom.setdefault(m, {"volume_m3": 0.0, "tonnage_t": 0.0})
        d["volume_m3"] += vol
        d["tonnage_t"] += ton
    return bom
