"""
rl/lca_gwp.py  —  LCA 碳足迹 (kgCO2e/m²); 碳【不折现】, 直接相加。
================================================================
建造碳 = Σ 各层吨位 × 碳因子。
养护碳 = 复用 lcc 的 schedule, 把 overlay 动作映射成新增 AC 厚度 -> 吨位 -> 碳。
放置位置: rl/lca_gwp.py
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Optional
from rl.bom import make_bom, DENSITY
from rl.carbon_factors import CarbonProvider, get_provider

# overlay 动作 -> 新增沥青层厚(米) (粗略映射; 可按你的处治定义校正)
OVERLAY_AC_M = {
    "thin_overlay":        0.038,   # ~1.5 in
    "structural_overlay":  0.051,   # ~2 in
    "mill_inlay_deep":     0.102,   # ~4 in
    # 非 AC / 表处类: 计为 0 (碳贡献忽略或另算)
    "routine": 0.0, "routine_minor": 0.0, "routine_plus_slurry": 0.0,
    "slurry_seal": 0.0, "crack_seal": 0.0, "chip_seal": 0.0,
    "fdr_reclamation": 0.0, "reconstruction": 0.0,
}


def embodied_gwp(bom: Dict[str, dict], provider: CarbonProvider) -> float:
    return float(sum(b["tonnage_t"] * provider.get_gwp(m) for m, b in bom.items()))


def maintenance_gwp(schedule: Optional[List[dict]], provider: CarbonProvider,
                    area_m2: float = 1.0) -> float:
    """养护碳(不折现): overlay 新增 AC 厚 × 密度 × AC 碳因子。"""
    if not schedule:
        return 0.0
    ac_factor = provider.get_gwp("AC_surface")
    g = 0.0
    for evt in schedule:
        h_ac = OVERLAY_AC_M.get(evt.get("action", ""), 0.0)
        if h_ac > 0:
            ton = h_ac * area_m2 * DENSITY["AC_surface"]
            g += ton * ac_factor
    return float(g)


def lca_gwp_for_design(thickness_m: Sequence[float], pavement_type: str = "flexible",
                       provider: CarbonProvider = None,
                       schedule: Optional[List[dict]] = None,
                       use_phase_gwp: float = 0.0, area_m2: float = 1.0) -> float:
    """总 GWP (kgCO2e/m²) = 建造碳 + 养护碳(+ 使用阶段, 默认0)。碳不折现。"""
    if provider is None:
        provider = get_provider("ec3", cache=True)
    g = embodied_gwp(make_bom(thickness_m, pavement_type, area_m2), provider)
    g += maintenance_gwp(schedule, provider, area_m2)
    return g + use_phase_gwp
