"""
rl/lcc_mo.py  —  LCC 封装 (零重复: 复用你现有的成本与 NPV 实现)
================================================================
construction = reward.CompositeReward._material_cost(thickness, modulus, type)  # CNY/m²
            -> /7.20 -> USD/m²
LCC NPV      = lifecycle_lcc_intl.lcc_npv_usd(C_usd, design_life, B1, B2, r)    # USD/m²
养护时刻表由 B1/B2 边际驱动 (与 env._compute_post_eval 完全一致)。
放置位置: rl/lcc_mo.py
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np

from rl.reward import CompositeReward
from rl.lifecycle_lcc_intl import lcc_npv_usd, cny_to_usd_per_m2

_REWARD = None  # 复用一个实例 (只用其 _material_cost)


def _reward():
    global _REWARD
    if _REWARD is None:
        _REWARD = CompositeReward()
    return _REWARD


def construction_cost_usd(thickness_m: Sequence[float], modulus_MPa: Sequence[float],
                          pavement_type: str = "flexible") -> float:
    """与 env 一致: CNY/m² 经 /7.20 转 USD/m²。"""
    c_cny = _reward()._material_cost(
        np.asarray(thickness_m, float), np.asarray(modulus_MPa, float),
        pavement_type=pavement_type)
    return cny_to_usd_per_m2(float(c_cny))


def lcc_for_design(thickness_m: Sequence[float], modulus_MPa: Sequence[float],
                   margins: Dict[str, float], pavement_type: str = "flexible",
                   design_life_years_lcc: float = 20.0,
                   discount_rate: float = 0.04) -> Dict:
    """
    返回 lcc_npv_usd 的完整 dict (含 'NPV_total_usd_m2' 与 'schedule')。
    margins: protocol.evaluate 得到的 margins (需含 B1_asphalt_fatigue / B2_semi_rigid_fatigue)。
    """
    C_usd = construction_cost_usd(thickness_m, modulus_MPa, pavement_type)
    mB1 = float(margins.get("B1_asphalt_fatigue", float("inf")))
    mB2 = float(margins.get("B2_semi_rigid_fatigue", float("inf")))
    return lcc_npv_usd(
        C_construction_usd_per_m2=C_usd,
        design_life_years=float(design_life_years_lcc),
        margin_B1=mB1, margin_B2=mB2, discount_rate=discount_rate)
