"""
rl/reward_mo.py
多目标加权奖励 (λ 扫描)。设计原则:
  - 可行: 不进奖励 (Guard 强制)。
  - 合规: “屏障”大惩罚(固定大值), 用不敏感性扫描辩护; 输出再由选择规则保证合规。
  - 成本/碳: 两个目标, 归一化后 λ 加权; λ 从 0->1 扫描, 每个 λ 一个策略 -> Pareto 一点。
放置位置: rl/reward_mo.py
"""
from __future__ import annotations
from dataclasses import dataclass
from rl.mo_objectives import ObjResult, Normalizer


@dataclass
class MORewardConfig:
    lam: float = 0.5         # 碳权重 λ; 成本权重 = 1-λ; 扫描此参数得前沿
    P_barrier: float = 10.0  # 合规屏障(违规惩罚); 做 {5,10,20,50} 扫描证明 >=阈值不敏感


def mo_reward(obj: ObjResult, norm: Normalizer, cfg: MORewardConfig) -> float:
    """
    reward = -[ λ·gwp_norm + (1-λ)·lcc_norm ] - P·1[非合规]
    (最大化 reward = 最小化 加权(成本,碳), 同时满足合规; 可行由 Guard 保证)
    """
    cost_term = norm.lcc_norm(obj.lcc)
    carb_term = norm.gwp_norm(obj.gwp)
    objective = -(cfg.lam * carb_term + (1.0 - cfg.lam) * cost_term)
    barrier = -cfg.P_barrier if obj.dsr < 1.0 else 0.0
    return objective + barrier
