"""
rl/mo_objectives.py  —  多目标统一评估 (钩子已接到你的真实管线)
================================================================
设计 -> (DSR, 可行, LCC[USD/m²], GWP[kgCO2e/m²], margins)。
加权奖励(reward_mo) 与 NSGA-II(nsga2_benchmark) 都调 evaluate_design, 保证一致。

钩子接法 (均按你真实代码核对):
  predict_responses : rl.surrogate_predictor.SurrogatePredictor.predict(thickness(m),modulus(MPa),E_subgrade,pavement_type)
  compliance_eval   : specs.get_protocol("JTG_D50_2017").evaluate(DesignInputs(...完整...), responses)
  is_feasible       : rl.guards.NumericalGuard(base_type).check_design(thickness,modulus,E_subgrade)
  dsr               : rl.dsr_patch.compute_dsr(margins)
  lcc               : rl.lcc_mo.lcc_for_design(...)  (复用 reward._material_cost + lcc_npv_usd)
  gwp               : rl.lca_gwp.lca_gwp_for_design(...)

设计变量 x = [h1..h5(米), E1..E5(MPa)] (10 维), 与 env 动作空间一致 (h 与 E 都可调)。
放置位置: rl/mo_objectives.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from rl.lcc_mo import lcc_for_design
from rl.lca_gwp import lca_gwp_for_design
from rl.carbon_factors import get_provider


@dataclass
class DesignCase:
    """一个设计问题的固定上下文 (对应一个 LTPP 段/需求)。默认值取自 EnvConfig。"""
    pavement_type: str = "flexible"            # 'flexible' | 'semi_rigid'
    protocol_name: str = "JTG_D50_2017"
    road_class: str = "expressway"
    traffic_level: str = "heavy"
    E_subgrade: float = 60.0
    nu_subgrade: float = 0.40
    poisson: List[float] = field(default_factory=lambda: [0.25, 0.30, 0.30, 0.25, 0.35])
    design_life: int = 15                       # 给 protocol 的设计年限
    design_life_years_lcc: float = 20.0         # 给 LCC 的分析期 (env: design_life_years_lcc)
    discount_rate: float = 0.04
    extras: Dict = field(default_factory=lambda: {"city": "", "climate_zone": "temperate",
                                                  "VFA_pct": 70.0,
                                                  "R_s_MPa": 1.0, "R_0_mm": 1.5})
    surrogate_ckpt: str = "output/surrogate_model/surrogate_v3.pt"


@dataclass
class ObjResult:
    dsr: float
    feasible: bool
    lcc: float                 # NPV, USD/m²
    gwp: float                 # kgCO2e/m²
    margins: Dict[str, float] = field(default_factory=dict)


# ── 缓存 (避免每次评估都重载) ─────────────────────────────────
_SURROGATE = None
_PROTOCOL = {}


def _surrogate(ckpt):
    global _SURROGATE
    if _SURROGATE is None:
        from rl.surrogate_predictor import SurrogatePredictor
        _SURROGATE = SurrogatePredictor(ckpt)   # 你的 __init__ 需要 ckpt 路径 (本地 AI 漏了)
    return _SURROGATE


def _protocol(name):
    if name not in _PROTOCOL:
        from specs import get_protocol
        _PROTOCOL[name] = get_protocol(name)
    return _PROTOCOL[name]


# ── 钩子 1: 力学响应 (NSGA-II 上千次评估 -> 用 surrogate, 快) ──
def predict_responses(thickness_m: Sequence[float], modulus_MPa: Sequence[float],
                      case: DesignCase) -> Dict[str, float]:
    sp = _surrogate(case.surrogate_ckpt)
    return sp.predict(thickness=list(thickness_m), modulus=list(modulus_MPa),
                      E_subgrade=case.E_subgrade, pavement_type=case.pavement_type)


# ── 钩子 2: 规范评估 -> DesignEvaluation (含 margins) ──
def compliance_eval(thickness_m: Sequence[float], modulus_MPa: Sequence[float],
                    case: DesignCase, responses: Dict[str, float]):
    from specs.protocol import DesignInputs
    proto = _protocol(case.protocol_name)
    inputs = DesignInputs(
        pavement_type=case.pavement_type,
        road_class=case.road_class,
        traffic_level=case.traffic_level,
        thickness=list(thickness_m),
        modulus=list(modulus_MPa),
        poisson=list(case.poisson),
        E_subgrade=case.E_subgrade,
        nu_subgrade=case.nu_subgrade,
        design_life=case.design_life,
        extras=dict(case.extras),
    )
    return proto.evaluate(inputs, responses)   # DesignEvaluation


# ── 钩子 3: 可行性 (Guard 边界) ──
def is_feasible(thickness_m: Sequence[float], modulus_MPa: Sequence[float],
                case: DesignCase) -> bool:
    from rl.guards import NumericalGuard, GuardViolation
    g = NumericalGuard(base_type=case.pavement_type)
    try:
        g.check_design(np.asarray(thickness_m, float),
                       np.asarray(modulus_MPa, float), float(case.E_subgrade))
        return True
    except GuardViolation:
        return False


# ── 统一评估 ──
def evaluate_design(x: Sequence[float], case: DesignCase = None,
                    provider=None) -> ObjResult:
    """x = [h1..h5(米), E1..E5(MPa)] (10 维)。"""
    if case is None:
        case = DesignCase()
    if provider is None:
        provider = get_provider("ec3", cache=True)
    x = list(x)
    thickness, modulus = x[:5], x[5:10]

    feasible = is_feasible(thickness, modulus, case)
    resp = predict_responses(thickness, modulus, case)
    ev = compliance_eval(thickness, modulus, case, resp)
    margins = {k: float(v) for k, v in ev.margins.items()}

    from rl.dsr_patch import compute_dsr
    dsr = compute_dsr(margins)

    lcc_d = lcc_for_design(thickness, modulus, margins,
                           pavement_type=case.pavement_type,
                           design_life_years_lcc=case.design_life_years_lcc,
                           discount_rate=case.discount_rate)
    gwp = lca_gwp_for_design(thickness, pavement_type=case.pavement_type,
                             provider=provider, schedule=lcc_d.get("schedule"))
    return ObjResult(dsr=dsr, feasible=feasible,
                     lcc=float(lcc_d["NPV_total_usd_m2"]), gwp=float(gwp),
                     margins=margins)


class Normalizer:
    """把 LCC/GWP 归一化到 [0,1] (在合规设计集上 fit)。归一化常数写进方法学。"""
    def __init__(self, lcc_min=None, lcc_max=None, gwp_min=None, gwp_max=None):
        self.lcc_min, self.lcc_max = lcc_min, lcc_max
        self.gwp_min, self.gwp_max = gwp_min, gwp_max

    def fit(self, results):
        lccs = [r.lcc for r in results]; gwps = [r.gwp for r in results]
        self.lcc_min, self.lcc_max = min(lccs), max(lccs)
        self.gwp_min, self.gwp_max = min(gwps), max(gwps)
        return self

    @staticmethod
    def _n(v, lo, hi):
        return 0.0 if (hi is None or lo is None or hi <= lo) else (v - lo) / (hi - lo)

    def lcc_norm(self, v): return self._n(v, self.lcc_min, self.lcc_max)
    def gwp_norm(self, v): return self._n(v, self.gwp_min, self.gwp_max)
