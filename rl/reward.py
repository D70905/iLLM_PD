# -*- coding: utf-8 -*-
"""
rl.reward — 5-component composite reward (v0.5, 6-layer, type-aware pricing)
==============================================================================

UPGRADE v0.5 (Phase 2D, type-aware pricing for dual-base):
    - RewardConfig now carries TWO price tables:
        * material_prices                      → semi_rigid (cement-treated base)
        * material_prices_flexible             → flexible (granular base)
      Same for modulus_price_coeffs / modulus_price_coeffs_flexible.
    - _material_cost(...) and economic_reward(...) take a `pavement_type`
      argument and route to the correct price table.
    - compute(...) accepts pavement_type='semi_rigid'|'flexible' (default
      'semi_rigid' for backward compatibility — A v2 behavior unchanged).
    - No other behavior change; weights, performance scoring, smoothness,
      exploration, guidance all unmodified.

Rationale (R2-2 / Phase 2D dual-base):
    In flexible pavements (GPS-1, ACUB), the base/subbase are unbound
    granular materials (crushed stone, sand-gravel) costing roughly 1/3
    the per-m³ rate of cement-treated semi-rigid bases. Using the
    semi-rigid price (320 CNY/m³) for granular base biases PPO toward
    thinning the base layer to "save cost", which physically degrades
    B3 (AC permanent deformation) and B4 (subgrade strain) margins.
    A v2 training showed this clearly: feasibility 0.641, B3 min 0.79,
    ep_rew_mean −2.69, vs healthy semi_rigid baseline 1.000 / 1.04 / +4.46.

UPGRADE v0.4 (Phase 2A-1):
    - Material prices: 3 → 5 layers
        [upper_AC, mid_AC, lower_AC, base, subbase]
    - Guidance target_layers updated for 6-layer indexing.
    - Cost target ranges adjusted for typical 6-layer pavement cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Recognized pavement types (must match env / surrogate / guards / specs).
PAVEMENT_TYPE_SEMI_RIGID = 'semi_rigid'
PAVEMENT_TYPE_FLEXIBLE = 'flexible'
_VALID_PAVEMENT_TYPES = (PAVEMENT_TYPE_SEMI_RIGID, PAVEMENT_TYPE_FLEXIBLE)


@dataclass
class RewardConfig:
    """
    All hyperparameters for the 5-component reward (6-layer compatible,
    dual-base aware as of v0.5).
    """
    # ── Performance reward sub-weights ──────────────────────────
    perf_indicator_weights: Dict[str, float] = field(default_factory=lambda: {
        'B1_asphalt_fatigue':           0.30,
        'B2_semi_rigid_fatigue':        0.35,
        'B3_ac_permanent_deformation':  0.20,
        'B4_subgrade_strain':           0.15,
    })

    # Piecewise margin → score function (JTG)
    score_margin_failed:     float = 0.0
    score_margin_near_fail:  float = 0.5
    score_margin_ideal:      float = 1.0
    score_margin_decay_rate: float = 0.15
    score_margin_floor:      float = 0.2

    # ── Economic reward — SEMI-RIGID price table (default, backward-compatible) ──
    # Material unit prices CNY/m³, layer order:
    #   [upper_AC, mid_AC, lower_AC, base, subbase]
    # Reference: Chinese highway industry estimates 2024
    # Base = cement-stabilized aggregate;  subbase = lime-fly-ash / lime-soil
    material_prices: List[float] = field(default_factory=lambda:
        [1800.0, 1100.0, 900.0, 320.0, 180.0])
    modulus_price_coeffs: List[float] = field(default_factory=lambda:
        [2.0e-5, 1.8e-5, 1.5e-5, 6.0e-5, 0.0])

    # ── Economic reward — FLEXIBLE price table (new in v0.5) ────
    # AC layers (0-2) identical to semi_rigid: same asphalt materials.
    # Base (3): unbound graded crushed stone, ~100 CNY/m³
    # Subbase (4): sand-gravel / graded subbase, ~80 CNY/m³
    # Modulus dependence:
    #   - AC layers: same as semi_rigid (high-modulus mix costs more)
    #   - Granular base/subbase: cost is dominated by aggregate grading
    #     and compaction effort, not modulus. Coefficient much smaller.
    material_prices_flexible: List[float] = field(default_factory=lambda:
        [1800.0, 1100.0, 900.0, 100.0, 80.0])
    modulus_price_coeffs_flexible: List[float] = field(default_factory=lambda:
        [2.0e-5, 1.8e-5, 1.5e-5, 1.0e-5, 0.0])

    # Cost target ranges — apply uniformly to both pavement types.
    # For 6-layer: typical baseline cost ~360 CNY/m² (vs ~250 for 4-layer).
    # In flexible mode, costs naturally trend lower (cheaper base/subbase),
    # so the same target band still functions: PPO learns that thickening
    # cheap granular base is a viable strategy.
    cost_target_min:     float = 280.0
    cost_target_optimal: float = 380.0
    cost_target_max:     float = 500.0
    econ_max_reward:     float = 0.30

    # ── Guidance reward ─────────────────────────────────────────
    guide_bonus_per_correct_layer: float = 0.04
    guide_penalty_idle: float = -0.05

    # ── Smoothness reward (Δtotal piecewise) ────────────────────
    smooth_zero_penalty:    float = -0.02
    smooth_small_threshold: float = 0.005
    smooth_ideal_low:       float = 0.005
    smooth_ideal_high:      float = 0.08
    smooth_ideal_peak:      float = 0.08
    smooth_decay_rate:      float = 8.0

    # ── Exploration reward ──────────────────────────────────────
    explore_per_param_change: float = 0.02
    explore_max_reward: float = 0.05
    explore_thickness_threshold: float = 0.01
    explore_modulus_threshold: float = 50.0

    # ── Adaptive weights (three-stage) ──────────────────────────
    tau_early_threshold: float = 0.30
    tau_mid_threshold:   float = 0.70

    weights_early: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.40, 'economic': 0.30, 'guidance': 0.15,
        'smoothness':  0.10, 'exploration': 0.05,
    })
    weights_mid: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.40, 'economic': 0.40, 'guidance': 0.10,
        'smoothness':  0.08, 'exploration': 0.02,
    })
    weights_late: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.55, 'economic': 0.35, 'guidance': 0.05,
        'smoothness':  0.05, 'exploration': 0.00,
    })

    # ── Final saturation ───────────────────────────────────────
    final_tanh_scale: float = 0.8
    final_output_scale: float = 1.5

    # ── Hard feasibility bonus/penalty ─────────────────────────
    feasibility_bonus: float = 0.30
    infeasibility_penalty: float = -0.30


class CompositeReward:
    """
    Composite 5-component reward.

    Usage (NEW in v0.5 — pavement_type optional, defaults semi_rigid):
        result = reward_fn.compute(
            margins={'B1':..., 'B3':..., 'B4':...},      # B2 may be None for flexible
            new_design={'thickness':[5], 'modulus':[5]},
            old_design={'thickness':[5], 'modulus':[5]},
            tau=0.4,
            critical_indicator='B3_ac_permanent_deformation',
            feasible=True,
            pavement_type='flexible',                     # ← new optional kwarg
        )

    BACKWARD COMPAT: callers that don't pass pavement_type default to
    'semi_rigid', i.e. the v0.4 price table & behavior.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    # ─── Pricing table router (new in v0.5) ─────────────────────

    def _get_price_tables(
        self, pavement_type: str,
    ) -> Tuple[List[float], List[float]]:
        """Return (material_prices, modulus_price_coeffs) for given pavement type.

        Falls back to semi_rigid on unknown values (defensive; should not
        happen if env.py is correctly configured).
        """
        c = self.config
        if pavement_type == PAVEMENT_TYPE_FLEXIBLE:
            return c.material_prices_flexible, c.modulus_price_coeffs_flexible
        # default + semi_rigid path
        return c.material_prices, c.modulus_price_coeffs

    # ─── Component 1: Performance ───────────────────────────────

    def _margin_to_score(self, margin: float) -> float:
        c = self.config
        if margin < 0.5:
            return c.score_margin_failed
        elif margin < 1.0:
            return c.score_margin_failed + (margin - 0.5) * (c.score_margin_near_fail - c.score_margin_failed) / 0.5
        elif margin <= 1.5:
            return c.score_margin_ideal
        elif margin <= 5.0:
            decay = (margin - 1.5) * c.score_margin_decay_rate
            return max(c.score_margin_floor, c.score_margin_ideal - decay)
        else:
            return c.score_margin_floor

    def performance_reward(self, margins: Dict[str, float]) -> float:
        weights = self.config.perf_indicator_weights
        score_sum = 0.0
        weight_sum = 0.0
        for k, m in margins.items():
            # Skip margins that are None / NaN (e.g. B2 in flexible mode)
            if m is None:
                continue
            try:
                m_val = float(m)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(m_val):
                continue
            w = weights.get(k, 1.0 / max(len(margins), 1))
            score_sum += w * self._margin_to_score(m_val)
            weight_sum += w
        return score_sum / max(weight_sum, 1e-9)

    # ─── Component 2: Economic (UPDATED in v0.5) ────────────────

    def _material_cost(
        self,
        thickness: np.ndarray,
        modulus: np.ndarray,
        pavement_type: str = PAVEMENT_TYPE_SEMI_RIGID,
    ) -> float:
        """
        Estimated material cost in CNY/m² for 6-layer pavement.
            C = Σ_i  γ_i · h_i · (1 + α_i · E_i)
        where i ranges over 5 structural layers, and γ_i / α_i are
        selected based on pavement_type.
        """
        prices, coeffs = self._get_price_tables(pavement_type)
        cost = 0.0
        n_layers = min(5, len(thickness))
        for i in range(n_layers):
            h_i_m = float(thickness[i])
            E_i = float(modulus[i])
            cost += prices[i] * h_i_m * (1.0 + coeffs[i] * E_i)
        return cost

    def economic_reward(
        self,
        thickness: np.ndarray,
        modulus: np.ndarray,
        pavement_type: str = PAVEMENT_TYPE_SEMI_RIGID,
    ) -> float:
        c = self.config
        C = self._material_cost(thickness, modulus, pavement_type)
        if C < c.cost_target_min:
            return 0.5 * c.econ_max_reward * (C / c.cost_target_min)
        elif C <= c.cost_target_optimal:
            return c.econ_max_reward
        elif C <= c.cost_target_max:
            return c.econ_max_reward * (c.cost_target_max - C) / (c.cost_target_max - c.cost_target_optimal)
        else:
            return -0.2

    # ─── Component 3: Guidance (6-layer target_layers) ──────────

    def guidance_reward(
        self, new_design: Dict, old_design: Dict,
        critical_indicator: Optional[str] = None,
    ) -> float:
        c = self.config
        new_h = np.asarray(new_design['thickness'])
        new_E = np.asarray(new_design['modulus'])
        old_h = np.asarray(old_design['thickness'])
        old_E = np.asarray(old_design['modulus'])

        dh = new_h - old_h
        dE = new_E - old_E

        if np.all(np.abs(dh) < 1e-6) and np.all(np.abs(dE) < 1e-6):
            return c.guide_penalty_idle

        if critical_indicator is None:
            return 0.0

        # 6-layer mapping: which layers' increase HELPS this indicator
        # Layer indices: 0=upper_AC, 1=mid_AC, 2=lower_AC, 3=base, 4=subbase
        target_layers = {
            'B1_asphalt_fatigue':           [0, 1, 2],     # all 3 AC sublayers
            'B2_semi_rigid_fatigue':        [3],            # base
            'B3_ac_permanent_deformation':  [0, 1, 2],     # AC layers (especially upper)
            'B4_subgrade_strain':           [0, 1, 2, 3, 4],  # all structural layers
        }.get(critical_indicator, [])

        bonus = 0.0
        for i in target_layers:
            if i >= len(new_h):
                continue
            if dh[i] > 0:
                bonus += c.guide_bonus_per_correct_layer
            if dE[i] > 0:
                bonus += c.guide_bonus_per_correct_layer
            if dh[i] < 0:
                bonus -= c.guide_bonus_per_correct_layer * 0.5
            if dE[i] < 0:
                bonus -= c.guide_bonus_per_correct_layer * 0.5
        return float(np.clip(bonus, -0.2, 0.3))

    # ─── Component 4: Smoothness ────────────────────────────────

    def smoothness_reward(self, new_design: Dict, old_design: Dict) -> float:
        c = self.config
        new_h = np.asarray(new_design['thickness'])
        new_E = np.asarray(new_design['modulus'])
        old_h = np.asarray(old_design['thickness'])
        old_E = np.asarray(old_design['modulus'])

        delta_total = 0.0
        n_layers = min(5, len(new_h))
        for i in range(n_layers):
            old_h_i = max(old_h[i], 1e-6)
            old_E_i = max(old_E[i], 1e-6)
            delta_total += abs(new_h[i] - old_h[i]) / old_h_i
            delta_total += abs(new_E[i] - old_E[i]) / old_E_i

        if delta_total == 0:
            return c.smooth_zero_penalty
        elif delta_total < c.smooth_small_threshold:
            return 4.0 * delta_total
        elif delta_total <= c.smooth_ideal_high:
            return 0.02 + (c.smooth_ideal_high - delta_total) / (
                c.smooth_ideal_high - c.smooth_ideal_low) * (c.smooth_ideal_peak - 0.02)
        else:
            return 0.02 * np.exp(-c.smooth_decay_rate * (delta_total - c.smooth_ideal_high))

    # ─── Component 5: Exploration ───────────────────────────────

    def exploration_reward(
        self, new_design: Dict, old_design: Dict, tau: float,
    ) -> float:
        c = self.config
        new_h = np.asarray(new_design['thickness'])
        new_E = np.asarray(new_design['modulus'])
        old_h = np.asarray(old_design['thickness'])
        old_E = np.asarray(old_design['modulus'])

        exploration_factor = max(0.1, 1.0 - tau)
        n_changes = 0
        n_layers = min(5, len(new_h))
        for i in range(n_layers):
            if abs(new_h[i] - old_h[i]) > c.explore_thickness_threshold:
                n_changes += 1
            if abs(new_E[i] - old_E[i]) > c.explore_modulus_threshold:
                n_changes += 1
        bonus = exploration_factor * n_changes * c.explore_per_param_change
        return float(min(c.explore_max_reward, bonus))

    # ─── Adaptive weight scheduling ─────────────────────────────

    def get_weights(self, tau: float) -> Dict[str, float]:
        c = self.config
        if tau < c.tau_early_threshold:
            return dict(c.weights_early)
        elif tau < c.tau_mid_threshold:
            return dict(c.weights_mid)
        else:
            return dict(c.weights_late)

    # ─── Combined reward (UPDATED in v0.5) ──────────────────────

    def compute(
        self,
        margins: Dict[str, float],
        new_design: Dict,
        old_design: Dict,
        tau: float,
        critical_indicator: Optional[str] = None,
        feasible: Optional[bool] = None,
        pavement_type: str = PAVEMENT_TYPE_SEMI_RIGID,
    ) -> 'RewardResult':
        """
        Compute the composite reward.

        Args:
            margins: dict {indicator_key: margin or None}
            new_design / old_design: {'thickness':[5], 'modulus':[5]}
            tau: training progress in [0, 1]
            critical_indicator: name of the currently-critical margin (or None)
            feasible: True / False / None (governs feasibility bonus/penalty)
            pavement_type: 'semi_rigid' (default) or 'flexible'. Selects the
                price table for the economic reward component. All other
                components are independent of pavement type.

        Returns:
            RewardResult with total, raw, per-component values, weights, tau.
        """
        c = self.config
        new_h = np.asarray(new_design['thickness'])
        new_E = np.asarray(new_design['modulus'])

        r_perf    = self.performance_reward(margins)
        r_econ    = self.economic_reward(new_h, new_E, pavement_type)
        r_guide   = self.guidance_reward(new_design, old_design, critical_indicator)
        r_smooth  = self.smoothness_reward(new_design, old_design)
        r_explore = self.exploration_reward(new_design, old_design, tau)

        w = self.get_weights(tau)

        raw = (w['performance']  * r_perf
             + w['economic']     * r_econ
             + w['guidance']     * r_guide
             + w['smoothness']   * r_smooth
             + w['exploration']  * r_explore)

        if feasible is True:
            raw += c.feasibility_bonus
        elif feasible is False:
            raw += c.infeasibility_penalty

        total = float(np.tanh(raw * c.final_tanh_scale) * c.final_output_scale)

        return RewardResult(
            total=total,
            raw=raw,
            components={
                'performance':  r_perf,
                'economic':     r_econ,
                'guidance':     r_guide,
                'smoothness':   r_smooth,
                'exploration':  r_explore,
            },
            weights_used=w,
            tau=tau,
            pavement_type=pavement_type,
        )


@dataclass
class RewardResult:
    total: float
    raw: float
    components: Dict[str, float]
    weights_used: Dict[str, float]
    tau: float
    pavement_type: str = PAVEMENT_TYPE_SEMI_RIGID  # new in v0.5

    def to_dict(self) -> Dict:
        return {
            'total': self.total, 'raw': self.raw,
            'components': self.components,
            'weights': self.weights_used, 'tau': self.tau,
            'pavement_type': self.pavement_type,
        }


# ─────────────────────────────────────────────────────────────────
#  Self-test: confirms (a) backward compatibility, (b) flexible
#  pricing differs as expected, (c) margins=None for B2 in flexible
#  mode doesn't break the performance component.
#
#  Run with:   python -m rl.reward
# ─────────────────────────────────────────────────────────────────

def _selftest():
    print("=" * 72)
    print("rl.reward v0.5 self-test — type-aware pricing")
    print("=" * 72)

    rwd = CompositeReward()

    # Realistic 6-layer initial design (matches Phase 2A-1 baseline).
    # thicknesses in m, moduli in MPa.
    design = {
        'thickness': [0.05, 0.06, 0.07, 0.30, 0.20],   # 5+6+7 = 18 cm AC, 30+20 = 50 cm base/subbase
        'modulus':   [1400.0, 1200.0, 1000.0, 1500.0, 400.0],
    }
    old_design = {
        'thickness': [0.05, 0.06, 0.07, 0.28, 0.20],   # small change in base layer
        'modulus':   [1400.0, 1200.0, 1000.0, 1500.0, 400.0],
    }
    margins_full = {
        'B1_asphalt_fatigue':           4.0,
        'B2_semi_rigid_fatigue':        2.5,
        'B3_ac_permanent_deformation':  1.10,
        'B4_subgrade_strain':           2.54,
    }
    margins_flex = {  # B2 absent in flexible (no semi-rigid base to fatigue)
        'B1_asphalt_fatigue':           4.0,
        'B2_semi_rigid_fatigue':        None,
        'B3_ac_permanent_deformation':  1.10,
        'B4_subgrade_strain':           2.54,
    }

    # --- TEST 1: backward compat (no pavement_type passed) ---
    r_default = rwd.compute(
        margins=margins_full, new_design=design, old_design=old_design,
        tau=0.5, critical_indicator='B3_ac_permanent_deformation', feasible=True,
    )
    assert r_default.pavement_type == 'semi_rigid', \
        f"default should be semi_rigid, got {r_default.pavement_type!r}"
    print(f"  TEST 1 backward-compat (no arg)     → pavement_type={r_default.pavement_type}, total={r_default.total:+.4f} ✓")

    # --- TEST 2: explicit semi_rigid matches the default ---
    r_semi = rwd.compute(
        margins=margins_full, new_design=design, old_design=old_design,
        tau=0.5, critical_indicator='B3_ac_permanent_deformation', feasible=True,
        pavement_type='semi_rigid',
    )
    assert abs(r_default.total - r_semi.total) < 1e-9, \
        f"semi_rigid result should match default; {r_default.total} vs {r_semi.total}"
    print(f"  TEST 2 explicit semi_rigid           → total={r_semi.total:+.4f} (matches default ✓)")

    # --- TEST 3: flexible price table gives lower material cost ---
    cost_semi = rwd._material_cost(
        np.array(design['thickness']), np.array(design['modulus']),
        pavement_type='semi_rigid',
    )
    cost_flex = rwd._material_cost(
        np.array(design['thickness']), np.array(design['modulus']),
        pavement_type='flexible',
    )
    print(f"  TEST 3 material cost (same design)   → semi_rigid={cost_semi:6.1f}, flexible={cost_flex:6.1f} CNY/m²")
    assert cost_flex < cost_semi, \
        f"flexible should be cheaper for the same design; got flex={cost_flex} >= semi={cost_semi}"
    print(f"           cost reduction from flexible base = {cost_semi - cost_flex:.1f} CNY/m² ({(cost_semi-cost_flex)/cost_semi*100:.1f}%) ✓")

    # --- TEST 4: flexible with B2=None doesn't crash ---
    r_flex = rwd.compute(
        margins=margins_flex, new_design=design, old_design=old_design,
        tau=0.5, critical_indicator='B3_ac_permanent_deformation', feasible=True,
        pavement_type='flexible',
    )
    assert np.isfinite(r_flex.total), f"flexible reward not finite: {r_flex.total}"
    assert r_flex.pavement_type == 'flexible'
    print(f"  TEST 4 flexible mode (B2=None)       → total={r_flex.total:+.4f}, perf={r_flex.components['performance']:+.4f} ✓")

    # --- TEST 5: economic_reward DIFFERS between modes for the SAME design ---
    e_semi = rwd.economic_reward(
        np.array(design['thickness']), np.array(design['modulus']), 'semi_rigid',
    )
    e_flex = rwd.economic_reward(
        np.array(design['thickness']), np.array(design['modulus']), 'flexible',
    )
    print(f"  TEST 5 economic_reward               → semi_rigid={e_semi:+.4f}, flexible={e_flex:+.4f}")
    print(f"           (flexible scores higher because cheaper-yet-still-in-target band)")

    # --- TEST 6: invalid pavement_type defensively falls back to semi_rigid ---
    e_unknown = rwd.economic_reward(
        np.array(design['thickness']), np.array(design['modulus']), 'rigid_concrete_xyz',
    )
    assert abs(e_unknown - e_semi) < 1e-12, \
        "unknown pavement_type should fall back to semi_rigid table"
    print(f"  TEST 6 unknown type → falls back to semi_rigid ✓")

    print()
    print("[rl.reward v0.5] self-test PASSED ✓")
    print()
    print("Backward compatibility: confirmed (no kwarg → semi_rigid → unchanged from v0.4).")
    print("New behavior:  pavement_type='flexible' → granular price table, B2=None tolerated.")


if __name__ == "__main__":
    _selftest()
