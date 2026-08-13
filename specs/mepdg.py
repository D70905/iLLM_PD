# -*- coding: utf-8 -*-
"""
specs.mepdg — ME-PDG (NCHRP 1-37A) Design Protocol — SIMPLIFIED
=================================================================

Implementation of (simplified):
    Mechanistic-Empirical Pavement Design Guide
    Based on NCHRP Report 1-37A (2004)
    Adopted by AASHTO as AASHTOWare Pavement ME Design (2008+)

★★★ THIS IS A SIMPLIFIED IMPLEMENTATION ★★★

Full ME-PDG (AASHTOWare Pavement ME) requires:
    1. Hourly climate data over 20-year design life
       (temperature, moisture, freeze-thaw cycles)
    2. Full axle load spectrum (axle type distribution, growth)
    3. Dynamic modulus master curve E*(f, T)
    4. Iterative damage accumulation across all seasons × temperatures
       via Miner's rule
    5. Multiple distress predictions: fatigue, rutting, IRI evolution,
       thermal cracking, top-down cracking

Our simplifications (each documented at point-of-use below):
    [S1] Climate:     single equivalent annual temperature
    [S2] Axles:       BZZ-100 single equivalent axle (for cross-spec
                      comparability with JTG D50-2017)
    [S3] Modulus:     single equivalent dynamic modulus at reference temp
    [S4] Damage:      single-pass evaluation at reference condition
                      (no Miner accumulation across seasons)
    [S5] Distresses:  bottom-up fatigue cracking + total rutting only
                      (NOT implementing: top-down, thermal, IRI)
    [S6] C-factor:    set to 1.0 (binder content data not in inputs)

Rationale for HARA framework demonstration:
    The HARA architecture's contribution is the AUDIT HARNESS, not the
    spec implementation. Full AASHTOWare Pavement ME calls would
    consume too much compute per RL step and require commercial license.
    Our simplification preserves the spec's MATHEMATICAL FORM and
    NATIONAL CALIBRATION COEFFICIENTS while making the per-step cost
    tractable. For production design, users should follow up with
    AASHTOWare Pavement ME.

★ COEFFICIENT VERIFICATION STATUS ★
    National calibration coefficients (β_f1, β_f2, β_f3, β_r1, β_r2)
    in specs/data/mepdg.json are VERIFIED against NCHRP 1-37A
    (widely cited; appear consistently across literature).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List

from specs.protocol import (
    DesignProtocol,
    DesignInputs,
    DesignEvaluation,
    margin_to_score,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_THIS_DIR, 'data', 'mepdg.json')


def _load_data(path: str = None) -> dict:
    p = path or _DEFAULT_DATA
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


class MEPDG_Simplified(DesignProtocol):
    """
    Simplified ME-PDG protocol — see module docstring for [S1]..[S6]
    simplifications.
    """

    name = "ME-PDG (NCHRP 1-37A, simplified)"
    citation = (
        "AASHTO. (2008-present). Mechanistic-Empirical Pavement Design Guide "
        "(MEPDG), originally NCHRP Report 1-37A (2004). "
        "Implemented commercially as AASHTOWare Pavement ME Design. "
        "★ This implementation uses simplified single-temperature, "
        "single-axle evaluation; see module docstring for details."
    )

    def __init__(self, data: dict = None, data_path: str = None):
        self.data = data if data is not None else _load_data(data_path)
        self._cfg = self.data

    # ─── Interface ───────────────────────────────────────────────

    def required_fea_outputs(self) -> List[str]:
        """
        ME-PDG needs:
          - ε_t at AC bottom (horizontal tensile strain) — bottom-up fatigue
          - ε_v at subgrade top (vertical compressive strain) — subgrade rutting

        Same physical positions as JTG D50-2017, so the same FEA call
        produces both spec's required outputs. Only the transfer functions
        differ.
        """
        return [
            'epsilon_a_microstrain',   # ε_t at AC bottom (same as JTG ε_a)
            'epsilon_z_microstrain',   # ε_v at subgrade top (same as JTG ε_z)
        ]

    # ─── Design ESAL ─────────────────────────────────────────────

    def _get_design_ESAL(self, inputs: DesignInputs) -> float:
        """
        Design ESAL over the design period.

        [S2] We use BZZ-100 equivalent axle (100 kN) instead of formal
        ESAL (80 kN, 18 kip), to maintain cross-spec comparability with
        JTG D50-2017. This introduces a ~25 % bias relative to formal
        ME-PDG; documented as simplification.
        """
        # Same lookup as JTG (BZZ-100 equivalent passes) for fair comparison
        Ne_table = {
            'light':       1.0e6,
            'medium':      8.0e6,
            'heavy':       3.0e7,
            'extra_heavy': 1.0e8,
        }
        base = Ne_table.get(inputs.traffic_level, 8.0e6)

        rc_factor = {
            'expressway':  1.5,
            'highway_1':   1.3,
            'highway_2':   1.0,
            'highway_3':   0.7,
            'highway_4':   0.5,
        }.get(inputs.road_class, 1.0)

        return base * rc_factor

    # ─── Bottom-up fatigue cracking ──────────────────────────────

    def _predict_fatigue_cracking_pct(
        self, inputs: DesignInputs, eps_t_microstrain: float,
    ) -> float:
        """
        Predict bottom-up fatigue cracking percentage at end of design life.

        NCHRP 1-37A Eq. 3.3.1-3.3.5:
          N_f = β_f1 · k1 · C · (1/ε_t)^β_f2 · (1/E)^β_f3
          Damage D = ESAL / N_f
          FC% = transfer_function(D)

        [S4] We use single-pass evaluation: damage is just ESAL/N_f at
        reference condition.
        [S6] C-factor = 1.0.

        Returns: predicted fatigue cracking percentage (% of lane area).
        """
        if eps_t_microstrain <= 0:
            return 0.0

        cfg = self._cfg['fatigue_cracking_model']
        coeffs = cfg['national_calibration_coefficients']
        beta_f1 = coeffs['beta_f1']
        beta_f2 = coeffs['beta_f2']
        beta_f3 = coeffs['beta_f3']

        # Thickness adjustment k1 (NCHRP 1-37A Eq. 3.3.4)
        h_AC_inch = inputs.thickness[0] * 39.3701   # m -> inch
        try:
            denom_k1 = 0.000398 + 0.003602 / (1.0 + math.exp(
                11.02 - 3.49 * h_AC_inch))
            k1 = 1.0 / denom_k1
        except OverflowError:
            k1 = 1.0
        k1 = max(0.1, min(k1, 100.0))

        # C-factor: [S6] simplification = 1.0
        C = 1.0

        # ε_t needs to be in absolute strain (in/in), and E in psi typically
        # ε_t microstrain → in/in: × 1e-6
        eps_t_abs = eps_t_microstrain * 1.0e-6
        E_AC_psi = inputs.modulus[0] * 145.038   # MPa -> psi

        # N_f
        try:
            N_f = (beta_f1 * k1 * C
                   * (1.0 / eps_t_abs) ** beta_f2
                   * (1.0 / E_AC_psi) ** beta_f3)
        except (OverflowError, ZeroDivisionError, ValueError):
            return 100.0  # severe failure

        # Damage = ESAL / N_f
        ESAL = self._get_design_ESAL(inputs)
        if N_f <= 0:
            return 100.0
        D = ESAL / N_f

        # [S4] Simplified transfer function: linear approximation
        # FC% ≈ min(100, D * 100)
        FC_pct = min(100.0, max(0.0, D * 100.0))
        return FC_pct

    # ─── Subgrade rutting (simplified) ───────────────────────────

    def _predict_rutting_mm(
        self, inputs: DesignInputs, eps_v_microstrain: float,
    ) -> float:
        """
        Predict total rutting depth at end of design life.

        Simplified — uses subgrade-only rutting contribution (largest
        component for typical flexible pavements).

        NCHRP 1-37A Section 3.3.7 (subgrade rutting, simplified form):
          ε_p / ε_v = β_r1 · ε_v^(β_r2) · N^(β_r3)
          RD_subgrade ≈ ε_p · h_influence

        [S1] Temperature effect on HMA rutting NOT included.
        [S4] Single-pass evaluation.

        Returns: predicted rutting depth in mm.
        """
        if eps_v_microstrain <= 0:
            return 0.0

        cfg = self._cfg['rutting_model']
        coeffs = cfg['national_calibration_coefficients_HMA']
        beta_r1 = coeffs['beta_r1']
        beta_r2 = coeffs['beta_r2']
        beta_r3 = coeffs['beta_r3']

        ESAL = self._get_design_ESAL(inputs)
        if ESAL <= 0:
            return 0.0

        eps_v_abs = eps_v_microstrain * 1.0e-6

        # Reference temperature in °F (for the transfer function)
        T_F = (
            inputs.extras.get('mean_annual_temp_C',
                              self._cfg['reference_climate']['default_MAT_C'])
            * 9.0 / 5.0 + 32.0
        )

        # Permanent strain ratio
        try:
            ratio = (10.0 ** beta_r1) * (T_F ** beta_r2) * (ESAL ** beta_r3)
        except (OverflowError, ValueError):
            return 50.0  # severe

        eps_p_subgrade = eps_v_abs * ratio

        # Approximate subgrade-influence depth ~ 0.5 m for typical pavement
        h_influence_mm = 500.0
        RD_subgrade_mm = eps_p_subgrade * h_influence_mm

        # Add a rough HMA rutting contribution (typically 1-5 mm)
        # Simplified: 30% of total
        RD_total_mm = RD_subgrade_mm / 0.7

        return max(0.0, min(RD_total_mm, 50.0))

    # ─── Performance criteria ────────────────────────────────────

    def _get_performance_limits(self, inputs: DesignInputs) -> Dict[str, float]:
        """
        Get performance limits by road class.
        """
        # Map iLLM-PD road_class to ME-PDG road class
        road_class_map = {
            'expressway':  'interstate',
            'highway_1':   'interstate',
            'highway_2':   'primary',
            'highway_3':   'primary',
            'highway_4':   'secondary',
        }
        mepdg_class = road_class_map.get(inputs.road_class, 'primary')
        tbl = self._cfg['performance_criteria_defaults']['by_road_class']
        return tbl.get(mepdg_class, tbl['primary'])

    # ─── Public methods ──────────────────────────────────────────

    def allowable_values(self, inputs: DesignInputs) -> Dict[str, float]:
        limits = self._get_performance_limits(inputs)
        return {
            'FC_pct_limit':         limits['fatigue_cracking_pct_max'],
            'RD_mm_limit':          limits['rutting_mm_max'],
            'IRI_limit_m_per_km':   limits['IRI_m_per_km_max'],
            'design_ESAL':          self._get_design_ESAL(inputs),
        }

    def evaluate(
        self, inputs: DesignInputs, fea_outputs: Dict[str, float],
    ) -> DesignEvaluation:
        eps_t = fea_outputs.get('epsilon_a_microstrain', None)
        eps_v = fea_outputs.get('epsilon_z_microstrain', None)

        if eps_t is None or eps_v is None:
            return DesignEvaluation(
                feasible=False, margins={}, responses={},
                allowable_values=self.allowable_values(inputs),
                critical_indicator='NONE',
                spec_name=self.name,
                details={'error': 'Missing required FEA outputs'},
            )

        # Predict distresses
        FC_pct = self._predict_fatigue_cracking_pct(inputs, eps_t)
        RD_mm = self._predict_rutting_mm(inputs, eps_v)

        # Limits
        limits = self._get_performance_limits(inputs)
        FC_limit = limits['fatigue_cracking_pct_max']
        RD_limit = limits['rutting_mm_max']

        # Margins (capacity/demand): for distresses, margin = limit / predicted
        # Larger predicted distress => smaller margin => closer to failing.
        margins = {
            'fatigue_cracking':   FC_limit / max(FC_pct, 0.01),
            'total_rutting':      RD_limit / max(RD_mm,  0.01),
        }

        feasible = (FC_pct <= FC_limit) and (RD_mm <= RD_limit)
        critical = min(margins, key=margins.get)

        return DesignEvaluation(
            feasible=feasible,
            margins=margins,
            responses={
                'epsilon_a_microstrain':   eps_t,
                'epsilon_z_microstrain':   eps_v,
                'predicted_FC_pct':        FC_pct,
                'predicted_RD_mm':         RD_mm,
            },
            allowable_values={
                'FC_pct_limit':            FC_limit,
                'RD_mm_limit':             RD_limit,
            },
            critical_indicator=critical,
            spec_name=self.name,
            details={
                'design_ESAL':             self._get_design_ESAL(inputs),
                'simplifications_applied': (
                    self._cfg['_metadata']['simplifications_applied']
                ),
            },
        )

    def reward_components(
        self, evaluation: DesignEvaluation,
    ) -> Dict[str, float]:
        if not evaluation.margins:
            return {'performance': 0.0, 'feasibility': 0.0}

        per_indicator = {
            k: margin_to_score(v) for k, v in evaluation.margins.items()
        }
        return {
            'performance':       sum(per_indicator.values()) / len(per_indicator),
            'feasibility':       1.0 if evaluation.feasible else 0.0,
            'critical_margin':   min(evaluation.margins.values()),
            **{('per_' + k): v for k, v in per_indicator.items()},
        }
