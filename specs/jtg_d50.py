# -*- coding: utf-8 -*-
"""
specs.jtg_d50 — JTG D50-2017 Design Protocol — 6-layer compatible (v0.4)
==========================================================================

UPGRADE v0.4 (Phase 2A-1):
    - thickness/modulus arrays now length 5 (top-down):
        [0] Upper AC
        [1] Mid AC
        [2] Lower AC
        [3] Base
        [4] Subbase
    - AC properties accessed via helper methods (no direct indexing):
        _h_AC_total_mm()             — total AC thickness in mm
        _E_AC_equivalent_MPa()       — thickness-weighted AC modulus
        _h_base_mm()                 — base layer thickness in mm
    - B.3 permanent deformation: NEW multi-sublayer implementation per
      JTG B.3.1 (3 AC sublayers). Falls back to single-sublayer if
      FEA does not provide p_AC_{upper,mid,lower}_mid_MPa.
    - All other JTG equations (B.1 fatigue, B.2 semi-rigid, B.4 subgrade)
      unchanged in formula, only thickness/modulus accessors updated.

Implementation of:
    《公路沥青路面设计规范》JTG D50-2017

All formulas cross-referenced against regulation PDF. See SPECS_VERIFICATION.md
for compliance checklist.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

from specs.protocol import (
    DesignProtocol,
    DesignInputs,
    DesignEvaluation,
    margin_to_score,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_THIS_DIR, 'data', 'jtg_d50.json')


def _load_data(path: str = None) -> dict:
    p = path or _DEFAULT_DATA
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


class JTG_D50_2017(DesignProtocol):
    """
    JTG D50-2017 design protocol (China, current).

    Input layer convention (6-layer model, top-down):
        thickness[0], modulus[0]: Upper AC (SMA-13)
        thickness[1], modulus[1]: Mid AC (AC-20)
        thickness[2], modulus[2]: Lower AC (AC-25)
        thickness[3], modulus[3]: Base (cement-stabilized aggregate)
        thickness[4], modulus[4]: Subbase (graded aggregate)
        E_subgrade, nu_subgrade: scalar subgrade properties

    DesignInputs.extras keys consumed (same as v0.3 — unchanged):
        ac_grade           : 'modified_asphalt_SBS' | 'neat_asphalt'
        base_type          : 'inorganic_stabilized_granular' | '_soil'
        construction_type  : 'new_construction' | 'rehabilitation_overlay'
        frost_zone         : 'non_frost' (default) | 'heavy_frost' | ...
        city               : 'beijing' | 'shanghai' | ... (Pinyin lowercase)
        climate_zone       : 'cold' | 'temperate' (default) | ... (fallback)
        VFA_pct            : default 70 (% voids filled with asphalt)
        R_s_MPa            : default 0.8 (semi-rigid flexural strength)
        R_0_mm             : default 1.5 (lab rutting test result)
    """

    name = "JTG D50-2017"
    citation = (
        "中华人民共和国交通运输部. 公路沥青路面设计规范 (JTG D50-2017). "
        "Specifications for Design of Highway Asphalt Pavement. "
        "Beijing: 人民交通出版社, 2017. "
        "Issued 2017-03-20, effective 2017-09-01."
    )

    # Number of AC sublayers (Phase 2A-1: 3, top-down indexes 0/1/2)
    N_AC_SUBLAYERS = 3
    # Layer indices in thickness/modulus arrays
    IDX_UPPER_AC = 0
    IDX_MID_AC   = 1
    IDX_LOWER_AC = 2
    IDX_BASE     = 3
    IDX_SUBBASE  = 4

    def __init__(self, data: dict = None, data_path: str = None):
        self.data = data if data is not None else _load_data(data_path)

    # ─────────────────────────────────────────────────────────────
    # Interface
    # ─────────────────────────────────────────────────────────────

    def required_fea_outputs(self) -> List[str]:
        return [
            'epsilon_a_microstrain',
            'sigma_t_MPa',
            'epsilon_z_microstrain',
            # Optional (for B.3 multi-sublayer):
            'p_AC_upper_mid_MPa',
            'p_AC_mid_mid_MPa',
            'p_AC_lower_mid_MPa',
        ]

    # ─────────────────────────────────────────────────────────────
    # Layer accessor helpers (6-layer model)
    # ─────────────────────────────────────────────────────────────

    def _h_AC_total_mm(self, inputs: DesignInputs) -> float:
        """Total AC thickness (sum of 3 AC sublayers) in mm."""
        return sum(inputs.thickness[i] for i in
                   (self.IDX_UPPER_AC, self.IDX_MID_AC, self.IDX_LOWER_AC)) * 1000.0

    def _E_AC_equivalent_MPa(self, inputs: DesignInputs) -> float:
        """Thickness-weighted equivalent AC modulus (MPa)."""
        h_list = [inputs.thickness[i] for i in
                  (self.IDX_UPPER_AC, self.IDX_MID_AC, self.IDX_LOWER_AC)]
        E_list = [inputs.modulus[i] for i in
                  (self.IDX_UPPER_AC, self.IDX_MID_AC, self.IDX_LOWER_AC)]
        h_total = sum(h_list)
        if h_total <= 0:
            return 0.0
        return sum(h * E for h, E in zip(h_list, E_list)) / h_total

    def _h_base_mm(self, inputs: DesignInputs) -> float:
        """Base layer thickness in mm."""
        return float(inputs.thickness[self.IDX_BASE]) * 1000.0

    def _h_AC_sublayer_mm(self, inputs: DesignInputs, sublayer: str) -> float:
        """Individual AC sublayer thickness in mm. sublayer in {'upper','mid','lower'}."""
        idx = {'upper': self.IDX_UPPER_AC,
               'mid':   self.IDX_MID_AC,
               'lower': self.IDX_LOWER_AC}[sublayer]
        return float(inputs.thickness[idx]) * 1000.0

    # ─────────────────────────────────────────────────────────────
    # Section 3 — Basic design parameters
    # ─────────────────────────────────────────────────────────────

    def _beta(self, inputs: DesignInputs) -> float:
        """Section 3.0.1, Table 3.0.1 — target reliability index β."""
        tbl = self.data['section_3_0_1_target_reliability']['by_road_class']
        return float(tbl.get(inputs.road_class, tbl['highway_2'])['beta'])

    def _design_life(self, inputs: DesignInputs) -> int:
        """Section 3.0.2, Table 3.0.2 — design life in years."""
        tbl = self.data['section_3_0_2_design_life_years']['by_road_class']
        return int(tbl.get(inputs.road_class, tbl['highway_2']))

    def _traffic_growth_factor(self, years: int, growth_rate: float) -> float:
        """Cumulative multiplier for an initial annual traffic value."""
        if years <= 0:
            return 0.0
        if abs(growth_rate) < 1e-12:
            return float(years)
        try:
            return (((1.0 + growth_rate) ** years) - 1.0) / growth_rate
        except (OverflowError, ZeroDivisionError, ValueError):
            return float(years)

    def _lane_distribution_factor(self, inputs: DesignInputs) -> Optional[float]:
        """Return LDF from explicit input or JTG Table A.2.5 defaults."""
        extras = inputs.extras or {}
        if extras.get('LDF') is not None:
            try:
                return float(extras['LDF'])
            except (TypeError, ValueError):
                return None

        lanes = extras.get('design_lane_count', extras.get('lane_count'))
        if lanes is None:
            return None
        try:
            n_lanes = int(lanes)
        except (TypeError, ValueError):
            return None

        table = (self.data['appendix_A_traffic_loading']
                 ['A25_lane_factor_LDF_table'])
        defaults = table.get('default_typical_by_lanes', {})
        if n_lanes <= 1:
            key = '1'
        elif n_lanes == 2:
            key = '2'
        elif n_lanes == 3:
            key = '3'
        else:
            key = '4+'
        return float(defaults.get(key, 0.75))

    def _indicator_ealf_prefix(self, indicator: str) -> str:
        """Map JTG indicator to the EALF table column prefix."""
        if indicator == 'semi_rigid_fatigue':
            return 'semi_rigid'
        if indicator == 'subgrade_strain':
            return 'subgrade'
        return 'asphalt'

    def _vehicle_distribution_pct(self, inputs: DesignInputs) -> Optional[Dict[str, float]]:
        """Return vehicle-class distribution percentages for classes 2..11."""
        extras = inputs.extras or {}
        direct = (extras.get('VCDF_by_vehicle_class_pct')
                  or extras.get('vehicle_class_distribution_pct')
                  or extras.get('VCDF_pct'))
        if isinstance(direct, dict):
            out = {}
            for k, v in direct.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            return out if out else None

        ttc = extras.get('TTC_class', extras.get('traffic_class_TTC'))
        if ttc is None:
            return None
        ttc_key = str(ttc).upper().replace('-', '')
        if ttc_key.startswith('TTC'):
            ttc_key = 'TTC' + ttc_key.replace('TTC', '')
        else:
            ttc_key = 'TTC' + ttc_key

        vcdf_root = (self.data['appendix_A_traffic_loading']
                     ['A26_TTC_classification']
                     ['VCDF_by_TTC_and_vehicle_class_pct'])
        row = vcdf_root.get(ttc_key)
        if not isinstance(row, dict):
            return None
        return {str(k): float(v) for k, v in row.items()
                if not str(k).startswith('_')}

    def _empty_full_share(self, vehicle_class: str) -> tuple:
        """Return typical empty/full shares from Table A.3.1-2."""
        ratios = (self.data['appendix_A_traffic_loading']
                  ['table_A312_empty_full_ratios']
                  ['by_vehicle_class'])
        row = ratios.get(str(vehicle_class), {})

        def mid(name, default):
            val = row.get(name)
            if isinstance(val, list) and len(val) == 2:
                return 0.5 * (float(val[0]) + float(val[1]))
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        empty = mid('empty', 0.5)
        full = mid('full', max(0.0, 1.0 - empty))
        total = empty + full
        if total <= 0:
            return 0.5, 0.5
        return empty / total, full / total

    def _ealf_for_vehicle_class(self, indicator: str, vehicle_class: str) -> float:
        """Return load-state weighted EALF from Table A.3.1-3."""
        table = (self.data['appendix_A_traffic_loading']
                 ['table_A313_EALF_per_vehicle_class']
                 ['data'])
        row = table.get(str(vehicle_class))
        if not isinstance(row, dict):
            return 0.0
        prefix = self._indicator_ealf_prefix(indicator)
        empty_share, full_share = self._empty_full_share(str(vehicle_class))
        empty = float(row.get(prefix + '_empty', 0.0))
        full = float(row.get(prefix + '_full', 0.0))
        return empty_share * empty + full_share * full

    def _appendix_a_Ne(self, inputs: DesignInputs, indicator: str) -> Optional[float]:
        """
        Full JTG Appendix A path when traffic inputs are available.

        Required extras:
            AADTT: two-way annual average daily traffic of 2+ axle,
                   6+ wheel vehicles, vehicles/day.
            DDF: directional distribution factor.
            LDF or design_lane_count/lane_count: design-lane factor.
            TTC_class or VCDF_by_vehicle_class_pct: vehicle-class mix.

        Formula:
            N_1 = AADTT * DDF * LDF * sum(VCDF_m * EALF_m)
            N_e = growth_factor(t, gamma) * 365 * N_1
        """
        extras = inputs.extras or {}
        aadtt = extras.get('AADTT', extras.get('AADTT_2plus_6wheel'))
        if aadtt is None:
            return None
        try:
            aadtt = float(aadtt)
        except (TypeError, ValueError):
            return None
        if aadtt <= 0:
            return None

        ddf = extras.get('DDF')
        if ddf is None:
            return None
        try:
            ddf = float(ddf)
        except (TypeError, ValueError):
            return None

        ldf = self._lane_distribution_factor(inputs)
        if ldf is None:
            return None

        vcdf = self._vehicle_distribution_pct(inputs)
        if not vcdf:
            return None

        ealf_sum = 0.0
        for cls, pct in vcdf.items():
            ealf_sum += (float(pct) / 100.0) * self._ealf_for_vehicle_class(
                indicator, cls)
        if ealf_sum <= 0:
            return None

        years = int(extras.get('traffic_design_life_years',
                               inputs.design_life or self._design_life(inputs)))
        growth = float(extras.get('traffic_growth_rate', 0.06))
        if growth > 1.0:
            growth = growth / 100.0

        n1 = aadtt * ddf * ldf * ealf_sum
        return max(365.0 * self._traffic_growth_factor(years, growth) * n1, 1.0)

    def _traffic_base_Ne_override(self, inputs: DesignInputs) -> Optional[float]:
        """
        Return a user-specified cumulative BZZ-100 equivalent traffic value.

        The override represents the asphalt-fatigue / AC-rutting equivalent
        cumulative axle passes. Indicator-specific N_e values are scaled from
        the simplified JTG table ratios unless explicit per-indicator values
        are supplied.
        """
        extras = inputs.extras or {}

        for key in ('total_ESAL_BZZ100', 'total_ESAL', 'design_ESAL',
                    'cumulative_ESAL_BZZ100', 'cumulative_ESAL'):
            val = extras.get(key)
            if val is not None:
                try:
                    return max(float(val), 1.0)
                except (TypeError, ValueError):
                    pass

        for key in ('annual_ESAL_BZZ100', 'annual_ESAL'):
            val = extras.get(key)
            if val is not None:
                try:
                    annual = max(float(val), 0.0)
                except (TypeError, ValueError):
                    continue
                years = int(extras.get('traffic_design_life_years',
                                       inputs.design_life or self._design_life(inputs)))
                growth = float(extras.get('traffic_growth_rate', 0.0))
                if growth > 1.0:
                    growth = growth / 100.0
                return max(annual * self._traffic_growth_factor(years, growth), 1.0)

        return None

    def _traffic_override_source(self, inputs: DesignInputs) -> str:
        extras = inputs.extras or {}
        if (extras.get('AADTT') is not None
                or extras.get('AADTT_2plus_6wheel') is not None):
            if (extras.get('DDF') is not None
                    and self._lane_distribution_factor(inputs) is not None
                    and self._vehicle_distribution_pct(inputs)):
                return 'appendix_A:AADTT+DDF+LDF+VCDF+EALF'
            return 'appendix_A:incomplete_traffic_inputs'
        for key in ('total_ESAL_BZZ100', 'total_ESAL', 'design_ESAL',
                    'cumulative_ESAL_BZZ100', 'cumulative_ESAL',
                    'annual_ESAL_BZZ100', 'annual_ESAL'):
            if extras.get(key) is not None:
                return key
        return 'traffic_level:' + str(inputs.traffic_level)

    def _N_e(self, inputs: DesignInputs, indicator: str) -> float:
        """Cumulative design axle passes N_e for a specific indicator."""
        ne_root = (self.data['appendix_A_traffic_loading']
                   ['simplified_Ne_by_traffic_level_per_indicator'])
        if indicator == 'asphalt_fatigue' or indicator == 'ac_rutting':
            tbl = ne_root['asphalt_fatigue_OR_ac_rutting_N_e']
        elif indicator == 'semi_rigid_fatigue':
            tbl = ne_root['semi_rigid_fatigue_N_e']
        elif indicator == 'subgrade_strain':
            tbl = ne_root['subgrade_strain_N_e']
        else:
            tbl = ne_root['asphalt_fatigue_OR_ac_rutting_N_e']
        extras = inputs.extras or {}
        explicit = {
            'asphalt_fatigue': ('N_e_asphalt', 'N_e_asphalt_fatigue'),
            'ac_rutting': ('N_e_ac_rutting', 'N_e_asphalt', 'N_e_asphalt_fatigue'),
            'semi_rigid_fatigue': ('N_e_semi_rigid', 'N_e_semi_rigid_fatigue'),
            'subgrade_strain': ('N_e_subgrade', 'N_e_subgrade_strain'),
        }.get(indicator, ())
        for key in explicit:
            if extras.get(key) is not None:
                try:
                    return max(float(extras[key]), 1.0)
                except (TypeError, ValueError):
                    pass

        appendix_a_ne = self._appendix_a_Ne(inputs, indicator)
        if appendix_a_ne is not None:
            return appendix_a_ne

        base_override = self._traffic_base_Ne_override(inputs)
        if base_override is not None:
            asphalt_tbl = ne_root['asphalt_fatigue_OR_ac_rutting_N_e']
            traffic_key = inputs.traffic_level if inputs.traffic_level in asphalt_tbl else 'heavy'
            asphalt_ref = float(asphalt_tbl.get(traffic_key, asphalt_tbl['heavy']))
            indicator_ref = float(tbl.get(traffic_key, tbl.get('heavy', asphalt_ref)))
            return max(base_override * indicator_ref / max(asphalt_ref, 1.0), 1.0)

        return float(tbl.get(inputs.traffic_level, tbl['medium']))

    def _k_a(self, inputs: DesignInputs) -> float:
        """Table B.1.1 — seasonal frost adjustment factor k_a."""
        zone = inputs.extras.get('frost_zone', 'non_frost')
        tbl = (self.data['appendix_B11_table_seasonal_frost_ka']['by_frost_zone'])
        entry = tbl.get(zone, tbl['non_frost'])
        return float(entry['k_a_typical'])

    # ─────────────────────────────────────────────────────────────
    # Appendix G — Temperature factors
    # ─────────────────────────────────────────────────────────────

    def _get_temperature_data(self, inputs: DesignInputs) -> Dict[str, float]:
        """Return {'kT_asphalt', 'kT_subgrade', 'T_xi'} for the design location."""
        extras = inputs.extras or {}

        # Direct continuous/derived override, useful for non-Chinese OOD cases.
        if all(k in extras for k in ('kT_asphalt', 'kT_subgrade', 'T_xi')):
            return {
                'kT_asphalt':  float(extras['kT_asphalt']),
                'kT_subgrade': float(extras['kT_subgrade']),
                'T_xi':        float(extras['T_xi']),
                '_source':     'direct_temperature_override',
            }

        # Continuous MAAT support: interpolate the verified city table by
        # annual mean air temperature. This preserves the JTG table basis while
        # letting OOD scripts exercise climate as a real numeric input.
        maat = None
        for key in ('MAAT_C', 'mean_annual_temp_C', 'annual_mean_air_temp_C'):
            if extras.get(key) is not None:
                try:
                    maat = float(extras[key])
                    break
                except (TypeError, ValueError):
                    pass
        if maat is not None:
            city_tbl = (self.data['appendix_G_temperature']['G12_city_table']['cities'])
            rows = sorted(
                (float(v['T_annual']), float(v['kT_asphalt']),
                 float(v['kT_subgrade']), float(v['T_xi']))
                for v in city_tbl.values()
            )
            if rows:
                if maat <= rows[0][0]:
                    _t, k1, k3, tx = rows[0]
                elif maat >= rows[-1][0]:
                    _t, k1, k3, tx = rows[-1]
                else:
                    for lo, hi in zip(rows[:-1], rows[1:]):
                        if lo[0] <= maat <= hi[0]:
                            span = max(hi[0] - lo[0], 1e-9)
                            w = (maat - lo[0]) / span
                            k1 = lo[1] + w * (hi[1] - lo[1])
                            k3 = lo[2] + w * (hi[2] - lo[2])
                            tx = lo[3] + w * (hi[3] - lo[3])
                            break
                return {
                    'kT_asphalt':  float(k1),
                    'kT_subgrade': float(k3),
                    'T_xi':        float(tx),
                    '_source':     'MAAT_C_interpolated:{:.2f}'.format(maat),
                }
        city = inputs.extras.get('city')
        if city:
            tbl = (self.data['appendix_G_temperature']['G12_city_table']['cities'])
            entry = tbl.get(city.lower())
            if entry:
                return {
                    'kT_asphalt':  float(entry['kT_asphalt']),
                    'kT_subgrade': float(entry['kT_subgrade']),
                    'T_xi':        float(entry['T_xi']),
                    '_source':     'city_lookup:' + city,
                }
        zone = inputs.extras.get('climate_zone', 'temperate')
        tbl = (self.data['appendix_G_temperature']['climate_zone_fallback'])
        entry = tbl.get(zone, tbl['temperate'])
        return {
            'kT_asphalt':  float(entry['kT_asphalt']),
            'kT_subgrade': float(entry['kT_subgrade']),
            'T_xi':        float(entry['T_xi']),
            '_source':     'climate_zone_fallback:' + zone,
        }

    def _k_T1(self, inputs: DesignInputs) -> float:
        return self._get_temperature_data(inputs)['kT_asphalt']

    def _k_T2(self, inputs: DesignInputs) -> float:
        return self._get_temperature_data(inputs)['kT_asphalt']

    def _k_T3(self, inputs: DesignInputs) -> float:
        return self._get_temperature_data(inputs)['kT_subgrade']

    def _T_pef(self, inputs: DesignInputs) -> float:
        """
        Eq. G.2.1: T_pef = T_ξ + 0.016 × h_a   (h_a = TOTAL AC thickness in mm)
        """
        T_xi = self._get_temperature_data(inputs)['T_xi']
        h_a_mm = self._h_AC_total_mm(inputs)
        return T_xi + 0.016 * h_a_mm

    # ─────────────────────────────────────────────────────────────
    # Appendix B.1 — AC fatigue
    # ─────────────────────────────────────────────────────────────

    def _k_b(self, E_a: float, VFA: float, h_a_mm: float) -> float:
        """Eq. B.1.1-2 — k_b loading-mode factor."""
        h_a_clip = min(max(h_a_mm, 1.0), 500.0)
        exp_term = math.exp(0.024 * h_a_clip - 5.41)
        num = 1.0 + 0.3 * (E_a ** 0.43) * (VFA ** -0.85) * exp_term
        den = 1.0 + exp_term
        if den <= 0 or num <= 0:
            return 1.0
        return (num / den) ** 3.33

    def _N_f1(self, inputs: DesignInputs, eps_a_microstrain: float) -> float:
        """
        Eq. B.1.1-1 — AC fatigue life N_f1.

            N_f1 = 6.32 × 10^(15.96 - 0.29β) · k_a · k_b · k_T1^(-1)
                   · (1/ε_a)^3.97 · (1/E_a)^1.58 · VFA^2.72

        E_a is the EQUIVALENT AC modulus (thickness-weighted across sublayers).
        h_a is the TOTAL AC thickness for k_b.
        """
        if eps_a_microstrain <= 0:
            return float('inf')
        beta = self._beta(inputs)
        k_a = self._k_a(inputs)
        E_a = self._E_AC_equivalent_MPa(inputs)
        h_a_mm = self._h_AC_total_mm(inputs)
        VFA = float(inputs.extras.get('VFA_pct', 70.0))
        k_b = self._k_b(E_a, VFA, h_a_mm)
        k_T1 = self._k_T1(inputs)
        try:
            log10_Nf = (
                math.log10(6.32)
                + (15.96 - 0.29 * beta)
                + math.log10(max(k_a, 1e-9))
                + math.log10(max(k_b, 1e-9))
                - math.log10(max(k_T1, 1e-9))
                + 3.97 * math.log10(1.0 / eps_a_microstrain)
                + 1.58 * math.log10(1.0 / E_a)
                + 2.72 * math.log10(VFA)
            )
            return 10.0 ** log10_Nf
        except (ValueError, OverflowError):
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # Appendix B.2 — Semi-rigid base fatigue
    # ─────────────────────────────────────────────────────────────

    def _k_c(self, c1: float, c2: float, c3: float,
             h_a_mm: float, h_b_mm: float) -> float:
        """Eq. B.2.1-2 — k_c field correction."""
        return c1 * math.exp(c2 * (h_a_mm + h_b_mm)) + c3

    def _N_f2(self, inputs: DesignInputs, sigma_t_MPa: float) -> float:
        """
        Eq. B.2.1-1 — semi-rigid base fatigue life N_f2.
        h_a = TOTAL AC thickness, h_b = base thickness.
        """
        if inputs.pavement_type != 'semi_rigid':
            return float('inf')
        if sigma_t_MPa <= 0:
            return float('inf')

        base_type = inputs.extras.get('base_type', 'inorganic_stabilized_granular')
        ab_tbl = (self.data['appendix_B21_semi_rigid_fatigue']
                  ['table_B211_material_a_b'])
        ab = ab_tbl.get(base_type, ab_tbl['inorganic_stabilized_granular'])
        a, b = float(ab['a']), float(ab['b'])

        const_type = inputs.extras.get('construction_type', 'new_construction')
        kc_root = (self.data['appendix_B21_semi_rigid_fatigue']
                   ['table_B212_kc_parameters'])
        kc_key = ('new_construction_OR_existing_layer'
                  if const_type == 'new_construction'
                  else 'rehabilitation_overlay')
        kc_subtype = 'granular' if 'granular' in base_type else 'soil'
        p = kc_root[kc_key][kc_subtype]
        c1, c2, c3 = float(p['c1']), float(p['c2']), float(p['c3'])

        h_a_mm = self._h_AC_total_mm(inputs)
        h_b_mm = self._h_base_mm(inputs)
        k_c = self._k_c(c1, c2, c3, h_a_mm, h_b_mm)

        # R_s
        R_s = inputs.extras.get('R_s_MPa')
        if R_s is None:
            R_s_tbl = (self.data['appendix_B21_semi_rigid_fatigue']
                       ['R_s_typical_values_MPa'])
            material_default = inputs.extras.get('R_s_material_type',
                'cement_stabilized_crushed_stone_grade_I')
            R_s = float(R_s_tbl.get(material_default, 0.8))
        R_s = float(R_s)

        k_a = self._k_a(inputs)
        k_T2 = self._k_T2(inputs)
        beta = self._beta(inputs)
        try:
            log10_Nf2 = (
                math.log10(max(k_a, 1e-9))
                - math.log10(max(k_T2, 1e-9))
                + a - b * (sigma_t_MPa / R_s) + k_c - 0.57 * beta
            )
            return 10.0 ** log10_Nf2
        except (ValueError, OverflowError):
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # Appendix B.3 — AC permanent deformation (MULTI-SUBLAYER, NEW)
    # ─────────────────────────────────────────────────────────────

    def _k_Ri(self, h_a_total_mm: float, z_i_mm: float) -> float:
        """Eq. B.3.2-2 to B.3.2-4 — k_Ri.

        h_a_total_mm: TOTAL AC thickness (cap at 200mm per regulation).
        z_i_mm:       depth of sublayer mid-point FROM AC SURFACE (mm).
        """
        h_a_eff = min(h_a_total_mm, 200.0)
        d_1 = -1.35e-4 * h_a_eff ** 2 + 8.18e-2 * h_a_eff - 14.50
        d_2 = 8.78e-7 * h_a_eff ** 2 - 1.50e-3 * h_a_eff + 0.90
        return (d_1 + d_2 * z_i_mm) * (0.9731 ** z_i_mm)

    def _R_a(self, inputs: DesignInputs,
             fea_responses: Optional[Dict[str, float]] = None) -> float:
        """
        Eq. B.3.2-1 — AC permanent deformation R_a, MULTI-SUBLAYER per B.3.1.

            R_a = Σ_i  2.31e-8 · k_Ri · T_pef^2.93 · p_i^1.80 · N_e3^0.48
                       · (h_i / h_0) · R_0_i

        Per JTG B.3.1, AC ≥ 200mm should be divided into multiple sublayers
        for calculation. Our 6-layer FEA model directly provides 3 sublayer
        mid-depth stresses. We use them when available; otherwise fall back
        to single-sublayer simplified mid-stress.

        z_i for each sublayer is measured from AC SURFACE downward.
        """
        h_upper_mm = self._h_AC_sublayer_mm(inputs, 'upper')
        h_mid_mm   = self._h_AC_sublayer_mm(inputs, 'mid')
        h_lower_mm = self._h_AC_sublayer_mm(inputs, 'lower')
        h_total_mm = h_upper_mm + h_mid_mm + h_lower_mm
        if h_total_mm <= 0:
            return 0.0

        T_pef = self._T_pef(inputs)
        N_e3 = self._N_e(inputs, 'ac_rutting')
        h_0 = 50.0
        R_0 = float(inputs.extras.get('R_0_mm', 1.5))

        # Determine p_i for each sublayer
        if fea_responses is not None and all(
                k in fea_responses for k in
                ('p_AC_upper_mid_MPa', 'p_AC_mid_mid_MPa', 'p_AC_lower_mid_MPa')):
            # FEA provided per-sublayer mid-depth stresses
            p_upper = float(fea_responses['p_AC_upper_mid_MPa'])
            p_mid   = float(fea_responses['p_AC_mid_mid_MPa'])
            p_lower = float(fea_responses['p_AC_lower_mid_MPa'])
        else:
            # Fallback: assume uniform 0.7 MPa (= contact pressure) attenuated
            # with depth — extremely rough, only used if FEA didn't provide.
            p_upper = 0.7
            p_mid   = 0.6
            p_lower = 0.4

        # z_i = depth of each sublayer mid-point from AC SURFACE
        z_upper_mm = h_upper_mm / 2.0
        z_mid_mm   = h_upper_mm + h_mid_mm / 2.0
        z_lower_mm = h_upper_mm + h_mid_mm + h_lower_mm / 2.0

        # Compute R_ai for each sublayer
        R_a_total = 0.0
        for h_i, z_i, p_i in [
            (h_upper_mm, z_upper_mm, p_upper),
            (h_mid_mm,   z_mid_mm,   p_mid),
            (h_lower_mm, z_lower_mm, p_lower),
        ]:
            try:
                k_Ri = self._k_Ri(h_total_mm, z_i)
                R_ai = (2.31e-8 * k_Ri
                        * (T_pef ** 2.93)
                        * (max(p_i, 1e-6) ** 1.80)
                        * (N_e3 ** 0.48)
                        * (h_i / h_0)
                        * R_0)
                R_a_total += max(R_ai, 0.0)
            except (ValueError, OverflowError):
                pass
        return R_a_total

    def _R_a_allowable(self, inputs: DesignInputs) -> float:
        """Table 3.0.6-1 — allowable R_a."""
        tbl = (self.data['appendix_B3_ac_permanent_deformation']
               ['table_3_0_6_1_allowable_R_a_mm'])
        if inputs.pavement_type == 'semi_rigid':
            key = 'semi_rigid_OR_concrete_base'
        else:
            key = 'other_base_types'
        if inputs.road_class in ('expressway', 'highway_1'):
            return float(tbl[key]['expressway_OR_highway_1'])
        else:
            return float(tbl[key]['highway_2_OR_3'])

    # ─────────────────────────────────────────────────────────────
    # Appendix B.4 — Subgrade strain
    # ─────────────────────────────────────────────────────────────

    def _epsilon_z_allowable(self, inputs: DesignInputs) -> float:
        """Eq. B.4.1 — [ε_z] = 1.25e4 · 10^(-0.1β) · (k_T3 · N_e4)^(-0.21)"""
        beta = self._beta(inputs)
        k_T3 = self._k_T3(inputs)
        N_e4 = self._N_e(inputs, 'subgrade_strain')
        try:
            return (1.25e4 * (10.0 ** (-0.1 * beta))
                    * ((k_T3 * N_e4) ** -0.21))
        except (ValueError, OverflowError):
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────

    def allowable_values(self, inputs: DesignInputs) -> Dict[str, float]:
        return {
            'N_e_asphalt':                       self._N_e(inputs, 'asphalt_fatigue'),
            'N_e_semi_rigid':                    self._N_e(inputs, 'semi_rigid_fatigue'),
            'N_e_subgrade':                      self._N_e(inputs, 'subgrade_strain'),
            'traffic_source':                    self._traffic_override_source(inputs),
            'target_beta':                       self._beta(inputs),
            'design_life_years':                 self._design_life(inputs),
            'k_T1':                              self._k_T1(inputs),
            'k_T3':                              self._k_T3(inputs),
            'T_pef_C':                           self._T_pef(inputs),
            'epsilon_z_allowable_microstrain':   self._epsilon_z_allowable(inputs),
            'R_a_allowable_mm':                  self._R_a_allowable(inputs),
            'h_AC_total_mm':                     self._h_AC_total_mm(inputs),
            'E_AC_equivalent_MPa':               self._E_AC_equivalent_MPa(inputs),
        }

    def evaluate(
        self, inputs: DesignInputs, fea_outputs: Dict[str, float],
    ) -> DesignEvaluation:
        eps_a   = fea_outputs.get('epsilon_a_microstrain')
        sigma_t = fea_outputs.get('sigma_t_MPa')
        eps_z   = fea_outputs.get('epsilon_z_microstrain')

        margins: Dict[str, float] = {}
        details: Dict[str, float] = {}
        responses: Dict[str, float] = {}

        # B.1 AC fatigue
        if eps_a is not None and eps_a > 0:
            N_f1 = self._N_f1(inputs, eps_a)
            N_e1 = self._N_e(inputs, 'asphalt_fatigue')
            margins['B1_asphalt_fatigue'] = N_f1 / max(N_e1, 1.0)
            details['B1_N_f1'] = N_f1
            details['B1_N_e'] = N_e1
            responses['epsilon_a_microstrain'] = eps_a

        # B.2 Semi-rigid fatigue
        if (inputs.pavement_type == 'semi_rigid'
                and sigma_t is not None and sigma_t > 0):
            N_f2 = self._N_f2(inputs, sigma_t)
            N_e2 = self._N_e(inputs, 'semi_rigid_fatigue')
            margins['B2_semi_rigid_fatigue'] = N_f2 / max(N_e2, 1.0)
            details['B2_N_f2'] = N_f2
            details['B2_N_e'] = N_e2
            responses['sigma_t_MPa'] = sigma_t

        # B.3 Permanent deformation (multi-sublayer, always computable)
        R_a = self._R_a(inputs, fea_responses=fea_outputs)
        R_a_allow = self._R_a_allowable(inputs)
        if R_a > 0:
            margins['B3_ac_permanent_deformation'] = R_a_allow / R_a
            details['B3_R_a_predicted_mm'] = R_a
            details['B3_R_a_allowable_mm'] = R_a_allow
            # Pass-through FEA sublayer stresses for audit
            for k in ('p_AC_upper_mid_MPa', 'p_AC_mid_mid_MPa',
                      'p_AC_lower_mid_MPa'):
                if k in fea_outputs:
                    responses[k] = fea_outputs[k]

        # B.4 Subgrade strain
        if eps_z is not None and eps_z > 0:
            eps_z_allow = self._epsilon_z_allowable(inputs)
            margins['B4_subgrade_strain'] = eps_z_allow / eps_z
            details['B4_epsilon_z_allowable_microstrain'] = eps_z_allow
            responses['epsilon_z_microstrain'] = eps_z

        if not margins:
            return DesignEvaluation(
                feasible=False, margins={}, responses=responses,
                allowable_values=self.allowable_values(inputs),
                critical_indicator='NONE',
                spec_name=self.name,
                details={'error': 'No valid FEA outputs received'},
            )

        feasible = all(m >= 1.0 for m in margins.values())
        critical = min(margins, key=margins.get)

        temp_data = self._get_temperature_data(inputs)

        return DesignEvaluation(
            feasible=feasible,
            margins=margins,
            responses=responses,
            allowable_values=self.allowable_values(inputs),
            critical_indicator=critical,
            spec_name=self.name,
            details={**details,
                     'beta':              self._beta(inputs),
                     'design_life_years': self._design_life(inputs),
                     'k_a_frost':         self._k_a(inputs),
                     'temperature_source': temp_data['_source'],
                     'traffic_source':     self._traffic_override_source(inputs),
                     'T_pef_C':           self._T_pef(inputs),
                     'h_AC_total_mm':     self._h_AC_total_mm(inputs),
                     'E_AC_equivalent_MPa': self._E_AC_equivalent_MPa(inputs),
                     'verification_status':
                        self.data['_metadata']['verification_status']},
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
