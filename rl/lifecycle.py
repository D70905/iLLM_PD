# -*- coding: utf-8 -*-
"""
rl/lifecycle.py — Life-Cycle Performance Transfer Functions (Phase 2D, R1-2)
=============================================================================

Converts FEA mechanical responses (ε, σ) into predicted fatigue life,
rutting depth, and subgrade strain capacity using JTG D50-2017 and
NCHRP 1-37A (ME-PDG) transfer functions.

Also provides a lightweight NPV-based life-cycle cost (LCC) estimator.

All functions are self-contained (no DesignInputs dependency) so they can
be called directly from rl/reward.py or from standalone analysis scripts.

Regulation references:
    JTG D50-2017:     Appendix B.1 (AC fatigue), B.2 (semi-rigid fatigue),
                      B.3 (AC permanent deformation), B.4 (subgrade strain),
                      Appendix G (temperature), Section 3 (reliability).
    NCHRP 1-37A:      Section 3.3.3 (fatigue cracking), 3.3.7 (rutting).

LCC methodology:
    NPV-based lightweight model adapted from FHWA LCCA guidelines.
    C_total = C_construction + Σ C_maintenance(t) / (1 + r)^t
    Maintenance schedule derived from predicted fatigue life margins.
"""

import math
from typing import Dict, Tuple, Optional

# ══════════════════════════════════════════════════════════════════════
# JTG D50-2017 transfer functions
# ══════════════════════════════════════════════════════════════════════

def ac_fatigue_life_Nf1(
    eps_a_microstrain: float,
    E_ac_MPa: float,
    h_ac_mm: float,
    VFA_pct: float = 70.0,
    beta: float = 1.65,
    k_a: float = 1.0,
    k_T1: float = 1.0,
) -> float:
    """
    JTG D50-2017 Eq. B.1.1-1 — AC bottom-up fatigue life.

    N_f1 = 6.32 × 10^(15.96−0.29β) · k_a · k_b · k_T1^(−1)
           · (1/ε_a)^3.97 · (1/E_a)^1.58 · VFA^2.72

    Args:
        eps_a_microstrain: AC bottom horizontal tensile strain (×10⁻⁶).
        E_ac_MPa:          Equivalent AC dynamic modulus (MPa).
        h_ac_mm:           Total AC thickness (mm), for k_b.
        VFA_pct:           Voids Filled with Asphalt (%).
        beta:              Target reliability index (Table 3.0.1).
        k_a:               Seasonal frost adjustment (Table B.1.1).
        k_T1:              Temperature factor (Appendix G, col 1).

    Returns:
        N_f1: predicted fatigue life in equivalent axle passes.
              Returns inf if eps_a ≤ 0.
    """
    if eps_a_microstrain <= 0:
        return float('inf')

    # Eq. B.1.1-2: k_b loading-mode factor
    h_clip = min(max(h_ac_mm, 1.0), 500.0)
    exp_term = math.exp(0.024 * h_clip - 5.41)
    num = 1.0 + 0.3 * (E_ac_MPa ** 0.43) * (VFA_pct ** -0.85) * exp_term
    den = 1.0 + exp_term
    k_b = (num / den) ** 3.33 if (den > 0 and num > 0) else 1.0

    try:
        log10_Nf = (
            math.log10(6.32)
            + (15.96 - 0.29 * beta)
            + math.log10(max(k_a, 1e-9))
            + math.log10(max(k_b, 1e-9))
            - math.log10(max(k_T1, 1e-9))
            + 3.97 * math.log10(1.0 / eps_a_microstrain)
            + 1.58 * math.log10(1.0 / E_ac_MPa)
            + 2.72 * math.log10(VFA_pct)
        )
        return 10.0 ** log10_Nf
    except (ValueError, OverflowError):
        return 0.0


def ac_fatigue_margin(
    eps_a_microstrain: float,
    E_ac_MPa: float,
    h_ac_mm: float,
    N_e_design: float,
    VFA_pct: float = 70.0,
    beta: float = 1.65,
    k_a: float = 1.0,
    k_T1: float = 1.0,
) -> float:
    """
    JTG B1 margin: N_f1 / N_e_design.

    Returns margin ≥ 1.0 for passing (fatigue life exceeds design passes).
    """
    N_f1 = ac_fatigue_life_Nf1(eps_a_microstrain, E_ac_MPa, h_ac_mm,
                                VFA_pct, beta, k_a, k_T1)
    if N_f1 >= float('inf') / 2:
        return float('inf')
    return N_f1 / max(N_e_design, 1.0)


def semi_rigid_fatigue_life_Nf2(
    sigma_t_MPa: float,
    R_s_MPa: float,
    h_ac_mm: float,
    h_base_mm: float,
    base_type: str = 'inorganic_stabilized_granular',
    construction_type: str = 'new_construction',
    beta: float = 1.65,
    k_a: float = 1.0,
    k_T2: float = 1.0,
) -> float:
    """
    JTG D50-2017 Eq. B.2.1-1 — semi-rigid base fatigue life.

    N_f2 = k_a · k_T2^(−1) · 10^(a − b·σ_t/R_s + k_c − 0.57β)

    Args:
        sigma_t_MPa:       Base bottom radial tensile stress (MPa).
        R_s_MPa:           Flexural tensile strength of base material (MPa).
        h_ac_mm, h_base_mm: AC and base thicknesses (mm), for k_c.
        base_type:         'inorganic_stabilized_granular' (default) or '_soil'.
        construction_type: 'new_construction' (default) or 'rehabilitation_overlay'.
        beta:              Target reliability index.
        k_a:               Seasonal frost adjustment.
        k_T2:              Temperature factor (Appendix G, col 1).

    Returns:
        N_f2: predicted fatigue life in equivalent axle passes.
    """
    if sigma_t_MPa <= 0:
        return float('inf')

    # Table B.2.1-1: a, b coefficients
    ab_table = {
        'inorganic_stabilized_granular':      {'a': 13.24, 'b': 12.52},
        'inorganic_stabilized_soil':          {'a': 12.18, 'b': 12.79},
    }
    ab = ab_table.get(base_type, ab_table['inorganic_stabilized_granular'])
    a, b = ab['a'], ab['b']

    # Table B.2.1-2: k_c field correction (c1, c2, c3)
    kc_key = ('new_construction_OR_existing_layer' if construction_type == 'new_construction'
              else 'rehabilitation_overlay')
    kc_subtype = 'granular' if 'granular' in base_type else 'soil'
    kc_table = {
        'new_construction_OR_existing_layer': {
            'granular': {'c1': 0.105, 'c2': -0.026, 'c3': -7.967},
            'soil':     {'c1': 0.094, 'c2': -0.024, 'c3': -6.980},
        },
        'rehabilitation_overlay': {
            'granular': {'c1': 0.108, 'c2': -0.027, 'c3': -8.307},
            'soil':     {'c1': 0.096, 'c2': -0.025, 'c3': -7.215},
        },
    }
    p = kc_table[kc_key][kc_subtype]
    c1, c2, c3 = p['c1'], p['c2'], p['c3']
    k_c = c1 * math.exp(c2 * (h_ac_mm + h_base_mm)) + c3

    try:
        log10_Nf2 = (
            math.log10(max(k_a, 1e-9))
            - math.log10(max(k_T2, 1e-9))
            + a - b * (sigma_t_MPa / R_s_MPa) + k_c - 0.57 * beta
        )
        return 10.0 ** log10_Nf2
    except (ValueError, OverflowError):
        return 0.0


def semi_rigid_fatigue_margin(
    sigma_t_MPa: float,
    R_s_MPa: float,
    h_ac_mm: float,
    h_base_mm: float,
    N_e_design: float,
    base_type: str = 'inorganic_stabilized_granular',
    construction_type: str = 'new_construction',
    beta: float = 1.65,
    k_a: float = 1.0,
    k_T2: float = 1.0,
) -> float:
    """JTG B2 margin: N_f2 / N_e_design."""
    N_f2 = semi_rigid_fatigue_life_Nf2(sigma_t_MPa, R_s_MPa, h_ac_mm, h_base_mm,
                                        base_type, construction_type,
                                        beta, k_a, k_T2)
    if N_f2 >= float('inf') / 2:
        return float('inf')
    return N_f2 / max(N_e_design, 1.0)


def ac_rutting_Ra_mm(
    p_AC_mid_MPa: float,
    h_ac_mm: float,
    T_pef_C: float,
    N_e_design: float,
    R_0_mm: float = 1.5,
) -> float:
    """
    JTG D50-2017 Eq. B.3.2-1 — AC permanent deformation (single-sublayer).

    R_a = 2.31×10⁻⁸ · k_R · T_pef^2.93 · p_i^1.80 · N_e^0.48 · (h_i/h_0) · R_0

    Args:
        p_AC_mid_MPa: Vertical compressive stress at AC mid-depth (MPa).
        h_ac_mm:      Total AC thickness (mm).
        T_pef_C:      Equivalent temperature (°C), per G.2.1.
        N_e_design:   Design cumulative axle passes.
        R_0_mm:       Lab rutting test result (mm), default 1.5.
    """
    if h_ac_mm <= 0:
        return 0.0

    h_a = h_ac_mm
    z_i = h_a / 2.0  # single-sublayer midpoint

    # Eq. B.3.2-2/3/4: k_Ri
    h_a_eff = min(h_a, 200.0)
    d_1 = -1.35e-4 * h_a_eff ** 2 + 8.18e-2 * h_a_eff - 14.50
    d_2 = 8.78e-7 * h_a_eff ** 2 - 1.50e-3 * h_a_eff + 0.90
    k_R = (d_1 + d_2 * z_i) * (0.9731 ** z_i)

    try:
        R_ai = (2.31e-8 * k_R * (T_pef_C ** 2.93)
                * (p_AC_mid_MPa ** 1.80)
                * (N_e_design ** 0.48)
                * (h_a / 50.0) * R_0_mm)
        return max(R_ai, 0.0)
    except (ValueError, OverflowError):
        return float('inf')


def subgrade_strain_allowable(
    N_e_design: float,
    beta: float = 1.65,
    k_T3: float = 1.0,
) -> float:
    """
    JTG D50-2017 Eq. B.4.1 — allowable subgrade top vertical compressive strain.

    [ε_z] = 1.25×10⁴ · 10^(−0.1β) · (k_T3 · N_e)^(−0.21)    [με]
    """
    try:
        return 1.25e4 * (10.0 ** (-0.1 * beta)) * ((k_T3 * N_e_design) ** -0.21)
    except (ValueError, OverflowError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════
# ME-PDG (NCHRP 1-37A) simplified transfer functions
# ══════════════════════════════════════════════════════════════════════

def mepdg_fatigue_life_Nf(
    eps_t_microstrain: float,
    E_ac_psi: float,
    h_ac_inch: float,
    beta_f1: float = 0.00432,
    beta_f2: float = 3.9492,
    beta_f3: float = 1.281,
) -> float:
    """
    NCHRP 1-37A Eq. 3.3.1 — bottom-up fatigue cracking life.

    N_f = k1 · β_f1 · C · (1/ε_t)^β_f2 · (1/E)^β_f3

    Simplified: C = 1.0 (typical binder content), single-temperature evaluation.
    """
    if eps_t_microstrain <= 0:
        return float('inf')

    # k1: thickness adjustment (NCHRP 1-37A Eq. 3.3.4)
    try:
        denom = 0.000398 + 0.003602 / (1.0 + math.exp(11.02 - 3.49 * h_ac_inch))
        k1 = 1.0 / denom
    except (OverflowError, ZeroDivisionError):
        k1 = 1.0
    k1 = max(0.1, min(k1, 100.0))

    C = 1.0   # simplified
    eps_abs = eps_t_microstrain * 1.0e-6

    try:
        N_f = (k1 * beta_f1 * C
               * (1.0 / eps_abs) ** beta_f2
               * (1.0 / E_ac_psi) ** beta_f3)
        return N_f
    except (OverflowError, ZeroDivisionError, ValueError):
        return float('inf')


def mepdg_rutting_RD_mm(
    eps_v_microstrain: float,
    ESAL: float,
    T_F: float,
    beta_r1: float = -3.4488,
    beta_r2: float = 1.5606,
    beta_r3: float = 0.4791,
) -> float:
    """
    NCHRP 1-37A — simplified total rutting depth.

    log10(ε_p/ε_v) = β_r1 + β_r2·log10(T_F) + β_r3·log10(N)

    (Equivalent log-form of the NCHRP Eq. 3.3.7; see mepdg.json for derivation.)

    Returns: predicted rutting depth in mm.
    """
    if eps_v_microstrain <= 0 or ESAL <= 0:
        return 0.0

    eps_v_abs = eps_v_microstrain * 1.0e-6
    ratio = (10.0 ** beta_r1) * (T_F ** beta_r2) * (ESAL ** beta_r3)
    eps_p_subgrade = eps_v_abs * ratio
    RD_subgrade_mm = eps_p_subgrade * 500.0   # approximate influence depth

    # Rough HMA rutting contribution divider
    RD_total_mm = RD_subgrade_mm / 0.7
    return max(0.0, min(RD_total_mm, 50.0))


# ══════════════════════════════════════════════════════════════════════
# Lightweight LCC (NPV-based)
# ══════════════════════════════════════════════════════════════════════

def lcc_npv(
    C_construction: float,
    design_life_years: float,
    margin_B1: float,
    margin_B2: float = float('inf'),
    discount_rate: float = 0.05,
) -> Dict[str, float]:
    """
    Lightweight NPV-based life-cycle cost estimate (FHWA LCCA-compatible).

    Maintenance schedule derived from fatigue margins:
      - B1 (AC fatigue) margin < 2.0  → AC overlay at year 12
      - B1 margin < 1.5                → AC overlay at year 8
      - B1 margin < 1.0                → AC overlay at year 5
      - B2 (semi-rigid) margin < 1.5   → base rehab at year 15
      - Both margins ≥ 2.0             → routine maintenance only

    C_maintenance estimates (CNY/m², rough order-of-magnitude):
      - AC overlay:  120 CNY/m²
      - Base rehab:  200 CNY/m²
      - Routine:      15 CNY/m² per application

    Returns:
        dict with NPV_total, C_construction, C_maintenance_NPV, schedule.
    """
    C_maint_NPV = 0.0
    schedule = []

    # AC overlay scheduling (based on B1 margin)
    if margin_B1 < 1.0:
        overlay_years = [5, 10, 15]
        overlay_cost = 120.0
    elif margin_B1 < 1.5:
        overlay_years = [8, 15]
        overlay_cost = 120.0
    elif margin_B1 < 2.0:
        overlay_years = [12]
        overlay_cost = 120.0
    else:
        overlay_years = []
        overlay_cost = 0.0

    for yr in overlay_years:
        if yr <= design_life_years:
            npv = overlay_cost / ((1.0 + discount_rate) ** yr)
            C_maint_NPV += npv
            schedule.append({'year': yr, 'action': 'AC overlay',
                             'cost': overlay_cost, 'npv': round(npv, 2)})

    # Base rehab (based on B2, semi-rigid only)
    if margin_B2 < 1.5 and margin_B2 != float('inf'):
        yr = min(15, int(design_life_years))
        rehab_cost = 200.0
        npv = rehab_cost / ((1.0 + discount_rate) ** yr)
        C_maint_NPV += npv
        schedule.append({'year': yr, 'action': 'base rehab',
                         'cost': rehab_cost, 'npv': round(npv, 2)})

    # Routine maintenance
    routine_interval = 3  # years
    routine_cost = 15.0
    for yr in range(routine_interval, int(design_life_years) + 1, routine_interval):
        npv = routine_cost / ((1.0 + discount_rate) ** yr)
        C_maint_NPV += npv
        schedule.append({'year': yr, 'action': 'routine',
                         'cost': routine_cost, 'npv': round(npv, 2)})

    return {
        'NPV_total':           round(C_construction + C_maint_NPV, 2),
        'C_construction':      round(C_construction, 2),
        'C_maintenance_NPV':   round(C_maint_NPV, 2),
        'discount_rate':       discount_rate,
        'design_life_years':   design_life_years,
        'schedule':            sorted(schedule, key=lambda x: x['year']),
    }


# ══════════════════════════════════════════════════════════════════════
# Composite lifecycle indicator (for reward.py integration)
# ══════════════════════════════════════════════════════════════════════

def compute_lifecycle_margins(
    epsilon_a_microstrain: float,
    sigma_t_MPa: float,
    epsilon_z_microstrain: float,
    E_ac_MPa: float,
    h_ac_mm: float,
    h_base_mm: float,
    N_e_B1: float,
    N_e_B2: float,
    pavement_type: str = 'semi_rigid',
    **kwargs,
) -> Dict[str, float]:
    """
    Compute lifecycle-aware margins (B1_fatigue, B2_fatigue) from raw FEA
    responses and design context. Ready to be called from rl/reward.py.

    Args:
        epsilon_a_microstrain: FEA AC bottom tensile strain.
        sigma_t_MPa:           FEA base bottom tensile stress.
        epsilon_z_microstrain: FEA subgrade top compressive strain.
        E_ac_MPa:              Equivalent AC modulus.
        h_ac_mm:               Total AC thickness (mm).
        h_base_mm:             Base thickness (mm).
        N_e_B1:                Design axle passes for B1 (asphalt fatigue).
        N_e_B2:                Design axle passes for B2 (semi-rigid fatigue).
        pavement_type:         'semi_rigid' or 'flexible'.
        **kwargs:              beta, k_a, k_T1, k_T2, R_s_MPa, etc.

    Returns:
        Dict with margins for B1 (fatigue life ratio) and B2 (fatigue life ratio),
        plus raw N_f values for audit.
    """
    beta   = kwargs.get('beta', 1.65)
    k_a    = kwargs.get('k_a', 1.0)
    k_T1   = kwargs.get('k_T1', 1.0)
    k_T2   = kwargs.get('k_T2', 1.0)
    VFA    = kwargs.get('VFA_pct', 70.0)
    R_s    = kwargs.get('R_s_MPa', 1.0)

    # B1: AC fatigue
    margin_B1 = ac_fatigue_margin(epsilon_a_microstrain, E_ac_MPa, h_ac_mm,
                                   N_e_B1, VFA, beta, k_a, k_T1)

    # B2: semi-rigid fatigue (only for semi-rigid)
    margin_B2 = float('inf')
    if pavement_type == 'semi_rigid' and sigma_t_MPa > 0:
        margin_B2 = semi_rigid_fatigue_margin(sigma_t_MPa, R_s, h_ac_mm,
                                               h_base_mm, N_e_B2,
                                               base_type=kwargs.get('base_type',
                                                    'inorganic_stabilized_granular'),
                                               construction_type=kwargs.get(
                                                   'construction_type', 'new_construction'),
                                               beta=beta, k_a=k_a, k_T2=k_T2)

    return {
        'margin_B1_ac_fatigue':       margin_B1,
        'margin_B2_semi_rigid_fatigue': margin_B2,
    }


# ══════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Quick sanity: Beijing expressway, heavy traffic, typical 6-layer FEA output
    eps_a  = 61.48    # με  (from run_with_spec_demo FEA)
    sigma_t = 0.0337  # MPa
    eps_z  = 103.9    # με
    E_ac   = 10778.0  # MPa (thickness-weighted equivalent)
    h_ac   = 180.0    # mm  (4+6+8 cm)
    h_base = 360.0    # mm
    N_e_B1 = 4.0e7    # heavy traffic, asphalt fatigue
    N_e_B2 = 7.0e9    # semi-rigid, b=13
    beta   = 1.65
    k_T1   = 1.23     # Beijing
    k_T2   = 1.23
    k_T3   = 1.09

    print('=' * 60)
    print('rl.lifecycle — self-test')
    print('=' * 60)

    # B1
    Nf1 = ac_fatigue_life_Nf1(eps_a, E_ac, h_ac, VFA_pct=70.0,
                               beta=beta, k_T1=k_T1)
    m1  = ac_fatigue_margin(eps_a, E_ac, h_ac, N_e_B1,
                             beta=beta, k_T1=k_T1)
    print(f'B1 AC fatigue:     N_f1 = {Nf1:.2e}  '
          f'margin = {m1:.1f}  (N_e = {N_e_B1:.1e})')

    # B2
    Nf2 = semi_rigid_fatigue_life_Nf2(sigma_t, R_s_MPa=1.0,
                                       h_ac_mm=h_ac, h_base_mm=h_base,
                                       beta=beta, k_T2=k_T2)
    m2  = semi_rigid_fatigue_margin(sigma_t, R_s_MPa=1.0,
                                     h_ac_mm=h_ac, h_base_mm=h_base,
                                     N_e_design=N_e_B2, beta=beta, k_T2=k_T2)
    print(f'B2 semi-rigid fat:  N_f2 = {Nf2:.2e}  '
          f'margin = {m2:.1f}  (N_e = {N_e_B2:.1e})')

    # B3
    Ra = ac_rutting_Ra_mm(p_AC_mid_MPa=0.5827, h_ac_mm=h_ac,
                           T_pef_C=22.98, N_e_design=N_e_B1)
    print(f'B3 AC rutting:      R_a = {Ra:.2f} mm  (allowable = 15 mm)')

    # B4
    eps_z_allow = subgrade_strain_allowable(N_e_B1, beta=beta, k_T3=k_T3)
    print(f'B4 subgrade strain: [eps_z] = {eps_z_allow:.1f} με  '
          f'(FEA = {eps_z:.1f} με, margin = {eps_z_allow/eps_z:.2f})')

    # LCC
    C_construction = 350.0  # CNY/m²  (typical flexible)
    lcc = lcc_npv(C_construction, design_life_years=15,
                   margin_B1=m1, margin_B2=m2)
    print(f'\nLCC: NPV_total = {lcc["NPV_total"]:.0f} CNY/m2  '
          f'(construction={lcc["C_construction"]:.0f}, '
          f'maintenance_NPV={lcc["C_maintenance_NPV"]:.0f})')
    print(f'  Maintenance events: {len(lcc["schedule"])}')
    for evt in lcc['schedule'][:5]:
        print(f'    yr {evt["year"]}: {evt["action"]:<12} '
              f'{evt["cost"]:.0f} CNY/m2 -> NPV = {evt["npv"]:.0f}')

    # Composite
    lm = compute_lifecycle_margins(eps_a, sigma_t, eps_z, E_ac, h_ac, h_base,
                                    N_e_B1, N_e_B2, pavement_type='semi_rigid',
                                    beta=beta, k_T1=k_T1, k_T2=k_T2)
    print(f'\nComposite: B1_margin = {lm["margin_B1_ac_fatigue"]:.1f}  '
          f'B2_margin = {lm["margin_B2_semi_rigid_fatigue"]:.1f}')
    print('=' * 60)
    print('All lifecycle functions ready.')
