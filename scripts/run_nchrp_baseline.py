# -*- coding: utf-8 -*-
"""
scripts/run_nchrp_baseline.py — NCHRP 1-37A reference baseline for LTPP sections.
================================================================================

Implements the **NCHRP 1-37A (2004) Mechanistic-Empirical Pavement Design Guide**
core distress prediction equations directly from the AASHTO official report,
without relying on PMED software. This serves as the third baseline alongside
LTPP As-built and AASHTO 1993 in the manuscript.

Method overview:
    1. For each LTPP section, define candidate 3-layer designs by grid search
       (h_AC ∈ [10, 30] cm, h_base ∈ [15, 40] cm, h_subbase ∈ [15, 30] cm).
    2. For each candidate, predict 20-year FC% / RD / IRI using NCHRP 1-37A
       Equations (3.3.1, 3.3.3, 3.3.4) at Level 3 climate input.
    3. Select the cheapest candidate that meets all three performance thresholds
       (FC% ≤ 25%, RD ≤ 0.75 inch, IRI ≤ 172 in/mi) — this is the NCHRP baseline
       design for that section.
    4. Map the 3-layer NCHRP design to a 5-layer model for FEA + JTG evaluation
       (apples-to-apples comparison with HARA and As-built).

Equations and parameter sources (NCHRP Report 1-37A, Part 3, Chapter 3):
    Eq.1  Bottom-up Fatigue Cracking : §3.3.1, Eqs. 3.3.1.1 — 3.3.1.7
    Eq.2  AC Rutting                 : §3.3.3, Eqs. 3.3.3.1 — 3.3.3.4
    Eq.3  Unbound base/subbase RD    : §3.3.3, Eqs. 3.3.3.7 — 3.3.3.11 (Tseng-Lytton)
    Eq.4  Subgrade RD                : §3.3.3, same form, subgrade params
    Eq.5  IRI evolution              : §3.3.4, Eq. 3.3.4.1

Reference:
    ARA Inc. ERES Consultants Division (2004). Guide for Mechanistic-Empirical
    Design of New and Rehabilitated Pavement Structures, Final Report,
    NCHRP Project 1-37A. Transportation Research Board, Washington DC.
    Available: https://onlinepubs.trb.org/onlinepubs/archive/mepdg/

Climate input level:
    Level 3 (simplified, per NCHRP 1-37A §10.2). We use representative MAAT
    for the 4 LTPP climate zones (DF/DNF/WF/WNF). Full Level 1/2 input would
    require EICM monthly climate files (part of PMED-EICM module). This is
    DISCLOSED as a limitation in the manuscript and is the standard approach
    for academic NCHRP 1-37A implementations without PMED access.

Usage:
    conda activate illm_pd
    python scripts/run_nchrp_baseline.py

Output:
    experiments/ltpp_data/deliverables/ltpp_nchrp/nchrp_summary_<timestamp>.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("nchrp_baseline")

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Project-internal imports (same as run_asbuilt_baseline.py)
from fea import run_fea
from specs import get_protocol, DesignInputs
from rl.lifecycle_lcc_intl import lcc_npv_usd
from rl import metrics as _metrics


# ============================================================================
# SECTION 1 — GPS family + climate zone configuration
# ============================================================================

_GPS_FAMILY: Dict[str, str] = {
    "04_1034": "GPS-1", "12_1060": "GPS-1", "16_1010": "GPS-1",
    "27_1085": "GPS-1", "48_0001": "GPS-1", "48_1076": "GPS-1",
    "04_1065": "GPS-2", "06_2004": "GPS-2", "12_4097": "GPS-2",
    "27_2023": "GPS-2", "30_7076": "GPS-2", "48_1109": "GPS-2",
}

# KESAL data extracted from LTPP TRF_ESAL tables (thousands of ESAL)
_KESAL: Dict[str, int] = {
    "04_1034": 57, "04_1065": 583, "06_2004": 120,
    "12_1060": 81, "12_4097": 200,
    "16_1010": 123, "27_1085": 10, "27_2023": 574,
    "30_7076": 94, "48_0001": 133, "48_1076": 164, "48_1109": 81,
}

# LTPP climate zone classification (per LTPP InfoPave manual + FHWA-HRT-13-049)
# MAAT = Mean Annual Air Temperature, °C (representative values)
# Freezing Index in °C-days
# Source: FHWA "LTPP Climate Tool" + NCHRP 1-37A Appendix CC (climate Level 3 input)
CLIMATE_ZONE_PROPS: Dict[str, Dict[str, float]] = {
    "DF":  {"MAAT_C": 7.0,  "freezing_index_C_days": 200,  "annual_precip_mm": 380},
    "DNF": {"MAAT_C": 18.0, "freezing_index_C_days": 0,    "annual_precip_mm": 380},
    "WF":  {"MAAT_C": 8.0,  "freezing_index_C_days": 600,  "annual_precip_mm": 1100},
    "WNF": {"MAAT_C": 19.0, "freezing_index_C_days": 0,    "annual_precip_mm": 1300},
}

# Section → climate zone mapping (from your existing LTPP master xlsx)
_SECTION_CLIMATE: Dict[str, str] = {
    "16_1010": "DF",  "30_7076": "DF",
    "04_1034": "DNF", "04_1065": "DNF", "06_2004": "DNF", "48_1076": "DNF",
    "27_1085": "WF",  "27_2023": "WF",
    "12_1060": "WNF", "12_4097": "WNF", "48_1109": "WNF", "48_0001": "WNF",
}


# ============================================================================
# SECTION 2 — NCHRP 1-37A performance thresholds
# ============================================================================
# Default Level 3 performance criteria for Interstate / freeway
# per NCHRP 1-37A Part 1, Chapter 2 (Design Inputs and Performance Criteria)

PERFORMANCE_THRESHOLDS: Dict[str, float] = {
    "FC_pct_max":        25.0,   # Bottom-up alligator cracking, %
    "RD_total_inch_max": 0.75,   # Total rutting, inch
    "IRI_in_mi_max":     172.0,  # Smoothness, in/mile (= 2.72 m/km)
    "thermal_crack_ft_per_mi_max": 1000.0,  # Not used (thermal model not impl.)
}

# Design reliability per NCHRP 1-37A §10.3.4 default for Interstate = 90%
DESIGN_RELIABILITY = 0.90

# Design analysis period (years)
DESIGN_LIFE_YEARS = 20

# Initial IRI at construction (in/mi)
IRI_INITIAL = 63.0


# ============================================================================
# SECTION 3 — Material defaults for NCHRP 1-37A Level 3 input
# ============================================================================

@dataclass
class ACProperties:
    """AC layer properties at Level 3 (default from NCHRP 1-37A defaults)."""
    Va_pct: float = 7.0          # Air voids by volume, %
    Vbe_pct: float = 11.0        # Effective binder content, % volume
    E_psi: float = 500_000.0     # Dynamic modulus E* at design conditions, psi
    poisson: float = 0.30
    # Calibration factors (per NCHRP 1-37A Appendix BB, global calibration)
    beta_f1: float = 1.0
    beta_f2: float = 1.0
    beta_f3: float = 1.0
    beta_r1: float = 1.0         # AC rutting calibration


@dataclass
class UnboundProperties:
    """Unbound (granular or subgrade) layer properties (Level 3 default)."""
    Mr_psi: float = 30_000.0     # Resilient modulus, psi
    poisson: float = 0.35
    beta_s1: float = 1.0         # Unbound rutting calibration
    epsilon_v_microstrain: float = 200.0  # Vertical compressive strain


# Material unit prices (CNY/m³, same as as-built script for consistency)
MATERIAL_PRICES_CNY_PER_M3 = {
    "flexible":   [1800, 1100, 900, 100, 80],     # AC1, AC2, AC3, granular base, granular subbase
    "semi_rigid": [1800, 1100, 900, 320, 180],    # AC1, AC2, AC3, stabilised base, subbase
}
CNY_TO_USD = 1.0 / 7.20


# ============================================================================
# SECTION 4 — NCHRP 1-37A core equations
# ============================================================================
# NOTE on units:
# NCHRP 1-37A equations are written in US customary units (psi, microstrain, °F).
# We implement them in those units internally, convert at I/O boundaries.

def k_kelvin_log_factor(T_C: float) -> float:
    """Compute temperature effect factor for AC modulus shifting (simplified
    Arrhenius-like adjustment per NCHRP 1-37A §3.3.1.4 Level 3 simplification).
    Returns a multiplicative factor for E_AC at design temperature.
    """
    # Reference T = 21.1°C (70°F), at which AC E_psi is calibrated
    T_F = T_C * 9/5 + 32
    T_ref_F = 70.0
    # Empirical sensitivity ~0.02/°F (typical for asphalt at intermediate T)
    return math.exp(-0.020 * (T_F - T_ref_F))


def nchrp_fatigue_cracking(
    epsilon_t_microstrain: float,
    E_psi: float,
    h_AC_inch: float,
    ac: ACProperties,
    total_traffic_ESAL: float,
) -> Tuple[float, float]:
    """
    Eq. 1 — NCHRP 1-37A §3.3.1: Bottom-up Fatigue Cracking.

    Returns:
        N_f      : allowable fatigue load repetitions
        FC_pct   : predicted alligator cracking percentage
    """
    # Eq. 3.3.1.4: laboratory fatigue life
    # N_f = 0.00432 × C × β_f1 × (1/ε_t)^(3.9492×β_f2) × (1/E)^(1.281×β_f3)

    # C (thickness correction): NCHRP 1-37A Eq. 3.3.1.5
    # C = 10^M
    # M = 4.84 × (Vbe / (Va + Vbe) - 0.69)
    M = 4.84 * (ac.Vbe_pct / (ac.Va_pct + ac.Vbe_pct) - 0.69)
    C = 10 ** M

    eps = max(epsilon_t_microstrain, 1.0)  # avoid div-by-zero
    E_msi = E_psi / 1e6  # convert to msi for compatibility with NCHRP formulation

    N_f = (
        0.00432 * C * ac.beta_f1
        * (1.0 / eps) ** (3.9492 * ac.beta_f2)
        * (1.0 / max(E_psi, 1e3)) ** (1.281 * ac.beta_f3)
    )
    N_f = max(N_f, 1.0)

    # Damage ratio D = total traffic / allowable
    D = min(total_traffic_ESAL / N_f, 1e6)

    # Eq. 3.3.1.6: alligator cracking transfer function (bottom-up)
    # FC% = 6000 / (1 + e^(C1' + C2' × log10(D × 100))) × (1/60)
    # NCHRP 1-37A §3.3.1.6, thickness-dependent C1', C2':
    C2_prime = -2.40874 - 39.748 * (1 + h_AC_inch) ** -2.856
    C1_prime = -2 * C2_prime

    log_D100 = math.log10(D * 100 + 1e-9)
    arg = C1_prime + C2_prime * log_D100
    arg = max(min(arg, 30), -30)  # prevent overflow
    FC_pct = (6000.0 / (1.0 + math.exp(arg))) * (1.0 / 60.0)
    FC_pct = max(0.0, min(FC_pct, 100.0))

    return N_f, FC_pct


def nchrp_ac_rutting(
    epsilon_r_microstrain: float,
    E_psi: float,
    h_AC_inch: float,
    ac: ACProperties,
    T_C: float,
    N_ESAL: float,
) -> float:
    """
    Eq. 2 — NCHRP 1-37A §3.3.3: AC permanent deformation (rutting).

    ε_p / ε_r = k_z × β_r1 × 10^(-3.4488) × T_F^(1.5606) × N^(0.479244)

    Returns:
        RD_AC_inch : AC layer rutting depth (inch)
    """
    T_F = T_C * 9/5 + 32

    # Depth correction factor k_z (NCHRP 1-37A Eq. 3.3.3.3, simplified mid-layer)
    # k_z = (C1 + C2 × depth) × 0.328196^depth, depth in inch from surface
    depth_in = h_AC_inch / 2.0  # mid-depth of AC
    C1 = -0.1039 * h_AC_inch ** 2 + 2.4868 * h_AC_inch - 17.342
    C2 = 0.0172 * h_AC_inch ** 2 - 1.7331 * h_AC_inch + 27.428
    k_z = (C1 + C2 * depth_in) * (0.328196 ** depth_in)
    k_z = max(0.5, min(k_z, 3.0))  # bound to reasonable range

    eps_r = epsilon_r_microstrain  # resilient strain
    eps_p_over_eps_r = (
        k_z * ac.beta_r1 * (10 ** (-3.4488))
        * (T_F ** 1.5606) * (max(N_ESAL, 1.0) ** 0.479244)
    )

    # Total AC rutting (integrate over AC thickness, simplified):
    RD_AC_inch = eps_p_over_eps_r * eps_r * h_AC_inch * 1e-6
    return max(0.0, RD_AC_inch)


def nchrp_unbound_rutting(
    epsilon_v_microstrain: float,
    Mr_psi: float,
    h_inch: float,
    N_ESAL: float,
    unbound: UnboundProperties,
    is_subgrade: bool = False,
) -> float:
    """
    Eq. 3/4 — NCHRP 1-37A §3.3.3: Unbound material rutting (Tseng-Lytton).

    δ_a = β_s1 × (ε0 / εr) × e^(-(ρ/N)^β) × ε_v × h

    Simplified Level 3 form per NCHRP 1-37A §3.3.3.7.

    Returns:
        RD_inch : layer rutting depth (inch)
    """
    # Simplified Tseng-Lytton parameters at Level 3
    # log(ε0/εr) = 0.74168 × (Wc^0.073) — depends on water content; use defaults
    if is_subgrade:
        eps0_over_epsr = 0.69  # fine-grained subgrade default
        beta = 1.673
        rho = 10 ** 7
    else:
        eps0_over_epsr = 0.39  # granular base/subbase default
        beta = 1.673
        rho = 10 ** 7.5

    N = max(N_ESAL, 1.0)
    decay = math.exp(-((rho / N) ** beta)) if rho / N < 100 else 0.0

    RD_inch = (
        unbound.beta_s1 * eps0_over_epsr * decay
        * epsilon_v_microstrain * h_inch * 1e-6
    )
    return max(0.0, RD_inch)


def nchrp_IRI(
    IRI_initial_in_mi: float,
    FC_pct: float,
    RD_total_inch: float,
    site_factor: float = 1.0,
    thermal_crack_ft_per_mi: float = 0.0,
) -> float:
    """
    Eq. 5 — NCHRP 1-37A §3.3.4: IRI evolution for flexible pavement.

    IRI = IRI0 + 0.0150 × SF + 0.400 × FC + 0.0080 × TC + 40.0 × RD

    Returns:
        IRI_in_mi : terminal IRI (in/mi)
    """
    IRI_final = (
        IRI_initial_in_mi
        + 0.0150 * site_factor
        + 0.400 * FC_pct
        + 0.0080 * thermal_crack_ft_per_mi
        + 40.0 * RD_total_inch
    )
    return max(IRI_initial_in_mi, IRI_final)


# ============================================================================
# SECTION 5 — Strain estimation (Burmister-like analytical approximation)
# ============================================================================
# NCHRP 1-37A uses JULEA (multilayer elastic) to compute ε_t, ε_v at critical
# locations. We use a fast Burmister approximation for the grid search, then
# verify the best candidate with full ABAQUS CAX4R FEA.

def estimate_strains_analytical(
    h_AC_m: float, E_AC_MPa: float,
    h_base_m: float, E_base_MPa: float,
    h_subbase_m: float, E_subbase_MPa: float,
    E_subgrade_MPa: float,
    pressure_MPa: float = 0.7,
    radius_m: float = 0.1065,
) -> Tuple[float, float, float]:
    """
    Quick Burmister-like strain estimates (used for grid search only).
    Returns (epsilon_t_AC_microstrain, epsilon_v_subgrade_microstrain,
             epsilon_v_base_microstrain).

    These are first-order approximations; final design is verified by ABAQUS FEA.
    """
    # Effective combined modulus (weighted by thickness, simplified)
    total_h = h_AC_m + h_base_m + h_subbase_m
    # Boussinesq elastic deflection at center: w = (1.5 × p × a) / E_eq
    # Strain at AC bottom (radial tensile): ε_t ∝ p / (E_AC × h_AC^1.5)
    # Strain at subgrade top (vertical compressive): ε_v ∝ p / E_eq

    # AC bottom tensile strain (simplified, monotonic in inputs)
    # Based on regression of multilayer-elastic solutions for typical pavements
    a_term = (E_AC_MPa / 1000.0)  # GPa
    h_AC_cm = h_AC_m * 100
    eps_t_AC = 150.0 * pressure_MPa / (a_term ** 0.4 * (h_AC_cm + 1) ** 0.6)

    # Subgrade vertical compressive strain
    # ε_v ∝ p × radius / (h_total × E_subgrade)
    h_above_sg_m = h_AC_m + h_base_m + h_subbase_m
    eps_v_SG = 600.0 * pressure_MPa / (
        (E_subgrade_MPa / 100.0) * (h_above_sg_m + 0.1) ** 1.2
    )

    # Base layer compressive strain (approx halfway between AC and SG)
    eps_v_base = 0.6 * eps_v_SG + 0.2 * eps_t_AC

    return eps_t_AC, eps_v_SG, eps_v_base


# ============================================================================
# SECTION 6 — Per-section design (grid search)
# ============================================================================

def get_traffic_growth_factor(annual_growth_pct: float, years: int) -> float:
    """Compound growth factor for cumulative ESAL over analysis period."""
    if annual_growth_pct <= 0:
        return float(years)
    g = annual_growth_pct / 100.0
    return ((1 + g) ** years - 1) / g


def design_one_section_nchrp(
    section: Dict[str, Any],
    candidate_grid: List[Tuple[float, float, float]],  # (h_AC_cm, h_base_cm, h_subbase_cm)
) -> Optional[Dict[str, Any]]:
    """
    Grid search to find the cheapest NCHRP-compliant design for one LTPP section.
    Returns the best candidate dict (or None if no candidate passes thresholds).
    """
    sid = section["section_id"]
    pavtype = section["pavement_type"]
    E_sub = section["E_subgrade"]
    climate = section["climate_zone"]
    ESAL_initial = section["ESAL_initial"]
    annual_growth = section.get("traffic_growth_pct", 2.0)

    # Cumulative traffic over analysis period
    growth_factor = get_traffic_growth_factor(annual_growth, DESIGN_LIFE_YEARS)
    N_total = ESAL_initial * growth_factor

    # Climate temperature for AC modulus adjustment
    T_design_C = CLIMATE_ZONE_PROPS[climate]["MAAT_C"]
    T_factor = k_kelvin_log_factor(T_design_C)

    # AC and unbound default properties
    ac = ACProperties()
    base_unbound = UnboundProperties(Mr_psi=20000.0 if pavtype == "flexible" else 50000.0)
    subbase_unbound = UnboundProperties(Mr_psi=15000.0 if pavtype == "flexible" else 30000.0)
    subgrade_unbound = UnboundProperties(Mr_psi=E_sub * 145.038)  # MPa → psi

    # Material moduli (consistent with HARA default)
    E_AC_MPa_ref = 11_000.0      # average across 3 AC sublayers
    E_base_MPa = 400.0 if pavtype == "flexible" else 1_500.0
    E_subbase_MPa = 200.0 if pavtype == "flexible" else 500.0

    candidates_evaluated = []

    for h_AC_cm, h_base_cm, h_subbase_cm in candidate_grid:
        h_AC_m = h_AC_cm / 100
        h_base_m = h_base_cm / 100
        h_subbase_m = h_subbase_cm / 100
        h_AC_inch = h_AC_cm / 2.54

        # Adjust E_AC for climate temperature
        E_AC_MPa = E_AC_MPa_ref * T_factor
        E_AC_psi = E_AC_MPa * 145.038

        # Estimate strains (analytical fast approximation)
        eps_t_AC, eps_v_SG, eps_v_base = estimate_strains_analytical(
            h_AC_m, E_AC_MPa,
            h_base_m, E_base_MPa,
            h_subbase_m, E_subbase_MPa,
            E_sub,
        )

        # Apply NCHRP 1-37A equations
        N_f, FC_pct = nchrp_fatigue_cracking(
            eps_t_AC, E_AC_psi, h_AC_inch, ac, N_total
        )

        RD_AC = nchrp_ac_rutting(
            eps_t_AC, E_AC_psi, h_AC_inch, ac, T_design_C, N_total
        )

        RD_base = nchrp_unbound_rutting(
            eps_v_base, E_base_MPa * 145.038,
            h_base_cm / 2.54, N_total, base_unbound, is_subgrade=False
        )

        RD_subbase = nchrp_unbound_rutting(
            eps_v_SG * 0.85, E_subbase_MPa * 145.038,
            h_subbase_cm / 2.54, N_total, subbase_unbound, is_subgrade=False
        )

        RD_SG = nchrp_unbound_rutting(
            eps_v_SG, E_sub * 145.038, 4.0,  # top 10 cm of subgrade
            N_total, subgrade_unbound, is_subgrade=True
        )

        RD_total_inch = RD_AC + RD_base + RD_subbase + RD_SG

        IRI = nchrp_IRI(IRI_INITIAL, FC_pct, RD_total_inch)

        # Compute construction cost (3-layer)
        # Use same per-m3 prices as HARA, with AC split 25/35/40
        prices = MATERIAL_PRICES_CNY_PER_M3[pavtype]
        # Split AC by NCHRP default ratio for 3-layer cost calculation
        cost_AC = h_AC_m * (0.25 * prices[0] + 0.35 * prices[1] + 0.40 * prices[2])
        cost_base = h_base_m * prices[3]
        cost_subbase = h_subbase_m * prices[4]
        cost_total_cny = cost_AC + cost_base + cost_subbase

        # Check thresholds
        passes_FC = FC_pct <= PERFORMANCE_THRESHOLDS["FC_pct_max"]
        passes_RD = RD_total_inch <= PERFORMANCE_THRESHOLDS["RD_total_inch_max"]
        passes_IRI = IRI <= PERFORMANCE_THRESHOLDS["IRI_in_mi_max"]
        passes_all = passes_FC and passes_RD and passes_IRI

        candidates_evaluated.append({
            "h_AC_cm": h_AC_cm,
            "h_base_cm": h_base_cm,
            "h_subbase_cm": h_subbase_cm,
            "FC_pct": FC_pct,
            "RD_total_inch": RD_total_inch,
            "RD_AC_inch": RD_AC,
            "RD_base_inch": RD_base,
            "RD_subbase_inch": RD_subbase,
            "RD_SG_inch": RD_SG,
            "IRI_in_mi": IRI,
            "N_f_allowable": N_f,
            "cost_cny": cost_total_cny,
            "eps_t_AC_microstrain": eps_t_AC,
            "eps_v_SG_microstrain": eps_v_SG,
            "passes_FC": passes_FC,
            "passes_RD": passes_RD,
            "passes_IRI": passes_IRI,
            "passes_all": passes_all,
            "T_design_C": T_design_C,
            "N_total_ESAL": N_total,
        })

    # Select cheapest passing candidate
    passing = [c for c in candidates_evaluated if c["passes_all"]]
    if not passing:
        logger.warning(
            f"[{sid}] No candidate passes NCHRP thresholds. "
            f"Best partial candidate by total cost will be reported."
        )
        # Fall back to cheapest overall (will be flagged in output)
        best = min(candidates_evaluated, key=lambda c: c["cost_cny"])
        best["selected_via"] = "fallback_cheapest"
    else:
        best = min(passing, key=lambda c: c["cost_cny"])
        best["selected_via"] = "min_cost_among_passing"

    best["n_candidates_total"] = len(candidates_evaluated)
    best["n_candidates_passing"] = len(passing)
    return best


# ============================================================================
# SECTION 7 — Map NCHRP 3-layer design to 5-layer FEA model
# ============================================================================

def map_nchrp_to_5_layer(
    h_AC_cm: float, h_base_cm: float, h_subbase_cm: float,
    pavtype: str,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Map 3-layer NCHRP design to 5-layer (h, E, ν) consistent with HARA model.
    AC is split 25/35/40 (surface / binder / lower).
    """
    h5 = [
        h_AC_cm / 100 * 0.25,   # AC upper
        h_AC_cm / 100 * 0.35,   # AC mid
        h_AC_cm / 100 * 0.40,   # AC lower
        h_base_cm / 100,
        h_subbase_cm / 100,
    ]
    E5 = [
        14000.0, 11000.0, 9000.0,
        400.0 if pavtype == "flexible" else 1500.0,
        200.0 if pavtype == "flexible" else 400.0,
    ]
    nu5 = [0.25, 0.30, 0.30, 0.25 if pavtype == "semi_rigid" else 0.35, 0.35]
    return h5, E5, nu5


# ============================================================================
# SECTION 8 — Section loading + main entry
# ============================================================================

def load_sections(xlsx_path: str) -> List[Dict[str, Any]]:
    """Load 12 LTPP sections with traffic + climate info."""
    df = pd.read_excel(xlsx_path)
    sections = []
    for _, row in df.iterrows():
        sid = str(row["section_id"]).strip()
        gps = _GPS_FAMILY.get(sid, "GPS-1")
        pavtype = "flexible" if gps == "GPS-1" else "semi_rigid"
        climate = _SECTION_CLIMATE.get(sid, "WNF")

        # Try multiple possible ESAL column names
        esal = None
        for c in ("annual_ESAL", "ESAL_initial", "AADTT", "annual_ESAL_million"):
            if c in df.columns and pd.notna(row.get(c)):
                v = float(row[c])
                if c == "annual_ESAL_million":
                    v *= 1e6
                esal = v
                break
        if esal is None and sid in _KESAL:
            esal = float(_KESAL[sid]) * 1000.0  # KESAL → ESAL
        if esal is None:
            logger.warning(f"[{sid}] No ESAL data found, using 200k/year default")
            esal = 200_000.0

        sections.append({
            "section_id":      sid,
            "state_name":      str(row.get("state_name", "")),
            "climate_zone":    climate,
            "gps_family":      gps,
            "pavement_type":   pavtype,
            "subgrade_bin":    str(row.get("subgrade_bin", "")),
            "E_subgrade":      float(row["E_subgrade_MPa"]),
            "is_baseline":     sid in ("48_0001", "48_1076"),
            "ESAL_initial":    esal,
            "traffic_growth_pct": 2.0,
        })
    return sections


def build_candidate_grid() -> List[Tuple[float, float, float]]:
    """Build the 3-layer thickness candidate grid."""
    h_AC_options = [10, 12, 15, 18, 20, 22, 25, 28]      # cm
    h_base_options = [15, 20, 25, 30, 35, 40]            # cm
    h_subbase_options = [15, 20, 25, 30]                 # cm
    return [(a, b, c) for a in h_AC_options
                       for b in h_base_options
                       for c in h_subbase_options]


def verify_with_fea_and_jtg(
    section: Dict[str, Any],
    nchrp_design: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map best NCHRP design to 5-layer, run ABAQUS FEA, evaluate under JTG D50-2017.
    Returns enriched result dict with FEA responses, JTG margins, DSR/SCR, LCC.
    """
    sid = section["section_id"]
    pavtype = section["pavement_type"]
    E_sub = section["E_subgrade"]

    h5, E5, nu5 = map_nchrp_to_5_layer(
        nchrp_design["h_AC_cm"],
        nchrp_design["h_base_cm"],
        nchrp_design["h_subbase_cm"],
        pavtype,
    )

    logger.info(
        f"  [{sid}] NCHRP design: AC={nchrp_design['h_AC_cm']}cm "
        f"base={nchrp_design['h_base_cm']}cm subbase={nchrp_design['h_subbase_cm']}cm "
        f"→ 5-layer: {[f'{h*100:.1f}' for h in h5]} cm"
    )

    try:
        fea_result = run_fea(
            thickness=h5,
            modulus=E5,
            poisson=nu5,
            E_subgrade=E_sub,
            nu_subgrade=0.40,
            load_pressure=0.7,
            load_radius=0.1065,
            num_cpus=4,
            verbose=False,
        )
        fea_responses = fea_result.get("responses", {})

        inputs = DesignInputs(
            pavement_type=pavtype,
            road_class="expressway",
            traffic_level="heavy",
            thickness=h5, modulus=E5, poisson=nu5,
            E_subgrade=E_sub, nu_subgrade=0.40,
            design_life=15,
            extras={"city": "beijing", "VFA_pct": 70.0,
                    "R_s_MPa": 1.0, "R_0_mm": 1.5},
        )
        protocol = get_protocol("JTG_D50_2017")
        evaluation = protocol.evaluate(inputs, fea_responses)

        margins = {k: float(v) for k, v in evaluation.margins.items()}
        dsr = _metrics.compute_dsr(margins)
        is_compliant = _metrics.compute_compliance(margins)
        scr = 1.0 if is_compliant else (
            sum(1 for v in margins.values() if v >= 1.0) / len(margins))

        cny_per_m3 = MATERIAL_PRICES_CNY_PER_M3[pavtype]
        C_const_cny = sum(cny_per_m3[i] * h5[i] for i in range(5))
        C_const_usd = C_const_cny * CNY_TO_USD

        margin_B1 = margins.get("B1_asphalt_fatigue", 99.0)
        margin_B2 = margins.get("B2_semi_rigid_fatigue", 99.0)
        lcc = lcc_npv_usd(
            C_construction_usd_per_m2=C_const_usd,
            design_life_years=20.0,
            margin_B1=margin_B1, margin_B2=margin_B2,
            discount_rate=0.04,
        )

        return {
            "h1_cm": round(h5[0] * 100, 1),
            "h2_cm": round(h5[1] * 100, 1),
            "h3_cm": round(h5[2] * 100, 1),
            "h4_cm": round(h5[3] * 100, 1),
            "h5_cm": round(h5[4] * 100, 1),
            "JTG_feasible": evaluation.feasible,
            "JTG_critical": evaluation.critical_indicator,
            "JTG_B1": round(margins.get("B1_asphalt_fatigue", 0), 2),
            "JTG_B2": round(margins.get("B2_semi_rigid_fatigue", 0), 2),
            "JTG_B3": round(margins.get("B3_ac_permanent_deformation", 0), 2),
            "JTG_B4": round(margins.get("B4_subgrade_strain", 0), 2),
            "JTG_DSR": round(dsr, 4),
            "JTG_SCR": round(scr, 4),
            "JTG_compliant": bool(is_compliant),
            "C_const_cny": round(C_const_cny, 1),
            "C_const_usd": round(C_const_usd, 2),
            "LCC_NPV_usd": round(lcc.get("NPV_total_usd_m2", 0), 2),
            "C_maint_NPV_usd": round(lcc.get("C_maintenance_NPV_usd_m2", 0), 2),
            "n_maint_events": lcc.get("n_events", 0),
            "FEA_eps_a_microstrain": round(
                fea_responses.get("epsilon_a_microstrain", 0), 2),
            "FEA_eps_z_microstrain": round(
                fea_responses.get("epsilon_z_microstrain", 0), 2),
        }
    except Exception as e:
        logger.error(f"  [{sid}] FEA/JTG evaluation FAILED: {e}")
        return {"FEA_jtg_status": "failed", "error": str(e)}


def main():
    xlsx = "experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx"
    out_dir = Path("experiments/ltpp_data/deliverables/ltpp_nchrp")
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = load_sections(xlsx)
    logger.info(f"Loaded {len(sections)} sections from {xlsx}")
    logger.info(f"Climate zones: " + ", ".join(
        f"{s['section_id']}={s['climate_zone']}" for s in sections))

    candidate_grid = build_candidate_grid()
    logger.info(f"Candidate grid size: {len(candidate_grid)} "
                f"(h_AC × h_base × h_subbase combinations)")

    results = []
    t_start = time.time()

    for i, sec in enumerate(sections):
        sid = sec["section_id"]
        logger.info(f"[{i+1}/{len(sections)}] Processing {sid} "
                    f"({sec['gps_family']}, {sec['pavement_type']}, "
                    f"climate={sec['climate_zone']}, E_sub={sec['E_subgrade']:.0f} MPa, "
                    f"ESAL={sec['ESAL_initial']:.0f}/yr)")

        # Step 1: NCHRP grid search
        t0 = time.time()
        best = design_one_section_nchrp(sec, candidate_grid)
        elapsed_design = time.time() - t0

        if best is None:
            logger.error(f"  [{sid}] NCHRP design failed")
            results.append({"section_id": sid, "status": "design_failed"})
            continue

        logger.info(
            f"  [{sid}] NCHRP best: h_AC={best['h_AC_cm']}cm "
            f"h_base={best['h_base_cm']}cm h_subbase={best['h_subbase_cm']}cm "
            f"FC={best['FC_pct']:.2f}% RD={best['RD_total_inch']:.3f}in "
            f"IRI={best['IRI_in_mi']:.1f} cost=¥{best['cost_cny']:.0f} "
            f"({elapsed_design:.1f}s, {best['n_candidates_passing']}/"
            f"{best['n_candidates_total']} pass) via={best['selected_via']}"
        )

        # Step 2: 5-layer FEA + JTG verification
        verification = verify_with_fea_and_jtg(sec, best)

        # Combine
        rec = {
            "section_id": sid,
            "state_name": sec["state_name"],
            "climate_zone": sec["climate_zone"],
            "gps_family": sec["gps_family"],
            "pavement_type": sec["pavement_type"],
            "subgrade_bin": sec["subgrade_bin"],
            "E_subgrade": sec["E_subgrade"],
            "is_baseline": sec["is_baseline"],
            "ESAL_annual": sec["ESAL_initial"],
            "ESAL_total_20yr": best["N_total_ESAL"],
            # NCHRP design (3-layer)
            "NCHRP_h_AC_cm": best["h_AC_cm"],
            "NCHRP_h_base_cm": best["h_base_cm"],
            "NCHRP_h_subbase_cm": best["h_subbase_cm"],
            # NCHRP performance predictions
            "NCHRP_FC_pct": round(best["FC_pct"], 2),
            "NCHRP_RD_total_inch": round(best["RD_total_inch"], 3),
            "NCHRP_RD_AC_inch": round(best["RD_AC_inch"], 3),
            "NCHRP_RD_base_inch": round(best["RD_base_inch"], 3),
            "NCHRP_RD_subbase_inch": round(best["RD_subbase_inch"], 3),
            "NCHRP_RD_SG_inch": round(best["RD_SG_inch"], 3),
            "NCHRP_IRI_in_mi": round(best["IRI_in_mi"], 1),
            "NCHRP_N_f_allowable": round(best["N_f_allowable"], 0),
            "NCHRP_T_design_C": best["T_design_C"],
            "NCHRP_selected_via": best["selected_via"],
            "NCHRP_n_pass": best["n_candidates_passing"],
            "NCHRP_n_total": best["n_candidates_total"],
            # NCHRP threshold checks
            "NCHRP_passes_FC": best["passes_FC"],
            "NCHRP_passes_RD": best["passes_RD"],
            "NCHRP_passes_IRI": best["passes_IRI"],
            # 5-layer FEA + JTG verification
            **verification,
            "status": "ok",
        }
        results.append(rec)

    # Save CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"nchrp_summary_{ts}.csv"

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(csv_path, index=False)

    elapsed_total = time.time() - t_start
    logger.info(f"DONE. Total time: {elapsed_total/60:.1f} min")
    logger.info(f"Summary CSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 105)
    print("NCHRP 1-37A BASELINE SUMMARY")
    print("=" * 105)
    print(f"{'Section':<10} {'GPS':<6} {'Clim':<5} {'h_AC':<6} {'h_b':<6} {'h_sb':<6} "
          f"{'FC%':<6} {'RD_in':<7} {'IRI':<6} {'DSR':<6} {'SCR':<6} {'NPV_usd':<10}")
    print("-" * 105)
    for r in results:
        if r.get("status") == "ok":
            print(
                f"{r['section_id']:<10} {r['gps_family']:<6} {r['climate_zone']:<5} "
                f"{r['NCHRP_h_AC_cm']:<6.0f} {r['NCHRP_h_base_cm']:<6.0f} "
                f"{r['NCHRP_h_subbase_cm']:<6.0f} "
                f"{r['NCHRP_FC_pct']:<6.2f} {r['NCHRP_RD_total_inch']:<7.3f} "
                f"{r['NCHRP_IRI_in_mi']:<6.1f} "
                f"{r.get('JTG_DSR', '?'):<6} {r.get('JTG_SCR', '?'):<6} "
                f"${r.get('LCC_NPV_usd', '?'):<9}"
            )
        else:
            print(f"{r['section_id']:<10} FAILED")
    print("=" * 105)


if __name__ == "__main__":
    main()
