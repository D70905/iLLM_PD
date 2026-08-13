# -*- coding: utf-8 -*-
"""
rl/lifecycle_climate.py — Climate-coupled life prediction (Miner accumulation)
==============================================================================

WHAT THIS DOES
--------------
Wires together three things WITHOUT rewriting your existing transfer functions:

    real monthly temperature  ->  E*(T) [rl.dynamic_modulus]
                              ->  FEA strain at that modulus [you supply]
                              ->  monthly fatigue life N_f [rl.lifecycle]
                              ->  Miner damage summed over the year/design life

This converts the current "single representative-condition CHECK" into a
"climate-resolved life prediction" — the step that lets real per-section climate
actually change the predicted life (and removes the beijing/heavy artefact).

It deliberately calls your EXISTING rl.lifecycle functions (ac_fatigue_life_Nf1
for JTG; mepdg_fatigue_life_Nf for ME-PDG) rather than reimplementing them.

THE ONE THING YOU MUST SUPPLY: a `strain_provider`
--------------------------------------------------
Fatigue uses the AC-bottom tensile strain, which depends on the (temperature
-dependent) AC modulus. So we need: strain = f(E_ac). Two ways:

  * REAL (on your ABAQUS machine): wrap fea.runner.run_fea so that, given the
    AC modulus for a month, it runs the FEA and returns epsilon_a. Use
    make_fea_strain_provider(...) below as a template.

  * ILLUSTRATIVE (for testing without ABAQUS): make_powerlaw_strain_provider()
    approximates eps_a(E) = eps_ref * (E_ref / E)**p. This is ONLY a stand-in to
    exercise the machinery and to get an order-of-magnitude diagnostic; it is
    NOT a substitute for FEA and must not be used for reported results.

UNITS: T in C, modulus in MPa, strain in microstrain, N in axle passes.
Pure standard-library Python. Requires rl/lifecycle.py and rl/dynamic_modulus.py.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from rl.dynamic_modulus import DynamicModulusMasterCurve, mpa_to_psi
from rl.lifecycle import ac_fatigue_life_Nf1, mepdg_fatigue_life_Nf

StrainProvider = Callable[[float], float]  # E_ac_MPa -> epsilon_a (microstrain)


# ──────────────────────────────────────────────────────────────────────
# Strain providers
# ──────────────────────────────────────────────────────────────────────
def make_powerlaw_strain_provider(
    eps_ref_microstrain: float,
    E_ref_MPa: float,
    exponent: float = 0.30,
) -> StrainProvider:
    """
    ILLUSTRATIVE ONLY. eps_a(E) = eps_ref * (E_ref / E)**exponent.

    As AC modulus rises (colder), the AC bottom tensile strain falls, so the
    exponent is positive (~0.2-0.4 is a rough range for AC-bottom strain). This
    is a stand-in to test the pipeline without ABAQUS — replace with real FEA.
    """
    if eps_ref_microstrain <= 0 or E_ref_MPa <= 0:
        raise ValueError("eps_ref and E_ref must be positive")

    def provider(E_ac_MPa: float) -> float:
        return eps_ref_microstrain * (E_ref_MPa / max(E_ac_MPa, 1e-6)) ** exponent

    return provider


def make_fea_strain_provider(run_fea_callable, base_design: Dict,
                             ac_layer_indices: Sequence[int] = (0, 1, 2)) -> StrainProvider:
    """
    TEMPLATE for the REAL provider on your ABAQUS machine (do not call here —
    fea.runner needs ABAQUS). Given a month's AC modulus, set every AC sublayer
    modulus to that value (or scale them), run the FEA, return epsilon_a.

        base_design = {
            'thickness': [...5...], 'modulus': [...5...], 'poisson': [...5...],
            'E_subgrade': ..., 'nu_subgrade': ...,
        }

    Example body (uncomment on the ABAQUS machine):

        def provider(E_ac_MPa):
            mod = list(base_design['modulus'])
            for i in ac_layer_indices:
                mod[i] = E_ac_MPa
            res = run_fea_callable(
                thickness=base_design['thickness'], modulus=mod,
                poisson=base_design['poisson'],
                E_subgrade=base_design['E_subgrade'],
                nu_subgrade=base_design['nu_subgrade'], verbose=False)
            return res['responses']['epsilon_a_microstrain']
        return provider
    """
    raise NotImplementedError(
        "make_fea_strain_provider is a template; implement its body on the "
        "ABAQUS machine (see docstring).")


# ──────────────────────────────────────────────────────────────────────
# Miner accumulation
# ──────────────────────────────────────────────────────────────────────
def miner_damage(n_applied: Sequence[float], N_allow: Sequence[float]) -> float:
    """Sum_i n_i / N_i  (Miner's linear damage rule)."""
    D = 0.0
    for n, N in zip(n_applied, N_allow):
        if N is None or N <= 0:
            continue
        if N >= float("inf") / 2:
            continue
        D += n / N
    return D


# ──────────────────────────────────────────────────────────────────────
# Climate-resolved AC fatigue (JTG B.1 by default; ME-PDG optional)
# ──────────────────────────────────────────────────────────────────────
def fatigue_life_climate(
    monthly_temps_C: Sequence[float],
    monthly_traffic: Sequence[float],
    master_curve: DynamicModulusMasterCurve,
    strain_provider: StrainProvider,
    h_ac_mm: float,
    *,
    spec: str = "JTG",
    design_years: float = 15.0,
    jtg_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Climate-resolved AC fatigue via Miner accumulation over a representative year.

    Parameters
    ----------
    monthly_temps_C : 12 representative AC temperatures (one per month).
    monthly_traffic : 12 axle-pass counts (one per month) for ONE year.
    master_curve    : DynamicModulusMasterCurve (T -> E_ac).
    strain_provider : E_ac_MPa -> epsilon_a (microstrain). REAL = FEA wrapper.
    h_ac_mm         : total AC thickness (mm).
    spec            : 'JTG' (uses ac_fatigue_life_Nf1) or 'MEPDG'
                      (uses mepdg_fatigue_life_Nf, modulus in psi).
    design_years    : design period; total damage is scaled by this many years.
    jtg_kwargs      : extra args forwarded to ac_fatigue_life_Nf1
                      (VFA_pct, beta, k_a, k_T1). NOTE: k_T1 should be left at
                      1.0 here because temperature now enters physically through
                      E*(T); do NOT double-count it via k_T1.

    Returns
    -------
    dict with per-month breakdown, annual damage, damage over design_years,
    and the implied fatigue life in years (design_years / D_design * design_years
    -> simplified: years_to_D1 = design_years / D_design).
    """
    jtg_kwargs = dict(jtg_kwargs or {})
    months = list(zip(monthly_temps_C, monthly_traffic))
    rows: List[Dict] = []
    n_list: List[float] = []
    N_list: List[float] = []

    for m, (T, n) in enumerate(months, start=1):
        E_mpa = master_curve.modulus_MPa(T)
        eps = strain_provider(E_mpa)
        if spec.upper() == "MEPDG":
            Nf = mepdg_fatigue_life_Nf(eps, mpa_to_psi(E_mpa), h_ac_mm / 25.4)
        else:
            Nf = ac_fatigue_life_Nf1(eps, E_mpa, h_ac_mm, **jtg_kwargs)
        rows.append({"month": m, "T_C": round(T, 2), "E_MPa": round(E_mpa, 1),
                     "eps_a_ue": round(eps, 2),
                     "Nf": Nf, "n_month": n})
        n_list.append(n)
        N_list.append(Nf)

    D_annual = miner_damage(n_list, N_list)
    D_design = D_annual * design_years
    years_to_failure = (1.0 / D_annual) if D_annual > 0 else float("inf")

    return {
        "spec": spec.upper(),
        "months": rows,
        "annual_traffic": sum(n_list),
        "D_annual": D_annual,
        "D_design": D_design,
        "design_years": design_years,
        "predicted_fatigue_life_years": years_to_failure,
        "passes_design_life": (years_to_failure >= design_years),
    }


def compare_fixed_vs_climate(
    fixed_temp_C: float,
    monthly_temps_C: Sequence[float],
    annual_traffic: float,
    master_curve: DynamicModulusMasterCurve,
    strain_provider: StrainProvider,
    h_ac_mm: float,
    *,
    spec: str = "JTG",
    design_years: float = 15.0,
    jtg_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Compare a single fixed-temperature check (the old beijing-style assumption)
    against the climate-resolved Miner result, holding total annual traffic equal.
    Traffic is split evenly across months for the climate case.
    """
    n_each = annual_traffic / 12.0
    climate = fatigue_life_climate(
        monthly_temps_C, [n_each] * 12, master_curve, strain_provider,
        h_ac_mm, spec=spec, design_years=design_years, jtg_kwargs=jtg_kwargs)

    # Fixed-temperature single check at the same total annual traffic
    fixed = fatigue_life_climate(
        [fixed_temp_C] * 12, [n_each] * 12, master_curve, strain_provider,
        h_ac_mm, spec=spec, design_years=design_years, jtg_kwargs=jtg_kwargs)

    life_fixed = fixed["predicted_fatigue_life_years"]
    life_clim = climate["predicted_fatigue_life_years"]
    ratio = (life_fixed / life_clim) if life_clim not in (0, float("inf")) else float("inf")

    return {
        "fixed_temp_C": fixed_temp_C,
        "fixed_life_years": life_fixed,
        "climate_life_years": life_clim,
        "fixed_over_climate_ratio": ratio,
        "fixed_detail": fixed,
        "climate_detail": climate,
    }


# ──────────────────────────────────────────────────────────────────────
# Self-test / demo (illustrative strain provider — NOT for reported results)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("rl.lifecycle_climate — self-test (ILLUSTRATIVE strain provider)")
    print("=" * 64)

    # Master curve anchored at 10,000 MPa @ 20 C
    mc = DynamicModulusMasterCurve(E_ref_MPa=10000.0, T_ref_C=20.0, freq_hz=10.0)

    # Illustrative strain: 200 ue at the 20 C modulus, falls as modulus rises
    sp = make_powerlaw_strain_provider(eps_ref_microstrain=200.0,
                                       E_ref_MPa=10000.0, exponent=0.30)

    # A cold-continental monthly AC-temperature profile (deg C), Jan..Dec
    cold_profile = [-8, -5, 2, 10, 18, 26, 30, 28, 21, 12, 3, -6]
    annual_N = 2.0e6  # axle passes per year

    print("\nMonthly fatigue breakdown (JTG B.1, k_T1=1 since T enters via E*):")
    res = fatigue_life_climate(cold_profile, [annual_N / 12] * 12, mc, sp,
                               h_ac_mm=180.0, spec="JTG", design_years=15.0,
                               jtg_kwargs={"VFA_pct": 70.0, "beta": 1.65})
    print("  {:>3} {:>7} {:>9} {:>9} {:>12}".format("Mo", "T(C)", "E(MPa)", "eps(ue)", "Nf"))
    for r in res["months"]:
        print("  {:>3} {:>7.1f} {:>9.0f} {:>9.1f} {:>12.3e}"
              .format(r["month"], r["T_C"], r["E_MPa"], r["eps_a_ue"], r["Nf"]))
    print("\n  annual damage D     = {:.4e}".format(res["D_annual"]))
    print("  predicted life      = {:.1f} years".format(res["predicted_fatigue_life_years"]))

    print("\nFixed-temp (beijing-style, 23 C) vs climate-resolved:")
    cmp = compare_fixed_vs_climate(
        fixed_temp_C=23.0, monthly_temps_C=cold_profile, annual_traffic=annual_N,
        master_curve=mc, strain_provider=sp, h_ac_mm=180.0, spec="JTG",
        design_years=15.0, jtg_kwargs={"VFA_pct": 70.0, "beta": 1.65})
    print("  fixed   23C  life = {:>8.1f} yr".format(cmp["fixed_life_years"]))
    print("  climate      life = {:>8.1f} yr".format(cmp["climate_life_years"]))
    print("  fixed / climate   = {:>8.2f}x  (>1 means fixed-T was optimistic)"
          .format(cmp["fixed_over_climate_ratio"]))
    print("\n  Interpretation: the gap is the size of the error the old fixed-")
    print("  climate assumption was hiding. Replace the illustrative strain")
    print("  provider with the FEA wrapper to get the real number.")
    print("=" * 64)
    print("lifecycle_climate ready.")
