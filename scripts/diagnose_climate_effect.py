# -*- coding: utf-8 -*-
"""
scripts/diagnose_climate_effect.py
==================================

Quick diagnostic to answer: "How much does using REAL per-section climate change
things, versus the old fixed beijing/heavy assumption?"

It reports TWO things:

  (A) ROBUST — the seasonal variation of the AC dynamic modulus.
      The master curve converts each month's representative temperature into a
      modulus. This part needs NO FEA and is fully trustworthy. The headline is
      the max/min modulus RATIO across the year: the old fixed-modulus setup
      collapsed all of that to a single number, which is precisely why it
      produced no real climate diversity.

  (B) ILLUSTRATIVE — fixed-vs-climate fatigue life.
      This uses a crude power-law strain stand-in (NOT FEA). IMPORTANT: the
      magnitude AND EVEN THE SIGN of the life effect depend on how the AC-bottom
      tensile strain co-varies with modulus, which ONLY the FEA can resolve.
      Treat (B) as "machinery works", not as a result. Replace the strain
      provider with the FEA wrapper (see lifecycle_climate.make_fea_strain_provider)
      to get the real number.

Run:
    python scripts/diagnose_climate_effect.py
"""

from __future__ import annotations

from rl.dynamic_modulus import DynamicModulusMasterCurve, mpa_to_psi
from rl.lifecycle_climate import (
    make_powerlaw_strain_provider, fatigue_life_climate, compare_fixed_vs_climate,
)


def main() -> None:
    # ── EDIT THESE FOR A REAL SECTION ──────────────────────────────────
    # Reference modulus at 20 C for this section's AC (MPa). For a first pass
    # use the section's known/typical AC modulus; later anchor to measured E*.
    E_ref_20C_MPa = 10000.0

    # Representative AC temperature per month (Jan..Dec), deg C. Replace with the
    # section's real values (from LTPP computed pavement temps, or sensors, or an
    # air->pavement model). Below is a cold-continental example.
    monthly_AC_temps_C = [-8, -5, 2, 10, 18, 26, 30, 28, 21, 12, 3, -6]

    # The single fixed temperature the OLD setup effectively assumed.
    old_fixed_temp_C = 23.0

    annual_traffic = 2.0e6   # axle passes / year
    h_ac_mm = 180.0          # total AC thickness (mm)
    design_years = 15.0
    # ───────────────────────────────────────────────────────────────────

    mc = DynamicModulusMasterCurve(E_ref_MPa=E_ref_20C_MPa, T_ref_C=20.0, freq_hz=10.0)

    print("=" * 70)
    print("CLIMATE DIAGNOSTIC")
    print("=" * 70)
    print("Section AC anchor: {:.0f} MPa @ 20 C, 10 Hz".format(E_ref_20C_MPa))
    print("Activation energy: {:.0f} J/mol (calibrate against measured E* later)"
          .format(mc.Ea))

    # ---- (A) ROBUST: seasonal modulus variation ----
    print("\n--- (A) Seasonal AC modulus variation  [ROBUST, no FEA needed] ---")
    print("  {:>3}  {:>7}  {:>11}  {:>12}".format("Mo", "T(C)", "|E*|(MPa)", "|E*|(psi)"))
    moduli = []
    for m, T in enumerate(monthly_AC_temps_C, start=1):
        E = mc.modulus_MPa(T)
        moduli.append(E)
        print("  {:>3}  {:>7.1f}  {:>11.0f}  {:>12.0f}".format(m, T, E, mpa_to_psi(E)))
    Emax, Emin = max(moduli), min(moduli)
    print("\n  modulus MAX (coldest month) = {:>8.0f} MPa".format(Emax))
    print("  modulus MIN (hottest month) = {:>8.0f} MPa".format(Emin))
    print("  >>> seasonal MAX/MIN ratio  = {:>8.2f}x".format(Emax / Emin))
    print("  The old fixed-modulus setup represented this whole {:.1f}x span"
          .format(Emax / Emin))
    print("  with a SINGLE number — that is the climate information that was lost.")

    # ---- (B) ILLUSTRATIVE: fixed vs climate fatigue life ----
    print("\n--- (B) Fixed-vs-climate fatigue life  [ILLUSTRATIVE — needs FEA] ---")
    sp = make_powerlaw_strain_provider(eps_ref_microstrain=200.0,
                                       E_ref_MPa=E_ref_20C_MPa, exponent=0.30)
    cmp = compare_fixed_vs_climate(
        fixed_temp_C=old_fixed_temp_C, monthly_temps_C=monthly_AC_temps_C,
        annual_traffic=annual_traffic, master_curve=mc, strain_provider=sp,
        h_ac_mm=h_ac_mm, spec="JTG", design_years=design_years,
        jtg_kwargs={"VFA_pct": 70.0, "beta": 1.65})
    print("  fixed   {:.0f}C  -> life = {:>8.2f} yr".format(
        old_fixed_temp_C, cmp["fixed_life_years"]))
    print("  climate      -> life = {:>8.2f} yr".format(cmp["climate_life_years"]))
    print("  ratio (fixed/climate) = {:>6.2f}x".format(cmp["fixed_over_climate_ratio"]))
    print("\n  *** CAVEAT ***  The strain here is a crude stand-in, not FEA. With")
    print("  the JTG B.1 transfer function the result is very sensitive to how the")
    print("  AC-bottom strain co-varies with modulus — so (B)'s size and even its")
    print("  sign are NOT reliable until you plug in the real FEA strain provider.")
    print("  Only (A) is a trustworthy takeaway from this script alone.")

    print("\n" + "=" * 70)
    print("NEXT: replace make_powerlaw_strain_provider with a FEA-backed provider")
    print("(see rl/lifecycle_climate.make_fea_strain_provider) on the ABAQUS box,")
    print("then (B) becomes the real, reportable fixed-vs-climate life comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
