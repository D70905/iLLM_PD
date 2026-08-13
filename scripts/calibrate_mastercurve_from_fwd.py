# -*- coding: utf-8 -*-
"""
scripts/calibrate_mastercurve_from_fwd.py
=========================================

Template: calibrate the dynamic-modulus master curve against your senior's FWD
data. Her workbook gives, per test point, the backcalculated AC layer modulus
both at the field temperature ("raw") and "corrected to 20 C". That pairing is
exactly what the master curve needs:

    * E_ref_20C_MPa  = the 20 C-corrected AC modulus  -> the anchor at T_ref=20
    * (raw_temp, raw_modulus) pairs -> the points used to fit activation energy Ea

This script shows the workflow with HARD-CODED example numbers. Replace them by
reading the AC layer (e.g. C1 or a thickness-weighted AC modulus) out of her
Excel. Keep modulus in MPa and temperature in C.

NOTE: FWD backcalculated moduli are LAYER moduli (whole-layer), not lab |E*|.
That is fine for fitting how THIS material's stiffness varies with temperature
(which is what we want for the temperature coupling); just report it as such.

Run:
    python scripts/calibrate_mastercurve_from_fwd.py
"""

from __future__ import annotations

from rl.dynamic_modulus import calibrate_activation_energy, DynamicModulusMasterCurve


def main() -> None:
    # ── REPLACE with values read from 师姐 Excel (AC layer) ─────────────
    # The 20 C-corrected AC modulus for the chosen point (anchor):
    E_ref_20C_MPa = 11000.0

    # Raw (field-temperature, modulus) pairs for that same point/section.
    # Example: warmer days -> lower backcalculated AC modulus.
    raw_temps_C   = [ 5.0,  12.0,  20.0,  28.0,  34.0]
    raw_moduli_MPa = [16500.0, 13200.0, 11000.0, 8200.0, 6300.0]
    # ───────────────────────────────────────────────────────────────────

    Ea, curve, rmse = calibrate_activation_energy(
        E_ref_MPa=E_ref_20C_MPa, T_ref_C=20.0,
        measured_temps_C=raw_temps_C, measured_moduli_MPa=raw_moduli_MPa,
        freq_hz=10.0)

    print("=" * 64)
    print("MASTER-CURVE CALIBRATION FROM FWD DATA")
    print("=" * 64)
    print("Anchor: {:.0f} MPa @ 20 C".format(E_ref_20C_MPa))
    print("Fitted activation energy Ea = {:.0f} J/mol".format(Ea))
    print("Fit RMSE (log10 MPa)        = {:.4f}".format(rmse))
    print("\nMeasured vs fitted:")
    print("  {:>7} {:>12} {:>12} {:>8}".format("T(C)", "meas(MPa)", "fit(MPa)", "ratio"))
    for T, E in zip(raw_temps_C, raw_moduli_MPa):
        Ef = curve.modulus_MPa(T)
        print("  {:>7.1f} {:>12.0f} {:>12.0f} {:>8.2f}".format(T, E, Ef, Ef / E))
    print("\nUse this fitted curve object in the climate pipeline:")
    print("  {}".format(curve))
    print("=" * 64)
    print("If the fit ratios are far from 1.0, the default asymptotes/gamma may")
    print("need adjusting for this material — pass them into calibrate_*() too.")


if __name__ == "__main__":
    main()
