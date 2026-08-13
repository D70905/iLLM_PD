# -*- coding: utf-8 -*-
"""
rl/dynamic_modulus.py — Asphalt dynamic-modulus master curve  E*(T)
====================================================================

WHY THIS FILE EXISTS
--------------------
The FEA (fea/runner.py) currently takes ONE fixed modulus per layer. Asphalt
is viscoelastic: its modulus drops sharply as temperature rises. With a fixed
modulus, changing the climate input changes NOTHING in the computed strain or
predicted life — which is exactly why the old "beijing/heavy" setup produced no
real climate diversity. This module makes the AC modulus a function of
temperature, so that real per-section climate finally enters the design.

WHAT MODEL THIS USES (and why NOT the raw Witczak equation)
-----------------------------------------------------------
We use the STANDARD sigmoidal dynamic-modulus master curve (the same functional
form AASHTOWare/MEPDG uses internally to represent E*), with an Arrhenius
time–temperature shift factor:

    log10|E*| = delta + alpha / (1 + exp(beta + gamma * log10(f_r)))
    f_r       = f * a_T                                   (reduced frequency)
    log10(a_T)= (Ea / (2.303*R)) * (1/T - 1/T_ref)        (Arrhenius shift)

We do NOT ship the Witczak 1-37A predictive equation (which predicts the master
-curve parameters from binder viscosity A-VTS + full gradation). Reason: its
exact regression coefficients could not be verified against a primary source at
build time, and shipping unverified constants would silently bias every modulus.
The Witczak / NCHRP 1-40D / Hirsch predictors remain the right tool IF you obtain
full gradation (p200, p4, p3/8, p3/4) + binder A-VTS or |G*|; in that case see
NCHRP 1-37A App. CC-4 / FHWA-HRT-10-035 and add it as an alternative builder.

THE KEY ADVANTAGE OF THE ANCHORED APPROACH
------------------------------------------
We ANCHOR the curve to a known modulus E_ref at a reference temperature T_ref
(e.g. the 20 C modulus you already have for a section). The curve is then
guaranteed to pass through that point, and the temperature dependence is set by
the (calibratable) activation energy + asymptotes. This is physically sane by
construction and needs only one modulus value to get started.

CALIBRATION
-----------
calibrate_activation_energy() fits Ea to measured (T, E) pairs — e.g. your
senior's FWD data, which provides backcalculated AC modulus both "raw" (at field
temperature) and "corrected to 20 C". That pairing is exactly what this needs.

UNITS
-----
T in degrees Celsius, f in Hz, modulus in MPa. Helpers mpa_to_psi / psi_to_mpa
provided because rl/lifecycle.py's ME-PDG functions expect psi.

This file is pure standard-library Python (math only). Run it directly:
    python rl/dynamic_modulus.py
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

# Universal gas constant times ln(10): 2.303 * R, R = 8.314 J/(mol K)
_2303R = 2.302585092994046 * 8.314462618  # ~ 19.147 J/(mol K)
_ABS_ZERO_C = 273.15

# ---- Default master-curve SHAPE for dense-graded HMA (calibratable) ----
# These describe the *shape* (asymptotes + slope); the absolute level is fixed
# by anchoring to E_ref at T_ref, so the defaults below only set how steeply the
# modulus changes with temperature. They are typical for dense-graded HMA and
# can be overridden / calibrated.
DEFAULT_LOG10_EMAX_MPA = math.log10(25000.0)  # glassy upper asymptote ~25,000 MPa
DEFAULT_LOG10_EMIN_MPA = math.log10(30.0)     # equilibrium lower asymptote ~30 MPa
DEFAULT_GAMMA = -0.50                          # sigmoid slope (negative)
DEFAULT_EA_J_PER_MOL = 200000.0                # activation energy (typical HMA)


def mpa_to_psi(e_mpa: float) -> float:
    """MPa -> psi (1 MPa = 145.037738 psi)."""
    return e_mpa * 145.0377377968587


def psi_to_mpa(e_psi: float) -> float:
    """psi -> MPa."""
    return e_psi / 145.0377377968587


class DynamicModulusMasterCurve:
    """
    Anchored sigmoidal |E*| master curve with Arrhenius temperature shift.

    Parameters
    ----------
    E_ref_MPa : float
        Known dynamic modulus at (T_ref_C, freq_hz). The curve passes through
        this point exactly. For a section you might use its 20 C modulus.
    T_ref_C : float
        Reference temperature for the anchor (default 20 C).
    freq_hz : float
        Design loading frequency (default 10 Hz, ~ highway speed at the AC mid
        -depth). Keep this consistent with how N (load repetitions) is defined.
    Ea_J_per_mol : float
        Arrhenius activation energy controlling temperature sensitivity.
    log10_Emax_MPa, log10_Emin_MPa, gamma : float
        Master-curve shape (upper/lower asymptote in log10 MPa, sigmoid slope).

    Notes
    -----
    Given the asymptotes + gamma + the anchor, the remaining sigmoid parameter
    `beta` is solved so the curve hits E_ref at (T_ref, f). delta = log10_Emin,
    alpha = log10_Emax - log10_Emin.
    """

    def __init__(
        self,
        E_ref_MPa: float,
        T_ref_C: float = 20.0,
        freq_hz: float = 10.0,
        Ea_J_per_mol: float = DEFAULT_EA_J_PER_MOL,
        log10_Emax_MPa: float = DEFAULT_LOG10_EMAX_MPA,
        log10_Emin_MPa: float = DEFAULT_LOG10_EMIN_MPA,
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        if E_ref_MPa <= 0:
            raise ValueError("E_ref_MPa must be positive")
        if freq_hz <= 0:
            raise ValueError("freq_hz must be positive")

        self.E_ref_MPa = float(E_ref_MPa)
        self.T_ref_C = float(T_ref_C)
        self.freq_hz = float(freq_hz)
        self.Ea = float(Ea_J_per_mol)
        self.delta = float(log10_Emin_MPa)
        self.alpha = float(log10_Emax_MPa) - float(log10_Emin_MPa)
        self.gamma = float(gamma)

        if self.alpha <= 0:
            raise ValueError("log10_Emax must exceed log10_Emin")

        log10_Eref = math.log10(self.E_ref_MPa)
        # Clamp the anchor strictly inside the asymptotes so the solve is valid.
        lo = self.delta + 1e-6
        hi = self.delta + self.alpha - 1e-6
        if not (lo < log10_Eref < hi):
            raise ValueError(
                "E_ref ({:.0f} MPa) must lie strictly between the asymptotes "
                "[{:.0f}, {:.0f}] MPa; adjust E_ref or the asymptotes."
                .format(self.E_ref_MPa,
                        10 ** self.delta, 10 ** (self.delta + self.alpha)))

        # Solve beta so that at T_ref (a_T = 1, log10 f_r = log10 f): E = E_ref
        # log10_Eref = delta + alpha / (1 + exp(beta + gamma*log10 f))
        frac = self.alpha / (log10_Eref - self.delta) - 1.0   # = exp(beta+gamma*log f)
        if frac <= 0:
            raise ValueError("Invalid anchor; check E_ref vs asymptotes.")
        self.beta = math.log(frac) - self.gamma * math.log10(self.freq_hz)

    # ---- shift factor & modulus ----
    def log10_shift(self, T_C: float) -> float:
        """log10(a_T) via Arrhenius, relative to T_ref."""
        T_K = T_C + _ABS_ZERO_C
        Tref_K = self.T_ref_C + _ABS_ZERO_C
        return (self.Ea / _2303R) * (1.0 / T_K - 1.0 / Tref_K)

    def modulus_MPa(self, T_C: float, freq_hz: Optional[float] = None) -> float:
        """Dynamic modulus |E*| (MPa) at temperature T_C (and optional freq)."""
        f = self.freq_hz if freq_hz is None else float(freq_hz)
        log10_fr = math.log10(f) + self.log10_shift(T_C)
        log10_E = self.delta + self.alpha / (1.0 + math.exp(self.beta + self.gamma * log10_fr))
        return 10.0 ** log10_E

    def modulus_psi(self, T_C: float, freq_hz: Optional[float] = None) -> float:
        return mpa_to_psi(self.modulus_MPa(T_C, freq_hz))

    def curve(self, temps_C: Sequence[float]) -> List[float]:
        """List of |E*| (MPa) for a sequence of temperatures."""
        return [self.modulus_MPa(T) for T in temps_C]

    def __repr__(self) -> str:
        return ("DynamicModulusMasterCurve(E_ref={:.0f}MPa @ {:.0f}C, f={:.0f}Hz, "
                "Ea={:.0f}, asymptotes=[{:.0f},{:.0f}]MPa, gamma={:.2f})"
                .format(self.E_ref_MPa, self.T_ref_C, self.freq_hz, self.Ea,
                        10 ** self.delta, 10 ** (self.delta + self.alpha), self.gamma))


def calibrate_activation_energy(
    E_ref_MPa: float,
    T_ref_C: float,
    measured_temps_C: Sequence[float],
    measured_moduli_MPa: Sequence[float],
    freq_hz: float = 10.0,
    log10_Emax_MPa: float = DEFAULT_LOG10_EMAX_MPA,
    log10_Emin_MPa: float = DEFAULT_LOG10_EMIN_MPA,
    gamma: float = DEFAULT_GAMMA,
    Ea_bounds: Tuple[float, float] = (80000.0, 320000.0),
) -> Tuple[float, DynamicModulusMasterCurve, float]:
    """
    Fit the Arrhenius activation energy Ea to measured (T, E) pairs, with the
    curve anchored at (T_ref_C, E_ref_MPa).

    Intended for your senior's FWD data: pass the 20 C-corrected AC modulus as
    E_ref_MPa (with T_ref_C = 20), and the raw (field-temperature, modulus)
    pairs as measured_temps_C / measured_moduli_MPa.

    Returns (Ea_best, fitted_curve, rmse_log10). Pure-Python golden-section
    search in log space; no scipy needed.
    """
    temps = list(measured_temps_C)
    mods = list(measured_moduli_MPa)
    if len(temps) != len(mods) or len(temps) == 0:
        raise ValueError("measured_temps_C and measured_moduli_MPa must align and be non-empty")

    def sse(Ea: float) -> float:
        mc = DynamicModulusMasterCurve(
            E_ref_MPa, T_ref_C, freq_hz, Ea,
            log10_Emax_MPa, log10_Emin_MPa, gamma)
        s = 0.0
        for T, E in zip(temps, mods):
            if E <= 0:
                continue
            s += (math.log10(mc.modulus_MPa(T)) - math.log10(E)) ** 2
        return s

    # Golden-section minimisation on Ea
    a, b = Ea_bounds
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = sse(d)
        if abs(b - a) < 1.0:
            break
    Ea_best = 0.5 * (a + b)
    curve = DynamicModulusMasterCurve(
        E_ref_MPa, T_ref_C, freq_hz, Ea_best,
        log10_Emax_MPa, log10_Emin_MPa, gamma)
    n = sum(1 for E in mods if E > 0)
    rmse = math.sqrt(sse(Ea_best) / max(n, 1))
    return Ea_best, curve, rmse


# ──────────────────────────────────────────────────────────────────────
# Self-test / demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("rl.dynamic_modulus — self-test")
    print("=" * 64)

    # Anchor at a typical dense-graded AC: 10,000 MPa at 20 C, 10 Hz
    mc = DynamicModulusMasterCurve(E_ref_MPa=10000.0, T_ref_C=20.0, freq_hz=10.0)
    print(mc)
    print("\n|E*| vs temperature (10 Hz):")
    print("  {:>6}  {:>12}  {:>10}".format("T(C)", "|E*|(MPa)", "|E*|(psi)"))
    last = None
    monotonic = True
    for T in [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        E = mc.modulus_MPa(T)
        print("  {:>6.0f}  {:>12.0f}  {:>10.0f}".format(T, E, mpa_to_psi(E)))
        if last is not None and E > last:  # must DECREASE as T rises
            monotonic = False
        last = E

    print("\nSanity checks:")
    print("  monotonic decrease with T:        {}".format("PASS" if monotonic else "FAIL"))
    e20 = mc.modulus_MPa(20.0)
    print("  passes through anchor (20C=10000): {}  (got {:.1f})"
          .format("PASS" if abs(e20 - 10000.0) < 1.0 else "FAIL", e20))
    e5, e40 = mc.modulus_MPa(5.0), mc.modulus_MPa(40.0)
    print("  cold(5C) > anchor > hot(40C):      {}  ({:.0f} > 10000 > {:.0f})"
          .format("PASS" if (e5 > 10000 > e40) else "FAIL", e5, e40))
    in_range = 700 <= e40 <= 8000 and 12000 <= e5 <= 25000
    print("  magnitudes physically plausible:   {}".format("PASS" if in_range else "CHECK"))

    # --- Calibration demo against synthetic "measured" data ---
    print("\nCalibration demo (fit Ea to synthetic measured points):")
    truth = DynamicModulusMasterCurve(10000.0, 20.0, 10.0, Ea_J_per_mol=185000.0)
    meas_T = [-5, 5, 15, 25, 35]
    meas_E = [truth.modulus_MPa(T) for T in meas_T]
    Ea_fit, fitted, rmse = calibrate_activation_energy(
        E_ref_MPa=10000.0, T_ref_C=20.0,
        measured_temps_C=meas_T, measured_moduli_MPa=meas_E, freq_hz=10.0)
    print("  true Ea = 185000  ->  fitted Ea = {:.0f}  (rmse_log10 = {:.4f})"
          .format(Ea_fit, rmse))
    print("  recovery: {}".format("PASS" if abs(Ea_fit - 185000.0) < 5000 else "FAIL"))
    print("=" * 64)
    print("dynamic_modulus ready.")
