# -*- coding: utf-8 -*-
"""
rl/fea_strain_provider.py — Climate-coupled FEA strain provider
================================================================

Concrete implementation of the `make_fea_strain_provider` template from
rl/lifecycle_climate.py. Given a month's temperature, it:

    1. Converts temperature -> each of the 3 AC-layer moduli via that layer's
       own dynamic-modulus master curve (rl.dynamic_modulus), anchored to the
       layer's 20 C reference modulus.
    2. Keeps base + subbase moduli fixed (granular layers don't change with T).
    3. Calls fea.runner.run_fea once.
    4. Returns the FEA responses (epsilon_a for fatigue, epsilon_z, sigma_t,
       sublayer stresses) plus the thickness-weighted equivalent AC modulus.

This is the piece that turns the climate diagnostic from "modulus varies"
(input side) into "epsilon_a / N_f actually change" (output side), because
epsilon_a now comes from a real FEA run at the temperature-adjusted moduli.

IMPORTANT — runs ABAQUS:
    By default this imports fea.runner.run_fea, which launches ABAQUS. It can
    therefore only run on your ABAQUS machine. For testing the plumbing without
    ABAQUS, pass run_fea_fn=<a mock> (see scripts/run_section_climate_loop.py).

UNITS: temperature C, thickness m (as fea.runner expects), modulus MPa.
Requires rl/dynamic_modulus.py and (at run time) fea/runner.py.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

from rl.dynamic_modulus import DynamicModulusMasterCurve

Number = Union[int, float]


def build_ac_master_curves(
    anchors_20C_MPa: Sequence[float],
    freq_hz: float = 10.0,
    Ea_J_per_mol: float = 200000.0,
) -> List[DynamicModulusMasterCurve]:
    """
    Build one master curve per AC layer, each anchored to its own 20 C modulus.

    anchors_20C_MPa : e.g. [14000, 11000, 9000] for (upper, mid, lower) AC.
    Ea_J_per_mol    : shared activation energy. If you ever get measured |E*|
                      for a layer, calibrate Ea per layer (see dynamic_modulus
                      .calibrate_activation_energy) and build per-layer curves.
    """
    return [
        DynamicModulusMasterCurve(E_ref_MPa=a, T_ref_C=20.0,
                                  freq_hz=freq_hz, Ea_J_per_mol=Ea_J_per_mol)
        for a in anchors_20C_MPa
    ]


def make_climate_strain_provider(
    base_design: Dict,
    ac_master_curves: Sequence[DynamicModulusMasterCurve],
    run_fea_fn: Optional[Callable] = None,
    verbose: bool = False,
) -> Callable[[Union[Number, Sequence[float]]], Dict]:
    """
    Build the climate-coupled FEA strain provider.

    Parameters
    ----------
    base_design : dict with
        'thickness'  : list[5] layer thicknesses (m) [upperAC, midAC, lowerAC, base, subbase]
        'modulus'    : list[5] moduli (MPa); indices 0,1,2 (AC) are OVERWRITTEN
                       per temperature, indices 3,4 (base, subbase) are kept.
        'poisson'    : list[5] Poisson ratios
        'E_subgrade' : float (MPa)
        'nu_subgrade': float
    ac_master_curves : 3 master curves (upper, mid, lower AC). Their 20 C anchors
                       should equal base_design['modulus'][0:3].
    run_fea_fn : callable like fea.runner.run_fea. If None, imports the real one
                 (ABAQUS). Pass a mock for offline testing.
    verbose : forwarded to run_fea.

    Returns
    -------
    provider(temps_C) -> dict
        temps_C : a single pavement temperature (applied to all 3 AC layers),
                  or a list of 3 per-layer temperatures.
        returns : {'temps_C', 'ac_moduli_MPa', 'E_ac_equiv_MPa',
                   'eps_a_microstrain', 'eps_z_microstrain', 'sigma_t_MPa',
                   'responses'}
    """
    if len(ac_master_curves) != 3:
        raise ValueError("expected 3 AC master curves (upper, mid, lower)")

    if run_fea_fn is None:
        from fea.runner import run_fea as run_fea_fn  # noqa: requires ABAQUS

    thickness = list(base_design["thickness"])
    poisson = list(base_design["poisson"])
    base_modulus = list(base_design["modulus"])
    E_sub = float(base_design["E_subgrade"])
    nu_sub = float(base_design["nu_subgrade"])
    if not (len(thickness) == len(poisson) == len(base_modulus) == 5):
        raise ValueError("thickness/modulus/poisson must each be length 5")
    h_ac = thickness[0:3]
    if sum(h_ac) <= 0:
        raise ValueError("AC thicknesses must be positive")

    def provider(temps_C: Union[Number, Sequence[float]]) -> Dict:
        if isinstance(temps_C, (int, float)):
            T3 = [float(temps_C)] * 3
        else:
            T3 = [float(t) for t in temps_C]
            if len(T3) != 3:
                raise ValueError("temps_C must be a scalar or length-3 list")

        ac_mod = [mc.modulus_MPa(T) for mc, T in zip(ac_master_curves, T3)]
        modulus = list(base_modulus)
        modulus[0], modulus[1], modulus[2] = ac_mod[0], ac_mod[1], ac_mod[2]

        res = run_fea_fn(
            thickness=thickness, modulus=modulus, poisson=poisson,
            E_subgrade=E_sub, nu_subgrade=nu_sub, verbose=verbose,
        )
        r = res.get("responses", res)

        # thickness-weighted equivalent AC modulus for the transfer function
        E_equiv = sum(h * E for h, E in zip(h_ac, ac_mod)) / sum(h_ac)

        return {
            "temps_C": T3,
            "ac_moduli_MPa": ac_mod,
            "E_ac_equiv_MPa": E_equiv,
            "eps_a_microstrain": r.get("epsilon_a_microstrain"),
            "eps_z_microstrain": r.get("epsilon_z_microstrain"),
            "sigma_t_MPa": r.get("sigma_t_MPa"),
            "responses": r,
        }

    return provider
