# -*- coding: utf-8 -*-
"""
fea.verify — Three-tier verification of FEA results (v0.5, 6-layer)
=====================================================================

Tier 1 — Boussinesq half-space (always; zero dependencies)
Tier 2 — PyMastic multilayer elastic theory (if available)
Tier 3 — Physics heuristics (always)

PyMastic notes:
  - Module file is Main/MLE.py inside the PYMASTIC-master folder.
  - Import is: `from Main.MLE import PyMastic`.
  - Sign convention: positive = compressive (opposite of ABAQUS).
    We negate PyMastic stresses/strains before comparing.
  - Line 83 in MLE.py: ``E *= 1000`` (hardcoded ksi→psi conversion).
    Under metric (MPa) this makes E 1000× too large → D 1000× too small.
    We compensate by multiplying displacement by 1e6 instead of 1e3.
    Stresses/strains are unaffected (linear elasticity: D ∝ q/E).
"""
import os
import sys
import json


def _resolve_pymastic_path(run_dir, override):
    if override:
        return override
    try:
        project_root = os.path.abspath(os.path.join(run_dir, '..', '..', '..'))
        guess = os.path.join(project_root, 'third_party', 'PyMastic-master')
        if os.path.isdir(guess):
            return guess
    except Exception:
        pass
    env_path = os.environ.get('PYMASTIC_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    return None


def _try_pymastic_tier(result, pymastic_path):
    """Run Tier 2 (PyMastic). Returns dict with 6-layer protocol comparisons."""
    out = {'status': 'pending', 'pymastic_path': pymastic_path}
    if not pymastic_path:
        out['status'] = 'skipped'
        out['reason'] = 'no pymastic_path provided/found'
        return out
    if not os.path.isdir(pymastic_path):
        out['status'] = 'skipped'
        out['reason'] = 'path does not exist: {}'.format(pymastic_path)
        return out

    if pymastic_path not in sys.path:
        sys.path.insert(0, pymastic_path)

    try:
        from Main.MLE import PyMastic
    except ImportError as e:
        out['status'] = 'fail'
        out['reason'] = (
            'Cannot import PyMastic from {}/Main/MLE.py. Error: {}'.format(
                pymastic_path, e))
        return out

    P_MPa       = result['load']['P_MPa']
    a_m         = result['load']['r_m']
    thicknesses = result['pavement']['thicknesses_m']    # 5 layers (6-layer model)
    moduli      = result['pavement']['moduli_MPa']       # 5 moduli
    poissons    = result['pavement']['poissons']         # 5 Poisson ratios
    E_sg        = result['pavement']['E_subgrade_MPa']
    nu_sg       = result['pavement']['nu_subgrade']

    # ── 6-layer evaluation depths (top-down) ──────────────────
    # [0] Upper AC  [1] Mid AC  [2] Lower AC  [3] Base  [4] Subbase
    z_ac_bot   = thicknesses[0] + thicknesses[1] + thicknesses[2]      # AC bottom
    z_base_bot = z_ac_bot  + thicknesses[3]                            # base bottom
    z_sg_top   = z_base_bot + thicknesses[4]                           # subgrade top

    z_pm = [0.0, z_ac_bot, z_base_bot, z_sg_top]
    H_pm = list(thicknesses)
    E_pm = list(moduli) + [E_sg]
    nu_pm = list(poissons) + [nu_sg]
    isBD = [1] * (len(E_pm) - 1)

    try:
        RS = PyMastic(
            P_MPa, a_m, [0.0], z_pm, H_pm, E_pm, nu_pm,
            7e-7, isBounded=isBD, iteration=10, inverser='solve',
        )
    except Exception as e:
        out['status'] = 'fail'
        out['reason'] = 'PyMastic call raised: {}'.format(e)
        return out

    if not isinstance(RS, dict):
        out['status'] = 'fail'
        out['reason'] = 'PyMastic returned {}, expected dict'.format(type(RS))
        return out

    try:
        # ── D: surface deflection (index 0) ─────────────────────
        D_pm_m  = float(RS['Displacement_Z'][0, 0])
        # PyMastic E*=1000 compensation (see module docstring)
        D_pm_mm = D_pm_m * 1_000_000.0     # ×1000 (m→mm) × 1000 (E correction)

        # ── σ_t: base bottom radial tensile stress (index 2) ───
        sig_R_base_bot = float(RS['Stress_R'][2, 0])
        sigma_t_pm_MPa = -sig_R_base_bot    # negate: PyMastic(+) = compr → ABAQUS(+) = tensile

        # ── ε_a: AC bottom radial tensile strain (index 1) ─────
        sig_R_ac_bot = float(RS['Stress_R'][1, 0])
        sig_T_ac_bot = float(RS['Stress_T'][1, 0])
        sig_Z_ac_bot = float(RS['Stress_Z'][1, 0])
        E_ac  = moduli[2]                   # lower AC modulus
        nu_ac = poissons[2]                 # lower AC Poisson
        eps_R_ac = (sig_R_ac_bot - nu_ac * (sig_T_ac_bot + sig_Z_ac_bot)) / E_ac
        # PyMastic(+) = compr → negate for tensile strain
        eps_a_pm_ue = -eps_R_ac * 1.0e6

        # ── ε_z: subgrade top vertical compressive strain (index 3)
        sig_Z_sg_top = float(RS['Stress_Z'][3, 0])
        sig_R_sg_top = float(RS['Stress_R'][3, 0])
        sig_T_sg_top = float(RS['Stress_T'][3, 0])
        eps_Z_sg = (sig_Z_sg_top - nu_sg * (sig_R_sg_top + sig_T_sg_top)) / E_sg
        # PyMastic(+) = compr → vertical compression is positive → keep sign
        eps_z_pm_ue = eps_Z_sg * 1.0e6     # positive = compressive
    except (KeyError, IndexError, TypeError) as e:
        out['status'] = 'fail'
        out['reason'] = 'Unexpected PyMastic result format: {}'.format(e)
        out['keys'] = list(RS.keys()) if isinstance(RS, dict) else None
        return out

    # ── FEA responses ──────────────────────────────────────────
    responses    = result.get('responses', {})
    D_fea_mm     = result['D_FEA_mm']
    sigma_fea    = result['sigma_FEA_MPa']                      # backward-compat
    eps_fea      = result['epsilon_FEA_microstrain']             # backward-compat

    # 6-layer protocol fields (preferred)
    sigma_t_fea  = responses.get('sigma_t_MPa', sigma_fea)
    eps_a_fea    = responses.get('epsilon_a_microstrain', eps_fea)
    eps_z_fea    = responses.get('epsilon_z_microstrain')

    def _pct(fea, ref):
        return None if ref == 0 else round((fea - ref) / abs(ref) * 100, 1)

    def _status(rel_pct):
        if rel_pct is None: return 'NA'
        a = abs(rel_pct)
        if a <= 10: return 'OK'
        if a <= 20: return 'WARN'
        return 'FAIL'

    out['status'] = 'ok'

    # Comparison 1: deflection
    out['D'] = {
        'PyMastic_mm': round(D_pm_mm, 4),
        'FEA_mm':      D_fea_mm,
        'diff_pct':    _pct(D_fea_mm, D_pm_mm),
        'status':      _status(_pct(D_fea_mm, D_pm_mm)),
    }

    # Comparison 2: base bottom tensile stress (σ_t, JTG B2)
    out['sigma_t_base_bottom'] = {
        'PyMastic_MPa': round(sigma_t_pm_MPa, 4),
        'FEA_MPa':      sigma_t_fea,
        'diff_pct':     _pct(sigma_t_fea, sigma_t_pm_MPa),
        'status':       _status(_pct(sigma_t_fea, sigma_t_pm_MPa)),
        '_note':        'Base bottom radial tensile stress (JTG B2)',
    }

    # Comparison 3: AC bottom tensile strain (ε_a, JTG B1)
    out['epsilon_a_AC_bottom'] = {
        'PyMastic_microstrain': round(eps_a_pm_ue, 2),
        'FEA_microstrain':      eps_a_fea,
        'diff_pct':             _pct(eps_a_fea, eps_a_pm_ue),
        'status':               _status(_pct(eps_a_fea, eps_a_pm_ue)),
        '_note':                'AC bottom radial tensile strain (JTG B1)',
    }

    # Comparison 4: subgrade top vertical compressive strain (ε_z, JTG B4)
    if eps_z_fea is not None and eps_z_fea > 0:
        out['epsilon_z_subgrade_top'] = {
            'PyMastic_microstrain': round(eps_z_pm_ue, 2),
            'FEA_microstrain':      eps_z_fea,
            'diff_pct':             _pct(eps_z_fea, eps_z_pm_ue),
            'status':               _status(_pct(eps_z_fea, eps_z_pm_ue)),
            '_note':                'Subgrade top vertical compressive strain (JTG B4)',
        }

    # Backward-compat (keep old keys for consumers that read these):
    out['sigma_surface_bottom'] = {
        'PyMastic_MPa': round(-float(RS['Stress_R'][1, 0]), 4),
        'FEA_MPa':      sigma_fea,
        'diff_pct':     _pct(sigma_fea, -float(RS['Stress_R'][1, 0])),
        'status':       _status(_pct(sigma_fea, -float(RS['Stress_R'][1, 0]))),
        '_note':        'Backward-compat legacy comparison',
    }
    sig_R_leg = float(RS['Stress_R'][2, 0])
    sig_T_leg = float(RS['Stress_T'][2, 0])
    sig_Z_leg = float(RS['Stress_Z'][2, 0])
    e_leg = moduli[1] if len(moduli) > 1 else moduli[0]
    nu_leg = poissons[1] if len(poissons) > 1 else poissons[0]
    eps_R_leg = (sig_R_leg - nu_leg * (sig_T_leg + sig_Z_leg)) / e_leg
    eps_leg_ue = -eps_R_leg * 1.0e6
    out['epsilon_base_bottom'] = {
        'PyMastic_microstrain': round(eps_leg_ue, 2),
        'FEA_microstrain':      eps_fea,
        'diff_pct':             _pct(eps_fea, eps_leg_ue),
        'status':               _status(_pct(eps_fea, eps_leg_ue)),
        '_note':                'Backward-compat legacy comparison',
    }

    return out


def verify_results(run_dir_or_result, pymastic_path=None, verbose=True):
    """
    Verify FEA results from fea.run_fea against analytical references.

    Args:
        run_dir_or_result: str (path to run directory) OR dict (result)
        pymastic_path: str or None
        verbose: bool

    Returns:
        dict with tier1_boussinesq, tier2_pymastic, tier3_heuristics
    """
    if isinstance(run_dir_or_result, str):
        run_dir = os.path.abspath(run_dir_or_result)
        with open(os.path.join(run_dir, 'pavement_result.json')) as f:
            result = json.load(f)
    elif isinstance(run_dir_or_result, dict):
        result = run_dir_or_result
        run_dir = result.get('run_dir', os.getcwd())
    else:
        raise TypeError('expected str or dict, got {}'.format(type(run_dir_or_result)))

    P_MPa       = result['load']['P_MPa']
    a_m         = result['load']['r_m']
    thicknesses = result['pavement']['thicknesses_m']
    moduli      = result['pavement']['moduli_MPa']
    E_sg        = result['pavement']['E_subgrade_MPa']
    nu_sg       = result['pavement']['nu_subgrade']
    D_fea       = result['D_FEA_mm']
    sigma_fea   = result['sigma_FEA_MPa']
    eps_fea     = result['epsilon_FEA_microstrain']
    basin       = result.get('deflection_basin_mm', {})

    # ── Tier 1: Boussinesq half-space ───────────────────────────
    P_Pa     = P_MPa * 1.0e6
    E_sg_Pa  = E_sg * 1.0e6
    D_bouss  = 2.0 * P_Pa * a_m * (1 - nu_sg ** 2) / E_sg_Pa * 1000.0
    ratio    = D_fea / D_bouss if D_bouss > 0 else 0
    if ratio < 0 or ratio > 1.0:
        t1_status = 'FAIL'
    elif ratio < 0.05:
        t1_status = 'WARN'
    elif 0.10 <= ratio <= 0.60:
        t1_status = 'OK'
    else:
        t1_status = 'WARN'

    total_h = sum(thicknesses)
    E_w = (moduli[0]*thicknesses[0] + moduli[1]*thicknesses[1]
           + moduli[2]*thicknesses[2]) / (thicknesses[0] + thicknesses[1] + thicknesses[2])
    D_bouss_w = 2.0 * P_Pa * a_m * (1 - 0.35**2) / (E_w * 1.0e6) * 1000.0

    tier1 = {
        'D_subgrade_only_mm':  round(D_bouss, 4),
        'D_FEA_mm':            D_fea,
        'ratio_pct':           round(ratio * 100, 1),
        'D_weighted_ref_mm':   round(D_bouss_w, 4),
        'E_weighted_MPa':      round(E_w, 1),
        'status':              t1_status,
    }

    # ── Tier 2: PyMastic ────────────────────────────────────────
    pym_path = _resolve_pymastic_path(run_dir, pymastic_path)
    tier2 = _try_pymastic_tier(result, pym_path)

    # ── Tier 3: Heuristics ──────────────────────────────────────
    tier3 = {}
    responses = result.get('responses', {})

    # basin monotonicity
    basin_items = sorted(
        [(float(k.replace('r_', '').replace('m', '')), v)
         for k, v in basin.items() if v is not None],
        key=lambda x: x[0])
    if len(basin_items) >= 2:
        monotonic = all(basin_items[i][1] >= basin_items[i+1][1] - 1e-6
                        for i in range(len(basin_items) - 1))
        tier3['basin_monotonic'] = monotonic
        r_far, d_far = basin_items[-1]
        d_centre = basin_items[0][1] if basin_items[0][0] == 0.0 else D_fea
        if d_centre > 0:
            tier3['far_field_pct'] = round(d_far / d_centre * 100, 1)

    # Use 6-layer responses when available
    eps_a_3 = responses.get('epsilon_a_microstrain', eps_fea)
    sigma_t_3 = responses.get('sigma_t_MPa', sigma_fea)
    eps_z_3 = responses.get('epsilon_z_microstrain')

    tier3['sigma_tensile']       = sigma_t_3 > 0 if sigma_t_3 is not None else sigma_fea > 0
    tier3['epsilon_tensile']     = eps_a_3 > 0 if eps_a_3 is not None else eps_fea > 0
    tier3['D_in_typical_range']  = 0.05 <= D_fea <= 2.0
    if eps_z_3 is not None:
        tier3['eps_z_compressive'] = eps_z_3 > 0
    tier3['_note'] = 'Tier 3 now uses 6-layer responses (sigma_t, epsilon_a, epsilon_z)'

    verification = {
        'tier1_boussinesq': tier1,
        'tier2_pymastic':   tier2,
        'tier3_heuristics': tier3,
    }

    out_path = os.path.join(run_dir, 'verification.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(verification, f, indent=2)
    except Exception:
        pass

    if verbose:
        _print_report(result, verification)

    return verification


def _print_report(result, v):
    responses = result.get('responses', {})
    eps_a_3  = responses.get('epsilon_a_microstrain')
    sigma_t_3 = responses.get('sigma_t_MPa')
    eps_z_3  = responses.get('epsilon_z_microstrain')

    print('=' * 78)
    print('fea.verify v0.5 (6-layer) — FEA Verification Report')
    print('-' * 78)
    print('  FEA (backward-compat): D={:.4f} mm  sigma={:.4f} MPa  epsilon={:.2f} ue'.format(
        result['D_FEA_mm'], result['sigma_FEA_MPa'],
        result['epsilon_FEA_microstrain']))
    if eps_a_3 is not None or sigma_t_3 is not None or eps_z_3 is not None:
        print('  FEA (6-layer protocol):', end='')
        if eps_a_3 is not None:
            print('  eps_a={:.2f} ue'.format(eps_a_3), end='')
        if sigma_t_3 is not None:
            print('  sigma_t={:.4f} MPa'.format(sigma_t_3), end='')
        if eps_z_3 is not None:
            print('  eps_z={:.2f} ue'.format(eps_z_3), end='')
        print()

    # Tier 1
    print()
    print('TIER 1  Boussinesq sanity')
    t1 = v['tier1_boussinesq']
    print('  D (subgrade only)    = {} mm'.format(t1['D_subgrade_only_mm']))
    print('  FEA / Boussinesq     = {} %'.format(t1['ratio_pct']))
    print('  D (weighted E ref)   = {} mm  (E_w = {} MPa)'.format(
        t1['D_weighted_ref_mm'], t1['E_weighted_MPa']))
    print('  Status               = {}'.format(t1['status']))

    # Tier 2
    print()
    print('TIER 2  PyMastic multilayer elastic (6-layer evaluation depths)')
    t2 = v['tier2_pymastic']
    if t2.get('status') == 'ok':
        # Protocol comparisons (new)
        for key, label, unit in [
            ('D',                     'D (surface defl)',         'mm'),
            ('sigma_t_base_bottom',   'sigma_t (base bot, B2)',   'MPa'),
            ('epsilon_a_AC_bottom',   'epsilon_a (AC bot, B1)',   'ue'),
        ]:
            d = t2[key]
            print('  {:<30} ref={:>8}  fea={:>8}  diff={:>+6} %  [{}]'.format(
                label, d['PyMastic_{}'.format(
                    'mm' if unit == 'mm' else 'MPa' if unit == 'MPa' else 'microstrain')],
                d['FEA_{}'.format(
                    'mm' if unit == 'mm' else 'MPa' if unit == 'MPa' else 'microstrain')],
                d['diff_pct'], d['status']))
        if 'epsilon_z_subgrade_top' in t2:
            d = t2['epsilon_z_subgrade_top']
            print('  {:<30} ref={:>8}  fea={:>8}  diff={:>+6} %  [{}]'.format(
                'epsilon_z (sg top, B4)', d['PyMastic_microstrain'],
                d['FEA_microstrain'], d['diff_pct'], d['status']))
    else:
        print('  Status: {}'.format(t2.get('status', 'unknown')))
        if 'reason' in t2:
            print('  Reason: {}'.format(t2['reason']))
        if t2.get('status') == 'skipped':
            print('  To enable: put PYMASTIC-master at')
            print('    <project_root>/third_party/PyMastic-master/')

    # Tier 3
    print()
    print('TIER 3  Physics heuristics')
    t3 = v['tier3_heuristics']
    for k, val in t3.items():
        if k.startswith('_'):
            continue
        print('  {:<25} = {}'.format(k, val))
    print('=' * 78)
