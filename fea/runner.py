# -*- coding: utf-8 -*-
"""
fea.runner — Axisymmetric pavement FEA via ABAQUS subprocess (v0.4, 6-layer)
=============================================================================
Each call to run_fea() creates an isolated subdirectory under
<base_dir>/output/runs/<run_name>/ so that all ABAQUS scratch files
(.inp/.odb/.log/.msg/.dat/.sta/.prt/.com) stay there and the project
root stays clean.

This file runs in REGULAR Python (3.10+). It does NOT import ABAQUS
modules. Those are only used in abaqus_script.py which is launched
as a subprocess.

UPGRADE v0.4 (Phase 2A-1):
    - 4 layers → 6 layers
    - thickness/modulus/poisson length: 3 → 5
    - Layer order (top-down):
        [0] Upper AC    (SMA-13 / fine surface)
        [1] Mid AC      (AC-20 / intermediate)
        [2] Lower AC    (AC-25 / coarse bottom)
        [3] Base        (cement-stabilized aggregate / inorganic stabilized)
        [4] Subbase     (graded aggregate / lime-fly-ash treated)
    - Subgrade is fixed semi-infinite (8 m deep, scalar E and nu)
    - New FEA outputs: 3 AC sublayer mid-depth vertical stresses (for B.3
      multi-sublayer permanent deformation calculation per JTG B.3.1).
"""
import os
import sys
import json
import shutil
import subprocess
import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ABAQUS_SCRIPT = os.path.join(_THIS_DIR, 'abaqus_script.py')


def run_fea(
    thickness,
    modulus,
    poisson,
    E_subgrade,
    nu_subgrade,
    load_pressure=0.7,
    load_radius=0.1065,
    sensor_offsets=None,
    num_cpus=4,
    run_name=None,
    base_dir=None,
    abaqus_command='abaqus',
    verbose=True,
):
    """
    Run one axisymmetric ABAQUS FEA for a 6-layer pavement.

    Args:
        thickness:  list[float], 5 layer thicknesses in m (top-down):
                    [upper_AC, mid_AC, lower_AC, base, subbase]
        modulus:    list[float], 5 layer moduli in MPa, same order.
        poisson:    list[float], 5 Poisson ratios, same order.
        E_subgrade: float, subgrade modulus in MPa.
        nu_subgrade:float, subgrade Poisson ratio.
        load_pressure:  float, contact pressure in MPa (default 0.7 = BZZ-100).
        load_radius:    float, contact-patch radius in m (default 0.1065).
        sensor_offsets: list[float] or None, radial offsets (m).
        num_cpus:   int, CPUs for ABAQUS solver.
        run_name:   str or None, subdirectory name (default: timestamp).
        base_dir:   str or None, project root (default: os.getcwd()).
                    Runs go under <base_dir>/output/runs/<run_name>/.
        abaqus_command: str, ABAQUS command (default 'abaqus').
        verbose:    bool, print progress.

    Returns:
        dict with FEA results plus 'run_dir' key.
        Keys (responses):
            epsilon_a_microstrain  (AC bottom, lower AC base, JTG B.1)
            sigma_t_MPa            (semi-rigid base bottom,    JTG B.2)
            epsilon_z_microstrain  (subgrade top,              JTG B.4)
            p_AC_upper_mid_MPa     (upper AC mid-depth vertical stress, JTG B.3)
            p_AC_mid_mid_MPa       (mid AC mid-depth vertical stress,   JTG B.3)
            p_AC_lower_mid_MPa     (lower AC mid-depth vertical stress, JTG B.3)
            p_AC_base_interface_MPa       (NCAT EPC: AC/base interface)
            p_base_subgrade_interface_MPa (NCAT EPC: base/subgrade interface)
            eps_AC_upper_mid_microstrain (upper AC mid-depth vertical elastic strain)
            eps_AC_mid_mid_microstrain   (mid AC mid-depth vertical elastic strain)
            eps_AC_lower_mid_microstrain (lower AC mid-depth vertical elastic strain)

    Raises:
        ValueError, FileNotFoundError, RuntimeError
    """
    # ── Validate ───────────────────────────────────────────────────
    if len(thickness) != 5 or len(modulus) != 5 or len(poisson) != 5:
        raise ValueError(
            'thickness, modulus, poisson must each have length 5 '
            '(upper_AC, mid_AC, lower_AC, base, subbase). '
            'Got lengths {}, {}, {}.'.format(
                len(thickness), len(modulus), len(poisson)))
    if any(h <= 0 for h in thickness):
        raise ValueError('thicknesses must all be positive')
    if any(E <= 0 for E in modulus) or E_subgrade <= 0:
        raise ValueError('moduli must all be positive')

    # ── Defaults ───────────────────────────────────────────────────
    if sensor_offsets is None:
        sensor_offsets = [0.0, 0.20, 0.30, 0.60, 0.90, 1.20, 1.50]
    if base_dir is None:
        base_dir = os.getcwd()
    if run_name is None:
        run_name = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # ── Run directory under output/runs/ ───────────────────────────
    run_dir = os.path.abspath(os.path.join(base_dir, 'output', 'runs', run_name))
    os.makedirs(run_dir, exist_ok=True)

    input_data = {
        'thickness':      list(thickness),
        'modulus':        list(modulus),
        'poisson':        list(poisson),
        'E_subgrade':     float(E_subgrade),
        'nu_subgrade':    float(nu_subgrade),
        'load_pressure':  float(load_pressure),
        'load_radius':    float(load_radius),
        'sensor_offsets': list(sensor_offsets),
        'num_cpus':       int(num_cpus),
    }
    with open(os.path.join(run_dir, 'pavement_input.json'), 'w') as f:
        json.dump(input_data, f, indent=2)

    if not os.path.exists(_ABAQUS_SCRIPT):
        raise FileNotFoundError(
            'Bundled abaqus_script.py not found at {}. '
            'The fea package is not installed correctly.'.format(_ABAQUS_SCRIPT))
    shutil.copy(_ABAQUS_SCRIPT, run_dir)

    if verbose:
        print('-' * 70)
        print('[fea.run_fea v0.4 6-layer] Run dir: {}'.format(run_dir))
        layer_labels = ['Upper AC', 'Mid AC  ', 'Lower AC', 'Base    ', 'Subbase ']
        for i in range(5):
            print('  {}: h={:.3f} m  E={:7.1f} MPa  nu={:.2f}'.format(
                layer_labels[i], thickness[i], modulus[i], poisson[i]))
        print('  Subgrade: E={} MPa  nu={}'.format(E_subgrade, nu_subgrade))
        print('  Load    : P={} MPa  r={} m'.format(load_pressure, load_radius))
        print('  Launching ABAQUS CAE...')
        print('-' * 70)

    # ── Spawn ABAQUS subprocess (with 5-minute hard timeout) ───────
    cmd = '{} cae nogui=abaqus_script.py'.format(abaqus_command)
    FEA_TIMEOUT_S = 300   # 5 minutes — typical FEA is 30-60s, anything > 5min is stuck

    abaqus_output = ''
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            timeout=FEA_TIMEOUT_S,
        )
        abaqus_output = proc.stdout or ''
        with open(os.path.join(run_dir, 'abaqus_stdout.txt'), 'w',
                  encoding='utf-8', errors='replace') as f:
            f.write(abaqus_output)
        if verbose and abaqus_output:
            print(abaqus_output)
    except subprocess.TimeoutExpired:
        # Kill any zombie ABAQUS workers spawned by this run
        import platform
        if platform.system() == 'Windows':
            for proc_name in ('standard.exe', 'abaqusW.exe', 'pre.exe', 'ABQcaeK.exe'):
                subprocess.run('taskkill /F /IM {} 2>nul'.format(proc_name),
                               shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        raise RuntimeError(
            'ABAQUS CAE subprocess exceeded {}s timeout in {}. '
            'Killed worker processes. Likely caused by non-convergent FEA input.'
            .format(FEA_TIMEOUT_S, run_dir))

    error_markers = (
        'Abaqus Error:',
        'FlexNet Licensing error',
        'The desired vendor daemon is down',
        'License path:',
    )
    abaqus_error = any(m in abaqus_output for m in error_markers)
    tail = os.linesep.join(abaqus_output.strip().splitlines()[-12:])

    if proc.returncode != 0 or abaqus_error:
        raise RuntimeError(
            'ABAQUS CAE subprocess failed in {} (returncode={}). '
            'Inspect abaqus_stdout.txt / iLLM_PD_FEA.log/.msg/.dat. '
            'Output tail:\n{}'.format(run_dir, proc.returncode, tail))

    result_path = os.path.join(run_dir, 'pavement_result.json')
    if not os.path.exists(result_path):
        raise RuntimeError(
            'pavement_result.json was not produced in {}. '
            'The ABAQUS subprocess exited without a detectable error but did '
            'not write results. Inspect abaqus_stdout.txt. Output tail:\n{}'
            .format(run_dir, tail))

    with open(result_path) as f:
        result = json.load(f)
    result['run_dir'] = run_dir

    if verbose:
        print('-' * 70)
        print('[fea.run_fea v0.4] Complete. Responses:')
        r = result.get('responses', {})
        print('  ε_a (AC bot)       = {} ue'.format(r.get('epsilon_a_microstrain', 'N/A')))
        print('  σ_t (base bot)     = {} MPa'.format(r.get('sigma_t_MPa', 'N/A')))
        print('  ε_z (subgrade top) = {} ue'.format(r.get('epsilon_z_microstrain', 'N/A')))
        print('  p_AC_upper_mid     = {} MPa'.format(r.get('p_AC_upper_mid_MPa', 'N/A')))
        print('  p_AC_mid_mid       = {} MPa'.format(r.get('p_AC_mid_mid_MPa', 'N/A')))
        print('  p_AC_lower_mid     = {} MPa'.format(r.get('p_AC_lower_mid_MPa', 'N/A')))
        print('  p_AC/base interface = {} MPa'.format(
            r.get('p_AC_base_interface_MPa', 'N/A')))
        print('  p_base/subgrade interface = {} MPa'.format(
            r.get('p_base_subgrade_interface_MPa', 'N/A')))
        print('  eps_AC_upper_mid   = {} ue'.format(r.get('eps_AC_upper_mid_microstrain', 'N/A')))
        print('  eps_AC_mid_mid     = {} ue'.format(r.get('eps_AC_mid_mid_microstrain', 'N/A')))
        print('  eps_AC_lower_mid   = {} ue'.format(r.get('eps_AC_lower_mid_microstrain', 'N/A')))
        print('  Output dir         = {}'.format(run_dir))
        print('-' * 70)

    return result
