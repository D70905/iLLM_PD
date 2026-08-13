# -*- coding: utf-8 -*-
"""
Demo (v0.4 6-layer): end-to-end FEA + spec evaluation under JTG / ME-PDG.

Run from project root (D:\\iLLM_PD_new\\):
    conda activate illm_pd
    python examples/run_with_spec_demo.py

UPGRADE v0.4 (Phase 2A-1):
    - 4-layer → 6-layer pavement model
    - 5 structural layers + subgrade
    - Tests B.3 multi-sublayer permanent deformation

This script:
  1. Runs FEA once for a 6-layer BZZ-100 configuration
  2. Evaluates the same FEA result under both JTG D50-2017 and ME-PDG
  3. Prints a side-by-side comparison
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea import run_fea
from specs import get_protocol, DesignInputs


def main():
    # ── Shared design context (6-layer) ────────────────────────
    # Layer order (top-down):
    #   [0] Upper AC    = SMA-13      4 cm  (modified asphalt)
    #   [1] Mid AC      = AC-20       6 cm  (modified asphalt)
    #   [2] Lower AC    = AC-25       8 cm  (neat 70# asphalt)
    #   [3] Base        = CTB        36 cm  (cement stab. crushed stone)
    #   [4] Subbase     = GAB        18 cm  (graded aggregate)
    common = dict(
        pavement_type='semi_rigid',
        road_class='expressway',
        traffic_level='heavy',
        thickness=[0.04, 0.06, 0.08, 0.36, 0.18],    # m
        modulus=[14000.0, 11000.0, 9000.0, 1500.0, 400.0],   # MPa
        poisson=[0.25, 0.30, 0.30, 0.25, 0.35],
        E_subgrade=60.0,
        nu_subgrade=0.40,
        design_life=15,
    )

    print('=' * 70)
    print('iLLM-PD v0.4 (6-layer) — End-to-end FEA + Spec Evaluation')
    print('=' * 70)
    print('Pavement section (top-down):')
    labels = ['Upper AC', 'Mid AC  ', 'Lower AC', 'Base    ', 'Subbase ']
    for i in range(5):
        print('  {}: h = {:.2f} cm, E = {:.0f} MPa, nu = {:.2f}'.format(
            labels[i], common['thickness'][i]*100,
            common['modulus'][i], common['poisson'][i]))
    print('  Subgrade: E = {:.0f} MPa, nu = {:.2f}'.format(
        common['E_subgrade'], common['nu_subgrade']))
    print('Total pavement thickness: {:.1f} cm'.format(
        sum(common['thickness']) * 100))
    print()

    # ── 1. Run FEA ────────────────────────────────────────────
    print('Running ABAQUS 6-layer FEA (expect ~11-15 sec)...')
    result = run_fea(
        thickness=common['thickness'],
        modulus=common['modulus'],
        poisson=common['poisson'],
        E_subgrade=common['E_subgrade'],
        nu_subgrade=common['nu_subgrade'],
        load_pressure=0.7,
        load_radius=0.1065,
        num_cpus=4,
        verbose=True,
    )

    # ── 2. Build DesignInputs ─────────────────────────────────
    inputs = DesignInputs(
        pavement_type=common['pavement_type'],
        road_class=common['road_class'],
        traffic_level=common['traffic_level'],
        thickness=common['thickness'],
        modulus=common['modulus'],
        poisson=common['poisson'],
        E_subgrade=common['E_subgrade'],
        nu_subgrade=common['nu_subgrade'],
        design_life=common['design_life'],
        extras={
            'city':       'beijing',
            'ac_grade':   'modified_asphalt_SBS',
            'base_type':  'inorganic_stabilized_granular',
            'construction_type': 'new_construction',
            'frost_zone': 'non_frost',
            'VFA_pct':    70.0,
            'R_s_MPa':    1.0,
            'R_0_mm':     1.5,
        },
    )

    fea_responses = result.get('responses', {})

    # ── 3. Evaluate under each protocol ───────────────────────
    for proto_name in ['JTG_D50_2017', 'MEPDG']:
        protocol = get_protocol(proto_name)
        evaluation = protocol.evaluate(inputs, fea_responses)
        rewards = protocol.reward_components(evaluation)

        print()
        print('=' * 70)
        print('[{}]  {}'.format(proto_name, protocol.name))
        print('=' * 70)
        print('  Design context: city=beijing  traffic=heavy  class=expressway')
        print('                  type=semi_rigid  life=15 yr')
        if hasattr(evaluation, 'details') and 'temperature_source' in evaluation.details:
            print('  Temperature source: {}'.format(evaluation.details['temperature_source']))
            print('  T_pef = {:.1f} C (h_AC_total = {:.0f} mm)'.format(
                evaluation.details.get('T_pef_C', 0),
                evaluation.details.get('h_AC_total_mm', 0)))
            print('  E_AC equivalent = {:.0f} MPa'.format(
                evaluation.details.get('E_AC_equivalent_MPa', 0)))
        print()
        print('  Feasible:           {}'.format(evaluation.feasible))
        print('  Critical indicator: {}'.format(evaluation.critical_indicator))
        print()
        print('  Margins (capacity / demand, >1 means pass):')
        for k, v in evaluation.margins.items():
            tag = '[OK]' if v >= 1.0 else ('[WARN]' if v >= 0.7 else '[FAIL]')
            print('    {:<35} {:8.3f}   {}'.format(k, v, tag))
        print()
        print('  FEA responses used:')
        for k, v in evaluation.responses.items():
            if isinstance(v, (int, float)):
                print('    {:<35} {:.4g}'.format(k, v))
        if hasattr(evaluation, 'details'):
            detail_keys = [k for k in evaluation.details.keys()
                           if not k.startswith('_')]
            if detail_keys:
                print()
                print('  Details:')
                for k in detail_keys:
                    v = evaluation.details[k]
                    if isinstance(v, (int, float)):
                        print('    {:<35} {:.4g}'.format(k, v))

    # ── 4. Summary ────────────────────────────────────────────
    print()
    print('=' * 70)
    print('CROSS-SPEC SUMMARY (v0.4 6-layer model)')
    print('=' * 70)
    print('  Same FEA result evaluated under:')
    print('    JTG D50-2017  (China,  4 indicators, B.3 multi-sublayer)')
    print('    ME-PDG         (US,    2 indicators)')
    print()
    print('  6-layer FEA provides 3 AC sublayer mid-depth stresses,')
    print('  enabling proper JTG B.3.1 multi-sublayer permanent deformation.')


if __name__ == '__main__':
    main()
