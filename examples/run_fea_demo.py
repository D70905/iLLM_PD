# -*- coding: utf-8 -*-
"""
Demo: end-to-end FEA + verification

Run from project root (D:\\iLLM_PD_new\\):
    conda activate illm_pd
    python examples/run_fea_demo.py

Produces:
    output/runs/run_<timestamp>/
        pavement_input.json
        pavement_result.json
        verification.json
        iLLM_PD_FEA.{inp,odb,log,msg,dat,sta,prt,com}
        abaqus_script.py (copy, for reproducibility)
"""
import os
import sys

# Make project root importable when running from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea import run_fea, verify_results


def main():
    # Default heavy-duty flexible pavement BZZ-100
    result = run_fea(
        thickness     = [0.15, 0.30, 0.20],
        modulus       = [1500.0, 600.0, 200.0],
        poisson       = [0.30, 0.25, 0.35],
        E_subgrade    = 50.0,
        nu_subgrade   = 0.40,
        load_pressure = 0.7,
        load_radius   = 0.1065,
        num_cpus      = 4,
        verbose       = True,
    )

    print()
    print('=' * 70)
    print('FEA finished. Running verification...')
    print('=' * 70)
    print()

    verification = verify_results(result['run_dir'], verbose=True)

    t2 = verification['tier2_pymastic']
    if t2.get('status') == 'ok':
        d_diff = abs(t2['D']['diff_pct'] or 0)
        if d_diff <= 15:
            print()
            print('[PASS] FEA deflection matches PyMastic within 15 %.')
        else:
            print()
            print('[CHECK] FEA / PyMastic deflection diff = {:.1f} %'.format(d_diff))
    else:
        print()
        print('[INFO] PyMastic tier skipped/failed.')
        print('       Put PYMASTIC-master folder at:')
        print('         {}/third_party/PyMastic-master/'.format(PROJECT_ROOT))


if __name__ == '__main__':
    main()
