# -*- coding: utf-8 -*-
"""
Smoke tests for fea package.

Note: full FEA tests require ABAQUS and run slow (~30-60 s each).
Mark them with @pytest.mark.slow and skip unless --runslow is given.
"""
import os
import sys

import pytest


def test_import_fea():
    """fea package should import without errors."""
    import fea
    assert hasattr(fea, 'run_fea')
    assert hasattr(fea, 'verify_results')


def test_run_fea_validates_inputs():
    """run_fea should reject malformed inputs early."""
    from fea import run_fea

    with pytest.raises(ValueError, match='length 3'):
        run_fea(
            thickness=[0.15, 0.30],  # wrong length
            modulus=[1500.0, 600.0, 200.0],
            poisson=[0.30, 0.25, 0.35],
            E_subgrade=50.0,
            nu_subgrade=0.40,
        )

    with pytest.raises(ValueError, match='positive'):
        run_fea(
            thickness=[0.15, 0.30, -0.20],  # negative
            modulus=[1500.0, 600.0, 200.0],
            poisson=[0.30, 0.25, 0.35],
            E_subgrade=50.0,
            nu_subgrade=0.40,
        )


def test_verify_results_accepts_dict():
    """verify_results should accept both run_dir and result dict."""
    from fea.verify import verify_results

    fake_result = {
        'D_FEA_mm': 0.2797,
        'sigma_FEA_MPa': 0.0984,
        'epsilon_FEA_microstrain': 72.29,
        'deflection_basin_mm': {
            'r_0.00m': 0.2797,
            'r_0.20m': 0.2002,
            'r_1.50m': 0.0684,
        },
        'load': {'P_MPa': 0.7, 'r_m': 0.1065},
        'pavement': {
            'thicknesses_m': [0.15, 0.30, 0.20],
            'moduli_MPa': [1500.0, 600.0, 200.0],
            'poissons': [0.30, 0.25, 0.35],
            'E_subgrade_MPa': 50.0,
            'nu_subgrade': 0.40,
        },
        'run_dir': '/tmp/fake_run',
    }
    v = verify_results(fake_result, verbose=False)
    assert 'tier1_boussinesq' in v
    assert 'tier3_heuristics' in v
    assert v['tier3_heuristics']['sigma_tensile'] is True


@pytest.mark.slow
def test_run_fea_actually_runs(tmp_path):
    """Full FEA roundtrip — requires ABAQUS, slow."""
    pytest.importorskip('numpy')
    from fea import run_fea

    result = run_fea(
        thickness=[0.15, 0.30, 0.20],
        modulus=[1500.0, 600.0, 200.0],
        poisson=[0.30, 0.25, 0.35],
        E_subgrade=50.0,
        nu_subgrade=0.40,
        base_dir=str(tmp_path),
        run_name='smoke',
        verbose=False,
    )
    assert result['D_FEA_mm'] > 0
    assert 0.05 < result['D_FEA_mm'] < 2.0
