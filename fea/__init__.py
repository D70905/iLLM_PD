"""
fea — Axisymmetric pavement FEA via ABAQUS
==============================================

Public API:
    run_fea(...)         — Run one axisymmetric ABAQUS FEA
    verify_results(...)  — Three-tier verification

Usage:
    from fea import run_fea, verify_results
    
    result = run_fea(
        thickness=[0.15, 0.30, 0.20],
        modulus=[1500.0, 600.0, 200.0],
        poisson=[0.30, 0.25, 0.35],
        E_subgrade=50.0,
        nu_subgrade=0.40,
    )
    verification = verify_results(result['run_dir'])
"""
from fea.runner import run_fea
from fea.verify import verify_results

__version__ = '0.2.0'
__all__ = ['run_fea', 'verify_results']
