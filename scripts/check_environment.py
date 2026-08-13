# -*- coding: utf-8 -*-
"""
Environment health check.

Run after `conda activate illm_pd`:
    python scripts/check_environment.py
"""
import os
import sys
import importlib
import subprocess


def check(label, ok, detail=''):
    mark = '[OK]  ' if ok else '[FAIL]'
    print('  {}  {:<35}  {}'.format(mark, label, detail))
    return ok


def main():
    print('=' * 70)
    print('iLLM-PD environment check')
    print('=' * 70)

    all_ok = True

    # Python version
    py = sys.version_info
    py_str = '{}.{}.{}'.format(py.major, py.minor, py.micro)
    all_ok &= check('Python >= 3.10', py >= (3, 10), 'detected {}'.format(py_str))

    # Phase 1 deps
    print()
    print('Phase 1 (FEA + verify):')
    for pkg in ['numpy', 'scipy', 'yaml']:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            all_ok &= check(pkg, True, 'v{}'.format(ver))
        except ImportError:
            all_ok &= check(pkg, False, 'NOT INSTALLED')

    # Phase 2 deps (informational)
    print()
    print('Phase 2 (RL) — optional:')
    for pkg in ['torch', 'stable_baselines3', 'gymnasium']:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            check(pkg, True, 'v{}'.format(ver))
        except ImportError:
            check(pkg, False, '(not installed; Phase 2)')

    # Phase 3 deps (informational)
    print()
    print('Phase 3 (Agents) — optional:')
    for pkg in ['langgraph', 'openai', 'anthropic']:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            check(pkg, True, 'v{}'.format(ver))
        except ImportError:
            check(pkg, False, '(not installed; Phase 3)')

    # ABAQUS on PATH
    print()
    print('External:')
    try:
        rc = subprocess.run(
            'abaqus information=release',
            shell=True, capture_output=True, text=True, timeout=20,
        )
        if rc.returncode == 0:
            ver_line = ''
            for line in rc.stdout.splitlines():
                if 'Abaqus' in line:
                    ver_line = line.strip()
                    break
            all_ok &= check('ABAQUS on PATH', True, ver_line or 'present')
        else:
            all_ok &= check('ABAQUS on PATH', False, 'not detected')
    except Exception as e:
        all_ok &= check('ABAQUS on PATH', False, 'check failed: {}'.format(e))

    # Project structure
    print()
    print('Project structure:')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ['fea', 'config', 'output', 'tests', 'examples', 'third_party']:
        path = os.path.join(project_root, d)
        check('  ' + d + '/', os.path.isdir(path),
              'exists' if os.path.isdir(path) else 'missing')

    # PyMastic
    pym = os.path.join(project_root, 'third_party', 'PyMastic-master',
                       'Main', 'MLE.py')
    if os.path.isfile(pym):
        check('PyMastic (Main/MLE.py)', True, 'found')
    else:
        check('PyMastic (Main/MLE.py)', False,
              'put PYMASTIC-master in third_party/')

    print()
    print('=' * 70)
    if all_ok:
        print('READY for Phase 1 (FEA + verification).')
        print('Try: python examples/run_fea_demo.py')
    else:
        print('Some Phase 1 requirements are missing — see [FAIL] above.')
    print('=' * 70)


if __name__ == '__main__':
    main()
