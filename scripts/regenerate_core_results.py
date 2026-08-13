"""
regenerate_core_results.py — A_v3/B × repaired env.py → 12段核心结果
==================================================================
用法:
  python scripts/regenerate_core_results.py                    # 默认 2048
  python scripts/regenerate_core_results.py --ckpt 4032        # 续跑版
  python scripts/regenerate_core_results.py --ckpt 2048        # 原版
"""

from stable_baselines3 import PPO
from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig
from scripts.delivered_result_helper import get_delivered
import numpy as np, time, sys

# ── Checkpoint selection ──
CKPT_STEP = sys.argv[sys.argv.index('--ckpt') + 1] if '--ckpt' in sys.argv else '002048'
FLEX_BASE = 'experiments/ltpp_data/deliverables/rl_models/A_v3_flex_2048ts/checkpoints'
FLEX_CKPT = f'{FLEX_BASE}/ckpt_final_step_{CKPT_STEP}/ppo_model.zip'
SEMI_BASE = 'experiments/ltpp_data/deliverables/rl_models/B_semi_rigid_1024ts/checkpoints'
SEMI_CKPT = f'{SEMI_BASE}/ckpt_final_step_001024/ppo_model.zip'

print(f'Using FLEX checkpoint: ckpt_final_step_{CKPT_STEP}')
flex_model = PPO.load(FLEX_CKPT)
semi_model = PPO.load(SEMI_CKPT)

SECTIONS = [
    ('16_1010','flexible',      78), ('27_1085','flexible',      86), ('48_1076','flexible',     115),
    ('04_1034','flexible',      91), ('48_0001','flexible',     700), ('12_1060','flexible',     286),
    ('30_7076','semi_rigid',    59), ('04_1065','semi_rigid',    91), ('27_2023','semi_rigid',   131),
    ('06_2004','semi_rigid',   112), ('48_1109','semi_rigid',   100), ('12_4097','semi_rigid',   286),
]

INIT_BY_TYPE = {
    'flexible':   dict(h=[0.04,0.06,0.08,0.30,0.25], E=[14000,11000,9000,350,250],  nu=[0.25,0.30,0.30,0.40,0.35]),
    'semi_rigid': dict(h=[0.04,0.06,0.08,0.36,0.18], E=[14000,11000,9000,1500,400], nu=[0.25,0.30,0.30,0.25,0.35]),
}
MK = ['B1_asphalt_fatigue','B2_semi_rigid_fatigue','B3_ac_permanent_deformation','B4_subgrade_strain']

print(f'{"section":<9}{"type":>8}{"E_sub":>7}{"AC_cm":>7}{"USD/m2":>8}{"LCC":>8}{"DSR":>6}{"SCR":>6}   B1/B2/B3/B4')
print('-'*110)
t0 = time.time()
all_rows = []
for sid, ptype, E_sub in SECTIONS:
    init = INIT_BY_TYPE[ptype]
    cfg = SurrogateEnvConfig(
        pavement_type=ptype,
        init_modulus_MPa=init['E'], init_thickness_m=init['h'], init_poisson=init['nu'],
        E_subgrade=float(E_sub), max_episode_steps=20,
        llm_enabled=False, fea_verbose=False, fea_keep_runs=False,
        climate_enabled=False,  # standard conditions, no temperature conversion
        use_surrogate=True,
        surrogate_model_path='./output/surrogate_model/surrogate_v3.pt',
        fea_validation_every=999,  # never validate — fastest
    )
    env = PavementEnvWithSurrogate(cfg)
    obs, info = env.reset()
    model_use = flex_model if ptype == 'flexible' else semi_model
    for _ in range(20):
        a, _ = model_use.predict(obs, deterministic=True)
        obs, r, done, tr, info = env.step(a)

    d = get_delivered(info)
    # Use delivered_margins if available (best compliant), else last-step margins
    m = info.get('delivered_margins') or d.get('margins', {})
    env.close()

    ac = float(np.sum(np.array(d['h_cm'][:3])))
    cost_cny = d.get('cost_cny')
    c = f'{cost_cny:.0f}' if cost_cny else 'NA'
    cost_usd = cost_cny / 7.2 if cost_cny else None
    cu = f'{cost_usd:.1f}' if cost_usd else 'NA'
    # LCC via lifecycle_lcc_intl
    lcc_usd = None
    try:
        from rl.lifecycle_lcc_intl import lcc_npv_usd
        if cost_usd and m:
            r = lcc_npv_usd(
                C_construction_usd_per_m2=cost_usd,
                design_life_years=20.0,
                margin_B1=float(m.get('B1_asphalt_fatigue', float('inf'))),
                margin_B2=float(m.get('B2_semi_rigid_fatigue', float('inf'))),
                discount_rate=0.04,
            )
            lcc_usd = r.get('NPV_total_usd_m2')
    except Exception:
        pass
    lu = f'{lcc_usd:.1f}' if lcc_usd else 'NA'
    sc = f'{d["scr_running"]:.3f}' if d.get('scr_running') else 'NA'
    flag = '' if d['compliant'] else ' <NON-COMPLIANT>'
    mrgs = ' / '.join(f'{m.get(k,float("nan")):.2f}' if m.get(k) is not None else '--' for k in MK[:4] if k in m or ptype == 'flexible')
    if ptype == 'flexible':
        mrgs = ' / '.join(f'{m.get(k,float("nan")):.2f}' if m.get(k) is not None else '--' for k in MK if k != 'B2_semi_rigid_fatigue')
    else:
        mrgs = ' / '.join(f'{m.get(k,float("nan")):.2f}' if m.get(k) is not None else '--' for k in MK)

    h_vals = d['h_cm']
    row = {
        'section': sid, 'type': ptype, 'E_sub': E_sub,
        'h1': h_vals[0], 'h2': h_vals[1], 'h3': h_vals[2], 'h4': h_vals[3], 'h5': h_vals[4],
        'AC_cm': round(ac, 1), 'cost_USD': round(cost_usd, 1) if cost_usd else None,
        'LCC_USD': round(lcc_usd, 1) if lcc_usd else None,
        'DSR': round(float(d['dsr']), 3), 'SCR': round(float(d.get('scr_running', 0)), 3),
    }
    all_rows.append(row)
    print(f'{sid:<9}{ptype:>8}{E_sub:>7}{ac:>7.1f}{cu:>8}{lu:>8}{d["dsr"]:>6.3f}{sc:>6}   {mrgs}{flag}')

# Save CSV
import csv, os
csv_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'core_results_2048_full.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(all_rows)
print(f'\nSaved: {csv_path}')
print(f'Elapsed: {time.time()-t0:.0f}s')