"""
run_climate_adaptive_design.py — 5段柔性 LTPP × 两模式(fixed23C / hot-month)
===========================================================
同段同E_sub, 仅改设计温度 → 看设计是否随气候自适应。
"""

from stable_baselines3 import PPO
from rl.env import PavementEnv, EnvConfig
import numpy as np

CKPT = 'output/rl_runs/diag_flex_16_1010_8000/checkpoints/ckpt_step_003600'
model = PPO.load(f'{CKPT}/ppo_model.zip')

FLEX = {
    '16_1010': dict(E_sub=78,  T_hot=20.3),
    '04_1065': dict(E_sub=91,  T_hot=22.2),
    '48_1076': dict(E_sub=115, T_hot=26.6),
    '48_0001': dict(E_sub=700, T_hot=29.5),
    '04_1034': dict(E_sub=91,  T_hot=35.0),
}
T_FIXED = 23.0
INIT_E = [14000., 11000., 9000., 350., 250.]
INIT_H = [0.04, 0.06, 0.08, 0.30, 0.25]
INIT_NU = [0.25, 0.30, 0.30, 0.40, 0.35]
MK = ['B1_asphalt_fatigue', 'B3_ac_permanent_deformation', 'B4_subgrade_strain']


def run_ep(E_sub, T_design):
    cfg = EnvConfig(
        pavement_type='flexible',
        init_modulus_MPa=INIT_E,
        init_thickness_m=INIT_H,
        init_poisson=INIT_NU,
        E_subgrade=float(E_sub),
        max_episode_steps=20,
        llm_enabled=False,
        fea_verbose=False,
        fea_keep_runs=False,
        climate_enabled=True,
        design_temperature_C=float(T_design),
    )
    env = PavementEnv(cfg)
    obs, info = env.reset()
    for _ in range(20):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, tr, info = env.step(a)

    d = info.get('delivered_design')
    if d is None:
        h = np.array(info['design_h_cm']) / 100
        E = np.array(info['design_E_MPa'])
        marg = info.get('margins', {})
        dsr = info.get('dsr')
        cost = None
        comp = False
    else:
        h = np.array(d['thickness'])
        E = np.array(d['modulus'])
        marg = info.get('delivered_margins', {})
        dsr = info.get('delivered_dsr')
        cost = info.get('delivered_cost_cny')
        comp = True

    lcc = None
    try:
        if cost is not None:
            r = env._lcc_evaluator(
                C_construction_usd_per_m2=float(cost) / 7.2,
                design_life_years=float(cfg.design_life_years_lcc),
                margin_B1=float(marg.get('B1_asphalt_fatigue', float('inf'))),
                margin_B2=float('inf'),
                discount_rate=0.04,
            )
            lcc = r.get('NPV_total_usd_m2')
    except Exception:
        pass
    env.close()

    return dict(
        ac=float(np.sum(h[:3]) * 100),
        h=(h * 100).round(1).tolist(),
        E_AC=[round(float(E[i]), 0) for i in range(3)],  # AC moduli (policy's 20C reference)
        cost=cost,
        lcc=lcc,
        dsr=float(dsr) if dsr is not None else float('nan'),
        m={k: round(float(marg.get(k, float('nan'))), 2) for k in MK},
        comp=comp,
    )


print(f'{"section":<9}{"T":>6}{"AC_cm":>7}{"E_AC":>8}{"cost":>8}{"DSR":>6}   B1/B3/B4')
print('-' * 95)
rows = {}
for sec, p in FLEX.items():
    fx = run_ep(p['E_sub'], T_FIXED)
    cl = run_ep(p['E_sub'], p['T_hot'])
    rows[sec] = (fx, cl)
    for tag, T, res in [('fix23', T_FIXED, fx), (f'hot{p["T_hot"]:.0f}', p['T_hot'], cl)]:
        c = f'{res["cost"]:.0f}' if res["cost"] else 'NA'
        e_ac = f'{np.mean(res["E_AC"]):.0f}' if res.get("E_AC") else 'NA'
        flag = '' if res['comp'] else ' <NON-COMPLIANT>'
        print(f'{sec:<9}{T:>6.0f}{res["ac"]:>7.1f}{e_ac:>8}{c:>8}{res["dsr"]:>6.2f}   {res["m"]}{flag}')
    print()

print('=== Climate - Fixed23 (same section, same E_sub) ===')
print(f'{"section":<9}{"T_hot":>6}{"dAC_cm":>8}{"dcost%":>8}   B3(fix->clim)')
for sec, p in FLEX.items():
    fx, cl = rows[sec]
    dac = cl['ac'] - fx['ac']
    dcost = (cl['cost'] / fx['cost'] - 1) * 100 if (cl['cost'] and fx['cost']) else float('nan')
    b3 = f'{fx["m"]["B3_ac_permanent_deformation"]:.2f}->{cl["m"]["B3_ac_permanent_deformation"]:.2f}'
    print(f'{sec:<9}{p["T_hot"]:>6.1f}{dac:>8.1f}{dcost:>8.1f}   {b3}')