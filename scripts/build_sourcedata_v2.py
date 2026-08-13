# -*- coding: utf-8 -*-
"""Build Source Data v2: fix 16_1010 cost, add layer thickness, format Fig5."""
import json, os
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd

PROJ = Path('d:/iLLM_PD_new')
DST = PROJ / 'SourceData_iLLM-PD.xlsx'
D = PROJ / 'experiments' / 'ltpp_data' / 'deliverables'

# === Build authoritative Table1/Fig3/S4.1 from JSONL ===
jsonl_path = D / 'ltpp_inference' / 'ltpp_inference_steps_20260625_133730.jsonl'
sel = {}
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line.strip())
        sid = str(r.get('section_id','?'))
        seed = int(r.get('seed',0))
        dsr = float(r.get('dsr',0))
        lcc_d = r.get('lcc')
        if lcc_d is None: continue
        cost = lcc_d.get('C_construction_usd_per_m2')
        lcc_v = lcc_d.get('NPV_total_usd_m2')
        if cost is None or lcc_v is None: continue
        if dsr >= 1.0:
            key = (sid, seed)
            if key not in sel or cost < sel[key]['cost']:
                sel[key] = {'cost': cost, 'lcc': lcc_v, 'step': r.get('step',-1)}

semi = {'30_7076','04_1065','27_2023','06_2004','12_4097','48_1109'}

# Merge with h1-h5 from old CSV for layer thickness
old_csv = PROJ / 'experiments' / 'core_results_2048_full.csv'
old = pd.read_csv(old_csv) if old_csv.exists() else None

rows = []
for sid in sorted(set(k[0] for k in sel.keys())):
    v = sel[(sid, 0)]
    p = 'semi_rigid' if sid in semi else 'flexible'
    row = {
        'Section': sid, 'Type': p,
        'h1_mm': '', 'h2_mm': '', 'h3_mm': '', 'h4_mm': '', 'h5_mm': '',
        'AC_cm': '', 'E_sub_MPa': '',
        'Construction_USD_m2': round(v['cost'],2),
        'LCC_NPV_USD_m2': round(v['lcc'],2),
        'DSR': 1.0, 'SCR_episode': '', 'Selected_Step': v['step'],
    }
    if old is not None:
        match = old[old['section'] == sid]
        if len(match) > 0:
            m = match.iloc[0]
            row['h1_mm'] = m.get('h1', '')*10 if 'h1' in m else ''
            row['h2_mm'] = m.get('h2', '')*10 if 'h2' in m else ''
            row['h3_mm'] = m.get('h3', '')*10 if 'h3' in m else ''
            row['h4_mm'] = m.get('h4', '')*10 if 'h4' in m else ''
            row['h5_mm'] = m.get('h5', '')*10 if 'h5' in m else ''
            row['AC_cm'] = m.get('AC_cm', '')
            row['E_sub_MPa'] = m.get('E_sub', '')
            row['SCR_episode'] = m.get('SCR', '')
    rows.append(row)

df_auth = pd.DataFrame(rows)

flex = df_auth[df_auth['Type'] == 'flexible']
semi_df = df_auth[df_auth['Type'] == 'semi_rigid']
summary = pd.DataFrame([
    {'Section': 'Flex mean +/- sd', 'Type': '', 'h1_mm':'','h2_mm':'','h3_mm':'','h4_mm':'','h5_mm':'',
     'AC_cm':'','E_sub_MPa':'',
     'Construction_USD_m2': f'{flex["Construction_USD_m2"].mean():.1f} +/- {flex["Construction_USD_m2"].std():.1f}',
     'LCC_NPV_USD_m2': f'{flex["LCC_NPV_USD_m2"].mean():.1f} +/- {flex["LCC_NPV_USD_m2"].std():.1f}',
     'DSR': 1.0, 'SCR_episode': '', 'Selected_Step': ''},
    {'Section': 'Semi mean +/- sd', 'Type': '', 'h1_mm':'','h2_mm':'','h3_mm':'','h4_mm':'','h5_mm':'',
     'AC_cm':'','E_sub_MPa':'',
     'Construction_USD_m2': f'{semi_df["Construction_USD_m2"].mean():.1f} +/- {semi_df["Construction_USD_m2"].std():.1f}',
     'LCC_NPV_USD_m2': f'{semi_df["LCC_NPV_USD_m2"].mean():.1f} +/- {semi_df["LCC_NPV_USD_m2"].std():.1f}',
     'DSR': 1.0, 'SCR_episode': '', 'Selected_Step': ''},
])

# === Write Excel ===
writer = pd.ExcelWriter(DST, engine='openpyxl')

# Sheet 1: Unified Table1/Fig3/S4.1 with layer thickness
df_auth.to_excel(writer, sheet_name='Table1_Fig3_S4.1', index=False)
summary.to_excel(writer, sheet_name='Table1_Fig3_S4.1', startrow=len(df_auth)+2, index=False)

# Sheet 2: Table2
t2 = D / 'ablation_inference' / 'ablation_table2.csv'
if t2.exists(): pd.read_csv(t2).to_excel(writer, sheet_name='Table2_Ablation', index=False)

# Sheet 3: Table3
ood_f = sorted((D / 'ood_stress').glob('ood_aggregate_*.csv'))[-1]
pd.read_csv(ood_f).to_excel(writer, sheet_name='Table3_OOD', index=False)

# Sheet 4: Fig2 trajectory
with open(jsonl_path, 'r', encoding='utf-8') as f:
    steps_16 = [json.loads(l) for l in f if l.strip() and json.loads(l.strip()).get('section_id')=='16_1010']
df_f2 = pd.DataFrame([{
    'section': r['section_id'], 'seed': r['seed'], 'step': r['step'],
    'DSR': r['dsr'], 'reward': r.get('reward',''), 'compliant': r.get('compliant',''),
    'C_construction_USD_m2': (r.get('lcc') or {}).get('C_construction_usd_per_m2',''),
    'LCC_NPV_USD_m2': (r.get('lcc') or {}).get('NPV_total_usd_m2',''),
} for r in steps_16])
df_f2.to_excel(writer, sheet_name='Fig2_trajectory_16_1010', index=False)

# Sheets 5-7: Fig4 baselines
for name, parts in [('Fig4_AASHTO1993', ('ltpp_aashto1993','aashto1993_summary_20260525_095437.csv')),
                     ('Fig4_AsBuilt',    ('ltpp_asbuilt','asbuilt_summary_20260525_005649.csv')),
                     ('Fig4c_ME-PDG',    ('ltpp_nchrp','nchrp_summary_20260525_145647.csv'))]:
    fp = D / parts[0] / parts[1]
    if fp.exists(): pd.read_csv(fp).to_excel(writer, sheet_name=name, index=False)

# Sheet 8: Fig5 surrogate drift — readable format
drift_rows = []
for p, label in [('flex_2048ts', 'Flexible_2048ts'), ('semi_rigid_1024ts', 'Semi-rigid_1024ts')]:
    jf = sorted(D.glob(f'surrogate_drift_*{p}*.json'))
    if jf:
        with open(jf[-1], encoding='utf-8') as f:
            d = json.load(f)
        items = d if isinstance(d, list) else [d]
        for it in items:
            # Real JSON keys: drift_<response>_<unit>_pct_mean_abs / _max_abs
            drift_rows.append({
                'Pavement_Type': label,
                'n_surrogate_calls': it.get('n_surrogate_calls',''),
                'n_fea_escalation': it.get('n_fea_escalation',''),
                'surrogate_fraction': it.get('surrogate_fraction',''),
                'epsilon_a_mean_pct': it.get('drift_epsilon_a_microstrain_pct_mean_abs',''),
                'epsilon_a_max_pct': it.get('drift_epsilon_a_microstrain_pct_max_abs',''),
                'epsilon_z_mean_pct': it.get('drift_epsilon_z_microstrain_pct_mean_abs',''),
                'epsilon_z_max_pct': it.get('drift_epsilon_z_microstrain_pct_max_abs',''),
                'sigma_t_mean_pct': it.get('drift_sigma_t_MPa_pct_mean_abs',''),
                'sigma_t_max_pct': it.get('drift_sigma_t_MPa_pct_max_abs',''),
                'p_AC_upper_mean_pct': it.get('drift_p_AC_upper_mid_MPa_pct_mean_abs',''),
                'p_AC_upper_max_pct': it.get('drift_p_AC_upper_mid_MPa_pct_max_abs',''),
                'p_AC_mid_mean_pct': it.get('drift_p_AC_mid_mid_MPa_pct_mean_abs',''),
                'p_AC_mid_max_pct': it.get('drift_p_AC_mid_mid_MPa_pct_max_abs',''),
                'p_AC_lower_mean_pct': it.get('drift_p_AC_lower_mid_MPa_pct_mean_abs',''),
                'p_AC_lower_max_pct': it.get('drift_p_AC_lower_mid_MPa_pct_max_abs',''),
            })
df_f5 = pd.DataFrame(drift_rows)
if not df_f5.empty:
    df_f5.to_excel(writer, sheet_name='Fig5_SurrogateDrift', index=False)

# Sheet 9: Fig6
f6 = PROJ / 'experiments' / 'batch_climate_12sections_summary.csv'
if f6.exists(): pd.read_csv(f6).to_excel(writer, sheet_name='Fig6_ClimateSensitivity', index=False)

# Sheet 10: Fig7
cl = D / 'cross_llm' / 'cross_llm_summary_20260525_230940.csv'
if cl.exists(): pd.read_csv(cl).to_excel(writer, sheet_name='Fig7_CrossLLM', index=False)

# README
readme = pd.DataFrame([
    ['Table1_Fig3_S4.1', 'Table 1, Fig 3, Supplementary S4.1',
     'Authoritative LCC from lifecycle_lcc_intl.py. M1 selected design from 0625 inference JSONL. Includes h1-h5 layer thickness from core_results.'],
    ['Table2_Ablation', 'Table 2', 'Component ablation (4 variants x 2 types x 3 seeds).'],
    ['Table3_OOD', 'Table 3', 'Out-of-distribution stress test.'],
    ['Fig2_trajectory_16_1010', 'Fig 2', 'Per-step trajectory. LCC is instantaneous per-step value, NOT selected design LCC.'],
    ['Fig4_AASHTO1993', 'Fig 4a/b', 'AASHTO 1993 baseline designs.'],
    ['Fig4_AsBuilt', 'Fig 4a/b', 'LTPP as-built section evaluation.'],
    ['Fig4c_ME-PDG', 'Fig 4c', 'ME-PDG rutting cross-check (escalation mode, real FEA). 33/36 pass.'],
    ['Fig5_SurrogateDrift', 'Fig 5b/c',
     'Manuscript Fig 5c cites 52% escalation from a representative single run (324 surrogate calls; Supplementary S3.2). '
     'This sheet reports the aggregate over the full run set (523 flexible / 708 semi-rigid calls), giving ~38% (flexible) and ~14% (semi-rigid). '
     'Both are correct; they describe different call populations.'],
    ['Fig6_ClimateSensitivity', 'Fig 6', 'Climate-resolved AC fatigue analysis.'],
    ['Fig7_CrossLLM', 'Fig 7', 'Cross-LLM robustness (6 backends).'],
    ['NOTE_16_1010_cost', '', '16_1010 selected design cost = 38.31 USD/m2 (0625 JSONL, authoritative). Manuscript should cite 38.3 (not 37.9 from older run).'],
], columns=['Sheet', 'Serves', 'Notes'])
readme.to_excel(writer, sheet_name='README', index=False)

writer.close()
print(f'Source Data v2: {DST}')
print(f'Fixes: (A)16_1010=38.31 authoritative, (B)h1-h5 columns added, (C)Fig5 readable format')