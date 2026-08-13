# Reproducibility Notes

This clean package separates three levels of reproducibility.

1. Stored-result inspection: the curated CSV/XLSX files under `experiments/`,
   `figures/source/`, and `Source_Data/` support checking the manuscript tables and
   figure values.
2. Python-level reruns: parser tests, specification calculations, ablations, and
   plotting scripts can be rerun after installing the Python environment.
3. Full mechanical reruns: ABAQUS-based finite-element evaluation requires a licensed
   ABAQUS installation and local solver configuration.

The submitted manuscript reports deterministic compliance metrics from the stored
mechanical/specification outputs. Language-model API outputs may vary by provider,
model version, and availability; therefore backend-substitution analyses are reported
as fixed-budget experimental results rather than as a required dependency for checking
the final compliance calculations.

## Minimal Checks

```powershell
$env:PYTHONPATH = "."
python -m py_compile rl\generator.py rl\train.py specs\jtg_d50.py
python -c "from specs.jtg_d50 import JTGSpecification; print('spec import OK')"
```

## Figure/Data Mapping

| Manuscript item | Primary files in this package |
| --- | --- |
| Fig. 2 | `Source_Data/SourceData_iLLM-PD_NC_final_0813.xlsx` sheet `Fig2_trajectory`; `output/fig2_04_1034_cost_heatmap_0813` in the working repository |
| Fig. 3 | `figures/final/Fig3_LTPP_only_true_delivered_redraw.png`; `figures/final/Fig3_LTPP_only_true_delivered_redraw_wrapped.svg`; `figures/source/Fig3_LTPP_only_true_delivered_source_data.csv` |
| Fig. 4 | `figures/final/Fig4_comparison_practice_v2.*`; `figures/source/Fig4_comparison_practice_v2_source_data*.csv` |
| Fig. 5 | `figures/final/fig6_climate_4panel.*`; `experiments/batch_climate_12sections_summary.csv` |
| Robustness/OOD figure | `figures/final/fig7_final.*`; `experiments/ltpp_data/deliverables/ood_stress/ood_aggregate_20260723_133102.csv` |
| Table 1 | `Source_Data/SourceData_iLLM-PD_NC_final_0813.xlsx` sheet `Table1_LTPP_sections` |
| Table 2 | `experiments/ltpp_data/deliverables/ablation_inference/ablation_table2.csv` |

The final Source Data workbook for the 0813 resubmission is:

```text
Source_Data/SourceData_iLLM-PD_NC_final_0813.xlsx
```

The workbook was generated from `Source_Data/_final_tables_0813/` and verified by
`scripts/verify_final_source_data_workbook_0813.py`.

The final numbering should be checked against the accepted manuscript file before
deposit because some historical script names retain earlier figure numbers.
