# NCAT Material-Type Sensitivity: Real-FEA Results

## Scope

This test uses three representative NCAT Cracking Group sections: N1, N5 and S6. For each section, the thickness was fixed at the best real-FEA point found in the earlier fixed-material audit, and material properties were varied. Mechanical stiffness variants were re-evaluated with real ABAQUS FEA. VFA and R0 were then swept as JTG material-parameter checks without rerunning FEA.

## Main Numerical Findings

- Real FEA runs: 24
- Total JTG evaluations including VFA/R0 sweeps: 288
- Compliant cases after including VFA/R0 material-parameter sweeps: 177
- Mechanical stiffness changes alone did not produce compliant designs for N1, N5 or S6 under VFA = 70% and R0 = 1.5 mm.

## Baseline And Mechanical-Only Best Cases

| Section | Baseline DSR | Baseline B1 | Baseline B3 | Best mechanical scenario | Best mechanical DSR | Best mechanical B1 | Best mechanical B3 | Feasible? |
|---|---:|---:|---:|---|---:|---:|---:|---|
| NCAT_CG_N1 | 0.843 | 1.530 | 0.843 | surface_stiffer_1p30 | 0.856 | 1.614 | 0.856 | False |
| NCAT_CG_N5 | 0.844 | 0.844 | 0.929 | all_ac_stiffer_1p30 | 0.933 | 1.203 | 0.933 | False |
| NCAT_CG_S6 | 0.815 | 1.323 | 0.815 | surface_stiffer_1p30 | 0.827 | 1.409 | 0.827 | False |

## Moderate Passing Material-Parameter Examples

These are not claimed as measured NCAT material parameters. They are moderate sensitivity examples showing what type of material improvement would be needed to clear the current B1/B3 bottleneck.

| Section | Scenario | VFA (%) | R0 (mm) | DSR | B1 | B3 | Cost change (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| NCAT_CG_N1 | baseline | 70 | 1.2 | 1.000 | 1.530 | 1.054 | 0.00 |
| NCAT_CG_N5 | all_ac_stiffer_1p15 | 70 | 1.2 | 1.000 | 1.016 | 1.164 | 1.12 |
| NCAT_CG_S6 | baseline | 70 | 1.2 | 1.000 | 1.323 | 1.019 | 0.00 |

## Interpretation

1. The earlier failure is not just a policy-search failure. In this fixed-material setting, the dominant limitation is material-performance representation: B1 and B3 cannot both be cleared by simple stiffness perturbations alone.
2. R0 is the most direct lever for B3. Because JTG B3 permanent deformation is proportional to R0, reducing R0 from 1.5 mm to roughly 1.2 mm is enough for N1 and S6 at the same best-audit thickness.
3. N5 is different because its baseline best-audit point is B1-controlled. It needs both B1 improvement and B3 improvement; in the tested grid, all-AC stiffness +15% combined with R0 = 1.2 mm reaches compliance with only about 1.12% cost increase.
4. This supports a more defensible manuscript claim: NCAT fixed-thickness/material validation should be reported as a boundary diagnosis, while material-type sensitivity can be used to show that the system identifies when material substitution or improved rutting resistance is required.

## Claim Boundary

The VFA/R0 sweep is a controlled design-sensitivity experiment, not a claim that NCAT measured R0 or VFA had those exact values. It should be framed as future-work or supplementary evidence for material-type sensitivity unless corresponding NCAT lab rutting parameters are obtained.



