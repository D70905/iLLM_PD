# iLLM-PD

This repository provides the code, configuration files, figure-generation scripts,
and curated result tables used for the 0813 revised Nature Communications manuscript
NCOMMS-25-89228.

## Scope

iLLM-PD is a human-auditable pavement structural-design workflow. Natural-language
project information can be parsed into structured inputs, while pavement structures
are evaluated through bounded numerical actions, finite-element mechanical responses,
and codified specification checks. The revised manuscript uses JTG D50-2017 as the
governing specification in the optimization loop and reports independent checks where
stated.

This 0813 clean package follows the latest manuscript figure set: Fig. 3 reports only
the twelve-section LTPP design evaluation, and the final Source Data workbook contains
the corresponding LTPP-only source table.

The package is intended for reviewer assessment and reproducibility checks. Large
commercial-solver working files, local logs, API keys, manuscript drafts, and raw
database files that are not required to inspect the submitted analyses are not included.

## Main Contents

| Path | Content |
| --- | --- |
| `rl/` | Reinforcement-learning environment, generator interface, guard logic, and training utilities |
| `fea/` | ABAQUS finite-element runner and mechanical-response helpers |
| `specs/` | JTG D50-2017 and ME-PDG/NCHRP 1-37A evaluation code |
| `scripts/` | Analysis, inference, benchmark, and data-processing scripts |
| `experiments/` | Curated input tables and result summaries used in the manuscript |
| `figures/final/` | Final figure files for the revised manuscript |
| `figures/source/` | Figure source-data files and plotting scripts |
| `Source_Data/` | Final 0813 Source Data workbook and staging tables |

`CLEANING_MANIFEST.csv` records which files were copied or excluded from the working
repository during package construction.

## Final Source Data For Upload

Use this workbook for the 0813 resubmission:

```text
Source_Data/SourceData_iLLM-PD_NC_final.xlsx
```

The workbook contains 10 flat worksheets: README, source index, Table 1, Table 2, and
source data for Figs. 2-6. The Fig. 3 worksheet is LTPP-only and contains no NCAT panel
data. No earlier Source Data workbooks are included in this upload package.

## Environment

The code was developed with Python 3.10. The finite-element workflow uses ABAQUS,
which requires a local commercial installation and is not redistributed in this
repository.

```powershell
conda env create -f environment.yml
conda activate illm_pd
pip install -r requirements.txt
```

For language-model-assisted runs, copy `env.example` to `.env` and provide local API
credentials. Parser and generator calls are optional for inspecting the deterministic
mechanics, specification checks, stored results, and figure source data.

## Reproducing Main Checks

Run commands from the repository root. On Windows PowerShell:

```powershell
$env:PYTHONPATH = "."
python -c "from specs.jtg_d50 import JTGSpecification; print('import OK')"
```

Representative analyses:

```powershell
python scripts/ablation_inference.py --seeds 0,1,2
python scripts/prepare_final_source_data_tables_0813.py
```

Several full finite-element or policy-evaluation runs require ABAQUS and trained
checkpoint files. Curated outputs used for the manuscript are provided under
`experiments/ltpp_data/deliverables/`.

## Figures

Final figure files and editable/source assets are collected in:

```text
figures/final/
figures/source/
```

The plotting scripts in `figures/source/` are copied from the working repository and
use repository-relative paths where possible. If a script requires external software
or data not included in the clean package, the corresponding curated source data are
provided alongside it.

## Data Availability Notes

The package includes curated LTPP section descriptors, manuscript result summaries,
figure source data, and the final 0813 Source Data workbook. Public or third-party
databases should be cited and accessed through their original providers when required
by their terms of use.

Repository URL: https://github.com/D70905/iLLM_PD

License: MIT. A Zenodo DOI should be added after the GitHub release is archived.
