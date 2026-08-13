# Clean Package Notes

Original working repository: private local working copy.

Clean package: this repository.

The original working repository was not modified. This package was reconstructed as
a reviewer-facing copy by excluding local/private materials, large solver artifacts,
raw database files, logs, caches, manuscript drafts, and historical response folders.

The 0813 update synchronizes the clean package with the latest manuscript changes:
Fig. 3 is LTPP-only, NCAT is not included as a main-figure Source Data sheet, and
the final upload workbook is `Source_Data/SourceData_iLLM-PD_NC_final.xlsx`.

## Excluded by construction

- `.env` and local API credentials
- `.git`, local assistant metadata, caches, and bytecode
- large ABAQUS working files such as `.odb`, `.inp`, `.stt`, `.dat`, `.msg`, `.sim`
- large LTPP Access database files (`.accdb`)
- `output/`, logs, backup folders, chat histories, manuscript drafts, and response drafts

## Items Requiring Final Human Confirmation

- Public repository URL
- Zenodo or equivalent archive DOI
- Code license
- Whether trained PPO checkpoints should be deposited separately if required by the editor
