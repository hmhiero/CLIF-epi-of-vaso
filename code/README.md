# code/

All analysis scripts for the vasopressin epidemiology project.

## Script order

| Script | Purpose | Run at |
|--------|---------|--------|
| `clif_extract.py` | Extract septic shock cohort and hourly features from CLIF 2.1.0 parquet tables | Each CLIF site |
| `site_summary.py` | Compute federated-safe aggregate statistics; write CSVs to `output/` | Each site |
| `site_threshold_sweep.py` | Per-feature threshold sweep vs clinician vasopressin action; optional RL comparison | Each site |

## Usage

```bash
# Extraction
python code/clif_extract.py                          # CLIF sites

# Federated summary — run at each site, share output/ CSVs only
python code/site_summary.py --dataset ucmc

# Analysis — run at each site after extraction
python code/site_threshold_sweep.py --dataset ucmc
```

## Outputs

`site_summary.py` and `site_threshold_sweep.py` write to `output/<dataset>/`.
