# code/

All analysis scripts for the vasopressin epidemiology project.

## Script order

| Script | Purpose | Run at |
|--------|---------|--------|
| `clif_extract.py` | Extract septic shock cohort and hourly features from CLIF 2.1.0 parquet tables | Each CLIF site |
| `mimic_extract.py` | Extract equivalent cohort from MIMIC-IV DuckDB | Coordinating site (internal) |
| `site_summary.py` | Compute federated-safe aggregate statistics; write CSVs to `output/` | Each site |
| `site_threshold_sweep.py` | Per-feature threshold sweep vs clinician vasopressin action; optional RL comparison | Each site |
| `cross_site_vasopressin_analysis.py` | Cross-site clinician vasopressin analysis (MIMIC + CLIF) | Coordinating site |

## Usage

```bash
# Extraction
python code/clif_extract.py                          # CLIF sites
python code/mimic_extract.py                         # MIMIC (internal)

# Federated summary — run at each site, share output/ CSVs only
python code/site_summary.py --dataset ucmc
python code/site_summary.py --dataset mimic

# Analysis — run at coordinating site after collecting all site outputs
python code/site_threshold_sweep.py --dataset ucmc
python code/site_threshold_sweep.py --dataset mimic
python code/cross_site_vasopressin_analysis.py
```

## Outputs

Scripts 35 and 36 write to `output/<dataset>/` and `output/clinician_vasopressin/`.
`site_summary.py` writes to `output/UCMC/` or `output/MIMIC/`.
