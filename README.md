# Epidemiology of Vasopressin in Septic Shock

Federated multi-site analysis of vasopressin initiation patterns in ICU patients meeting septic shock criteria (Sepsis-3 + norepinephrine + lactate > 2 mmol/L).

## CLIF VERSION

2.1.0

## Objective

Characterize clinician vasopressin initiation behavior across sites, identify feature-threshold decision rules that explain initiation timing, and compare clinician practice to a reinforcement-learning (RL) policy. The project supports federated execution: each site runs extraction and summary scripts locally and shares only aggregate outputs.

## Required CLIF tables and fields

The following tables are required:

1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`
3. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = `'map'`
4. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value`
   - `lab_category` = `'lactate'`, `'creatinine'`, `'bun'`
5. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`
   - `med_category` = `'norepinephrine'`, `'vasopressin'`, `'epinephrine'`, `'phenylephrine'`, `'dopamine'`, `'angiotensin'`, `'hydrocortisone'`, `'dexamethasone'`, `'methylprednisolone'`
6. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`
7. **patient_assessments** (SOFA): `hospitalization_id`, `recorded_dttm`, `numerical_value`

The [clifpy](https://common-longitudinal-icu-data-format.github.io/clifpy/) package is used for SOFA score computation and outlier handling.

## Cohort identification

**Inclusion:**
- First ICU stay per patient
- Sepsis-3 criteria: suspected infection + SOFA ≥ 2 at or near ICU admission
- Norepinephrine started within 24 hours of ICU admission (≥ 2 administration records)
- Lactate > 2.0 mmol/L within 24 hours of suspected infection

**Trajectory:** Up to 120 hours from shock onset (norepinephrine start), sampled hourly.

## Expected Results

Final aggregate outputs (no patient-level data) are written to `output/UCMC/` (CLIF sites) or `output/MIMIC/` (MIMIC-IV). The following files are produced per site:

| File | Contents |
|------|----------|
| `cohort_filter_counts.csv` | Patient counts at each inclusion step |
| `split_counts.csv` | Train/val/test counts by ever-vasopressin group |
| `baseline_table1.csv` | Baseline characteristics stratified by vasopressin use |
| `feature_at_initiation.csv` | Feature values (median [IQR]) at first vasopressin initiation |
| `feature_thresholds_youden.csv` | Per-feature threshold performance (AUC, sens, spec) |
| `feature_roc_curves.csv` | ROC curve points on fixed grid for coordinating-site replot |

> [!WARNING]
> **Never upload patient-level data to Box.** Only **aggregate** results may be placed in
> `output/` and shared with the project PI / consortium:
> - No `patient_id`, `stay_id`, or any row-level records.
> - Minimum cell size **n ≥ 11** for every reported statistic (cells with n < 11 are suppressed).
> - No raw `.parquet` / patient-level data files.
>
> See `output/README.md` for details.

## Detailed instructions for running the project

### 1. Configure `config.py`

```bash
cp config/config.example.py config.py
# Edit config.py: set CLIF_DIR and OUTPUT_DIR for your site
```

See [`config/README.md`](config/README.md) for details.

### 2. Set up the Python environment

```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

### 3. Extract cohort data

**CLIF sites:**
```bash
python code/clif_extract.py
```
Writes `Data/UCMC/cohort.parquet`, `Data/UCMC/features.parquet`, `Data/UCMC/cohort_filter_counts.csv`.

**MIMIC-IV (internal use):**
```bash
python code/mimic_extract.py
```
Writes `Data/MIMIC/cohort.parquet`, `Data/MIMIC/features.parquet`, `Data/MIMIC/cohort_filter_counts.csv`.

### 4. Run federated summary (site_summary.py)

```bash
# For CLIF sites:
python code/site_summary.py --dataset ucmc

# For MIMIC:
python code/site_summary.py --dataset mimic
```

Writes aggregate CSVs to `output/UCMC/` or `output/MIMIC/`. **Share only these files** — not the raw parquet data.

### 5. Run analysis scripts (coordinating site)

After collecting aggregate outputs from all sites:

```bash
# Threshold rule analysis
python code/35_threshold_policy_comparison.py --dataset ucmc
python code/35_threshold_policy_comparison.py --dataset mimic

# Cross-site clinician vasopressin analysis
python code/36_feature_threshold_rules.py
```

See [`code/README.md`](code/README.md) for full script documentation.

## Directory structure

```
.
├── code/                        # All analysis scripts
│   ├── clif_extract.py          # CLIF 2.1.0 cohort extraction
│   ├── mimic_extract.py         # MIMIC-IV cohort extraction (internal)
│   ├── site_summary.py          # Federated aggregate summary (run at each site)
│   ├── 35_threshold_policy_comparison.py
│   ├── 36_feature_threshold_rules.py
│   └── README.md
├── config/                      # Configuration templates
│   ├── config.example.py        # Copy to config.py and fill in site paths
│   └── README.md
├── docs/                        # Documentation
│   └── clif_extract.md          # CLIF extraction script reference
├── output/                      # Generated outputs (gitignored)
│   ├── UCMC/                    # CLIF site aggregate results
│   ├── MIMIC/                   # MIMIC-IV aggregate results
│   └── README.md
├── Data/                        # Local cohort data (gitignored, never share)
│   ├── UCMC/
│   └── MIMIC/
├── config.py                    # Site-specific config (gitignored, copy from config/)
└── requirements.txt
```
