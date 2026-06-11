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

Final aggregate outputs (no patient-level data) are written to `output/<SITE>/`. The following files are produced per site:

| File | Contents | Source script |
|------|----------|---------------|
| `cohort_filter_counts.csv` | Patient counts at each inclusion step | `site_summary.py` |
| `split_counts.csv` | Train/val/test counts by ever-vasopressin group | `site_summary.py` |
| `baseline_table1.csv` | Baseline characteristics stratified by vasopressin use | `site_summary.py` |
| `feature_at_initiation.csv` | Feature values (median [IQR]) at first vasopressin initiation | `site_summary.py` |
| `feature_thresholds_youden.csv` | Per-feature threshold performance (AUC, sens, spec) | `site_summary.py` |
| `feature_roc_curves.csv` | ROC curve points on fixed grid for coordinating-site replot | `site_summary.py` |
| `threshold_comparison_table.csv` | Per-feature optimal threshold, kappa, AUROC (step-level) | `site_threshold_sweep.py` |
| `patient_level_table.csv` | Per-feature patient-level threshold performance | `site_threshold_sweep.py` |
| `patient_level_confounders.csv` | Clinical profile (mortality, SOFA, age, LOS, lactate) by threshold group per feature | `site_threshold_sweep.py` |
| `threshold_sweep_data.csv` | Full kappa sweep curves per feature (for coordinating-site replotting) | `site_threshold_sweep.py` |
| `plots/` | Threshold sweep and decision-tree fidelity figures | `site_threshold_sweep.py` |


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

```bash
python code/clif_extract.py
```

Writes intermediate files to `Data/<SITE>/`: `cohort.parquet`, `features.parquet`, `cohort_filter_counts.csv`.

### 4. Run federated summary (site_summary.py)

```bash
python code/site_summary.py --dataset ucmc
```

Writes aggregate CSVs to `output/<SITE>/`. **Share only these files** — not the raw parquet data.

### 5. Run threshold analysis at your site (site_threshold_sweep.py)

Each site runs this locally (it reads the intermediate parquet files, not the aggregate CSVs):

```bash
python code/site_threshold_sweep.py --dataset ucmc
```

Writes `output/<SITE>/threshold_comparison_table.csv`, `output/<SITE>/patient_level_table.csv`, and plots to `output/<SITE>/plots/`. **Share these files** along with the outputs from step 4.

### 6. Run cross-site analysis (coordinating site only)

After collecting aggregate outputs from **all** sites:

```bash
python code/cross_site_vasopressin_analysis.py
```

See [`code/README.md`](code/README.md) for full script documentation.

## Directory structure

```
.
├── code/                        # All analysis scripts
│   ├── clif_extract.py          # CLIF 2.1.0 cohort extraction
│   ├── site_summary.py          # Federated aggregate summary (run at each site)
│   ├── site_threshold_sweep.py          # Per-feature threshold sweep (run at each site)
│   ├── cross_site_vasopressin_analysis.py  # Cross-site combined analysis (coordinating site)
│   └── README.md
├── config/                      # Configuration templates
│   ├── config.example.py        # Copy to config.py and fill in site paths
│   └── README.md
├── docs/                        # Documentation
│   └── clif_extract.md          # CLIF extraction script reference
├── output/                      # Generated outputs (gitignored)
│   └── <SITE>/                  # Aggregate results per site
├── Data/                        # Local cohort data (gitignored, never share)
│   └── <SITE>/
├── config.py                    # Site-specific config (gitignored, copy from config/)
└── requirements.txt
```
