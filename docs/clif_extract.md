# clif_extract.py

**Purpose:** CLIF 2.1.0 cohort extraction. Identifies septic shock patients from a CLIF site and builds hourly feature tables.

Outputs drop into `Data/UCMC/` (configured via `OUTPUT_DIR` in `config.py`):
- `cohort.parquet` — one row per patient, cohort-level columns
- `features.parquet` — one row per patient-hour, hourly features
- `cohort_filter_counts.csv` — patient counts at each inclusion filter step

---

## Prerequisites

### 1. CLIF 2.1.0 parquet files

Expected at `CLIF_DIR` (edit at top of script). Required tables:

| Table | Used for |
|-------|----------|
| `clif_adt.parquet` | ICU admission/discharge times, anchor year |
| `clif_patient.parquet` | Demographics (sex, race, death date) |
| `clif_hospitalization.parquet` | Discharge disposition, age at admission |
| `clif_medication_admin_intermittent.parquet` | Antibiotics (sepsis criteria), steroids |
| `clif_medication_admin_continuous.parquet` | NE, vasopressin, all vasopressors, fluids |
| `clif_microbiology_culture.parquet` | Blood cultures (sepsis criteria) |
| `clif_vitals.parquet` | MAP, HR, SpO2, temperature, weight |
| `clif_labs.parquet` | Creatinine, BUN, lactate, WBC, platelet |
| `clif_respiratory_support.parquet` | IMV / trach collar (ventilation flag) |
| `clif_crrt_therapy.parquet` | Continuous RRT |
| `clif_intermittent_hemodialysis.parquet` | Intermittent HD |
| `clif_patient_assessments.parquet` | GCS (optional; zero-filled if absent) |

### 2. Python environment

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy `config/config.example.py` to `config.py` (repo root) and fill in your site's paths — this is the **only file you need to edit**:

```bash
cp config/config.example.py config.py
# then open config.py and set CLIF_DIR and OUTPUT_DIR
```

`clif_extract.py` automatically loads `config.py` at startup and will exit with a clear error if `CLIF_DIR` or `OUTPUT_DIR` are missing.

All other parameters reflect the OVISS inclusion criteria and should not need adjustment.

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIF_DIR` | *(set per site)* | Root directory of CLIF parquet files |
| `OUTPUT_DIR` | *(set per site)* | Where outputs are written |
| `TRAJECTORY_HOURS` | 120 | Maximum trajectory length (hours) |
| `NE_WINDOW_HOURS` | 24 | NE must start within this many hours of ICU admit |
| `MIN_NE_RECORDS` | 2 | Minimum NE administration records required |
| `SOFA_THRESHOLD` | 2.0 | Minimum SOFA score at sepsis onset |
| `LACTATE_THRESHOLD` | 2.0 | Minimum lactate (mmol/L) within 24h of infection |

---

## Usage

```bash
python code/clif_extract.py
```

No arguments. The script runs Phase A (cohort) then Phase B (features) sequentially.

If `cohort.parquet` already exists in `OUTPUT_DIR` and has all required columns, Phase A is skipped and the cached cohort is used directly.


## Phase A: Cohort identification

Seven sequential inclusion criteria (Sepsis-3 / OVISS logic):

| Step | Criterion | Source |
|------|-----------|--------|
| 0 | All hospitalizations in CLIF site (baseline count) | `clif_hospitalization` |
| 1 | IV CMS-qualifying antibiotic + blood culture within 24h of each other → `presumed_infection_dttm` = earliest event | `medication_admin_intermittent`, `microbiology_culture` |
| 2 | ICU admission from ADT (`location_category == "icu"`) within ±24h of infection | `clif_adt` |
| 3 | First NE within 24h of ICU admit, ≥2 records (`med_category == "norepinephrine"`, unit `mcg/kg/min`) | `medication_admin_continuous` |
| 4 | SOFA ≥ 2 in 24h window around infection onset (via clifpy) | labs, vitals, resp support, meds |
| 5 | Lactate > 2 mmol/L within 24h of infection | `clif_labs` |
| 6 | Trajectory bounds: `trajectory_start = max(icu_intime, first_norepi_time, presumed_infection_dttm)`; `trajectory_end = min(icu_outtime, start + 120h, deathtime)` | all above |

Demographics (age, sex, race) and weight (median vitals during trajectory) are appended after step 6.

Patient counts at each step are saved to `cohort_filter_counts.csv` (see below).

### Output: `cohort.parquet`

| Column | Description |
|--------|-------------|
| `stay_id` | Hospitalization ID |
| `hospital_death` | 1 if died in hospital |
| `anchor_year_group` | Year of ICU admission (from ADT `in_dttm`) |
| `icu_intime` / `icu_outtime` | ICU admission and discharge timestamps |
| `deathtime` | Death timestamp (NaT if survived) |
| `trajectory_start` / `trajectory_end` | Shock-onset reference window |
| `traj_hours` | Trajectory length (integer hours, capped at 120) |
| `first_norepi_time` | Timestamp of first NE administration |
| `age` | Age at admission |
| `gender` | M / F |
| `race` | Race category (normalized) |
| `weight` | Median weight (kg) during trajectory |
| `sepsis_onset_sofa` | SOFA score at infection onset (24h window) |
| `initial_lactate` | First lactate value in 24h infection window |

### Output: `cohort_filter_counts.csv`

Two-column table written when the cohort is built (not when loaded from cache). Useful for consort/flow diagrams.

| Column | Description |
|--------|-------------|
| `step` | Human-readable description of the inclusion criterion |
| `n_hospitalizations` | Number of hospitalizations remaining after this filter |

Example rows:

| step | n_hospitalizations |
|------|--------------------|
| Total hospitalizations in CLIF site | — |
| IV CMS qualifying antibiotic + blood culture within 24h | — |
| ICU admission (ADT) within 24h of suspected infection | — |
| Norepinephrine within 24h of ICU admit (>=2 records) | — |
| SOFA >= 2.0 within 24h of infection | — |
| Lactate > 2.0 mmol/L within 24h of infection (final septic shock cohort) | — |

---

## Phase B: Hourly feature extraction

A per-patient hourly grid is built from `time_hour = 0` (trajectory start / shock onset) to `traj_hours`. Each feature function joins against this grid by `(stay_id, time_hour)`.

### Output: `features.parquet`

| Column | Source | Aggregation |
|--------|--------|-------------|
| `norepinephrine` | Continuous meds (`med_category == "norepinephrine"`, mcg/kg/min) | Mean dose over hour; 0 if no record |
| `vaso_dose` | Continuous meds (`med_category == "vasopressin"`) | Mean dose over hour; 0 if no record |
| `nee` | All vasopressors: NE + Epi + Phe/10 + Dopa/100 + Metaraminol/8 + AngII×10 + Vaso×2.5 (U/min) | Sum of time-overlapping contributions |
| `action_vaso` | Binary: `vaso_dose > 0` | — |
| `mbp` | Vitals (`vital_category == "map"`) | Mean over hour |
| `heart_rate` | Vitals | Last value in hour |
| `spo2` | Vitals | Last value in hour |
| `temperature` | Vitals (`temp_c`) | Last value in hour |
| `ventil` | Respiratory support (IMV or Trach Collar) | 1 if any record in hour |
| `rrt` | CRRT + intermittent HD | 1 if any record in hour |
| `steroid` | Intermittent meds (hydrocortisone, dexamethasone, etc.) | 1 if given this epoch **or the immediately preceding epoch**; resets to 0 otherwise |
| `fluids` | Continuous meds (`med_group == "fluids_electrolytes"`, mL/hr or mL/kg/hr only) | Rate × overlap hours, summed across concurrent infusions (mL) |
| `bun`, `creatinine`, `lactate`, `wbc`, `platelet` | Labs | Last observed value up to end of hour (LOCF) |
| `gcs` | Patient assessments (`gcs_total`) | Last observed value up to end of hour (LOCF) |
| `sofa` | clifpy (per patient over full trajectory) | Single value broadcast to all hours |
| `norepi_explicitly_stopped` | Continuous meds: dose = 0 or MAR stop action | 1 if any cessation event in hour |
| `vaso_explicitly_stopped` | Continuous meds: dose = 0 or MAR stop action | 1 if any cessation event in hour |
| `ne_mar_action` | Continuous meds (`mar_action_group` or `mar_action_name`) | Last action string in hour (NaN if no record) |
| `vaso_mar_action` | Continuous meds (`mar_action_group` or `mar_action_name`) | Last action string in hour (NaN if no record) |
| `urine_output` | Not available in CLIF 2.1.0 | Zero-filled |
| `death` | `hospital_death` from cohort (Expired discharge or death_dttm present) | Broadcast to all hours |

#### Fluids unit handling
- `mL/hr` → used directly
- `mL/kg/hr` → multiplied by cohort weight (70 kg fallback)
- `mEq/hr`, `mEq/kg/hr` → excluded (cannot convert to volume without concentration)

#### MAR action columns
`ne_mar_action` and `vaso_mar_action` record the last `mar_action_group` (falling back to `mar_action_name`) within each clock hour. A `NaN` means no record exists that hour — distinguishable from a "Given" record with dose = 0. Useful for identifying charting gaps vs. true cessation. The fallback to `NaN` if neither MAR column is present makes this safe across sites with different schemas.

---

## Notes for federated sites

- Update `CLIF_DIR` and `OUTPUT_DIR` at the top of the script.
- All other parameters match OVISS inclusion criteria and should not need adjustment.
- Phase A is cached: if `cohort.parquet` already exists in `OUTPUT_DIR` with all required columns, it will not be rebuilt. `cohort_filter_counts.csv` is only written on a fresh build.
- `gcs` is gracefully skipped if `clif_patient_assessments.parquet` is absent.
- `ne_mar_action` / `vaso_mar_action` are NaN-filled if no MAR action column is present in your continuous meds table.
