# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
CLIF 2.1.0 external validation: cohort identification and hourly feature extraction.

Outputs (in OUTPUT_DIR):
  cohort_clif.parquet   — matches MIMIC data/cohort.parquet schema
  features_clif.parquet — matches MIMIC data/features.parquet schema

"""

import sys
import io
import warnings

# Force UTF-8 stdout on Windows to handle Unicode in print statements
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from clifpy import ClifOrchestrator
    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE
except ModuleNotFoundError:
    print("Install clifpy: pip install clifpy")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLIF_DIR = Path(r"C:\Users\hhieromnimon\Box\04-CLIF-2.1\2.1.0")
OUTPUT_DIR = Path(r"C:\Users\hhieromnimon\Box\Research\rl-clinical-concordance\0_MIMIC_RAW_ABLATION\data_clif")
TIMEZONE = "UTC"
TRAJECTORY_HOURS = 120
NE_WINDOW_HOURS = 24   # NE must start within 24h of ICU admit
MIN_NE_RECORDS = 2     # ≥2 NE administrations (matches OVISS criterion)
SOFA_THRESHOLD = 2.0
LACTATE_THRESHOLD = 2.0
MAP_THRESHOLD = 65.0

STEROID_CATEGORIES = [
    "hydrocortisone", "dexamethasone", "methylprednisolone",
    "fludrocortisone", "prednisolone", "prednisone",
]
VASOPRESSOR_CATEGORIES = [
    "norepinephrine", "epinephrine", "phenylephrine",
    "vasopressin", "dopamine", "angiotensin",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def tz_coerce(s: pd.Series, tz: str) -> pd.Series:
    """Localize naive datetimes; convert already-aware ones."""
    if pd.api.types.is_datetime64_any_dtype(s):
        if s.dt.tz is None:
            return s.dt.tz_localize(tz)
        return s.dt.tz_convert(tz)
    return s


def tz_strip(s: pd.Series) -> pd.Series:
    """Remove timezone info — returns tz-naive UTC datetimes for comparison."""
    if pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s


def to_naive_utc(s: pd.Series) -> pd.Series:
    """Robustly convert any datetime-like series to tz-naive UTC.

    Handles tz-aware, tz-naive, and object-dtype (mixed) columns — including
    the case where pd.concat of tz-aware + tz-naive parquets produces object dtype.
    """
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


# ---------------------------------------------------------------------------
# Phase A-1: Suspected infection (antibiotics + blood culture within 24h)
# ---------------------------------------------------------------------------
def identify_suspected_infection(clif_dir: Path, time_window_hours: int = 24) -> pd.DataFrame:
    meds_i = pd.read_parquet(clif_dir / "clif_medication_admin_intermittent.parquet")
    cultures = pd.read_parquet(clif_dir / "clif_microbiology_culture.parquet")

    abx = meds_i[meds_i["med_group"] == "CMS_sepsis_qualifying_antibiotics"][["hospitalization_id", "admin_dttm"]].copy()

    blood_cx = cultures[
        (cultures["fluid_category"] == "blood_buffy") &
        cultures["collect_dttm"].notna() &
        (cultures["method_category"] == "culture")
    ][["hospitalization_id", "collect_dttm"]].copy()

    merged = abx.merge(blood_cx, on="hospitalization_id", how="inner")
    merged["time_diff"] = (
        (merged["admin_dttm"] - merged["collect_dttm"]).dt.total_seconds().abs() / 3600
    )
    merged = merged[merged["time_diff"] <= time_window_hours].copy()
    merged["presumed_infection_dttm"] = merged[["admin_dttm", "collect_dttm"]].min(axis=1)

    return (merged.groupby("hospitalization_id")
                  .agg(presumed_infection_dttm=("presumed_infection_dttm", "min"))
                  .reset_index())


# ---------------------------------------------------------------------------
# Phase A-2: ICU admission/discharge times from ADT
# ---------------------------------------------------------------------------
def get_icu_times(clif_dir: Path) -> pd.DataFrame:
    adt = pd.read_parquet(clif_dir / "clif_adt.parquet")
    icu = (adt[adt["location_category"] == "icu"]
           .sort_values("in_dttm")
           .groupby("hospitalization_id")
           .agg(icu_intime=("in_dttm", "first"), icu_outtime=("out_dttm", "last"))
           .reset_index())
    return icu


# ---------------------------------------------------------------------------
# Phase A-3: NE-specific criteria (first NE within 24h of ICU admit, ≥2 records)
# ---------------------------------------------------------------------------
def get_ne_criteria(clif_dir: Path, icu_df: pd.DataFrame) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    ne = meds[meds["med_category"] == "norepinephrine"][
        ["hospitalization_id", "admin_dttm", "med_dose"]
    ].copy()

    ne = ne.merge(icu_df[["hospitalization_id", "icu_intime"]], on="hospitalization_id")
    ne["admin_dttm"] = to_naive_utc(ne["admin_dttm"])
    ne["icu_intime"] = to_naive_utc(ne["icu_intime"])

    window_end = ne["icu_intime"] + pd.Timedelta(hours=NE_WINDOW_HOURS)
    ne_win = ne[(ne["admin_dttm"] >= ne["icu_intime"]) & (ne["admin_dttm"] <= window_end)]

    ne_agg = (ne_win.groupby("hospitalization_id")
                    .agg(first_norepi_time=("admin_dttm", "min"),
                         n_ne=("admin_dttm", "count"))
                    .reset_index())
    return ne_agg[ne_agg["n_ne"] >= MIN_NE_RECORDS].drop(columns="n_ne")


# ---------------------------------------------------------------------------
# Phase A-4: Mortality and trajectory bounds
# ---------------------------------------------------------------------------
def get_mortality(clif_dir: Path) -> pd.DataFrame:
    patient = pd.read_parquet(clif_dir / "clif_patient.parquet")[["patient_id", "death_dttm"]]
    hosp = pd.read_parquet(clif_dir / "clif_hospitalization.parquet")[
        ["patient_id", "hospitalization_id", "discharge_category", "discharge_dttm",
         "age_at_admission"]
    ]
    m = hosp.merge(patient, on="patient_id", how="left")
    m["hospital_death"] = (
        (m["discharge_category"] == "Expired") | m["death_dttm"].notna()
    ).astype(int)
    m["deathtime"] = m.apply(
        lambda r: r["death_dttm"] if pd.notna(r["death_dttm"])
        else (r["discharge_dttm"] if r["hospital_death"] == 1 else pd.NaT),
        axis=1,
    )
    return m[["hospitalization_id", "hospital_death", "deathtime", "age_at_admission"]]


# ---------------------------------------------------------------------------
# Phase A-4b: Patient demographics (age, sex, race, weight)
# ---------------------------------------------------------------------------
def get_demographics(clif_dir: Path, stay_ids: set) -> pd.DataFrame:
    """Age, gender (M/F), and normalized race for cohort patients."""
    patient = pd.read_parquet(clif_dir / "clif_patient.parquet")[
        ["patient_id", "sex_category", "race_category", "ethnicity_category"]
    ]
    hosp = pd.read_parquet(clif_dir / "clif_hospitalization.parquet")[
        ["patient_id", "hospitalization_id"]
    ]
    demo = hosp[hosp["hospitalization_id"].isin(stay_ids)].merge(patient, on="patient_id", how="left")
    demo = demo.rename(columns={"hospitalization_id": "stay_id"})

    demo["gender"] = demo["sex_category"].map({"Male": "M", "Female": "F"})

    demo["race"] = demo.apply(
        lambda r: "Hispanic" if r["ethnicity_category"] == "Hispanic" else r["race_category"],
        axis=1,
    )
    return demo[["stay_id", "gender", "race"]]


def get_vaso_pretraj(clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """1 if vasopressin was administered in the 24h before trajectory_start, else 0."""
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    vaso = meds[meds["med_category"] == "vasopressin"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    vaso = vaso.rename(columns={"hospitalization_id": "stay_id"})
    vaso["admin_dttm"] = to_naive_utc(vaso["admin_dttm"])

    bounds = cohort[["stay_id", "trajectory_start"]].copy()
    bounds["traj_start"] = pd.to_datetime(bounds["trajectory_start"], utc=True).dt.tz_localize(None)
    vaso = vaso.merge(bounds[["stay_id", "traj_start"]], on="stay_id", how="inner")

    in_window = (
        (vaso["admin_dttm"] >= vaso["traj_start"] - pd.Timedelta(hours=24)) &
        (vaso["admin_dttm"] <  vaso["traj_start"])
    )
    pretraj_ids = set(vaso.loc[in_window, "stay_id"])
    return pd.DataFrame({
        "stay_id": cohort["stay_id"],
        "vaso_before_traj": cohort["stay_id"].isin(pretraj_ids).astype(int),
    })


def get_weight_at_onset(clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Median weight_kg per patient during their trajectory window."""
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    w = vitals[vitals["vital_category"] == "weight_kg"][
        ["hospitalization_id", "recorded_dttm", "vital_value"]
    ].rename(columns={"hospitalization_id": "stay_id", "vital_value": "weight"})
    w["recorded_dttm"] = to_naive_utc(w["recorded_dttm"])

    bounds = cohort[["stay_id", "trajectory_start", "trajectory_end"]].copy()
    bounds["trajectory_start"] = to_naive_utc(pd.to_datetime(bounds["trajectory_start"], utc=True))
    bounds["trajectory_end"]   = to_naive_utc(pd.to_datetime(bounds["trajectory_end"],   utc=True))
    w = w.merge(bounds, on="stay_id")
    w = w[(w["recorded_dttm"] >= w["trajectory_start"]) & (w["recorded_dttm"] <= w["trajectory_end"])]
    return w.groupby("stay_id")["weight"].median().reset_index()


# ---------------------------------------------------------------------------
# Phase A-5: clifpy SOFA helpers
# ---------------------------------------------------------------------------
def _load_sofa_tables(co: ClifOrchestrator, hosp_ids: list) -> None:
    co.load_table("labs", filters={
        "hospitalization_id": hosp_ids,
        "lab_category": ["creatinine", "platelet_count", "po2_arterial",
                         "bilirubin_total", "lactate"],
    })
    co.load_table("vitals", filters={
        "hospitalization_id": hosp_ids,
        "vital_category": ["map", "spo2", "weight_kg"],
    })
    co.load_table("patient_assessments", filters={
        "hospitalization_id": hosp_ids,
        "assessment_category": ["gcs_total"],
    })
    co.load_table("medication_admin_continuous", filters={"hospitalization_id": hosp_ids})
    co.load_table("respiratory_support", filters={"hospitalization_id": hosp_ids})

    # Mark meds as pre-converted (NE is already in mcg/kg/min)
    med_df = co.medication_admin_continuous.df.copy()
    med_df = med_df[med_df["med_dose"].notna() & med_df["med_dose_unit"].notna()]
    co.medication_admin_continuous.df = med_df
    co.medication_admin_continuous.df_converted = med_df.copy()
    co.medication_admin_continuous.df_converted["_convert_status"] = "success"


def _add_missing_med_cols(co: ClifOrchestrator) -> None:
    for col in ["norepinephrine_mcg_kg_min", "epinephrine_mcg_kg_min",
                "dopamine_mcg_kg_min", "dobutamine_mcg_kg_min"]:
        if col not in co.wide_df.columns:
            co.wide_df[col] = 0.0


# ---------------------------------------------------------------------------
# Phase A: Full cohort assembly
# ---------------------------------------------------------------------------
def build_cohort(clif_dir: Path, co: ClifOrchestrator) -> tuple:
    print("\nStep 1: Suspected infection (IV abx + blood culture within 24h)...")
    suspected = identify_suspected_infection(clif_dir)
    print(f"  {len(suspected):,} patients")

    print("Step 2: ICU admission times from ADT...")
    icu = get_icu_times(clif_dir)
    suspected = suspected.merge(icu, on="hospitalization_id", how="inner")
    suspected["presumed_infection_dttm"] = tz_coerce(suspected["presumed_infection_dttm"], TIMEZONE)
    suspected["icu_intime"] = tz_coerce(suspected["icu_intime"], TIMEZONE)
    diff_h = (suspected["presumed_infection_dttm"] - suspected["icu_intime"]).dt.total_seconds() / 3600
    suspected = suspected[diff_h.abs() <= 24].copy()
    print(f"  {len(suspected):,} with infection within 24h of ICU admit")

    print("Step 3: NE criteria (first NE ≤24h of ICU admit, ≥2 records)...")
    ne = get_ne_criteria(clif_dir, icu)
    suspected = suspected.merge(ne, on="hospitalization_id", how="inner")
    print(f"  {len(suspected):,} patients")

    print("Step 4: SOFA ≥ 2 via clifpy...")
    hosp_ids = suspected["hospitalization_id"].tolist()
    _load_sofa_tables(co, hosp_ids)

    suspected["start_time"] = suspected["presumed_infection_dttm"]
    suspected["end_time"] = suspected["presumed_infection_dttm"] + pd.Timedelta(hours=24)

    co.create_wide_dataset(
        category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df=suspected,
        return_dataframe=True,
    )
    _add_missing_med_cols(co)
    sofa = co.compute_sofa_scores(
        wide_df=co.wide_df,
        id_name="hospitalization_id",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        create_new_wide_df=False,
    )
    suspected = suspected.merge(sofa[["hospitalization_id", "sofa_total"]],
                                on="hospitalization_id", how="left")
    suspected = suspected[suspected["sofa_total"] >= SOFA_THRESHOLD].copy()
    print(f"  {len(suspected):,} with SOFA ≥ {SOFA_THRESHOLD}")

    print("Step 5: Hemodynamic criteria (NE already applied; vasopressor criterion met)...")
    print(f"  {len(suspected):,} patients")

    print("Step 6: Elevated lactate > 2 within 24h of infection...")
    lactate = co.labs.df[co.labs.df["lab_category"] == "lactate"].copy()
    lactate["lab_result_dttm"] = tz_coerce(lactate["lab_result_dttm"], TIMEZONE)
    lac_window = lactate.merge(suspected[["hospitalization_id", "start_time", "end_time"]],
                               on="hospitalization_id")
    lac_window = lac_window[
        (lac_window["lab_result_dttm"] >= lac_window["start_time"]) &
        (lac_window["lab_result_dttm"] < lac_window["end_time"]) &
        lac_window["lab_value_numeric"].notna()
    ]
    initial_lac = (
        lac_window.sort_values("lab_result_dttm")
        .groupby("hospitalization_id")["lab_value_numeric"]
        .first()
        .reset_index()
        .rename(columns={"lab_value_numeric": "initial_lactate"})
    )
    lac = lac_window[lac_window["lab_value_numeric"] > LACTATE_THRESHOLD]
    elevated_ids = set(lac["hospitalization_id"])
    cohort = suspected[suspected["hospitalization_id"].isin(elevated_ids)].copy()
    cohort = cohort.merge(initial_lac, on="hospitalization_id", how="left")
    print(f"  {len(cohort):,} septic shock patients")

    print("Step 7: Mortality and trajectory bounds...")
    mortality = get_mortality(clif_dir)
    cohort = cohort.merge(mortality, on="hospitalization_id", how="left")
    cohort = cohort.rename(columns={"age_at_admission": "age"})

    for col in ["icu_intime", "icu_outtime", "first_norepi_time",
                "presumed_infection_dttm", "deathtime"]:
        if col in cohort.columns:
            cohort[col] = tz_coerce(cohort[col], TIMEZONE)

    cohort["trajectory_start"] = cohort[
        ["icu_intime", "first_norepi_time", "presumed_infection_dttm"]
    ].max(axis=1)

    traj_cap = cohort["trajectory_start"] + pd.Timedelta(hours=TRAJECTORY_HOURS)

    # Ensure all candidate end times are tz-aware UTC before taking min
    def to_utc(s):
        s = pd.to_datetime(s, utc=False)
        if s.dt.tz is None:
            s = s.dt.tz_localize(TIMEZONE)
        else:
            s = s.dt.tz_convert(TIMEZONE)
        return s

    icu_out = to_utc(cohort["icu_outtime"])
    cap     = to_utc(traj_cap)
    death   = to_utc(cohort["deathtime"])

    cohort["trajectory_end"] = pd.concat(
        [icu_out.rename("e"), cap.rename("e"), death.rename("e")], axis=1
    ).min(axis=1)

    cohort["traj_hours"] = (
        (cohort["trajectory_end"] - cohort["trajectory_start"])
        .dt.total_seconds() / 3600
    ).clip(lower=0).astype(int)

    # Rename sofa to sepsis_onset_sofa (already on cohort from step 4)
    if "sofa_total" in cohort.columns:
        cohort = cohort.rename(columns={"sofa_total": "sepsis_onset_sofa"})

    # Year of ICU admission from ADT in_dttm
    cohort["anchor_year_group"] = cohort["icu_intime"].dt.year.astype(str)
    cohort = cohort.rename(columns={"hospitalization_id": "stay_id"})

    print("  Adding demographics (sex, race)...")
    demo = get_demographics(clif_dir, set(cohort["stay_id"]))
    cohort = cohort.merge(demo, on="stay_id", how="left")

    print("  Adding weight from vitals...")
    weight_df = get_weight_at_onset(clif_dir, cohort)
    cohort = cohort.merge(weight_df, on="stay_id", how="left")

    print("  Checking vasopressin in 24h before trajectory start...")
    vaso_pretraj = get_vaso_pretraj(clif_dir, cohort)
    cohort = cohort.merge(vaso_pretraj, on="stay_id", how="left")
    cohort["vaso_before_traj"] = cohort["vaso_before_traj"].fillna(0).astype(int)

    return cohort, co


# ---------------------------------------------------------------------------
# Phase B: Hourly feature extraction
# ---------------------------------------------------------------------------
def build_hourly_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build per-patient hourly rows. All timestamps are tz-naive UTC."""
    rows = []
    for _, row in cohort.iterrows():
        ts_raw = row["trajectory_start"]
        traj_start = pd.to_datetime(ts_raw, utc=True).tz_localize(None)
        for h in range(int(row["traj_hours"]) + 1):
            rows.append({
                "stay_id":    row["stay_id"],
                "time_hour":  h,
                "start_time": traj_start + pd.Timedelta(hours=h),
                "end_time":   traj_start + pd.Timedelta(hours=h + 1),
            })
    return pd.DataFrame(rows)


def add_ne_dose(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    ne = meds[
        (meds["med_category"] == "norepinephrine") &
        meds["med_dose"].notna() &
        (meds["med_dose_unit"] == "mcg/kg/min")
    ][["hospitalization_id", "admin_dttm", "med_dose"]].copy()
    ne = ne.rename(columns={"hospitalization_id": "stay_id"})
    ne["admin_dttm"] = to_naive_utc(ne["admin_dttm"])
    ne = ne.sort_values(["stay_id", "admin_dttm"])
    ne["end_dttm"] = (ne.groupby("stay_id")["admin_dttm"]
                        .shift(-1)
                        .fillna(pd.Timestamp("2100-01-01")))
    ne.loc[ne["med_dose"] == 0.0, "end_dttm"] = ne.loc[ne["med_dose"] == 0.0, "admin_dttm"]

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    ne_g = ne.merge(g, on="stay_id")
    ne_hr = ne_g[(ne_g["admin_dttm"] < ne_g["end_time"]) &
                 (ne_g["end_dttm"]   > ne_g["start_time"])]
    ne_agg = (ne_hr.groupby(["stay_id", "time_hour"])["med_dose"]
                   .mean().reset_index()
                   .rename(columns={"med_dose": "norepinephrine"}))
    grid = grid.merge(ne_agg, on=["stay_id", "time_hour"], how="left")
    grid["norepinephrine"] = grid["norepinephrine"].fillna(0.0)
    return grid


def add_vaso_dose(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")
    vaso = meds[
        (meds["med_category"] == "vasopressin") &
        meds["med_dose"].notna()
    ][["hospitalization_id", "admin_dttm", "med_dose"]].copy()
    vaso = vaso.rename(columns={"hospitalization_id": "stay_id"})
    vaso["admin_dttm"] = to_naive_utc(vaso["admin_dttm"])
    vaso = vaso.sort_values(["stay_id", "admin_dttm"])
    vaso["end_dttm"] = (vaso.groupby("stay_id")["admin_dttm"]
                            .shift(-1)
                            .fillna(pd.Timestamp("2100-01-01")))
    vaso.loc[vaso["med_dose"] == 0.0, "end_dttm"] = vaso.loc[vaso["med_dose"] == 0.0, "admin_dttm"]

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    vaso_g = vaso.merge(g, on="stay_id")
    vaso_hr = vaso_g[(vaso_g["admin_dttm"] < vaso_g["end_time"]) &
                     (vaso_g["end_dttm"]   > vaso_g["start_time"])]
    vaso_agg = (vaso_hr.groupby(["stay_id", "time_hour"])["med_dose"]
                       .mean().reset_index()
                       .rename(columns={"med_dose": "vaso_dose"}))
    grid = grid.merge(vaso_agg, on=["stay_id", "time_hour"], how="left")
    grid["vaso_dose"] = grid["vaso_dose"].fillna(0.0)
    return grid


def add_mbp(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    mbp = vitals[vitals["vital_category"] == "map"][
        ["hospitalization_id", "recorded_dttm", "vital_value"]
    ].copy()
    mbp = mbp.rename(columns={"hospitalization_id": "stay_id"})
    mbp["recorded_dttm"] = to_naive_utc(mbp["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    mbp_g = mbp.merge(g, on="stay_id")
    mbp_hr = mbp_g[(mbp_g["recorded_dttm"] >= mbp_g["start_time"]) &
                   (mbp_g["recorded_dttm"] < mbp_g["end_time"])]
    mbp_agg = (mbp_hr.groupby(["stay_id", "time_hour"])["vital_value"]
                     .mean().reset_index()
                     .rename(columns={"vital_value": "mbp"}))
    grid = grid.merge(mbp_agg, on=["stay_id", "time_hour"], how="left")
    return grid


def add_ventil(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    resp = pd.read_parquet(clif_dir / "clif_respiratory_support.parquet")
    imv = resp[resp["device_category"].isin(["IMV"])][
        ["hospitalization_id", "recorded_dttm"]
    ].copy()
    imv = imv.rename(columns={"hospitalization_id": "stay_id"})
    imv["recorded_dttm"] = to_naive_utc(imv["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    imv_g = imv.merge(g, on="stay_id")
    imv_hr = (imv_g[(imv_g["recorded_dttm"] >= imv_g["start_time"]) &
                    (imv_g["recorded_dttm"] < imv_g["end_time"])]
              .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n"))
    imv_hr["ventil"] = 1
    grid = grid.merge(imv_hr[["stay_id", "time_hour", "ventil"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["ventil"] = grid["ventil"].fillna(0).astype(int)
    return grid


def add_rrt(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    crrt = pd.read_parquet(clif_dir / "clif_crrt_therapy.parquet")[
        ["hospitalization_id", "recorded_dttm"]
    ].copy()
    ihd = pd.read_parquet(clif_dir / "clif_intermittent_hemodialysis.parquet")[
        ["hospitalization_id", "recorded_dttm"]
    ].copy()
    rrt_all = pd.concat([crrt, ihd], ignore_index=True)
    rrt_all = rrt_all.rename(columns={"hospitalization_id": "stay_id"})
    rrt_all["recorded_dttm"] = to_naive_utc(rrt_all["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    rrt_g = rrt_all.merge(g, on="stay_id")
    rrt_hr = (rrt_g[(rrt_g["recorded_dttm"] >= rrt_g["start_time"]) &
                    (rrt_g["recorded_dttm"] < rrt_g["end_time"])]
              .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n"))
    rrt_hr["rrt"] = 1
    grid = grid.merge(rrt_hr[["stay_id", "time_hour", "rrt"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["rrt"] = grid["rrt"].fillna(0).astype(int)
    return grid


def add_steroids(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    meds_i = pd.read_parquet(clif_dir / "clif_medication_admin_intermittent.parquet")
    steroids = meds_i[meds_i["med_category"].isin(STEROID_CATEGORIES)][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    steroids = steroids.rename(columns={"hospitalization_id": "stay_id"})
    steroids["admin_dttm"] = to_naive_utc(steroids["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    st_g = steroids.merge(g, on="stay_id")
    st_hr = (
        st_g[
            (st_g["admin_dttm"] >= st_g["start_time"]) &
            (st_g["admin_dttm"] <  st_g["end_time"])
        ]
        .groupby(["stay_id", "time_hour"]).size().reset_index(name="_n")
    )
    st_hr["_given"] = 1

    grid = grid.sort_values(["stay_id", "time_hour"])
    grid = grid.merge(st_hr[["stay_id", "time_hour", "_given"]],
                      on=["stay_id", "time_hour"], how="left")
    grid["_given"] = grid["_given"].fillna(0).astype(int)

    # steroid = 1 if given this epoch OR the immediately preceding epoch
    grid["steroid"] = (
        (grid["_given"] | grid.groupby("stay_id")["_given"].shift(1, fill_value=0))
        .astype(int)
    )
    grid = grid.drop(columns="_given")
    return grid


def add_vitals(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """Add heart_rate, spo2, temperature (last value within each hour window).

    Avoids a full cartesian merge by binning each vital observation to its
    trajectory hour before joining on (stay_id, time_hour).
    """
    vitals = pd.read_parquet(clif_dir / "clif_vitals.parquet")
    target = vitals[vitals["vital_category"].isin(["heart_rate", "spo2", "temp_c"])][
        ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]
    ].copy()
    target = target.rename(columns={"hospitalization_id": "stay_id"})
    target["recorded_dttm"] = to_naive_utc(target["recorded_dttm"])

    # trajectory_start for each patient = start_time at time_hour == 0
    traj_start = (grid[grid["time_hour"] == 0][["stay_id", "start_time"]]
                  .rename(columns={"start_time": "traj_start"}))
    target = target.merge(traj_start, on="stay_id", how="inner")
    target["time_hour"] = ((target["recorded_dttm"] - target["traj_start"])
                           .dt.total_seconds() / 3600).astype(int)
    # Keep only records inside the trajectory window
    max_hour = grid.groupby("stay_id")["time_hour"].max().rename("max_hour")
    target = target.merge(max_hour, on="stay_id", how="inner")
    target = target[(target["time_hour"] >= 0) & (target["time_hour"] <= target["max_hour"])]

    v_last = (target.sort_values("recorded_dttm")
              .groupby(["stay_id", "time_hour", "vital_category"])["vital_value"]
              .last().reset_index())
    v_wide = (v_last.pivot_table(index=["stay_id", "time_hour"],
                                  columns="vital_category",
                                  values="vital_value")
              .reset_index())
    v_wide.columns.name = None
    v_wide = v_wide.rename(columns={"temp_c": "temperature"})
    grid = grid.merge(v_wide, on=["stay_id", "time_hour"], how="left")
    return grid


# NEE conversion factors (mcg/kg/min scale; vasopressin handled separately)
# Formula: NE + Epi + Phe/10 + Dopa/100 + Metaraminol/8 + AngII*10 + Vaso*2.5 (U/min)
_NEE_FACTORS_MCG = {
    "norepinephrine": 1.0,
    "epinephrine":    1.0,
    "phenylephrine":  0.1,
    "dopamine":       0.01,
    "metaraminol":    0.125,
    "angiotensin ii": 10.0,
}
# Vasopressin: 2.5 mcg/kg/min NE equiv per U/min
_VASO_FACTOR_U_MIN = 2.5
_ASSUMED_WEIGHT_KG = 70.0


def add_nee(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """Norepinephrine equivalent dose (mcg/kg/min):
    NE + Epi + Phe/10 + Dopa/100 + Metaraminol/8 + AngII×10 + Vaso×2.5(U/min).
    """
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")

    # ── Catecholamines + metaraminol + angiotensin II ─────────────────────────
    cath = meds[meds["med_category"].isin(_NEE_FACTORS_MCG) & meds["med_dose"].notna()][
        ["hospitalization_id", "admin_dttm", "med_category", "med_dose", "med_dose_unit"]
    ].copy()
    cath = cath.rename(columns={"hospitalization_id": "stay_id"})
    cath["admin_dttm"] = to_naive_utc(cath["admin_dttm"])
    # mcg/min without /kg → divide by assumed weight to get mcg/kg/min
    per_min = (cath["med_dose_unit"].str.lower().str.contains("mcg/min", na=False) &
               ~cath["med_dose_unit"].str.lower().str.contains("kg", na=False))
    cath.loc[per_min, "med_dose"] = cath.loc[per_min, "med_dose"] / _ASSUMED_WEIGHT_KG
    cath["nee_contrib"] = cath["med_dose"] * cath["med_category"].map(_NEE_FACTORS_MCG)

    # ── Vasopressin (U/min equivalent) ────────────────────────────────────────
    vaso_cats = {"vasopressin", "vasopressine", "terlipressin"}
    vaso = meds[meds["med_category"].str.lower().isin(vaso_cats) &
                meds["med_dose"].notna()][
        ["hospitalization_id", "admin_dttm", "med_dose", "med_dose_unit"]
    ].copy()
    vaso = vaso.rename(columns={"hospitalization_id": "stay_id"})
    vaso["admin_dttm"] = to_naive_utc(vaso["admin_dttm"])
    unit_lo = vaso["med_dose_unit"].str.lower().fillna("")
    # Convert U/hr → U/min
    is_per_hr = unit_lo.str.contains(r"u?nit[s]?/h|u/h", na=False)
    vaso.loc[is_per_hr, "med_dose"] = vaso.loc[is_per_hr, "med_dose"] / 60.0
    vaso["nee_contrib"] = vaso["med_dose"] * _VASO_FACTOR_U_MIN
    vaso["med_category"] = "vasopressin"

    # ── Combine, expand to intervals, and aggregate ───────────────────────────
    contrib = pd.concat([
        cath[["stay_id", "admin_dttm", "med_category", "nee_contrib"]],
        vaso[["stay_id", "admin_dttm", "med_category", "nee_contrib"]],
    ], ignore_index=True)
    contrib = contrib.sort_values(["stay_id", "med_category", "admin_dttm"])
    contrib["end_dttm"] = (contrib.groupby(["stay_id", "med_category"])["admin_dttm"]
                                   .shift(-1)
                                   .fillna(pd.Timestamp("2100-01-01")))
    contrib.loc[contrib["nee_contrib"] == 0.0, "end_dttm"] = contrib.loc[contrib["nee_contrib"] == 0.0, "admin_dttm"]

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    c_g = contrib.merge(g, on="stay_id")
    c_hr = c_g[(c_g["admin_dttm"] < c_g["end_time"]) &
               (c_g["end_dttm"]   > c_g["start_time"])]
    nee_agg = (c_hr.groupby(["stay_id", "time_hour"])["nee_contrib"]
                   .sum().reset_index()
                   .rename(columns={"nee_contrib": "nee"}))
    grid = grid.merge(nee_agg, on=["stay_id", "time_hour"], how="left")
    grid["nee"] = grid["nee"].fillna(0.0)
    return grid


def add_labs(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    labs = pd.read_parquet(clif_dir / "clif_labs.parquet")
    target = labs[labs["lab_category"].isin(["bun", "creatinine", "lactate", "wbc", "platelet_count"])][
        ["hospitalization_id", "lab_result_dttm", "lab_category", "lab_value_numeric"]
    ].copy()
    target = target.rename(columns={"hospitalization_id": "stay_id"})
    target["lab_result_dttm"] = to_naive_utc(target["lab_result_dttm"])

    g = grid[["stay_id", "time_hour", "end_time"]]
    lab_g = target.merge(g, on="stay_id")
    # Last observed value up to end of each hour (LOCF basis)
    lab_win = lab_g[lab_g["lab_result_dttm"] < lab_g["end_time"]]
    lab_last = (lab_win.sort_values("lab_result_dttm")
                .groupby(["stay_id", "time_hour", "lab_category"])["lab_value_numeric"]
                .last().reset_index())
    lab_wide = (lab_last.pivot_table(index=["stay_id", "time_hour"],
                                     columns="lab_category",
                                     values="lab_value_numeric")
                .reset_index())
    lab_wide.columns.name = None
    if "platelet_count" in lab_wide.columns:
        lab_wide = lab_wide.rename(columns={"platelet_count": "platelet"})
    grid = grid.merge(lab_wide, on=["stay_id", "time_hour"], how="left")
    return grid


def add_gcs(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """GCS total (last observed up to end of each hour) from patient_assessments."""
    pa_path = clif_dir / "clif_patient_assessments.parquet"
    if not pa_path.exists():
        grid["gcs"] = np.nan
        return grid
    pa = pd.read_parquet(pa_path)
    gcs = pa[pa["assessment_category"] == "gcs_total"][
        ["hospitalization_id", "recorded_dttm", "numerical_value"]
    ].copy()
    gcs = gcs.rename(columns={"hospitalization_id": "stay_id"})
    gcs["recorded_dttm"] = to_naive_utc(gcs["recorded_dttm"])

    g = grid[["stay_id", "time_hour", "end_time"]]
    gcs_g = gcs.merge(g, on="stay_id")
    gcs_win = gcs_g[gcs_g["recorded_dttm"] < gcs_g["end_time"]]
    gcs_last = (gcs_win.sort_values("recorded_dttm")
                .groupby(["stay_id", "time_hour"])["numerical_value"]
                .last().reset_index()
                .rename(columns={"numerical_value": "gcs"}))
    grid = grid.merge(gcs_last, on=["stay_id", "time_hour"], how="left")
    return grid


def add_fluids(grid: pd.DataFrame, clif_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Total fluid volume (mL) delivered per clock hour.

    Source: clif_medication_admin_continuous where med_group == 'fluids_electrolytes'.
    Only mL-based units are converted to volume; mEq units are excluded (no concentration).
      mL/hr    → volume = rate × overlap_hours
      mL/kg/hr → volume = rate × weight_kg × overlap_hours (cohort weight; 70 kg fallback)
    Multiple concurrent fluids sum independently.
    Carry-forward: each dose runs until the next record for that (patient, med_category).
    Stop signal: mar_action_category == 'stop' OR dose == 0.
    end_dttm is computed over ALL records (including null-dose stops) so stop events
    correctly terminate the preceding infusion even when dose is null.
    """
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")

    # All fluids_electrolytes records — needed to compute correct end_dttm via shift(-1)
    all_fl = meds[meds["med_group"] == "fluids_electrolytes"][
        ["hospitalization_id", "admin_dttm", "med_category",
         "med_dose", "med_dose_unit", "mar_action_category"]
    ].copy()
    all_fl = all_fl.rename(columns={"hospitalization_id": "stay_id"})
    all_fl["admin_dttm"] = to_naive_utc(all_fl["admin_dttm"])
    all_fl = all_fl.sort_values(["stay_id", "med_category", "admin_dttm"])

    # end_dttm = next record's admin_dttm within the same (patient, fluid type)
    all_fl["end_dttm"] = (
        all_fl.groupby(["stay_id", "med_category"])["admin_dttm"]
        .shift(-1)
        .fillna(pd.Timestamp("2100-01-01"))
    )

    # Terminate at stop records (mar_action_category == 'stop' or dose == 0)
    action_lo = all_fl["mar_action_category"].str.lower().fillna("").str.strip()
    is_stop = (action_lo == "stop") | (all_fl["med_dose"].fillna(-1) == 0.0)
    all_fl.loc[is_stop, "end_dttm"] = all_fl.loc[is_stop, "admin_dttm"]

    # Keep only valid dose records for volume calculation
    unit_norm = all_fl["med_dose_unit"].str.lower().str.strip()
    fluids = all_fl[
        all_fl["med_dose"].notna() &
        unit_norm.isin(["ml/hr", "ml/kg/hr"]) &
        ~is_stop
    ].copy()

    # Resolve mL/kg/hr → mL/hr using cohort weight (70 kg if missing)
    weight_map = cohort.set_index("stay_id")["weight"].fillna(_ASSUMED_WEIGHT_KG)
    fluids["weight_kg"] = fluids["stay_id"].map(weight_map).fillna(_ASSUMED_WEIGHT_KG)
    per_kg = fluids["med_dose_unit"].str.lower().str.strip() == "ml/kg/hr"
    fluids.loc[per_kg, "med_dose"] = fluids.loc[per_kg, "med_dose"] * fluids.loc[per_kg, "weight_kg"]
    fluids = fluids.drop(columns=["med_dose_unit", "weight_kg", "mar_action_category"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]
    f_g = fluids.merge(g, on="stay_id")
    f_hr = f_g[
        (f_g["admin_dttm"] < f_g["end_time"]) &
        (f_g["end_dttm"]   > f_g["start_time"])
    ].copy()

    overlap_start = f_hr[["admin_dttm", "start_time"]].max(axis=1)
    overlap_end   = f_hr[["end_dttm",   "end_time"]].min(axis=1)
    f_hr["volume_ml"] = (overlap_end - overlap_start).dt.total_seconds() / 3600 * f_hr["med_dose"]

    fluids_agg = (
        f_hr.groupby(["stay_id", "time_hour"])["volume_ml"]
        .sum().reset_index()
        .rename(columns={"volume_ml": "fluids"})
    )
    grid = grid.merge(fluids_agg, on=["stay_id", "time_hour"], how="left")
    grid["fluids"] = grid["fluids"].fillna(0.0)
    return grid


def add_explicitly_stopped(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """Detect explicit NE/vasopressin cessation events from mar_action_name or dose=0."""
    meds_path = clif_dir / "clif_medication_admin_continuous.parquet"
    meds = pd.read_parquet(meds_path)[
        ["hospitalization_id", "admin_dttm", "med_category", "med_dose", "mar_action_name"]
    ].copy()
    meds = meds.rename(columns={"hospitalization_id": "stay_id"})
    meds["admin_dttm"] = to_naive_utc(meds["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]

    for med_cat, col_name in [
        ("norepinephrine", "norepi_explicitly_stopped"),
        ("vasopressin",    "vaso_explicitly_stopped"),
    ]:
        stopped = meds[
            (meds["med_category"] == med_cat) &
            (
                meds["mar_action_name"].str.lower().str.contains(
                    "stop|discontinu|held|off|cancel|ended", na=False
                ) |
                (meds["med_dose"].fillna(-1) == 0)
            )
        ].copy()
        stopped_g = stopped.merge(g, on="stay_id")
        stopped_hr = stopped_g[
            (stopped_g["admin_dttm"] >= stopped_g["start_time"]) &
            (stopped_g["admin_dttm"] <  stopped_g["end_time"])
        ].groupby(["stay_id", "time_hour"]).size().reset_index(name="_n")
        stopped_hr[col_name] = 1
        grid = grid.merge(stopped_hr[["stay_id", "time_hour", col_name]],
                          on=["stay_id", "time_hour"], how="left")
        grid[col_name] = grid[col_name].fillna(0).astype(int)
    return grid


def add_mar_actions(grid: pd.DataFrame, clif_dir: Path) -> pd.DataFrame:
    """Last mar_action_group per hour for NE and vasopressin.

    Columns: ne_mar_action, vaso_mar_action (str, NaN if no record that hour).
    Lets downstream code distinguish true absence of dose records from
    held/stopped/running entries — important for detecting gaps in continuous meds.
    Falls back to mar_action_name if mar_action_group is not present.
    """
    meds = pd.read_parquet(clif_dir / "clif_medication_admin_continuous.parquet")

    action_col = next(
        (c for c in ["mar_action_group", "mar_action_name"] if c in meds.columns),
        None,
    )
    if action_col is None:
        grid["ne_mar_action"]   = np.nan
        grid["vaso_mar_action"] = np.nan
        return grid

    meds = meds[["hospitalization_id", "admin_dttm", "med_category", action_col]].copy()
    meds = meds.rename(columns={"hospitalization_id": "stay_id"})
    meds["admin_dttm"] = to_naive_utc(meds["admin_dttm"])

    g = grid[["stay_id", "time_hour", "start_time", "end_time"]]

    for med_cat, col_name in [
        ("norepinephrine", "ne_mar_action"),
        ("vasopressin",    "vaso_mar_action"),
    ]:
        med_sub = meds[meds["med_category"] == med_cat]
        med_g   = med_sub.merge(g, on="stay_id")
        med_hr  = med_g[
            (med_g["admin_dttm"] >= med_g["start_time"]) &
            (med_g["admin_dttm"] <  med_g["end_time"])
        ]
        last_action = (
            med_hr.sort_values("admin_dttm")
            .groupby(["stay_id", "time_hour"])[action_col]
            .last()
            .reset_index()
            .rename(columns={action_col: col_name})
        )
        grid = grid.merge(last_action, on=["stay_id", "time_hour"], how="left")

    return grid


def add_hourly_sofa(grid: pd.DataFrame, co: ClifOrchestrator) -> pd.DataFrame:
    """Compute SOFA per patient using clifpy, broadcast to all patient-hours."""
    cohort_ids = grid["stay_id"].unique().tolist()

    # Build one row per patient covering their full trajectory window
    traj = (grid.groupby("stay_id")
            .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
            .reset_index()
            .rename(columns={"stay_id": "hospitalization_id"}))
    traj["start_time"] = pd.to_datetime(traj["start_time"], utc=True)
    traj["end_time"]   = pd.to_datetime(traj["end_time"],   utc=True)

    _load_sofa_tables(co, cohort_ids)
    co.create_wide_dataset(
        category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df=traj,
        return_dataframe=True,
    )
    _add_missing_med_cols(co)

    # Drop string columns that DuckDB can't aggregate numerically
    str_cols = [c for c in co.wide_df.columns
                if co.wide_df[c].dtype == object and c != "hospitalization_id"]
    if str_cols:
        co.wide_df = co.wide_df.drop(columns=str_cols)

    sofa_scores = co.compute_sofa_scores(
        wide_df=co.wide_df,
        id_name="hospitalization_id",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        create_new_wide_df=False,
    )
    sofa_scores = (sofa_scores[["hospitalization_id", "sofa_total"]]
                   .rename(columns={"hospitalization_id": "stay_id", "sofa_total": "sofa"}))
    grid = grid.merge(sofa_scores, on="stay_id", how="left")
    grid["sofa"] = grid["sofa"].fillna(0.0)
    return grid


# ---------------------------------------------------------------------------
# Phase B: Assemble full feature table
# ---------------------------------------------------------------------------
def build_features(cohort: pd.DataFrame, co: ClifOrchestrator, clif_dir: Path) -> pd.DataFrame:
    print("\nBuilding hourly grid...")
    grid = build_hourly_grid(cohort)
    print(f"  {len(grid):,} patient-hours across {cohort['stay_id'].nunique()} patients")

    print("Adding NE dose...")
    grid = add_ne_dose(grid, clif_dir)

    print("Adding vasopressin dose...")
    grid = add_vaso_dose(grid, clif_dir)

    print("Adding MBP...")
    grid = add_mbp(grid, clif_dir)

    print("Adding ventilation (IMV)...")
    grid = add_ventil(grid, clif_dir)

    print("Adding RRT (CRRT + IHD)...")
    grid = add_rrt(grid, clif_dir)

    print("Adding steroids...")
    grid = add_steroids(grid, clif_dir)

    print("Adding vitals (heart rate, SpO2, temperature)...")
    grid = add_vitals(grid, clif_dir)

    print("Adding NEE (norepinephrine equivalent dose)...")
    grid = add_nee(grid, clif_dir)

    print("Adding labs (BUN, creatinine, lactate)...")
    grid = add_labs(grid, clif_dir)

    print("Adding GCS (last observed per hour)...")
    grid = add_gcs(grid, clif_dir)

    print("Adding explicit cessation flags (NE/vasopressin stopped events)...")
    grid = add_explicitly_stopped(grid, clif_dir)

    print("Adding MAR action group (NE/vasopressin, last per hour)...")
    grid = add_mar_actions(grid, clif_dir)

    print("Adding SOFA via clifpy (per patient, broadcast to all hours)...")
    grid = add_hourly_sofa(grid, co)

    print("Adding fluids (mL/hr × overlap hours from fluids_electrolytes meds)...")
    grid = add_fluids(grid, clif_dir, cohort)

    # Unavailable in CLIF 2.1.0 — zero-fill
    grid["urine_output"] = 0.0

    # action_vaso: vasopressin > 0 at this hour (mirrors MIMIC SQL vaso_hourly.action_vaso)
    grid["action_vaso"] = (grid["vaso_dose"] > 0).astype(int)

    # death: set to 0 here — 02_preprocess.py assigns it from cohort deathtime
    grid["death"] = 0

    grid = grid.drop(columns=["start_time", "end_time"])
    grid = grid.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    co = ClifOrchestrator(
        data_directory=str(CLIF_DIR),
        filetype="parquet",
        timezone=TIMEZONE,
        output_directory=str(OUTPUT_DIR),
    )

    cohort_path = OUTPUT_DIR / "cohort_clif.parquet"
    _REQUIRED_COLS = {"age", "gender", "race", "sepsis_onset_sofa", "initial_lactate"}

    _rebuild = True
    if cohort_path.exists():
        print("PHASE A: loading cached cohort...")
        cohort_out = pd.read_parquet(cohort_path)
        if _REQUIRED_COLS.issubset(set(cohort_out.columns)):
            cohort = cohort_out.copy()
            print(f"  {len(cohort):,} patients loaded from cache")
            _rebuild = False
        else:
            print("  Cached cohort missing demographic columns — rebuilding...")
            cohort_path.unlink()

    if _rebuild:
        print("=" * 60)
        print("PHASE A: COHORT IDENTIFICATION")
        print("=" * 60)
        cohort, co = build_cohort(CLIF_DIR, co)

        print(f"\nFinal cohort: {len(cohort):,} patients")
        print(f"  Mortality:        {cohort['hospital_death'].mean():.1%}")
        print(f"  Median traj_hours: {cohort['traj_hours'].median():.0f}h")

        cohort_out = cohort[[
            "stay_id", "hospital_death", "anchor_year_group",
            "icu_intime", "icu_outtime", "deathtime",
            "trajectory_start", "trajectory_end", "traj_hours",
            "first_norepi_time",
            "age", "gender", "race", "weight",
            "sepsis_onset_sofa", "initial_lactate",
            "vaso_before_traj",
        ]]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cohort_out.to_parquet(cohort_path, index=False)
        print(f"Saved {cohort_path}")

    print("\n" + "=" * 60)
    print("PHASE B: FEATURE EXTRACTION")
    print("=" * 60)
    features = build_features(cohort, co, CLIF_DIR)

    print(f"\nFeatures summary:")
    print(f"  Rows: {len(features):,}")
    print(f"  NE>0 steps: {(features['norepinephrine'] > 0).mean():.1%}")
    print(f"  MBP missing: {features['mbp'].isna().mean():.1%}")
    print(f"  SOFA mean: {features['sofa'].mean():.1f}")

    features.to_parquet(OUTPUT_DIR / "features_clif.parquet", index=False)
    print(f"Saved {OUTPUT_DIR / 'features_clif.parquet'}")

    print("\nDone. Outputs written to:")
    print(f"  {OUTPUT_DIR / 'cohort_clif.parquet'}")
    print(f"  {OUTPUT_DIR / 'features_clif.parquet'}")


if __name__ == "__main__":
    main()
