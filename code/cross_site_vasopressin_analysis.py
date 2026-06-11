#!/usr/bin/env python3
"""cross_site_vasopressin_analysis.py

Coordinating-site: clinician vasopressin analysis across MIMIC and CLIF.

Column clarification: 'norepinephrine' in both MIMIC processed data and CLIF features_clif
refers to raw norepinephrine dose (NE, µg/kg/min). Verified: correlation between MIMIC
processed 'norepinephrine' and raw features 'norepinephrine' is 0.999. The separate 'nee'
column is norepinephrine equivalent dose (NE + vasopressin equivalents) and is present in
both datasets under the same name. No cross-dataset column remapping is needed.

Analysis 1 — Vasopressin initiation stats
  Who gets vasopressin? Baseline characteristics (vaso vs no-vaso patients).
  Feature values at initiation (median [IQR]) and time to initiation from shock onset.
  Confounding assessment: Mann–Whitney / chi-square p-values; propensity discussion.
  Outputs:
    initiation_stats_{mimic,clif}.csv    median [IQR] of each feature at initiation
    baseline_comparison_{mimic,clif}.csv vaso vs no-vaso baseline characteristics
    plots/baseline_comparison_{mimic,clif}.png
    plots/initiation_features_{mimic,clif}.png

Analysis 2 — Per-feature threshold rules predicting clinician vasopressin action
  Threshold selected on MIMIC train (max Youden's J vs action_vaso).
  Evaluated on MIMIC test and CLIF:
    step-level: κ, AUROC, sens, spec at each timestep
    patient-level: does the patient ever receive vasopressin? (threshold on max feature value)
  Outputs:
    thresholds.csv
    step_eval_mimic.csv / step_eval_clif.csv
    patient_eval_mimic.csv / patient_eval_clif.csv
    plots/sweep_{feat}.png (one per continuous feature)
    plots/auroc_step_level.png / plots/auroc_patient_level.png

Analysis 3 — Decision tree over multiple features
  Depths 1–6 predicting action_vaso (step-level) and ever-vasopressin (patient-level).
  Trained on MIMIC, validated on MIMIC test and CLIF.
  Best-performing tree structure printed to console.
  Outputs:
    tree_eval.csv
    plots/tree_fidelity.png

Usage:
    python code/cross_site_vasopressin_analysis.py [--n-thresholds 100]
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE    = Path(__file__).parent
OUT_DIR = BASE.parent / "output" / "clinician_vasopressin"
PLOTS   = OUT_DIR / "plots"

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = BASE.parent / "Data"
MIMIC_COH  = DATA_DIR / "MIMIC" / "cohort.parquet"
MIMIC_FEAT = DATA_DIR / "MIMIC" / "features.parquet"
CLIF_FEAT  = DATA_DIR / "UCMC"  / "features.parquet"
CLIF_COH   = DATA_DIR / "UCMC"  / "cohort.parquet"

# ─── Feature config ────────────────────────────────────────────────────────────
# (column_name, display_label, is_binary)
# Both MIMIC and CLIF use the same column names; no remapping needed.
ANALYSIS_FEATURES = [
    ("time_hour",      "Time (h)",              False),
    ("norepinephrine", "NE (µg/kg/min)",        False),
    ("nee",            "NEE (µg/kg/min)",       False),
    ("mbp",            "MAP (mmHg)",            False),
    ("sofa",           "SOFA",                  False),
    ("lactate",        "Lactate (mmol/L)",      False),
    ("urine_output",   "Urine output (mL/h)",   False),
    ("creatinine",     "Creatinine (mg/dL)",    False),
    ("bun",            "BUN (mg/dL)",           False),
    ("ventil",         "Ventilation",           True),
    ("rrt",            "RRT",                   True),
    ("steroid",        "Corticosteroid",        True),
    ("fluids",         "Fluids (mL/hr)",        False),
]
FEAT_NAMES   = [f for f, _, _ in ANALYSIS_FEATURES]
FEAT_LABELS  = {f: l for f, l, _ in ANALYSIS_FEATURES}
BINARY_FEATS = {f for f, _, b in ANALYSIS_FEATURES if b}

# Baseline characteristics for vaso vs no-vaso comparison
MIMIC_BL_COLS = ["age", "sepsis_onset_sofa", "initial_lactate",
                 "charlson_comorbidity_index", "weight", "hospital_death"]
CLIF_BL_COLS  = ["age", "sepsis_onset_sofa", "initial_lactate", "weight", "hospital_death"]
BL_LABELS = {
    "age":                        "Age (years)",
    "sepsis_onset_sofa":          "SOFA at sepsis onset",
    "initial_lactate":            "Initial lactate (mmol/L)",
    "charlson_comorbidity_index": "Charlson CCI",
    "weight":                     "Weight (kg)",
    "hospital_death":             "Hospital mortality",
}
BL_BINARY = {"hospital_death"}

PCT_LO = 5
PCT_HI = 95

# ─── Dataset colors ────────────────────────────────────────────────────────────
# MIMIC is always lightgray.  Add new CLIF sites to CLIF_SITES (ordered);
# they receive colors from CLIF_PALETTE in order.
CLIF_PALETTE = ["#ffd328", "#ffb00e", "#fe8211", "#f04122", "#cb0824",
                "#ad1c54", "#6b1461", "#39114b"]
CLIF_SITES   = ["clif_ucmc"]   # extend when adding new sites
DS_COLORS = {site: CLIF_PALETTE[i % len(CLIF_PALETTE)]
             for i, site in enumerate(CLIF_SITES)}
DS_COLORS["mimic"] = "#b0b0b0"
DS_LABELS = {"mimic": "MIMIC-IV", "clif_ucmc": "CLIF (UCMC)"}


# ─── Helper functions ──────────────────────────────────────────────────────────

def _split_by_stay_id(
    df: pl.DataFrame,
    fracs: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Random 70/15/15 train/val/test split by stay_id."""
    ids = df["stay_id"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * fracs[0])
    n_va = int(n * fracs[1])
    tr_ids = set(ids[:n_tr].tolist())
    va_ids = set(ids[n_tr:n_tr + n_va].tolist())
    te_ids = set(ids[n_tr + n_va:].tolist())
    return (
        df.filter(pl.col("stay_id").is_in(list(tr_ids))),
        df.filter(pl.col("stay_id").is_in(list(va_ids))),
        df.filter(pl.col("stay_id").is_in(list(te_ids))),
    )


def _med_iqr(arr: np.ndarray) -> str:
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return "—"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]"


def _youden(pred: np.ndarray, target: np.ndarray) -> float:
    tp = int(((pred == 1) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens + spec - 1.0)


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    if a.sum() == 0 or a.sum() == len(a) or b.sum() == 0 or b.sum() == len(b):
        return float("nan")
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        po = float((a == b).mean())
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0


def _apply_rule(vals: np.ndarray, tau: float, direction: str) -> np.ndarray:
    return (vals > tau).astype(np.int32) if direction == "pos" else (vals < tau).astype(np.int32)


def _select_threshold(
    vals: np.ndarray, clin: np.ndarray, binary: bool, n_thresholds: int = 100
) -> tuple[float, str, float, list]:
    """Return (tau, direction, youden_j, sweep_records) maximising Youden's J vs clin."""
    if binary:
        rule = (vals > 0.5).astype(np.int32)
        return 0.5, "pos", _youden(rule, clin), []
    vals_c = np.where(np.isfinite(vals), vals, float(np.nanmedian(vals)))
    p5, p95 = np.nanpercentile(vals_c, PCT_LO), np.nanpercentile(vals_c, PCT_HI)
    if p5 >= p95:
        tau = float(np.median(vals_c))
        return tau, "pos", _youden((vals_c > tau).astype(np.int32), clin), []
    thresholds = np.linspace(p5, p95, n_thresholds)
    best_j, best_tau, best_dir = -1.0, thresholds[len(thresholds) // 2], "pos"
    sweep = []
    for tau in thresholds:
        for d in ("pos", "neg"):
            rule = _apply_rule(vals_c, tau, d)
            j = _youden(rule, clin)
            sweep.append((float(tau), d, float(j)))
            if j > best_j:
                best_j, best_tau, best_dir = j, float(tau), d
    return best_tau, best_dir, best_j, sweep


def _eval_step(vals: np.ndarray, clin: np.ndarray, tau: float, direction: str) -> dict:
    vals_c = np.where(np.isfinite(vals), vals, float(np.nanmedian(vals)))
    pred = _apply_rule(vals_c, tau, direction)
    kappa = _kappa(pred, clin)
    agree = float((pred == clin).mean())
    try:
        auc = roc_auc_score(clin, vals_c)
        auroc = max(auc, 1.0 - auc)
    except Exception:
        auroc = float("nan")
    cm = confusion_matrix(clin, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"kappa": kappa, "agreement": agree, "auroc": auroc, "sens": sens, "spec": spec,
            "_pred": pred}


def _eval_patient(df: pl.DataFrame, feat: str, tau: float, direction: str) -> dict:
    """Patient-level: threshold on max(feat) → predicts ever-vasopressin."""
    pat = (
        df.group_by("stay_id")
        .agg([
            (pl.col("action_vaso") == 1).any().cast(pl.Int32).alias("ever_vaso"),
            pl.col(feat).drop_nulls().max().alias("feat_max"),
        ])
        .drop_nulls(subset=["feat_max"])
    )
    ever_vaso = pat["ever_vaso"].to_numpy().astype(int)
    feat_max  = pat["feat_max"].to_numpy().astype(float)
    feat_max  = np.where(np.isfinite(feat_max), feat_max, float(np.nanmedian(feat_max)))

    pred  = _apply_rule(feat_max, tau, direction)
    kappa = _kappa(pred, ever_vaso)
    agree = float((pred == ever_vaso).mean())
    try:
        auc   = roc_auc_score(ever_vaso, feat_max)
        auroc = max(auc, 1.0 - auc)
    except Exception:
        auroc = float("nan")
    cm = confusion_matrix(ever_vaso, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    n_vaso   = int(ever_vaso.sum())
    n_novaso = int((ever_vaso == 0).sum())
    return {"kappa": kappa, "agreement": agree, "auroc": auroc,
            "sens": sens, "spec": spec, "n_vaso": n_vaso, "n_novaso": n_novaso}


# ─── Analysis 1: Vasopressin initiation stats ─────────────────────────────────

def _prior_vaso_exclusions(feat_df: pl.DataFrame, coh_df: pl.DataFrame) -> set:
    """Return stay_ids of patients who had vasopressin at t=0 or within 24h before t=0.

    MIMIC: uses action_vaso=1 at time_hour=0  UNION  first_vasopressin_time in (-24h, 0).
    CLIF:  uses action_vaso=1 at time_hour=0  UNION  vaso_before_traj=1 (no 24h granularity
           available in CLIF cohort; this conservatively excludes all pre-trajectory cases).
    """
    # Patients already on vasopressin at trajectory start (time_hour=0)
    at_t0 = set(
        feat_df.filter((pl.col("time_hour") == 0) & (pl.col("action_vaso") == 1))
        ["stay_id"].to_list()
    )

    # MIMIC: first_vasopressin_time in (trajectory_start - 24h, trajectory_start)
    prior_window: set = set()
    if "first_vasopressin_time" in coh_df.columns and "trajectory_start" in coh_df.columns:
        prior_window = set(
            coh_df.filter(
                pl.col("first_vasopressin_time").is_not_null()
                & (pl.col("first_vasopressin_time") < pl.col("trajectory_start"))
                & (pl.col("first_vasopressin_time") >= pl.col("trajectory_start") - pl.duration(hours=24))
            )["stay_id"].to_list()
        )

    # CLIF: vaso_before_traj flag (any pre-trajectory vasopressin)
    elif "vaso_before_traj" in coh_df.columns:
        prior_window = set(
            coh_df.filter(pl.col("vaso_before_traj") == 1)["stay_id"].to_list()
        )

    excl = at_t0 | prior_window
    return excl, len(at_t0), len(prior_window), len(excl)


def analysis_initiation(
    mimic_all: pl.DataFrame,
    mimic_coh: pl.DataFrame,
    clif_df:   pl.DataFrame,
    clif_coh:  pl.DataFrame,
) -> None:
    print("\n" + "=" * 65)
    print("ANALYSIS 1: Vasopressin initiation stats")
    print("  Exclusion: patients with vasopressin at t=0 or within 24h before t=0")
    print("=" * 65)

    datasets = [
        ("MIMIC", mimic_all, mimic_coh, MIMIC_BL_COLS),
        ("CLIF",  clif_df,   clif_coh,  CLIF_BL_COLS),
    ]

    collected = {}
    for ds_name, feat_df, coh_df, bl_cols in datasets:
        print(f"\n--- {ds_name} ---")

        # Exclude patients with prior/prevalent vasopressin
        excl_ids, n_at_t0, n_prior, n_total_excl = _prior_vaso_exclusions(feat_df, coh_df)
        n_before = feat_df["stay_id"].n_unique()
        feat_df  = feat_df.filter(~pl.col("stay_id").is_in(list(excl_ids)))
        coh_df   = coh_df.filter(~pl.col("stay_id").is_in(list(excl_ids)))
        n_after  = feat_df["stay_id"].n_unique()
        print(f"  Excluded {n_total_excl} patients  "
              f"(at t=0: {n_at_t0}; prior 24h window: {n_prior}; "
              f"union: {n_total_excl})")
        print(f"  Cohort after exclusion: {n_after}/{n_before} patients")

        # First vasopressin initiation per patient (0→1 transition)
        shifted = feat_df.sort(["stay_id", "time_hour"]).with_columns(
            pl.col("action_vaso").shift(1).over("stay_id").alias("prev_vaso")
        )
        inits = (
            shifted
            .filter((pl.col("action_vaso") == 1) & (pl.col("prev_vaso") == 0))
            .group_by("stay_id").first().sort("stay_id")
        )

        n_vaso  = inits["stay_id"].n_unique()
        n_total = feat_df["stay_id"].n_unique()
        print(f"  Patients with vasopressin: {n_vaso}/{n_total} ({100*n_vaso/n_total:.1f}%)")

        # Feature values at initiation
        init_rows = [{"feature": "Time to initiation (h)",
                      "median_iqr": _med_iqr(inits["time_hour"].to_numpy().astype(float))}]
        for f, lbl, binary in ANALYSIS_FEATURES:
            if f == "time_hour" or f not in inits.columns:
                continue
            arr = inits[f].cast(pl.Float64).fill_null(float("nan")).to_numpy()
            if binary:
                valid = arr[np.isfinite(arr)]
                val_str = f"{100 * valid.mean():.1f}%" if len(valid) > 0 else "—"
            else:
                val_str = _med_iqr(arr)
            init_rows.append({"feature": lbl, "median_iqr": val_str})

        pl.DataFrame(init_rows).write_csv(OUT_DIR / f"initiation_stats_{ds_name.lower()}.csv")
        print(f"  Saved: initiation_stats_{ds_name.lower()}.csv")
        for r in init_rows[:5]:
            print(f"    {r['feature']:<30} {r['median_iqr']}")

        # Baseline comparison: vaso vs no-vaso
        vaso_ids = set(inits["stay_id"].to_list())
        coh_ann  = coh_df.with_columns(
            pl.col("stay_id").is_in(list(vaso_ids)).cast(pl.Int8).alias("received_vaso")
        )
        bl_rows = []
        for bl_col in bl_cols:
            if bl_col not in coh_ann.columns:
                continue
            vaso_arr   = coh_ann.filter(pl.col("received_vaso") == 1)[bl_col].drop_nulls().cast(pl.Float64).to_numpy()
            novaso_arr = coh_ann.filter(pl.col("received_vaso") == 0)[bl_col].drop_nulls().cast(pl.Float64).to_numpy()
            if bl_col in BL_BINARY:
                vaso_str   = f"{100 * vaso_arr.mean():.1f}%"
                novaso_str = f"{100 * novaso_arr.mean():.1f}%"
                ct = np.array([[int((vaso_arr == 1).sum()),   int((vaso_arr == 0).sum())],
                               [int((novaso_arr == 1).sum()), int((novaso_arr == 0).sum())]])
                try:    _, pval, _, _ = stats.chi2_contingency(ct)
                except: pval = float("nan")
            else:
                vaso_str   = _med_iqr(vaso_arr)
                novaso_str = _med_iqr(novaso_arr)
                try:
                    _, pval = stats.mannwhitneyu(
                        vaso_arr[np.isfinite(vaso_arr)],
                        novaso_arr[np.isfinite(novaso_arr)],
                        alternative="two-sided",
                    )
                except: pval = float("nan")
            bl_rows.append({
                "characteristic": BL_LABELS.get(bl_col, bl_col),
                "vaso_n": len(vaso_arr), "novaso_n": len(novaso_arr),
                "vaso_stat": vaso_str, "novaso_stat": novaso_str,
                "p_value": round(float(pval), 4) if np.isfinite(pval) else None,
            })

        pl.DataFrame(bl_rows).write_csv(OUT_DIR / f"baseline_comparison_{ds_name.lower()}.csv")
        print(f"  Saved: baseline_comparison_{ds_name.lower()}.csv")

        collected[ds_name] = {
            "init_rows": init_rows, "bl_rows": bl_rows,
            "n_vaso": n_vaso, "n_novaso": n_total - n_vaso,
        }

    # Combined PNG tables (MIMIC + CLIF side by side)
    _plot_combined_baseline(
        collected["MIMIC"]["bl_rows"], collected["CLIF"]["bl_rows"],
        collected["MIMIC"]["n_vaso"], collected["MIMIC"]["n_novaso"],
        collected["CLIF"]["n_vaso"],  collected["CLIF"]["n_novaso"],
    )
    _plot_combined_initiation(
        collected["MIMIC"]["init_rows"], collected["CLIF"]["init_rows"],
        collected["MIMIC"]["n_vaso"],    collected["CLIF"]["n_vaso"],
    )


def _table_fig(col_labels: list, rows: list, title: str, col_widths: list | None = None) -> plt.Figure:
    """Render a formatted matplotlib table figure."""
    nrows = len(rows)
    ncols = len(col_labels)
    row_h = 0.42  # inches per row
    fig_h = max(3.0, row_h * (nrows + 2) + 1.2)
    fig_w = 11

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(nrows + 1):
                tbl[i, j].set_width(w)

    # Header
    for j in range(ncols):
        c = tbl[0, j]
        c.set_facecolor("#2d3a4a")
        c.set_edgecolor("white")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
        c.get_text().set_horizontalalignment("center")

    # Data rows
    for i in range(nrows):
        bg = "#f2f5f7" if i % 2 == 0 else "white"
        for j in range(ncols):
            c = tbl[i + 1, j]
            c.set_facecolor(bg)
            c.set_edgecolor("#d8dde2")
            c.get_text().set_horizontalalignment(
                "left" if j == 0 else "center"
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14, loc="left")
    fig.tight_layout(pad=0.4)
    return fig


def _pstr(pval) -> str:
    if pval is None or not np.isfinite(float(pval) if pval is not None else float("nan")):
        return "—"
    pval = float(pval)
    if pval < 0.001:  return "<0.001***"
    if pval < 0.01:   return f"{pval:.3f}**"
    if pval < 0.05:   return f"{pval:.3f}*"
    return f"{pval:.3f}"


def _plot_combined_baseline(
    mimic_rows: list[dict], clif_rows: list[dict],
    n_mv: int, n_mn: int, n_cv: int, n_cn: int,
) -> None:
    """One table with MIMIC + CLIF columns separated by a narrow dark divider."""
    if not mimic_rows and not clif_rows:
        return

    # Index both datasets by characteristic label
    m_idx = {r["characteristic"]: r for r in mimic_rows}
    c_idx = {r["characteristic"]: r for r in clif_rows}
    all_chars = list(dict.fromkeys(
        [r["characteristic"] for r in mimic_rows]
        + [r["characteristic"] for r in clif_rows]
    ))

    SEP = ""   # separator column content
    col_labels = [
        "Characteristic",
        f"Vaso\n(n={n_mv})", f"No-vaso\n(n={n_mn})", "p",
        SEP,
        f"Vaso\n(n={n_cv})", f"No-vaso\n(n={n_cn})", "p",
    ]
    rows = []
    for ch in all_chars:
        m = m_idx.get(ch)
        c = c_idx.get(ch)
        mv = m["vaso_stat"]   if m else "—"
        mn = m["novaso_stat"] if m else "—"
        mp = _pstr(m["p_value"] if m else None)
        cv = c["vaso_stat"]   if c else "—"
        cn = c["novaso_stat"] if c else "—"
        cp = _pstr(c["p_value"] if c else None)
        rows.append([ch, mv, mn, mp, SEP, cv, cn, cp])

    nrows = len(rows)
    ncols = len(col_labels)
    row_h = 0.42
    fig_h = max(3.0, row_h * (nrows + 2) + 1.5)
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Column widths: char(0.20), 4×mimic(0.10), sep(0.015), 4×clif(0.10) — normalised
    widths = [0.20, 0.115, 0.115, 0.07, 0.015, 0.115, 0.115, 0.07]
    for j, w in enumerate(widths):
        for i in range(nrows + 1):
            tbl[i, j].set_width(w)

    MIMIC_HDR  = "#1a3a5c"   # dark blue
    CLIF_HDR   = "#5c2a00"   # dark orange-brown
    SEP_COLOR  = "#222222"   # near-black separator
    ALT_A      = "#eef3f8"
    ALT_B      = "white"

    for j in range(ncols):
        c = tbl[0, j]
        if j == 4:                          # separator header
            c.set_facecolor(SEP_COLOR)
            c.set_edgecolor(SEP_COLOR)
        elif j in (0, 1, 2, 3):            # MIMIC header
            c.set_facecolor(MIMIC_HDR)
            c.set_edgecolor("white")
        else:                               # CLIF header
            c.set_facecolor(CLIF_HDR)
            c.set_edgecolor("white")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
        c.get_text().set_horizontalalignment("center")

    for i in range(nrows):
        bg = ALT_A if i % 2 == 0 else ALT_B
        for j in range(ncols):
            c = tbl[i + 1, j]
            if j == 4:
                c.set_facecolor(SEP_COLOR)
                c.set_edgecolor(SEP_COLOR)
                c.get_text().set_text("")
            else:
                c.set_facecolor(bg)
                c.set_edgecolor("#d0d5da")
                c.get_text().set_horizontalalignment(
                    "left" if j == 0 else "center"
                )

    title = (
        "Baseline Characteristics: Vasopressin vs No-Vasopressin  "
        "(MIMIC-IV  |  CLIF)\n"
        "Continuous: median [IQR], Mann-Whitney U;  Binary: %, chi-square"
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14, loc="left")
    fig.tight_layout(pad=0.4)
    out = PLOTS / "baseline_comparison_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def _plot_combined_initiation(
    mimic_rows: list[dict], clif_rows: list[dict],
    n_mimic: int, n_clif: int,
) -> None:
    """One table: feature values at initiation for MIMIC and CLIF side-by-side."""
    m_idx = {r["feature"]: r["median_iqr"] for r in mimic_rows}
    c_idx = {r["feature"]: r["median_iqr"] for r in clif_rows}
    all_feats = list(dict.fromkeys(
        [r["feature"] for r in mimic_rows] + [r["feature"] for r in clif_rows]
    ))

    SEP = ""
    col_labels = [
        "Feature",
        f"MIMIC-IV  (n={n_mimic})\nMedian [IQR] or %",
        SEP,
        f"CLIF  (n={n_clif})\nMedian [IQR] or %",
    ]
    rows = [[f, m_idx.get(f, "—"), SEP, c_idx.get(f, "—")] for f in all_feats]

    nrows = len(rows)
    ncols = 4
    row_h = 0.42
    fig_h = max(3.0, row_h * (nrows + 2) + 1.2)
    fig_w = 11
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    widths = [0.30, 0.33, 0.015, 0.33]
    for j, w in enumerate(widths):
        for i in range(nrows + 1):
            tbl[i, j].set_width(w)

    MIMIC_HDR = "#1a3a5c"
    CLIF_HDR  = "#5c2a00"
    SEP_COLOR = "#222222"
    ALT_A = "#eef3f8"
    ALT_B = "white"

    for j in range(ncols):
        c = tbl[0, j]
        if j == 2:
            c.set_facecolor(SEP_COLOR); c.set_edgecolor(SEP_COLOR)
        elif j in (0, 1):
            c.set_facecolor(MIMIC_HDR); c.set_edgecolor("white")
        else:
            c.set_facecolor(CLIF_HDR); c.set_edgecolor("white")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
        c.get_text().set_horizontalalignment("center")

    for i in range(nrows):
        bg = ALT_A if i % 2 == 0 else ALT_B
        for j in range(ncols):
            c = tbl[i + 1, j]
            if j == 2:
                c.set_facecolor(SEP_COLOR); c.set_edgecolor(SEP_COLOR)
                c.get_text().set_text("")
            else:
                c.set_facecolor(bg); c.set_edgecolor("#d0d5da")
                c.get_text().set_horizontalalignment(
                    "left" if j == 0 else "center"
                )

    ax.set_title(
        "Feature Values at Vasopressin Initiation  (MIMIC-IV  |  CLIF)",
        fontsize=10, fontweight="bold", pad=14, loc="left",
    )
    fig.tight_layout(pad=0.4)
    out = PLOTS / "initiation_features_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── Analysis 2: Per-feature threshold rules ──────────────────────────────────

def analysis_thresholds(
    mimic_train: pl.DataFrame,
    mimic_test:  pl.DataFrame,
    clif_train:  pl.DataFrame,
    clif_test:   pl.DataFrame,
    n_thresholds: int = 100,
) -> dict:
    """Separate threshold selection on each dataset's train split."""
    print("\n" + "=" * 65)
    print("ANALYSIS 2: Per-feature thresholds vs clinician vasopressin action")
    print("  (thresholds selected independently on each dataset's train split)")
    print("=" * 65)

    thresh_rows = []
    step_rows   = []
    pat_rows    = []

    for feat, lbl, binary in ANALYSIS_FEATURES:
        print(f"\n  Feature: {lbl}")

        # ── MIMIC threshold ───────────────────────────────────────────────────
        if feat not in mimic_train.columns:
            print(f"    [skip MIMIC] not in train")
            m_tau, m_dir, m_j, m_sweep = None, None, None, []
        else:
            tr_vals = mimic_train[feat].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()
            tr_clin = mimic_train["action_vaso"].to_numpy().astype(int)
            m_tau, m_dir, m_j, m_sweep = _select_threshold(tr_vals, tr_clin, binary, n_thresholds)
            print(f"    MIMIC  tau={m_tau:.4g} ({'>' if m_dir=='pos' else '<'})  J={m_j:.4f}")
            thresh_rows.append({"dataset": "mimic", "feature": feat, "label": lbl,
                                 "threshold": m_tau, "direction": m_dir, "youden_j": m_j})

        # ── CLIF threshold ────────────────────────────────────────────────────
        if feat not in clif_train.columns:
            print(f"    [skip CLIF] not in train")
            c_tau, c_dir, c_j, c_sweep = None, None, None, []
        else:
            cl_vals = clif_train[feat].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()
            cl_clin = clif_train["action_vaso"].to_numpy().astype(int)
            c_tau, c_dir, c_j, c_sweep = _select_threshold(cl_vals, cl_clin, binary, n_thresholds)
            print(f"    CLIF   tau={c_tau:.4g} ({'>' if c_dir=='pos' else '<'})  J={c_j:.4f}")
            thresh_rows.append({"dataset": "clif_ucmc", "feature": feat, "label": lbl,
                                 "threshold": c_tau, "direction": c_dir, "youden_j": c_j})

        # ── Combined sweep plot ───────────────────────────────────────────────
        _plot_sweep_combined(feat, lbl, [
            ("mimic",     m_sweep, m_tau, m_j),
            ("clif_ucmc", c_sweep, c_tau, c_j),
        ])

        # ── MIMIC test — step + patient ───────────────────────────────────────
        if m_tau is not None and feat in mimic_test.columns:
            te_vals = mimic_test[feat].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()
            te_clin = mimic_test["action_vaso"].to_numpy().astype(int)
            r = _eval_step(te_vals, te_clin, m_tau, m_dir)
            step_rows.append({"dataset": "mimic", "feature": feat, "label": lbl,
                              **{k: v for k, v in r.items() if k != "_pred"}})
            print(f"    MIMIC test step:     κ={r['kappa']:.3f}  AUROC={r['auroc']:.3f}  "
                  f"sens={r['sens']:.3f}  spec={r['spec']:.3f}")
            pr = _eval_patient(mimic_test, feat, m_tau, m_dir)
            pat_rows.append({"dataset": "mimic", "feature": feat, "label": lbl, **pr})
            print(f"    MIMIC test patient:  κ={pr['kappa']:.3f}  AUROC={pr['auroc']:.3f}  "
                  f"n_vaso={pr['n_vaso']}  n_novaso={pr['n_novaso']}")

        # ── CLIF test — step + patient ────────────────────────────────────────
        if c_tau is not None and feat in clif_test.columns:
            cl_vals = clif_test[feat].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()
            cl_clin = clif_test["action_vaso"].to_numpy().astype(int)
            r = _eval_step(cl_vals, cl_clin, c_tau, c_dir)
            step_rows.append({"dataset": "clif_ucmc", "feature": feat, "label": lbl,
                              **{k: v for k, v in r.items() if k != "_pred"}})
            print(f"    CLIF test step:      κ={r['kappa']:.3f}  AUROC={r['auroc']:.3f}  "
                  f"sens={r['sens']:.3f}  spec={r['spec']:.3f}")
            pr = _eval_patient(clif_test, feat, c_tau, c_dir)
            pat_rows.append({"dataset": "clif_ucmc", "feature": feat, "label": lbl, **pr})
            print(f"    CLIF test patient:   κ={pr['kappa']:.3f}  AUROC={pr['auroc']:.3f}  "
                  f"n_vaso={pr['n_vaso']}  n_novaso={pr['n_novaso']}")
        elif feat not in clif_test.columns:
            print(f"    CLIF: '{feat}' not available — skipped")

    # Save tables
    pl.DataFrame(thresh_rows).write_csv(OUT_DIR / "thresholds.csv")
    step_df = pl.DataFrame(step_rows)
    for ds in step_df["dataset"].unique().to_list():
        step_df.filter(pl.col("dataset") == ds).drop("dataset").write_csv(
            OUT_DIR / f"step_eval_{ds}.csv")
    pat_df = pl.DataFrame(pat_rows)
    for ds in pat_df["dataset"].unique().to_list():
        pat_df.filter(pl.col("dataset") == ds).drop("dataset").write_csv(
            OUT_DIR / f"patient_eval_{ds}.csv")

    print("\n  Saved: thresholds.csv, step_eval_*.csv, patient_eval_*.csv")
    _plot_auroc_bars(step_df, "step_level")
    _plot_auroc_bars(pat_df,  "patient_level")

    return thresh_rows


def _plot_sweep_combined(
    feat: str,
    lbl: str,
    datasets: list[tuple],  # (ds_key, sweep, tau, j) — tau/j may be None
) -> None:
    """Single sweep plot with one curve per dataset, color-coded optimal τ."""
    has_data = any(sw for _, sw, _, _ in datasets)
    if not has_data:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds_key, sweep, tau, j in datasets:
        if not sweep or tau is None:
            continue
        color = DS_COLORS.get(ds_key, "gray")
        label = DS_LABELS.get(ds_key, ds_key)
        # Collapse both directions: keep max J at each tau value
        tau_map: dict = {}
        for tv, _, jv in sweep:
            if tv not in tau_map or jv > tau_map[tv]:
                tau_map[tv] = jv
        ts = sorted(tau_map)
        js = [tau_map[t] for t in ts]
        ax.plot(ts, js, color=color, lw=2.0, alpha=0.88, label=label)
        ax.axvline(tau, ls="--", color=color, lw=1.3, alpha=0.85)
        # Annotate with slightly offset text to avoid overlap
        ax.annotate(
            f"τ={tau:.4g}  J={j:.3f}",
            xy=(tau, j),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=color, lw=0.5),
        )
    ax.set_xlabel(lbl, fontsize=10)
    ax.set_ylabel("Youden's J", fontsize=10)
    ax.set_title(
        f"Threshold sweep — {lbl}\n"
        "Optimal τ selected on each dataset's train split",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"sweep_{feat}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_auroc_bars(eval_df: pl.DataFrame, level: str) -> None:
    feats    = eval_df["feature"].unique().to_list()
    datasets = sorted(eval_df["dataset"].unique().to_list())

    mean_auroc = {
        f: float(np.nanmean(eval_df.filter(pl.col("feature") == f)["auroc"].to_numpy()))
        for f in feats
    }
    feats_sorted = sorted(feats, key=lambda f: mean_auroc[f])
    n = len(feats_sorted)
    x = np.arange(n)
    w = 0.36 if len(datasets) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(max(9, n * 0.9), 5.5))
    for i, ds in enumerate(datasets):
        sub  = eval_df.filter(pl.col("dataset") == ds)
        aurocs = []
        for f in feats_sorted:
            row = sub.filter(pl.col("feature") == f)
            val = float(row["auroc"][0]) if len(row) > 0 and row["auroc"][0] is not None else 0.5
            aurocs.append(val)
        offset = (i - (len(datasets) - 1) / 2) * w
        ax.barh([xi + offset for xi in x], aurocs, height=w,
                color=DS_COLORS.get(ds, "gray"), alpha=0.88,
                label=DS_LABELS.get(ds, ds))
        for xi, a in zip(x, aurocs):
            ax.text(a + 0.005, xi + offset, f"{a:.3f}", va="center", fontsize=7)

    ax.set_yticks(x)
    ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in feats_sorted], fontsize=9)
    ax.axvline(0.5, color="gray", lw=1.0, ls="--", label="Chance (0.5)")
    ax.axvline(0.7, color="#555", lw=0.7, ls=":", alpha=0.5, label="AUROC=0.7")
    ax.set_xlabel("AUROC — predicting clinician vasopressin", fontsize=10)
    ax.set_xlim(0.40, 1.08)
    title_level = "step-level (action_vaso)" if level == "step_level" else "patient-level (ever-vasopressin)"
    ax.set_title(f"Per-feature discriminability — {title_level}\n"
                 "(each dataset's threshold trained on its own 70% train split)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = PLOTS / f"auroc_{level}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── Analysis 3: Decision tree ────────────────────────────────────────────────

def analysis_decision_tree(
    mimic_train: pl.DataFrame,
    mimic_test:  pl.DataFrame,
    clif_train:  pl.DataFrame,
    clif_test:   pl.DataFrame,
) -> None:
    """Train separate trees on each dataset's train split; evaluate on each test split."""
    print("\n" + "=" * 65)
    print("ANALYSIS 3: Decision tree — predicting clinician vasopressin action")
    print("  (separate trees trained on MIMIC train and CLIF train)")
    print("=" * 65)

    mimic_feats = [f for f in FEAT_NAMES if f in mimic_train.columns and f in mimic_test.columns]
    clif_feats  = [f for f in FEAT_NAMES if f in clif_train.columns  and f in clif_test.columns]
    print(f"  MIMIC features ({len(mimic_feats)}): {', '.join(mimic_feats)}")
    print(f"  CLIF  features ({len(clif_feats)}): {', '.join(clif_feats)}")

    def _arr(df, feats):
        X = df.select(feats).fill_null(0.0).fill_nan(0.0).to_numpy().astype(float)
        y = df["action_vaso"].to_numpy().astype(int)
        return X, y

    def _pat_arr(df, feats):
        pat = (
            df.group_by("stay_id")
            .agg(
                [(pl.col("action_vaso") == 1).any().cast(pl.Int32).alias("ever_vaso")]
                + [pl.col(f).cast(pl.Float64).mean().alias(f) for f in feats]
            )
            .fill_null(0.0)
        )
        X = np.where(np.isfinite(pat.select(feats).to_numpy()), pat.select(feats).to_numpy(), 0.0)
        y = pat["ever_vaso"].to_numpy().astype(int)
        return X, y

    def _metrics(clf, X, y):
        pred = clf.predict(X)
        k    = _kappa(pred, y)
        agr  = float((pred == y).mean())
        try:   auc = max(roc_auc_score(y, clf.predict_proba(X)[:, 1]), 0.5)
        except: auc = float("nan")
        return k, agr, auc

    depths    = list(range(1, 7))
    tree_rows = []
    best_trees: dict = {}  # depth → {"mimic": (step, pat), "clif_ucmc": (step, pat)}

    for depth in depths:
        entry: dict = {"depth": depth}
        best_trees[depth] = {}

        for ds_key, tr_feats, tr_df, te_df in [
            ("mimic",     mimic_feats, mimic_train, mimic_test),
            ("clif_ucmc", clif_feats,  clif_train,  clif_test),
        ]:
            X_tr, y_tr = _arr(tr_df, tr_feats)
            X_te, y_te = _arr(te_df, tr_feats)
            X_tr_p, y_tr_p = _pat_arr(tr_df, tr_feats)
            X_te_p, y_te_p = _pat_arr(te_df, tr_feats)

            dt_s = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt_s.fit(X_tr, y_tr)
            dt_p = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt_p.fit(X_tr_p, y_tr_p)
            best_trees[depth][ds_key] = (dt_s, dt_p, tr_feats)

            sk, sa, sau = _metrics(dt_s, X_te, y_te)
            pk, pa, pau = _metrics(dt_p, X_te_p, y_te_p)
            entry[f"step_kappa_{ds_key}"]   = sk
            entry[f"step_auroc_{ds_key}"]   = sau
            entry[f"patient_kappa_{ds_key}"]= pk
            entry[f"patient_auroc_{ds_key}"]= pau
            print(f"  depth={depth} {ds_key:<10} step κ={sk:.3f} AUROC={sau:.3f} | "
                  f"patient κ={pk:.3f} AUROC={pau:.3f}")

        tree_rows.append(entry)

    pl.DataFrame(tree_rows).write_csv(OUT_DIR / "tree_eval.csv")
    print("\n  Saved: tree_eval.csv")

    def _safe_kappa(r, key):
        v = r.get(key)
        return v if (v is not None and np.isfinite(v)) else -1.0

    for ds_key, tr_feats_key in [("mimic", mimic_feats), ("clif_ucmc", clif_feats)]:
        lbl = DS_LABELS.get(ds_key, ds_key)
        fl  = [FEAT_LABELS.get(f, f) for f in tr_feats_key]

        sk_key = f"step_kappa_{ds_key}"
        pk_key = f"patient_kappa_{ds_key}"
        best_s = max(tree_rows, key=lambda r: _safe_kappa(r, sk_key))["depth"]
        best_p = max(tree_rows, key=lambda r: _safe_kappa(r, pk_key))["depth"]

        print(f"\n  [{lbl}] Best step depth={best_s}  "
              f"κ={tree_rows[best_s-1][sk_key]:.3f}")
        print(export_text(best_trees[best_s][ds_key][0], feature_names=fl, max_depth=4))

        print(f"  [{lbl}] Best patient depth={best_p}  "
              f"κ={tree_rows[best_p-1][pk_key]:.3f}")
        print(export_text(best_trees[best_p][ds_key][1], feature_names=fl, max_depth=4))

    _plot_tree_fidelity(tree_rows)


def _plot_tree_fidelity(tree_rows: list[dict]) -> None:
    depths = [r["depth"] for r in tree_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        (axes[0], "step_kappa",   "step_auroc",   "Step-level (action_vaso)"),
        (axes[1], "patient_kappa","patient_auroc", "Patient-level (ever-vasopressin)"),
    ]

    for ax, k_pfx, a_pfx, title in panels:
        for ds_key in ["mimic", "clif_ucmc"]:
            color = DS_COLORS.get(ds_key, "gray")
            label = DS_LABELS.get(ds_key, ds_key)
            kv = [r.get(f"{k_pfx}_{ds_key}", float("nan")) for r in tree_rows]
            av = [r.get(f"{a_pfx}_{ds_key}", float("nan")) for r in tree_rows]
            ax.plot(depths, kv, "o-",  color=color, lw=2,   ms=7, label=f"κ  {label}")
            ax.plot(depths, av, "s--", color=color, lw=1.5, ms=6, alpha=0.6,
                    label=f"AUROC  {label}")
            for d, k in zip(depths, kv):
                if np.isfinite(k):
                    ax.annotate(f"{k:.3f}", (d, k), textcoords="offset points",
                                xytext=(0, 8), fontsize=7.5, color=color, ha="center")

        ax.set_xlabel("Tree max_depth", fontsize=10)
        ax.set_ylabel("Metric", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(depths)
        ax.legend(fontsize=8, loc="lower right")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Decision tree: predicting clinician vasopressin\n"
                 "Solid=Cohen's κ, Dashed=AUROC  |  Each dataset trained on own 70% split",
                 fontsize=11)
    fig.tight_layout()
    out = PLOTS / "tree_fidelity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-thresholds", type=int, default=100,
                    help="Threshold candidates per feature (default: 100)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    # ── Load MIMIC ────────────────────────────────────────────────────────────
    print("Loading MIMIC data...")
    mimic_coh = pl.read_parquet(MIMIC_COH)
    mimic_all = (
        pl.read_parquet(MIMIC_FEAT)
        .join(mimic_coh.select(["stay_id", "hospital_death"]), on="stay_id", how="left")
    )

    # 70/15/15 random split by stay_id
    train_df, _mimic_val, test_df = _split_by_stay_id(mimic_all, seed=42)

    # ── Load CLIF ─────────────────────────────────────────────────────────────
    print("Loading CLIF data...")
    clif_coh  = pl.read_parquet(CLIF_COH)
    clif_df   = (
        pl.read_parquet(CLIF_FEAT)
        .join(clif_coh.select(["stay_id", "hospital_death"]), on="stay_id", how="left")
    )

    # 70/15/15 random split by stay_id for threshold + tree training
    clif_train, clif_val, clif_test = _split_by_stay_id(clif_df, seed=42)

    print(f"  MIMIC train: {train_df['stay_id'].n_unique():,} pts  ({len(train_df):,} steps)")
    print(f"  MIMIC test:  {test_df['stay_id'].n_unique():,} pts  ({len(test_df):,} steps)")
    print(f"  MIMIC all:   {mimic_all['stay_id'].n_unique():,} pts  ({len(mimic_all):,} steps)")
    print(f"  CLIF all:    {clif_df['stay_id'].n_unique():,} pts  ({len(clif_df):,} steps)")
    print(f"  CLIF train:  {clif_train['stay_id'].n_unique():,} pts  ({len(clif_train):,} steps)")
    print(f"  CLIF test:   {clif_test['stay_id'].n_unique():,} pts  ({len(clif_test):,} steps)")

    # ── Run analyses ──────────────────────────────────────────────────────────
    analysis_initiation(mimic_all, mimic_coh, clif_df, clif_coh)
    analysis_thresholds(train_df, test_df, clif_train, clif_test, n_thresholds=args.n_thresholds)
    analysis_decision_tree(train_df, test_df, clif_train, clif_test)

    print(f"\n{'=' * 65}")
    print(f"All outputs in: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
