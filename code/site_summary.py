#!/usr/bin/env python3
"""
analysis/site_summary.py

Federated-safe aggregate summary for a septic shock cohort.

Reads:
  Data/<DATASET>/cohort.parquet
  Data/<DATASET>/features.parquet
  Data/<DATASET>/cohort_filter_counts.csv

Writes only aggregate CSVs to outputs/<DATASET>/ — no patient-level data leaves the site.

Privacy guarantees
  - No row-level values, identifiers, free text, or exact dates in any output.
  - Cell suppression: n < K=11 → count shown as "<11", derived stats blank.
  - Continuous summaries rounded to 3 decimal places.
  - ROC/threshold curves evaluated on a fixed 100-point grid derived from
    rounded [p5, p95] of training data — NOT one point per observation.

Outcome for analyses 4 and 5
  Per-timestep imminent initiation: at each "at-risk" hour (patient not yet on
  vasopressin, i.e. previous action_vaso = 0), outcome = action_vaso at that hour.
  Threshold selected by max Youden's J on TRAIN split; carried unchanged to val/test.

Usage:
    python analysis/site_summary.py --dataset ucmc
    python analysis/site_summary.py --dataset mimic
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR         = Path(__file__).parent.parent

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

SUPPRESS_K       = 11       # suppress cells derived from fewer than K patients
ROUND_N          = 3        # decimal places for continuous statistics
N_THRESH_GRID    = 100      # grid points per continuous feature

# Clinical features for analyses 4 and 5
# urine_output excluded: all zero in CLIF 2.1.0
ANALYSIS_FEATURES_CONT = [
    "time_hour", "norepinephrine", "nee", "mbp", "sofa",
    "lactate", "creatinine", "bun", "fluids",
]
ANALYSIS_FEATURES_BIN  = ["ventil", "rrt", "steroid"]
ANALYSIS_FEATURES      = ANALYSIS_FEATURES_CONT + ANALYSIS_FEATURES_BIN

# Baseline table
BL_CONTINUOUS  = ["age", "weight", "sepsis_onset_sofa", "initial_lactate", "traj_hours"]
BL_BINARY      = ["hospital_death"]
BL_CATEGORICAL = ["gender", "race"]

# ============================================================
# HELPERS
# ============================================================

def _r(x):
    """Round to ROUND_N decimals; pass through None/NaN."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, ROUND_N)
    except (TypeError, ValueError):
        return None


def _n_str(n):
    """Return count as string; suppress if < K."""
    return str(int(n)) if int(n) >= SUPPRESS_K else f"<{SUPPRESS_K}"


def _suppress(val, n):
    """Return val if n >= K, else None."""
    return val if int(n) >= SUPPRESS_K else None


def _smd_cont(m1, s1, m2, s2):
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    return _r((m1 - m2) / pooled) if pooled > 0 else None


def _smd_bin(p1, p2):
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    return _r((p1 - p2) / denom) if denom > 0 else None


# ============================================================
# DATA LOADING AND SPLITTING
# ============================================================

def load_and_split():
    """Load cohort + features; add ever_vaso flag; split 70/15/15 by patient."""
    coh  = pd.read_parquet(INPUT_DIR / "cohort.parquet")
    feat = pd.read_parquet(INPUT_DIR / "features.parquet")

    # LOCF per patient for continuous features before any analysis
    feat = feat.sort_values(["stay_id", "time_hour"])
    for col in ANALYSIS_FEATURES_CONT:
        if col in feat.columns:
            feat[col] = feat.groupby("stay_id")[col].ffill()

    # Patient-level: ever received vasopressin during trajectory
    ever_vaso = (
        feat.groupby("stay_id")["action_vaso"]
        .max()
        .rename("ever_vaso")
        .reset_index()
    )
    coh = coh.merge(ever_vaso, on="stay_id", how="left")
    coh["ever_vaso"] = coh["ever_vaso"].fillna(0).astype(int)

    # Patient-level split (deterministic)
    ids = coh["stay_id"].values.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    ids = ids[rng.permutation(len(ids))]
    n     = len(ids)
    n_tr  = int(n * TRAIN_FRAC)
    n_va  = int(n * VAL_FRAC)
    tr_ids = set(ids[:n_tr])
    va_ids = set(ids[n_tr : n_tr + n_va])

    def _split_label(s):
        if s in tr_ids:
            return "train"
        if s in va_ids:
            return "val"
        return "test"

    coh["split"] = coh["stay_id"].map(_split_label)
    feat = feat.merge(coh[["stay_id", "split", "ever_vaso"]], on="stay_id", how="left")

    return coh, feat


# ============================================================
# OUTPUT 1: cohort_filter_counts.csv (copy as-is)
# ============================================================

def write_filter_counts():
    src = INPUT_DIR / "cohort_filter_counts.csv"
    dst = OUTPUT_DIR / "cohort_filter_counts.csv"
    shutil.copy2(src, dst)
    print(f"  Copied  {dst.name}")


# ============================================================
# OUTPUT 2: split_counts.csv
# ============================================================

def write_split_counts(coh):
    rows = []
    for split in ["train", "val", "test", "all"]:
        sub = coh if split == "all" else coh[coh["split"] == split]
        n_tot  = len(sub)
        n_vaso = int((sub["ever_vaso"] == 1).sum())
        n_none = int((sub["ever_vaso"] == 0).sum())
        rows.extend([
            {"split": split, "outcome_group": "ever_vaso_yes", "n_patients": _n_str(n_vaso)},
            {"split": split, "outcome_group": "ever_vaso_no",  "n_patients": _n_str(n_none)},
            {"split": split, "outcome_group": "total",         "n_patients": _n_str(n_tot)},
        ])
    out = OUTPUT_DIR / "split_counts.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 3: baseline_table1.csv
# ============================================================

def _cont_row(variable, col, groups_df, group_n):
    stats = {}
    for g, df in groups_df.items():
        arr = pd.to_numeric(df[col], errors="coerce").dropna()
        n       = len(arr)
        n_miss  = group_n[g] - n
        if n >= SUPPRESS_K:
            stats[g] = {
                "n": n, "n_missing": n_miss,
                "mean":   _r(arr.mean()),   "sd":     _r(arr.std()),
                "median": _r(arr.median()), "q25":    _r(arr.quantile(0.25)),
                "q75":    _r(arr.quantile(0.75)),
                "min":    _r(arr.min()),    "max":    _r(arr.max()),
                "pct": None,
                "_mean": float(arr.mean()), "_sd": float(arr.std()), "_n": n,
            }
        else:
            stats[g] = {k: None for k in ("mean", "sd", "median", "q25", "q75", "min", "max", "pct")}
            stats[g].update({"n": _n_str(n), "n_missing": n_miss, "_mean": None, "_sd": None, "_n": n})

    # SMD (vaso vs no_vaso)
    m1, s1 = stats["vaso"].get("_mean"), stats["vaso"].get("_sd")
    m2, s2 = stats["no_vaso"].get("_mean"), stats["no_vaso"].get("_sd")
    smd = _smd_cont(m1, s1, m2, s2) if all(v is not None for v in [m1, s1, m2, s2]) else None

    row = {"variable": variable, "level": "", "type": "continuous", "smd": smd}
    for g in ("vaso", "no_vaso", "overall"):
        for k in ("n", "n_missing", "mean", "sd", "median", "q25", "q75", "min", "max", "pct"):
            row[f"{g}_{k}"] = stats[g].get(k)
    return row


def _bin_row(variable, col, groups_df, group_n):
    stats = {}
    for g, df in groups_df.items():
        arr    = pd.to_numeric(df[col], errors="coerce").dropna()
        n      = len(arr)
        n_miss = group_n[g] - n
        pos    = int(arr.sum())
        pct    = _suppress(round(pos / n * 100, ROUND_N) if n > 0 else None, pos)
        stats[g] = {"n": _n_str(n), "n_missing": n_miss, "pct": pct,
                    "_n": n, "_pos": pos}

    p1 = stats["vaso"]["_pos"] / max(stats["vaso"]["_n"], 1)
    p2 = stats["no_vaso"]["_pos"] / max(stats["no_vaso"]["_n"], 1)
    smd = _smd_bin(p1, p2)

    row = {"variable": variable, "level": "=1", "type": "binary", "smd": smd}
    for g in ("vaso", "no_vaso", "overall"):
        row[f"{g}_n"]        = stats[g]["n"]
        row[f"{g}_n_missing"] = stats[g]["n_missing"]
        row[f"{g}_pct"]      = stats[g]["pct"]
        for k in ("mean", "sd", "median", "q25", "q75", "min", "max"):
            row[f"{g}_{k}"] = None
    return row


def _cat_rows(variable, col, groups_df, group_n):
    all_levels = sorted(
        set(groups_df["overall"][col].dropna().unique())
    )
    rows = []
    for level in all_levels:
        stats = {}
        for g, df in groups_df.items():
            n_grp  = group_n[g]
            cnt    = int((df[col].dropna() == level).sum())
            pct    = _suppress(round(cnt / n_grp * 100, ROUND_N) if n_grp > 0 else None, cnt)
            stats[g] = {"n": _n_str(cnt), "n_missing": None, "pct": pct,
                        "_cnt": cnt, "_n": n_grp}

        p1  = stats["vaso"]["_cnt"] / max(stats["vaso"]["_n"], 1)
        p2  = stats["no_vaso"]["_cnt"] / max(stats["no_vaso"]["_n"], 1)
        c1  = stats["vaso"]["_cnt"]
        c2  = stats["no_vaso"]["_cnt"]
        smd = _smd_bin(p1, p2) if (c1 >= SUPPRESS_K and c2 >= SUPPRESS_K) else None

        row = {"variable": variable, "level": level, "type": "categorical", "smd": smd}
        for g in ("vaso", "no_vaso", "overall"):
            row[f"{g}_n"]         = stats[g]["n"]
            row[f"{g}_n_missing"] = None
            row[f"{g}_pct"]       = stats[g]["pct"]
            for k in ("mean", "sd", "median", "q25", "q75", "min", "max"):
                row[f"{g}_{k}"] = None
        rows.append(row)
    return rows


def write_baseline_table1(coh):
    groups_df = {
        "vaso":    coh[coh["ever_vaso"] == 1],
        "no_vaso": coh[coh["ever_vaso"] == 0],
        "overall": coh,
    }
    group_n = {g: len(df) for g, df in groups_df.items()}

    rows = []
    for col in BL_CONTINUOUS:
        if col in coh.columns:
            rows.append(_cont_row(col, col, groups_df, group_n))
    for col in BL_BINARY:
        if col in coh.columns:
            rows.append(_bin_row(col, col, groups_df, group_n))
    for col in BL_CATEGORICAL:
        if col in coh.columns:
            rows.extend(_cat_rows(col, col, groups_df, group_n))

    out = OUTPUT_DIR / "baseline_table1.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 4: feature_at_initiation.csv
# ============================================================

def write_feature_at_initiation(feat):
    """Aggregate feature values at the first vasopressin initiation timestep."""
    fs = feat.sort_values(["stay_id", "time_hour"]).copy()
    fs["prev_vaso"] = fs.groupby("stay_id")["action_vaso"].shift(1).fillna(0)

    # First 0→1 transition per patient
    init = (
        fs[(fs["action_vaso"] == 1) & (fs["prev_vaso"] == 0)]
        .drop_duplicates(subset=["stay_id"], keep="first")
    )
    n_total = len(init)

    rows = []
    for col in ANALYSIS_FEATURES:
        if col not in init.columns:
            continue
        arr = pd.to_numeric(init[col], errors="coerce").dropna()
        n     = len(arr)
        n_miss = n_total - n
        row   = {"feature": col, "n": _n_str(n), "n_missing": n_miss}
        if n >= SUPPRESS_K:
            row.update({
                "mean":   _r(arr.mean()),   "sd":     _r(arr.std()),
                "median": _r(arr.median()), "q25":    _r(arr.quantile(0.25)),
                "q75":    _r(arr.quantile(0.75)),
                "min":    _r(arr.min()),    "max":    _r(arr.max()),
            })
        else:
            row.update({k: None for k in ("mean", "sd", "median", "q25", "q75", "min", "max")})
        rows.append(row)

    out = OUTPUT_DIR / "feature_at_initiation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Wrote   {out.name}")


# ============================================================
# OUTPUT 5a+5b: feature_thresholds_youden.csv + feature_roc_curves.csv
# ============================================================

def _threshold_grid(train_vals, is_binary):
    """Fixed grid that cannot reconstruct individual observations."""
    if is_binary:
        # Three points give proper ROC corners for a binary predictor
        return np.array([-0.5, 0.5, 1.5])

    p5  = float(np.nanpercentile(train_vals[np.isfinite(train_vals)], 5))
    p95 = float(np.nanpercentile(train_vals[np.isfinite(train_vals)], 95))

    # Round to 2 significant figures so thresholds don't reveal individual values
    def _sig2(x):
        if x == 0:
            return 0.0
        mag = int(np.floor(np.log10(abs(x))))
        return round(x, -(mag - 1))

    lo, hi = _sig2(p5), _sig2(p95)
    if lo >= hi:
        hi = lo + abs(lo * 0.1) + 1e-9
    return np.linspace(lo, hi, N_THRESH_GRID)


def _eval_thresh(y_true, y_score, threshold, direction):
    pred = (y_score > threshold if direction == "gt" else y_score < threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"tpr": sens, "fpr": 1 - spec,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "youden_j": sens + spec - 1.0}


_integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _auc(fprs, tprs):
    """Trapezoidal AUC with (0,0) and (1,1) corner anchors added.

    The threshold grid covers only [p5, p95] of training values, so without
    explicit corners the computed area misses the tails.  Adding (0,0) and
    (1,1) gives the correct AUROC under the assumption that the threshold can
    extend to ±∞ (standard convention).
    """
    fprs_ext = list(fprs) + [0.0, 1.0]
    tprs_ext = list(tprs) + [0.0, 1.0]
    idx = np.argsort(fprs_ext)
    f   = np.array(fprs_ext)[idx]
    t   = np.array(tprs_ext)[idx]
    return float(_integrate(t, f))


def write_roc_outputs(feat):
    # Per-timestep "at-risk" dataset: hours where patient was not yet on vasopressin
    fs = feat.sort_values(["stay_id", "time_hour"]).copy()
    fs["prev_vaso"] = fs.groupby("stay_id")["action_vaso"].shift(1).fillna(0)
    pred = fs[fs["prev_vaso"] == 0].copy()
    pred["outcome"] = pred["action_vaso"].astype(int)

    thresh_rows = []
    roc_rows    = []

    for col in ANALYSIS_FEATURES:
        if col not in pred.columns:
            continue
        is_bin = col in ANALYSIS_FEATURES_BIN

        tr      = pred[pred["split"] == "train"]
        tr_vals = pd.to_numeric(tr[col], errors="coerce").fillna(0).values
        tr_y    = tr["outcome"].values.astype(int)

        grid = _threshold_grid(tr_vals, is_bin)

        # Evaluate all grid points on train; determine direction
        gt_fprs, gt_tprs = [], []
        for tau in grid:
            m = _eval_thresh(tr_y, tr_vals, tau, "gt")
            gt_fprs.append(m["fpr"])
            gt_tprs.append(m["tpr"])

        auc_gt = _auc(gt_fprs, gt_tprs)
        direction = "gt" if auc_gt >= 0.5 else "lt"

        if direction == "lt":
            lt_fprs, lt_tprs = [], []
            for tau in grid:
                m = _eval_thresh(tr_y, tr_vals, tau, "lt")
                lt_fprs.append(m["fpr"])
                lt_tprs.append(m["tpr"])
            fprs_for_j, tprs_for_j = lt_fprs, lt_tprs
        else:
            fprs_for_j, tprs_for_j = gt_fprs, gt_tprs

        # Optimal threshold by Youden's J on train
        j_vals  = [t - f for t, f in zip(tprs_for_j, fprs_for_j)]
        opt_i   = int(np.argmax(j_vals))
        opt_tau = float(grid[opt_i])

        # Report on each split
        for split in ("train", "val", "test"):
            sp      = pred[pred["split"] == split]
            sv      = pd.to_numeric(sp[col], errors="coerce").fillna(0).values
            sy      = sp["outcome"].values.astype(int)
            n       = len(sy)
            n_pos   = int(sy.sum())
            n_neg   = n - n_pos

            if n < SUPPRESS_K or n_pos < SUPPRESS_K or n_neg < SUPPRESS_K:
                thresh_rows.append({
                    "feature": col, "split": split,
                    "optimal_threshold": _r(opt_tau), "direction": direction,
                    "auc": None, "sensitivity": None, "specificity": None,
                    "youden_j": None, "ppv": None, "npv": None,
                    "n": _n_str(n), "n_pos": _n_str(n_pos), "n_neg": _n_str(n_neg),
                })
                continue

            # ROC curve on grid
            sp_fprs, sp_tprs = [], []
            for tau in grid:
                m = _eval_thresh(sy, sv, tau, direction)
                sp_fprs.append(m["fpr"])
                sp_tprs.append(m["tpr"])
                roc_rows.append({
                    "feature": col, "split": split,
                    "threshold":   _r(float(tau)),
                    "tpr":         _r(m["tpr"]),
                    "fpr":         _r(m["fpr"]),
                    "sensitivity": _r(m["sens"]),
                    "specificity": _r(m["spec"]),
                })

            auc = _auc(sp_fprs, sp_tprs)

            # Apply train-derived threshold
            m_opt = _eval_thresh(sy, sv, opt_tau, direction)
            thresh_rows.append({
                "feature": col, "split": split,
                "optimal_threshold": _r(opt_tau), "direction": direction,
                "auc":         _r(auc),
                "sensitivity": _r(m_opt["sens"]),
                "specificity": _r(m_opt["spec"]),
                "youden_j":    _r(m_opt["youden_j"]),
                "ppv":         _r(m_opt["ppv"]),
                "npv":         _r(m_opt["npv"]),
                "n": n, "n_pos": n_pos, "n_neg": n_neg,
            })

    out1 = OUTPUT_DIR / "feature_thresholds_youden.csv"
    out2 = OUTPUT_DIR / "feature_roc_curves.csv"
    pd.DataFrame(thresh_rows).to_csv(out1, index=False)
    pd.DataFrame(roc_rows).to_csv(out2, index=False)
    print(f"  Wrote   {out1.name}")
    print(f"  Wrote   {out2.name}")


# ============================================================
# README
# ============================================================

README = f"""# Site Summary Outputs — Federated Privacy Manifest

Generated by: analysis/site_summary.py

## Privacy guarantees
- No patient-level records, identifiers, free text, or exact dates.
- Cell suppression: any statistic derived from fewer than {SUPPRESS_K} patients is
  reported as "<{SUPPRESS_K}"; all derived statistics for that cell are blank.
- Continuous summaries rounded to {ROUND_N} decimal places.
- ROC/threshold curves evaluated on a fixed {N_THRESH_GRID}-point grid derived from
  2-significant-figure rounded [p5, p95] of training data — NOT one per observation.
  AUC is computed with standard (0,0) and (1,1) corner anchors added to the grid-based
  curve, so it reflects the full AUROC rather than only the [p5, p95] excerpt.

## Analysis design
- Patient-level 70/15/15 train/val/test split, seed={RANDOM_SEED}.
- Outcome (analyses 4-5): per-timestep imminent vasopressin initiation.
  At-risk set = patient-hours where vasopressin was NOT active in previous hour.
  Outcome = action_vaso at current hour. Threshold selected on TRAIN only.
- Features (analyses 4-5): {", ".join(ANALYSIS_FEATURES)}.

## File schemas

### cohort_filter_counts.csv
Cohort inclusion flowchart. Columns: step, n_hospitalizations.

### split_counts.csv
Patients per split by ever-vasopressin group.
Columns: split, outcome_group (ever_vaso_yes/no/total), n_patients.

### baseline_table1.csv
Baseline characteristics stratified by eventual vasopressin (vaso/no_vaso) and overall.
One row per variable (or per level for categoricals).
Columns: variable, level, type, smd,
  {{group}}_n, {{group}}_n_missing, {{group}}_mean, {{group}}_sd,
  {{group}}_median, {{group}}_q25, {{group}}_q75, {{group}}_min, {{group}}_max, {{group}}_pct
where group ∈ {{vaso, no_vaso, overall}}.
Note: min/max are included per specification; consider suppressing in final sharing if group
sizes are near K.

### feature_at_initiation.csv
Feature values at the first vasopressin initiation timestep per initiating patient.
Columns: feature, n, n_missing, mean, sd, median, q25, q75, min, max.

### feature_thresholds_youden.csv
Per-feature (× split) threshold performance for imminent initiation.
Optimal threshold fixed from TRAIN; applied unchanged to val and test.
Columns: feature, split, optimal_threshold, direction (gt/lt),
  auc, sensitivity, specificity, youden_j, ppv, npv, n, n_pos, n_neg.

### feature_roc_curves.csv
Full ROC curve for coordinating site to replot. No patient-level data.
Columns: feature, split, threshold, tpr, fpr, sensitivity, specificity.
"""


def write_readme():
    out = OUTPUT_DIR / "README.md"
    out.write_text(README, encoding="utf-8")
    print(f"  Wrote   {out.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    global INPUT_DIR, OUTPUT_DIR

    ap = argparse.ArgumentParser(description="Federated-safe aggregate summary")
    ap.add_argument(
        "--dataset", choices=["mimic", "ucmc"], default="ucmc",
        help="Dataset to summarize (default: ucmc)"
    )
    args = ap.parse_args()

    INPUT_DIR  = BASE_DIR / "Data" / args.dataset.upper()
    OUTPUT_DIR = BASE_DIR / "output" / args.dataset.upper()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    print("[1/6] Filter counts")
    write_filter_counts()

    print("[2/6] Loading + splitting")
    coh, feat = load_and_split()
    tr, va, te = [(coh["split"] == s).sum() for s in ("train", "val", "test")]
    print(f"  {len(coh):,} patients  train={tr}  val={va}  test={te}")
    print(f"  Ever-vaso: {coh['ever_vaso'].sum():,}  "
          f"Never-vaso: {(coh['ever_vaso']==0).sum():,}")

    print("[3/6] Split counts")
    write_split_counts(coh)

    print("[4/6] Baseline table 1")
    write_baseline_table1(coh)

    print("[5/6] Feature at initiation")
    write_feature_at_initiation(feat)

    print("[6/6] ROC / threshold analysis")
    write_roc_outputs(feat)

    write_readme()

    print(f"\nDone. Outputs in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>6.1f} KB")


if __name__ == "__main__":
    main()
