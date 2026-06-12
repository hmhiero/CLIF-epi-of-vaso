"""
site_threshold_sweep.py

Per-feature threshold sweep predicting clinician vasopressin action (action_vaso).
Run at each site — reads the site's intermediate parquet files from
output/patient_level_data_<SITE>/, writes aggregate CSV and plot outputs to
output/upload_to_box_<SITE>/ for sharing with the coordinating site.

Optionally compares against an RL policy if a trained FQI model is available.

For each feature, asks: "If we triggered vasopressin above/below a threshold on this
feature, how well would that match what clinicians actually do?"

Metrics per feature (vs clinician action):
  - AUROC, Kappa, Sensitivity, Specificity at optimal threshold
  - Patient-level: threshold on max(feature) → does patient ever receive vasopressin?
  - DM value gap (only when --model is supplied)

Additional analyses:
  - Feature-action density (requires --model for RL curve)
  - Decision tree fidelity curve (depth 1-6 vs agreement with clinician)

Outputs (in output/upload_to_box_<SITE>/):
  - threshold_sweep.png
  - decision_tree_fidelity.png
  - threshold_comparison_table.csv
  - patient_level_table.csv

Usage:
    uv run python code/site_threshold_sweep.py
    uv run python code/site_threshold_sweep.py --model path/to/fqi_model.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE     = Path(__file__).parent

# Load config from config/config.py (by file path, not import, so the
# config/ directory is not mistaken for an empty namespace package)
def _load_site_config():
    import importlib.util as _ilu
    cfg_path = BASE.parent / "config" / "config.py"
    if not cfg_path.exists():
        return None
    spec = _ilu.spec_from_file_location("clif_site_config", cfg_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cfg = _load_site_config()
if _cfg is None:
    raise SystemExit(
        "ERROR: config/config.py not found.\n"
        "Copy config/config.example.py to config/config.py and set SITE_NAME, CLIF_DIR, OUTPUT_ROOT."
    )
SITE_NAME   = getattr(_cfg, "SITE_NAME", "UCMC")
OUTPUT_ROOT = getattr(_cfg, "OUTPUT_ROOT", None)
if OUTPUT_ROOT is None:
    raise SystemExit("ERROR: OUTPUT_ROOT is not set in config/config.py.")
OUTPUT_ROOT = Path(OUTPUT_ROOT)

# Patient-level intermediate (PHI, local); shareable aggregates go to upload_to_box.
PATIENT_LEVEL_DIR = OUTPUT_ROOT / "output" / f"patient_level_data_{SITE_NAME}"
UPLOAD_DIR        = OUTPUT_ROOT / "output" / f"upload_to_box_{SITE_NAME}"

# (column_name, display_label, is_binary)
ANALYSIS_FEATURES = [
    ("time_hour",      "Hours since shock onset",    False),
    ("sofa",           "SOFA score",                 False),
    ("norepinephrine", "NE dose (µg/kg/min)",        False),
    ("nee",            "NEE (µg/kg/min)",            False),
    ("lactate",        "Serum lactate (mmol/L)",     False),
    ("urine_output",   "Urine output (mL/kg/h)",     False),
    ("creatinine",     "Creatinine (mg/dL)",         False),
    ("bun",            "BUN (mg/dL)",                False),
    ("mbp",            "MAP (mmHg)",                 False),
    ("fluids",         "IV fluids (mL/hr)",          False),
    ("ventil",         "Mech. ventilation",          True),
    ("rrt",            "RRT",                        True),
    ("steroid",        "Glucocorticoids",            True),
]

CONTINUOUS_FEATURES = [(c, l) for c, l, b in ANALYSIS_FEATURES if not b]
BINARY_FEATURES     = [(c, l) for c, l, b in ANALYSIS_FEATURES if b]

N_SWEEP  = 300   # threshold resolution
PCT_LO   = 5     # clip percentile for continuous sweep
PCT_HI   = 95
N_BINS   = 15    # bins for feature-action density plot


# ---------------------------------------------------------------------------
# Q-value helpers
# ---------------------------------------------------------------------------

def compute_q(model_data: dict, states: np.ndarray):
    """Return (q0, q1) arrays for a single-model FQI."""
    model = model_data["model"]
    s0 = np.hstack([states, np.zeros((len(states), 1))])
    s1 = np.hstack([states, np.ones((len(states), 1))])
    return model.predict(s0), model.predict(s1)


# ---------------------------------------------------------------------------
# Per-feature threshold sweep
# ---------------------------------------------------------------------------

def sweep_continuous(values: np.ndarray, rl_actions: np.ndarray):
    """Sweep threshold for a continuous feature in both directions.

    Returns dict with:
      thresholds, kappa_ge, kappa_le, sens_best, spec_best,
      opt_thresh, opt_dir, opt_kappa, auroc, agree_at_opt
    """
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 10:
        return None
    lo, hi = np.percentile(values[finite_mask], [PCT_LO, PCT_HI])
    if lo >= hi:
        return None

    # Use finite values for sweep; fill NaN with median for prediction
    fill_val = float(np.median(values[finite_mask]))
    values = np.where(finite_mask, values, fill_val)

    thresholds = np.linspace(lo, hi, N_SWEEP)
    kappa_ge = np.full(N_SWEEP, np.nan)
    kappa_le = np.full(N_SWEEP, np.nan)
    agree_ge = np.full(N_SWEEP, np.nan)
    agree_le = np.full(N_SWEEP, np.nan)

    for i, t in enumerate(thresholds):
        for arr, kappa_arr, agree_arr, pred in [
            (kappa_ge, kappa_ge, agree_ge, (values >= t).astype(int)),
            (kappa_le, kappa_le, agree_le, (values <= t).astype(int)),
        ]:
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            try:
                kappa_ge[i] = cohen_kappa_score(rl_actions, (values >= t).astype(int))
                agree_ge[i] = ((values >= t).astype(int) == rl_actions).mean()
                kappa_le[i] = cohen_kappa_score(rl_actions, (values <= t).astype(int))
                agree_le[i] = ((values <= t).astype(int) == rl_actions).mean()
            except Exception:
                pass

    # AUROC (direction-agnostic)
    try:
        auc = roc_auc_score(rl_actions, values)
        auroc = max(auc, 1.0 - auc)
    except Exception:
        auroc = np.nan

    # Best threshold across both directions
    best_i_ge = int(np.nanargmax(kappa_ge)) if not np.all(np.isnan(kappa_ge)) else -1
    best_i_le = int(np.nanargmax(kappa_le)) if not np.all(np.isnan(kappa_le)) else -1

    if best_i_ge == -1 and best_i_le == -1:
        return None

    k_ge_best = kappa_ge[best_i_ge] if best_i_ge != -1 else -np.inf
    k_le_best = kappa_le[best_i_le] if best_i_le != -1 else -np.inf

    if k_ge_best >= k_le_best:
        opt_dir   = ">="
        opt_i     = best_i_ge
        opt_kappa = k_ge_best
        opt_pred  = (values >= thresholds[opt_i]).astype(int)
    else:
        opt_dir   = "<="
        opt_i     = best_i_le
        opt_kappa = k_le_best
        opt_pred  = (values <= thresholds[opt_i]).astype(int)

    cm = confusion_matrix(rl_actions, opt_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return dict(
        thresholds=thresholds,
        kappa_ge=kappa_ge,
        kappa_le=kappa_le,
        agree_ge=agree_ge,
        agree_le=agree_le,
        opt_thresh=thresholds[opt_i],
        opt_dir=opt_dir,
        opt_kappa=opt_kappa,
        opt_pred=opt_pred,
        sens=sens,
        spec=spec,
        auroc=auroc,
        agree_at_opt=(opt_pred == rl_actions).mean(),
    )


def sweep_binary(values: np.ndarray, rl_actions: np.ndarray):
    """For binary features: evaluate both policy directions (1→act, 0→act)."""
    results = {}
    for direction, pred in [("1_acts", values.astype(int)),
                             ("0_acts", (1 - values).astype(int))]:
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        try:
            kappa = cohen_kappa_score(rl_actions, pred)
        except Exception:
            kappa = np.nan
        cm = confusion_matrix(rl_actions, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])
        results[direction] = dict(
            kappa=kappa,
            sens=tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            spec=tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            agree=(pred == rl_actions).mean(),
            pred=pred,
        )

    try:
        auc = roc_auc_score(rl_actions, values)
        auroc = max(auc, 1.0 - auc)
    except Exception:
        auroc = np.nan

    if not results:
        return None

    best_dir = max(results, key=lambda k: results[k]["kappa"] if not np.isnan(results[k]["kappa"]) else -np.inf)
    r = results[best_dir]
    return dict(
        opt_dir=best_dir,
        opt_kappa=r["kappa"],
        opt_pred=r["pred"],
        sens=r["sens"],
        spec=r["spec"],
        auroc=auroc,
        agree_at_opt=r["agree"],
    )


# ---------------------------------------------------------------------------
# DM policy value
# ---------------------------------------------------------------------------

def dm_value(q0: np.ndarray, q1: np.ndarray, actions: np.ndarray) -> float:
    """Mean Q(s, a) over observed timesteps — direct method policy value estimate."""
    return np.where(actions == 1, q1, q0).mean()


# ---------------------------------------------------------------------------
# Figure 1: Threshold sweep curves
# ---------------------------------------------------------------------------

def _draw_sweep_panel(ax, t, r, lbl, fontsize_title=8, fontsize_ax=8, fontsize_tick=7, fontsize_legend=6):
    ax.plot(t, r["kappa_ge"], color="#1f77b4", lw=1.5, alpha=0.9,  label="κ (≥ threshold)")
    ax.plot(t, r["kappa_le"], color="#d62728", lw=1.5, alpha=0.9,  ls="--", label="κ (≤ threshold)")
    ax.axvline(r["opt_thresh"], color="black", lw=1.2, ls=":", label=f"opt {r['opt_dir']} {r['opt_thresh']:.2g}")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(lbl, fontsize=fontsize_ax)
    ax.set_ylabel("Cohen's Kappa", fontsize=fontsize_ax)
    ax.set_title(f"{lbl}\nAUROC={r['auroc']:.3f}  κ*={r['opt_kappa']:.3f}  "
                 f"Agr={r['agree_at_opt']:.2%}", fontsize=fontsize_title)
    ax.tick_params(labelsize=fontsize_tick)
    ax.legend(fontsize=fontsize_legend, loc="lower right")


def plot_threshold_sweep(sweep_results: list, out_path: Path):
    cont_results = [(col, lbl, r) for col, lbl, r in sweep_results if r is not None and "thresholds" in r]
    n = len(cont_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows))
    axes = axes.ravel()

    for i, (col, lbl, r) in enumerate(cont_results):
        _draw_sweep_panel(axes[i], r["thresholds"], r, lbl)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Threshold sweep: Kappa vs threshold for each feature\n"
                 "(target = clinician vasopressin action; black line = optimal threshold)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Individual per-feature plots
    indiv_dir = out_path.parent / "threshold_sweep_individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    for col, lbl, r in cont_results:
        fig_i, ax_i = plt.subplots(figsize=(5.5, 3.8))
        _draw_sweep_panel(ax_i, r["thresholds"], r, lbl,
                          fontsize_title=10, fontsize_ax=10, fontsize_tick=9, fontsize_legend=8)
        fig_i.suptitle("Threshold sweep: Kappa vs threshold\n"
                        "(target = clinician vasopressin action; black line = optimal threshold)",
                        fontsize=10)
        fig_i.tight_layout()
        feat_path = indiv_dir / f"threshold_sweep_{col}.png"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        fig_i.savefig(feat_path, dpi=500, bbox_inches="tight")
        plt.close(fig_i)
        print(f"Saved: {feat_path}")


def write_threshold_sweep_data(sweep_results: list, out_path: Path):
    """Write per-feature threshold sweep curves to CSV for coordinating-site replotting.

    Columns: feature, threshold, kappa_ge, kappa_le, agree_ge, agree_le, opt_thresh, opt_dir
    One row per (feature, threshold grid point). Binary features are excluded
    (their sweep is non-continuous and captured in threshold_comparison_table.csv).
    """
    rows = []
    for col, lbl, r in sweep_results:
        if r is None or "thresholds" not in r:
            continue
        for t, kge, kle, age, ale in zip(
            r["thresholds"], r["kappa_ge"], r["kappa_le"],
            r["agree_ge"], r["agree_le"],
        ):
            rows.append({
                "feature":    col,
                "threshold":  round(float(t),   4),
                "kappa_ge":   round(float(kge), 4) if np.isfinite(kge) else None,
                "kappa_le":   round(float(kle), 4) if np.isfinite(kle) else None,
                "agree_ge":   round(float(age), 4) if np.isfinite(age) else None,
                "agree_le":   round(float(ale), 4) if np.isfinite(ale) else None,
                "opt_thresh": round(float(r["opt_thresh"]), 4),
                "opt_dir":    r["opt_dir"],
            })
    import pandas as _pd
    _pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Feature-action density
# ---------------------------------------------------------------------------

def plot_feature_action_density(df: pl.DataFrame, sweep_results: list,
                                 q0: np.ndarray, q1: np.ndarray, out_path: Path):
    all_features = [(col, lbl, r) for col, lbl, r in sweep_results]
    n = len(all_features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows))
    axes = axes.ravel()

    rl_actions  = df["rl_action"].to_numpy()
    clin_actions = df["action_vaso"].to_numpy()
    adv = q1 - q0  # Q-value advantage

    for i, (col, lbl, r) in enumerate(all_features):
        ax = axes[i]
        vals = df[col].to_numpy().astype(float)
        is_binary = col in {c for c, _, b in ANALYSIS_FEATURES if b}

        if is_binary:
            for v_idx, v_label in [(0, "Off (0)"), (1, "On (1)")]:
                mask = vals == v_idx
                if mask.sum() > 0:
                    bar_x = v_idx
                    ax.bar(bar_x - 0.18, rl_actions[mask].mean(), width=0.35,
                           color="steelblue", alpha=0.75, label="RL" if v_idx == 0 else None)
                    ax.bar(bar_x + 0.18, clin_actions[mask].mean(), width=0.35,
                           color="tomato", alpha=0.75, label="Clinician" if v_idx == 0 else None)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Off (0)", "On (1)"], fontsize=8)
            ax.set_xlim(-0.6, 1.6)
        else:
            # Quantile bins
            non_nan = vals[~np.isnan(vals)]
            if len(non_nan) < 10:
                axes[i].set_visible(False)
                continue
            bins = np.nanpercentile(vals, np.linspace(0, 100, N_BINS + 1))
            bins = np.unique(bins)
            if len(bins) < 3:
                axes[i].set_visible(False)
                continue

            bin_mids, rl_means, clin_means, adv_means = [], [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (vals >= lo) & (vals < hi)
                if mask.sum() < 3:
                    continue
                bin_mids.append((lo + hi) / 2)
                rl_means.append(rl_actions[mask].mean())
                clin_means.append(clin_actions[mask].mean())
                adv_means.append(adv[mask].mean())

            if not bin_mids:
                axes[i].set_visible(False)
                continue

            ax.plot(bin_mids, rl_means,   color="steelblue", lw=1.8, marker="o", ms=3, label="RL")
            ax.plot(bin_mids, clin_means, color="tomato",    lw=1.8, marker="s", ms=3, ls="--", label="Clinician")

            # Q-value advantage on secondary axis
            ax2 = ax.twinx()
            ax2.plot(bin_mids, adv_means, color="mediumpurple", lw=1.2, ls=":", label="B(s)=Q1-Q0")
            ax2.axhline(0, color="mediumpurple", lw=0.4, alpha=0.4)
            ax2.set_ylabel("Mean B(s)", color="mediumpurple", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="mediumpurple", labelsize=6)

            # Optimal threshold line
            if r is not None and "opt_thresh" in r:
                ax.axvline(r["opt_thresh"], color="gold", lw=1.2, ls=":",
                           label=f"opt thresh={r['opt_thresh']:.2g}")

            ax.set_xlabel(lbl, fontsize=8)

        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("P(action=1)", fontsize=8)
        if r is not None:
            ax.set_title(f"{lbl}\nκ*={r['opt_kappa']:.3f}  AUROC={r['auroc']:.3f}", fontsize=8)
        else:
            ax.set_title(lbl, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature-action density: P(vasopressin=1) by feature value\n"
                 "Blue=RL policy, Red=Clinician  |  Purple=Q-value advantage B(s)=Q1-Q0 (right axis)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Decision tree fidelity curve
# ---------------------------------------------------------------------------

def plot_decision_tree_fidelity(X: np.ndarray, y: np.ndarray,
                                 feat_names: list, sweep_results: list,
                                 out_path: Path, max_depth: int = 7):
    depths = list(range(1, max_depth + 1))
    fidelities = []
    trees = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X, y)
        fidelities.append((dt.predict(X) == y).mean())
        trees.append(dt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: agreement curve
    ax = axes[0]
    ax.plot(depths, fidelities, marker="o", color="steelblue", lw=2, ms=7)
    for d, f in zip(depths, fidelities):
        ax.annotate(f"{f:.3f}", (d, f), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.set_xlabel("Decision tree max_depth", fontsize=10)
    ax.set_ylabel("Agreement with clinician vasopressin action", fontsize=10)
    ax.set_title("Decision tree agreement vs complexity\n"
                 "(depth=1 = single-feature threshold rule; target = clinician action)", fontsize=10)
    ax.set_xticks(depths)
    ax.set_ylim(min(fidelities) - 0.05, 1.02)
    ax.grid(axis="y", alpha=0.3)

    # Right: AUROC per feature from threshold sweep (honest discriminability measure)
    # sklearn feature_importances_ is misleading for degenerate depth-1 trees since
    # it always sums to 1.0 for the chosen split feature regardless of actual predictiveness.
    ax2 = axes[1]
    auroc_by_feat = {}
    for col, lbl, r in sweep_results:
        if r is not None and "auroc" in r and not np.isnan(r["auroc"]):
            auroc_by_feat[col] = (lbl, r["auroc"])

    # Sort by AUROC ascending (for horizontal bar chart)
    sorted_feats = sorted(auroc_by_feat.items(), key=lambda x: x[1][1])
    labels  = [v[0] for _, v in sorted_feats]
    aurocs  = [v[1] for _, v in sorted_feats]
    colors  = ["tomato" if a < 0.6 else ("gold" if a < 0.7 else "steelblue") for a in aurocs]

    bars = ax2.barh(labels, aurocs, color=colors, alpha=0.80)
    ax2.axvline(0.5, color="gray", lw=1.0, ls="--")
    ax2.axvline(0.7, color="steelblue", lw=0.8, ls=":", alpha=0.6)
    for bar, a in zip(bars, aurocs):
        ax2.text(a + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{a:.3f}", va="center", fontsize=8)
    ax2.set_xlim(0.45, max(aurocs) + 0.08)
    ax2.set_xlabel("AUROC (threshold on feature predicting clinician vasopressin)", fontsize=10)
    ax2.set_title("Per-feature discriminability of clinician vasopressin action\n"
                  "(red < 0.6 = near-chance, gold 0.6-0.7, blue > 0.7)", fontsize=10)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print tree structures to console
    print("\nDepth-1 decision tree (best single-feature rule for clinician action):")
    print(export_text(trees[0], feature_names=feat_names))
    print("\nDepth-2 decision tree:")
    print(export_text(trees[1], feature_names=feat_names))

    return trees, fidelities


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _patient_level_eval(step_data: pl.DataFrame, sweep_results: list) -> pl.DataFrame:
    """Patient-level: threshold on max(feature) → predicts ever-vasopressin."""
    from sklearn.metrics import roc_auc_score, cohen_kappa_score

    rows = []
    for col, lbl, r in sweep_results:
        if r is None or col not in step_data.columns:
            continue
        is_binary = col in {c for c, _, b in ANALYSIS_FEATURES if b}

        pat = (
            step_data
            .group_by("stay_id")
            .agg([
                (pl.col("action_vaso") == 1).any().cast(pl.Int32).alias("ever_vaso"),
                pl.col(col).cast(pl.Float64).drop_nulls().max().alias("feat_max"),
            ])
            .drop_nulls(subset=["feat_max"])
        )
        ever_vaso = pat["ever_vaso"].to_numpy().astype(int)
        feat_max  = pat["feat_max"].to_numpy().astype(float)
        feat_max  = np.where(np.isfinite(feat_max), feat_max,
                             float(np.nanmedian(feat_max)))

        # Apply the same direction found in step-level sweep
        if is_binary:
            tau, direction = 0.5, "pos"
        else:
            tau       = r["opt_thresh"]
            direction = r["opt_dir"]  # ">=" maps to "pos", "<=" to "neg"
            direction = "pos" if ">=" in direction else "neg"

        pred = (feat_max > tau).astype(int) if direction == "pos" else (feat_max < tau).astype(int)

        try:
            kappa = float(cohen_kappa_score(ever_vaso, pred))
        except Exception:
            kappa = float("nan")
        agree = float((pred == ever_vaso).mean())
        try:
            auc   = roc_auc_score(ever_vaso, feat_max)
            auroc = max(auc, 1.0 - auc)
        except Exception:
            auroc = float("nan")

        rows.append({
            "feature":         col,
            "label":           lbl,
            "n_vaso_patients": int(ever_vaso.sum()),
            "n_novaso_patients": int((ever_vaso == 0).sum()),
            "patient_kappa":   round(kappa, 4) if np.isfinite(kappa) else None,
            "patient_auroc":   round(auroc, 4) if np.isfinite(auroc) else None,
            "patient_agree":   round(agree, 4),
        })

    return pl.DataFrame(rows).sort("patient_auroc", descending=True, nulls_last=True)


_SUPPRESS_K = 11  # suppress cells with fewer than this many patients


def _patient_level_confounders(
    step_data: pl.DataFrame, sweep_results: list, cohort: pl.DataFrame
) -> pl.DataFrame:
    """For each feature at its optimal threshold, compare clinical profile of
    threshold-positive vs threshold-negative patients.

    Cohort columns used: hospital_death, sepsis_onset_sofa, age, traj_hours, initial_lactate.
    Cells with n < _SUPPRESS_K are suppressed (stats set to None).
    """
    # Compute traj_hours if not present (MIMIC cohort stores trajectory_start/end)
    if "traj_hours" not in cohort.columns and "trajectory_start" in cohort.columns and "trajectory_end" in cohort.columns:
        cohort = cohort.with_columns(
            ((pl.col("trajectory_end") - pl.col("trajectory_start")).dt.total_seconds() / 3600)
            .alias("traj_hours")
        )
    coh_cols = ["stay_id", "hospital_death", "sepsis_onset_sofa", "age", "initial_lactate"]
    if "traj_hours" in cohort.columns:
        coh_cols.append("traj_hours")
    coh = cohort.select(coh_cols)

    rows = []
    for col, lbl, r in sweep_results:
        if r is None or col not in step_data.columns:
            continue
        is_binary = col in {c for c, _, b in ANALYSIS_FEATURES if b}

        pat = (
            step_data
            .group_by("stay_id")
            .agg([
                (pl.col("action_vaso") == 1).any().cast(pl.Int32).alias("ever_vaso"),
                pl.col(col).cast(pl.Float64).drop_nulls().max().alias("feat_max"),
            ])
            .drop_nulls(subset=["feat_max"])
            .join(coh, on="stay_id", how="left")
        )

        feat_max = pat["feat_max"].to_numpy().astype(float)
        feat_max = np.where(np.isfinite(feat_max), feat_max, float(np.nanmedian(feat_max)))

        if is_binary:
            tau, direction = 0.5, "pos"
        else:
            tau       = r["opt_thresh"]
            direction = "pos" if ">=" in r["opt_dir"] else "neg"

        pred = (feat_max > tau).astype(int) if direction == "pos" else (feat_max < tau).astype(int)
        pat  = pat.with_columns(pl.Series("threshold_group", pred))

        for grp_val, grp_label in [(1, "threshold_positive"), (0, "threshold_negative")]:
            g = pat.filter(pl.col("threshold_group") == grp_val)
            n = len(g)

            def _mean(series_name):
                if n < _SUPPRESS_K:
                    return None
                vals = g[series_name].drop_nulls().cast(pl.Float64)
                return round(float(vals.mean()), 3) if len(vals) > 0 else None

            rows.append({
                "feature":              col,
                "label":                lbl,
                "threshold_group":      grp_label,
                "n":                    n if n >= _SUPPRESS_K else f"<{_SUPPRESS_K}",
                "ever_vaso_pct":        round(float(g["ever_vaso"].mean()) * 100, 1) if n >= _SUPPRESS_K else None,
                "hospital_mortality_pct": round(float(g["hospital_death"].drop_nulls().cast(pl.Float64).mean()) * 100, 1) if n >= _SUPPRESS_K else None,
                "mean_sofa":            _mean("sepsis_onset_sofa"),
                "mean_age":             _mean("age"),
                "mean_traj_hours":      _mean("traj_hours") if "traj_hours" in g.columns else None,
                "mean_initial_lactate": _mean("initial_lactate"),
            })

    return pl.DataFrame(rows)


def build_summary_table(sweep_results: list, q0: np.ndarray, q1: np.ndarray,
                         rl_actions: np.ndarray) -> pl.DataFrame:
    rl_dm = dm_value(q0, q1, np.argmax(np.stack([q0, q1], axis=1), axis=1))

    rows = []
    for col, lbl, r in sweep_results:
        if r is None:
            continue
        is_binary = col in {c for c, _, b in ANALYSIS_FEATURES if b}

        if is_binary:
            thresh_str = r["opt_dir"]  # "1_acts" or "0_acts"
        else:
            thresh_str = f"{r['opt_dir']} {r['opt_thresh']:.3g}"

        thresh_dm = dm_value(q0, q1, r["opt_pred"])
        dm_gap_pct = (thresh_dm - rl_dm) / abs(rl_dm) * 100 if rl_dm != 0 else np.nan

        rows.append({
            "feature":      col,
            "label":        lbl,
            "opt_threshold":thresh_str,
            "agreement":    round(r["agree_at_opt"], 4),
            "kappa":        round(r["opt_kappa"], 4) if not np.isnan(r["opt_kappa"]) else None,
            "auroc":        round(r["auroc"], 4)      if not np.isnan(r["auroc"])     else None,
            "sensitivity":  round(r["sens"], 4)       if not np.isnan(r["sens"])      else None,
            "specificity":  round(r["spec"], 4)       if not np.isnan(r["spec"])      else None,
            "dm_value":     round(float(thresh_dm), 4),
            "dm_gap_pct":   round(float(dm_gap_pct), 2),
        })

    df = pl.DataFrame(rows).sort("kappa", descending=True, nulls_last=True)
    return df, rl_dm


def print_summary(df: pl.DataFrame, rl_dm: float):
    print(f"\n{'='*95}")
    print(f"  RL baseline DM value: {rl_dm:.4f}  (kappa/auroc/sens/spec are vs clinician action)")
    print(f"{'='*95}")
    header = f"{'Feature':<20} {'Threshold':>14} {'Agree':>7} {'Kappa':>7} "
    header += f"{'AUROC':>7} {'Sens':>7} {'Spec':>7} {'DM val':>10} {'DM gap%':>9}"
    print(header)
    print("-" * 95)
    for row in df.iter_rows(named=True):
        print(
            f"{row['feature']:<20} {row['opt_threshold']:>14} "
            f"{row['agreement']:>7.3f} {(row['kappa'] or 0):>7.3f} "
            f"{(row['auroc'] or 0):>7.3f} {(row['sensitivity'] or 0):>7.3f} "
            f"{(row['specificity'] or 0):>7.3f} {row['dm_value']:>10.4f} "
            f"{row['dm_gap_pct']:>+9.2f}%"
        )
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="Path to fqi_model.pkl for RL comparison (optional)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: output/upload_to_box_<SITE_NAME>/)")
    args = ap.parse_args()

    feat_path = PATIENT_LEVEL_DIR / "features.parquet"
    coh_path  = PATIENT_LEVEL_DIR / "cohort.parquet"

    out_dir = Path(args.out_dir) if args.out_dir else UPLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found.\nRun code/clif_extract.py first."
        )

    # Load features and cohort
    print(f"Site:       {SITE_NAME}")
    cohort    = pl.read_parquet(coh_path)
    step_data = (
        pl.read_parquet(feat_path)
        .join(cohort.select(["stay_id", "hospital_death"]), on="stay_id", how="left")
    )

    # Exclude patients on vasopressin at or before t=0
    vaso_at_t0_ids = (
        step_data.filter((pl.col("time_hour") <= 0) & (pl.col("action_vaso") == 1))
        ["stay_id"].unique()
    )
    if len(vaso_at_t0_ids) > 0:
        print(f"Excluding:  {len(vaso_at_t0_ids):,} patients with action_vaso=1 at t<=0")
        step_data = step_data.filter(~pl.col("stay_id").is_in(vaso_at_t0_ids))
        cohort    = cohort.filter(~pl.col("stay_id").is_in(vaso_at_t0_ids))

    # Forward-fill (LOCF) continuous features per patient before any analysis
    cont_cols = [
        col for col, _, is_bin in ANALYSIS_FEATURES
        if not is_bin and col in step_data.columns
    ]
    step_data = step_data.sort(["stay_id", "time_hour"]).with_columns([
        pl.col(c).forward_fill().over("stay_id") for c in cont_cols
    ])

    print(f"Patients:   {step_data['stay_id'].n_unique():,}")
    print(f"Steps:      {len(step_data):,}")

    clin_actions = step_data["action_vaso"].to_numpy().astype(int)
    print(f"Clin P(action=1): {clin_actions.mean():.3f}")

    # Optional: load RL model for Q-value / DM analyses
    model_data, q0, q1, rl_actions = None, None, None, None
    model_path = Path(args.model) if args.model else None
    if model_path and model_path.exists():
        model_data = pickle.load(open(model_path, "rb"))
        feats = model_data["features"]
        available = [f for f in feats if f in step_data.columns]
        missing = [f for f in feats if f not in step_data.columns]
        if missing:
            print(f"  [warn] model features not in data: {missing} — padding with 0")
        S = step_data.select(available).fill_null(0.0).fill_nan(0.0).to_numpy().astype(float)
        q0, q1 = compute_q(model_data, S)
        rl_actions = (q1 > q0).astype(int)
        print(f"RL model:   {model_path.name}")
        print(f"RL P(action=1): {rl_actions.mean():.3f}")
        print(f"Concordance:    {(rl_actions == clin_actions).mean():.3f}")
    elif args.model:
        print(f"  [warn] model not found at {args.model} — skipping RL analyses")

    # All state features available in this dataset (excluding outcome columns)
    EXCLUDE = {"stay_id", "time_hour", "hospital_death", "death",
               "norepi_explicitly_stopped", "vaso_explicitly_stopped",
               "ne_mar_action", "vaso_mar_action", "action_vaso", "vaso_dose"}
    all_feat_cols = [c for c in step_data.columns if c not in EXCLUDE]

    # Per-feature threshold sweep targeting CLINICIAN action
    print("\nRunning threshold sweeps (target: clinician vasopressin action)...")
    sweep_results = []
    for col, lbl, is_binary in ANALYSIS_FEATURES:
        if col not in step_data.columns:
            print(f"  SKIP {col}: not in data")
            continue
        vals = step_data[col].to_numpy().astype(float)
        if is_binary:
            r = sweep_binary(vals, clin_actions)
        else:
            r = sweep_continuous(vals, clin_actions)
        sweep_results.append((col, lbl, r))
        if r is not None:
            print(f"  {col:<18}: AUROC={r['auroc']:.3f}  k*={r['opt_kappa']:.3f}  "
                  f"agree={r['agree_at_opt']:.3f}")

    # Decision tree predicting clinician action (depth 1-7)
    print("\nFitting decision trees targeting clinician vasopressin action (depth 1-7)...")
    tree_feats = [col for col, _, _ in ANALYSIS_FEATURES if col in step_data.columns]
    X_tree = step_data.select(tree_feats).fill_null(0.0).fill_nan(0.0).to_numpy().astype(float)
    trees, fidelities = plot_decision_tree_fidelity(
        X_tree, clin_actions, tree_feats, sweep_results,
        out_dir / "plots" / "decision_tree_fidelity.png",
    )

    # Step-level summary table
    if q0 is not None:
        table, rl_dm = build_summary_table(sweep_results, q0, q1, rl_actions)
        print_summary(table, rl_dm)
    else:
        # Build table without DM columns
        rows = []
        for col, lbl, r in sweep_results:
            if r is None:
                continue
            is_binary = col in {c for c, _, b in ANALYSIS_FEATURES if b}
            thresh_str = (r["opt_dir"] if is_binary
                          else f"{r['opt_dir']} {r['opt_thresh']:.3g}")
            rows.append({
                "feature": col, "label": lbl,
                "opt_threshold": thresh_str,
                "agreement": round(r["agree_at_opt"], 4),
                "kappa":     round(r["opt_kappa"], 4) if not np.isnan(r["opt_kappa"]) else None,
                "auroc":     round(r["auroc"], 4)     if not np.isnan(r["auroc"])     else None,
                "sensitivity": round(r["sens"], 4)    if not np.isnan(r["sens"])      else None,
                "specificity": round(r["spec"], 4)    if not np.isnan(r["spec"])      else None,
            })
        table = pl.DataFrame(rows).sort("kappa", descending=True, nulls_last=True)
        rl_dm = None

    table.write_csv(out_dir / "threshold_comparison_table.csv")
    print(f"\nSaved: {out_dir / 'threshold_comparison_table.csv'}")

    # Patient-level analysis: threshold on max(feature) → ever-vasopressin
    print("\nComputing patient-level metrics (threshold on max feature -> ever-vasopressin)...")
    pat_table = _patient_level_eval(step_data, sweep_results)
    pat_table.to_pandas().to_csv(out_dir / "patient_level_table.csv", index=False)
    print(f"Saved: {out_dir / 'patient_level_table.csv'}")

    print("Computing patient-level confounders by threshold group...")
    conf_table = _patient_level_confounders(step_data, sweep_results, cohort)
    conf_table.to_pandas().to_csv(out_dir / "patient_level_confounders.csv", index=False)
    print(f"Saved: {out_dir / 'patient_level_confounders.csv'}")
    print(f"\n{'Feature':<20} {'Pat κ':>8} {'Pat AUROC':>10} {'Pat Agree':>10} {'n_vaso':>8}")
    print("-" * 60)
    for row in pat_table.iter_rows(named=True):
        print(f"  {row['feature']:<18} {(row['patient_kappa'] or 0):>8.3f} "
              f"{(row['patient_auroc'] or 0):>10.3f} {row['patient_agree']:>10.3f} "
              f"{row['n_vaso_patients']:>8}")

    # Sweep data CSV (for coordinating-site replotting)
    write_threshold_sweep_data(sweep_results, out_dir / "threshold_sweep_data.csv")

    # Figures
    print("\nGenerating figures...")
    plots_dir = out_dir / "plots"
    plot_threshold_sweep(sweep_results, plots_dir / "threshold_sweep.png")
    if q0 is not None and rl_actions is not None:
        plot_feature_action_density(step_data, sweep_results, q0, q1,
                                     plots_dir / "feature_action_density.png")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
