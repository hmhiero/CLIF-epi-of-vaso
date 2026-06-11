"""Extract MIMIC-IV septic shock cohort and hourly features.

Reads from the existing DuckDB built in rl-clinical-concordance.
Uses the same SQL templates as that project so the output schema
matches Data/UCMC/ exactly.

Outputs (in Data/MIMIC/):
  cohort.parquet          — cohort demographics and outcomes
  features.parquet        — hourly feature time series
  cohort_filter_counts.csv — patient counts at each filter step

Usage:
    python mimic_extract.py
    python mimic_extract.py --output-dir Data/MIMIC
"""

import argparse
from pathlib import Path

import duckdb
import polars as pl
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# DuckDB built by rl-clinical-concordance/0_MIMIC_RAW_ABLATION/setup_duckdb.py
DUCKDB_PATH = Path(
    r"C:\Users\hhieromnimon\Box\Research\rl-clinical-concordance\0_MIMIC_RAW_ABLATION\mimic4.duckdb"
)

# SQL templates shared with rl-clinical-concordance
SQL_DIR = Path(
    r"C:\Users\hhieromnimon\Box\Research\rl-clinical-concordance\0_MIMIC_RAW_ABLATION\sql"
)

# Default output location
DEFAULT_OUT = Path(__file__).parent.parent / "Data" / "MIMIC"

# Template variables (match OVISS defaults)
TEMPLATE_VARS = {
    "trajectory_hours": 120,
    "uo_type": "normalized",
    "fluid_categories": "bolus_colloid",
}


# ---------------------------------------------------------------------------
# SQL rendering
# ---------------------------------------------------------------------------

def render_sql(template_path: Path, **kwargs) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
    )
    return env.get_template(template_path.name).render(**kwargs)


# ---------------------------------------------------------------------------
# Filter counts (cohort inclusion flowchart)
# ---------------------------------------------------------------------------

def get_filter_counts(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Count patients at each inclusion step to match cohort_filter_counts.csv format."""
    rows = []

    n = conn.execute(
        "SELECT COUNT(*) FROM mimiciv_derived.icustay_detail WHERE first_icu_stay = true"
    ).fetchone()[0]
    rows.append({"step": "Total first ICU stays in MIMIC-IV", "n_hospitalizations": n})

    n = conn.execute("""
        SELECT COUNT(DISTINCT se.subject_id)
        FROM mimiciv_derived.sepsis3 se
        WHERE se.sofa_score >= 2 AND se.sepsis3 = true
    """).fetchone()[0]
    rows.append({"step": "Sepsis-3 criteria (SOFA >= 2, first per patient)", "n_hospitalizations": n})

    n = conn.execute("""
        SELECT COUNT(DISTINCT se.subject_id)
        FROM mimiciv_derived.sepsis3 se
        JOIN mimiciv_derived.icustay_detail id ON se.stay_id = id.stay_id
        WHERE se.sofa_score >= 2 AND se.sepsis3 = true
          AND id.first_icu_stay = true
          AND se.suspected_infection_time
              BETWEEN id.icu_intime - INTERVAL '24 hours'
                  AND id.icu_intime + INTERVAL '24 hours'
    """).fetchone()[0]
    rows.append({"step": "ICU admission (first ICU stay) within 24h of suspected infection",
                 "n_hospitalizations": n})

    n = conn.execute("""
        SELECT COUNT(DISTINCT se.subject_id)
        FROM mimiciv_derived.sepsis3 se
        JOIN mimiciv_derived.icustay_detail id ON se.stay_id = id.stay_id
        JOIN (
            SELECT ne.stay_id
            FROM mimiciv_derived.norepinephrine ne
            JOIN mimiciv_derived.icustay_detail id2 ON ne.stay_id = id2.stay_id
            WHERE ne.starttime >= id2.icu_intime
              AND ne.starttime <= id2.icu_intime + INTERVAL '24 hours'
            GROUP BY ne.stay_id
            HAVING COUNT(*) >= 2
        ) nec ON se.stay_id = nec.stay_id
        WHERE se.sofa_score >= 2 AND se.sepsis3 = true
          AND id.first_icu_stay = true
          AND se.suspected_infection_time
              BETWEEN id.icu_intime - INTERVAL '24 hours'
                  AND id.icu_intime + INTERVAL '24 hours'
    """).fetchone()[0]
    rows.append({"step": "Norepinephrine within 24h of ICU admit (>=2 records)",
                 "n_hospitalizations": n})

    # Final count from the already-built cohort temp table
    n = conn.execute("SELECT COUNT(*) FROM cohort_septic_shock").fetchone()[0]
    rows.append({"step": "Lactate > 2.0 mmol/L (final septic shock cohort)",
                 "n_hospitalizations": n})

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT),
                    help=f"Output directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB not found: {DUCKDB_PATH}\n"
            "Run setup_duckdb.py in rl-clinical-concordance/0_MIMIC_RAW_ABLATION first."
        )
    if not SQL_DIR.exists():
        raise FileNotFoundError(f"SQL template directory not found: {SQL_DIR}")

    print(f"DuckDB: {DUCKDB_PATH}")
    print(f"Output: {out}")
    print(f"Template vars: {TEMPLATE_VARS}")

    conn = duckdb.connect(str(DUCKDB_PATH))

    # Ensure charlson comorbidity table exists
    charlson_exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = 'mimiciv_derived' AND table_name = 'charlson'"
    ).fetchone()[0]
    if not charlson_exists:
        print("Building mimiciv_derived.charlson...")
        charlson_sql = (
            Path(r"C:\Users\hhieromnimon\Box\Research\rl-clinical-concordance\0_MIMIC_RAW_ABLATION")
            / "mimic-code" / "mimic-iv" / "concepts_duckdb" / "comorbidity" / "charlson.sql"
        )
        if charlson_sql.exists():
            conn.execute(charlson_sql.read_text())
        else:
            raise FileNotFoundError(f"charlson.sql not found: {charlson_sql}")

    # --- Cohort ---
    print("\n[1/2] Extracting cohort...")
    cohort_sql = render_sql(SQL_DIR / "01_cohort.sql.j2", **TEMPLATE_VARS).rstrip().rstrip(";")
    conn.execute("DROP TABLE IF EXISTS cohort_septic_shock")
    conn.execute(f"CREATE TEMP TABLE cohort_septic_shock AS ({cohort_sql})")

    cohort = pl.from_arrow(conn.execute("SELECT * FROM cohort_septic_shock").arrow())
    cohort.write_parquet(out / "cohort.parquet")
    print(f"  {len(cohort):,} patients | "
          f"mortality {cohort['hospital_death'].mean():.1%} | "
          f"median age {cohort['age'].median():.0f}")
    print(f"  Saved {out / 'cohort.parquet'}")

    # --- Filter counts ---
    print("  Building filter counts...")
    filter_rows = get_filter_counts(conn)
    import csv
    with open(out / "cohort_filter_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "n_hospitalizations"])
        writer.writeheader()
        writer.writerows(filter_rows)
    print(f"  Saved {out / 'cohort_filter_counts.csv'}")
    for r in filter_rows:
        print(f"    {r['step']:<60} n={r['n_hospitalizations']:,}")

    # --- Features ---
    print("\n[2/2] Extracting hourly features...")
    features_sql = render_sql(SQL_DIR / "02_features.sql.j2", **TEMPLATE_VARS)
    features = pl.from_arrow(conn.execute(features_sql).arrow())
    features.write_parquet(out / "features.parquet")
    print(f"  {len(features):,} rows for {features['stay_id'].n_unique():,} patients")
    print(f"  Saved {out / 'features.parquet'}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
