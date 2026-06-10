# config.example.py
#
# Copy this file to config.py and fill in your site's paths.
# clif_extract.py automatically loads config.py at startup.
#
# Only CLIF_DIR and OUTPUT_DIR need to change per site.
# All other values reflect the OVISS inclusion criteria and
# should remain at their defaults unless your protocol differs.

from pathlib import Path

# --- Required: set these for your site ---

# Root directory containing all CLIF 2.1.0 parquet files
CLIF_DIR = Path("/path/to/your/clif/2.1.0")
# e.g. Windows: Path(r"C:\data\clif\2.1.0")

# Directory where output parquet files will be written
# (created automatically if it does not exist)
OUTPUT_DIR = Path("/path/to/your/output/data_clif")
# e.g. Windows: Path(r"C:\projects\epi-of-vaso\data_clif")

# --- Optional: adjust only if your protocol differs ---

TIMEZONE = "UTC"          # Timezone for all datetime parsing
TRAJECTORY_HOURS = 120    # Max hours of trajectory per patient
NE_WINDOW_HOURS = 24      # NE must start within this many hours of ICU admit
MIN_NE_RECORDS = 2        # Min NE administration records required
SOFA_THRESHOLD = 2.0      # Min SOFA score at sepsis onset
LACTATE_THRESHOLD = 2.0   # Min lactate (mmol/L) within 24h of infection
MAP_THRESHOLD = 65.0      # MAP threshold (mmHg) for vasopressor indication

# Medication categories — must match your site's CLIF med_category values exactly
STEROID_CATEGORIES = [
    "hydrocortisone", "dexamethasone", "methylprednisolone",
    "fludrocortisone", "prednisolone", "prednisone",
]
VASOPRESSOR_CATEGORIES = [
    "norepinephrine", "epinephrine", "phenylephrine",
    "vasopressin", "dopamine", "angiotensin",
]
