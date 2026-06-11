# config/

Site-specific configuration for CLIF extraction.

## Setup

Copy `config.example.py` to `config.py` at the **repo root** and fill in your paths:

```bash
cp config/config.example.py config.py
```

Then edit `config.py`:

```python
# Root directory containing your CLIF 2.1.0 parquet files
CLIF_DIR = Path(r"C:\path\to\your\clif\2.1.0")

# Output directory for cohort and features parquet files
OUTPUT_DIR = Path(r"C:\path\to\your\project\Data\UCMC")
```

`config.py` is gitignored — it stays local and is never committed.

## Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLIF_DIR` | Yes | — | Root of CLIF 2.1.0 parquet tables |
| `OUTPUT_DIR` | Yes | — | Where cohort/features parquet files are written |
| `TIMEZONE` | No | `"UTC"` | Timezone for datetime parsing |
| `TRAJECTORY_HOURS` | No | `120` | Max trajectory length per patient |
| `NE_WINDOW_HOURS` | No | `24` | Hours from ICU admit for NE start |
| `MIN_NE_RECORDS` | No | `2` | Minimum NE administration records |
| `SOFA_THRESHOLD` | No | `2.0` | Minimum SOFA at sepsis onset |
| `LACTATE_THRESHOLD` | No | `2.0` | Minimum lactate (mmol/L) |
| `MAP_THRESHOLD` | No | `65.0` | MAP threshold (mmHg) |
