from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
DATASET_DIR = ROOT / "Dataset"
DATA_RAW = DATASET_DIR / "Tetuan_City_power_consumption.csv"

# Outputs
OUT = ROOT / "outputs"
FIGURES_DIR = OUT / "figures"
TABLES_DIR = OUT / "tables"
MODELS_DIR = OUT / "models"
PROCESSED_DIR = OUT / "processed"

# Notebook 07 historical high-demand quantile, fitted on designated training labels.
PEAK_QUANTILE = 0.95

# Proposed target-date splits; Notebook 04 defines daily batch issues and label availability.
TRAIN_END = "2017-08-31 23:59:59"
VAL_END   = "2017-10-31 23:59:59"
TEST_END  = "2017-12-30 23:50:00"   # last available timestamp; 31 Dec is absent from the public CSV

# Zone column names after cleaning
ZONES = ["zone1", "zone2", "zone3"]

# Data-quality note (see EDA weather notebook)
WIND_NOTE = (
    "wind_speed is clustered: 57.01% of readings < 0.1 and "
    "38.62% between 4 and 6 inclusive in the checked local file. "
    "These are descriptive ranges, not confirmed sensor faults. "
    "Raw demand correlations overlap strongly with calendar patterns."
)
