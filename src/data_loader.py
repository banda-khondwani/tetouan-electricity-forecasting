"""Load and standardise the Tetuan City power-consumption dataset."""
from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW

# Final, canonical column names (all lowercase, no spaces)
RENAME = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "zone1",
    "Zone 2 Power Consumption": "zone2",   # double space collapsed upstream
    "Zone 3 Power Consumption": "zone3",
}


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, normalise column names, parse datetimes, sort."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Raw file not found: {DATA_RAW}")

    df = pd.read_csv(DATA_RAW)

    # Collapse the double space in 'Zone 2  Power Consumption' and any stray whitespace
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )

    # Rename to canonical snake_case
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing from CSV: {missing}")

    df = df.rename(columns=RENAME)

    # Parse datetimes (file uses M/D/YYYY H:MM — dayfirst must be False)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=False)

    # Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    return df