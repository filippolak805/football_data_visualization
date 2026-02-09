from __future__ import annotations

from pathlib import Path
import pandas as pd

COUNTRY_ALIASES = {
    "England": "United Kingdom",
    "Scotland": "United Kingdom"
}  # mapping to Natural Earth map country names

DATASET_COUNTRY_ALIASES = {
    "United Kingdom": ["England", "Scotland"]
}  # mapping back to match country names

EXCLUDED_FROM_UI = {"India", "Australia", "Saudi Arabia", "World"}


def to_map_country(country: str) -> str:
    return COUNTRY_ALIASES.get(country, country)


def countries_with_data_from_comps_csv(comps_csv_path: str | Path) -> list[str]:
    comps_csv_path = Path(comps_csv_path)
    df = pd.read_csv(comps_csv_path)

    if "country" not in df.columns:
        raise ValueError(f"Expected 'country' column in {comps_csv_path}, got: {df.columns.tolist()}")

    countries = []
    for c in df["country"].dropna().astype(str).unique().tolist():
        if c in EXCLUDED_FROM_UI:
            continue
        countries.append(to_map_country(c))

    # unique + sorted
    return sorted(set(countries))
