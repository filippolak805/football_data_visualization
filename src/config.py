# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    matches_csv_path: Path
    duckdb_path: Path
    data_download_url: str
    data_is_zip: bool
    match_datetime_column: str


def load_settings() -> Settings:
    load_dotenv()

    matches_csv_path = Path(os.getenv("MATCHES_CSV_PATH", "data/external/matches.csv"))
    duckdb_path = Path(os.getenv("DUCKDB_PATH", "data/cache/football.duckdb"))
    data_download_url = os.getenv("DATA_DOWNLOAD_URL", "").strip()
    data_is_zip = os.getenv("DATA_IS_ZIP", "0").strip() in {"1", "true", "True", "yes", "YES"}

    match_datetime_column = os.getenv("MATCH_DATETIME_COLUMN", "date")

    return Settings(
        matches_csv_path=matches_csv_path,
        duckdb_path=duckdb_path,
        data_download_url=data_download_url,
        data_is_zip=data_is_zip,
        match_datetime_column=match_datetime_column,
    )
