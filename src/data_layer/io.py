from __future__ import annotations

import duckdb
from src.config import load_settings


def get_connection() -> duckdb.DuckDBPyConnection:
    s = load_settings()
    if not s.duckdb_path.exists():
        raise FileNotFoundError(
            f"DuckDB not found at {s.duckdb_path}. Run: python scripts/run_dev.py"
        )
    return duckdb.connect(str(s.duckdb_path), read_only=True)
