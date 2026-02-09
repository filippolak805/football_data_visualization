from __future__ import annotations

import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402


MAPPINGS_DIR = ROOT / "data" / "mappings"
TEAMS_CSV = MAPPINGS_DIR / "all-teams_25-10-16.csv"
COMPS_CSV = MAPPINGS_DIR / "all-comps_25-10-16.csv"


def main() -> None:
    s = load_settings()

    if not s.matches_csv_path.exists():
        raise FileNotFoundError(f"Matches CSV not found: {s.matches_csv_path}")

    if not TEAMS_CSV.exists():
        raise FileNotFoundError(f"Teams mapping not found: {TEAMS_CSV}")

    if not COMPS_CSV.exists():
        raise FileNotFoundError(f"Competitions mapping not found: {COMPS_CSV}")

    s.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    if s.duckdb_path.exists() and s.duckdb_path.stat().st_size > 0:
        print(f"[duckdb] OK: DuckDB already exists at {s.duckdb_path}")
        return

    print(f"[duckdb] Building DuckDB at {s.duckdb_path} ...")

    con = duckdb.connect(str(s.duckdb_path))
    try:
        # Performance pragmas (safe defaults for local dev)
        con.execute("PRAGMA threads=4;")
        con.execute("PRAGMA enable_progress_bar=true;")

        # Load mappings (CSV auto-detect; header assumed)
        con.execute("DROP TABLE IF EXISTS teams;")
        con.execute(
            f"""
            CREATE TABLE teams AS
            SELECT *
            FROM read_csv_auto('{TEAMS_CSV.as_posix()}', header=true);
            """
        )

        con.execute("DROP TABLE IF EXISTS comps;")
        con.execute(
            f"""
            CREATE TABLE comps AS
            SELECT *
            FROM read_csv_auto('{COMPS_CSV.as_posix()}', header=true);
            """
        )

        # Load matches
        dt_col = s.match_datetime_column

        con.execute("DROP TABLE IF EXISTS matches;")
        con.execute(
            f"""
            CREATE TABLE matches AS
            SELECT *
            FROM read_csv_auto(
                '{s.matches_csv_path.as_posix()}',
                header=true,
                sample_size=200000
            );
            """
        )

        # Drop old views if rebuilding
        con.execute("DROP VIEW IF EXISTS matches_base;")
        con.execute("DROP VIEW IF EXISTS matches_named;")

        # Canonical base view: parse datetime ONCE
        con.execute(
            f"""
            CREATE VIEW matches_base AS
            SELECT
                *,
                CAST({dt_col} AS TIMESTAMP) AS match_datetime
            FROM matches
            WHERE {dt_col} IS NOT NULL
            """
        )

        # Human-readable names on top of canonical base
        con.execute(
            """
            CREATE VIEW matches_named AS
            SELECT
                mb.*,
                ht.name AS home_team_name,
                awt.name AS away_team_name,
                c.name AS comp_name,
                c.country AS comp_country
            FROM matches_base mb
            LEFT JOIN teams ht ON ht.team_id = mb.home_team_id
            LEFT JOIN teams awt ON awt.team_id = mb.away_team_id
            LEFT JOIN comps c ON c.comp_id = mb.comp_id
            """
        )

        print("[duckdb] Done.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
