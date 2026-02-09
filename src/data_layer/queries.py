from __future__ import annotations

from datetime import datetime
from src.data_layer.io import get_connection
from src.utils.time_axis import TimeAxis


def list_countries_from_matches(limit: int = 200) -> list[str]:
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT DISTINCT country
            FROM matches
            WHERE country IS NOT NULL AND country <> ''
            ORDER BY country
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def list_teams_for_comp(comp_id: int) -> list[tuple[int, str]]:
    con = get_connection()
    try:
        rows = con.execute(
            """
            WITH team_ids AS (
                SELECT home_team_id AS team_id FROM matches_base WHERE comp_id = ?
                UNION
                SELECT away_team_id AS team_id FROM matches_base WHERE comp_id = ?
            )
            SELECT t.team_id, t.name
            FROM team_ids x
            JOIN teams t ON t.team_id = x.team_id
            ORDER BY t.name
            """,
            [comp_id, comp_id],
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]
    finally:
        con.close()


def get_global_time_bounds():
    con = get_connection()
    try:
        row = con.execute(
            """
            SELECT
                MIN(match_datetime),
                MAX(match_datetime)
            FROM matches_base
            """
        ).fetchone()
        return row[0], row[1]
    finally:
        con.close()


def get_time_axis() -> TimeAxis:
    min_dt, max_dt = get_global_time_bounds()
    if min_dt is None or max_dt is None:
        raise RuntimeError("Global time bounds unavailable.")
    max_days = (max_dt - min_dt).days
    return TimeAxis(origin=min_dt, max_days=max_days)


def matches_per_month(countries: list[str] | None,
                      comp_id: int | None,
                      team_id: int | None,
                      start_dt: datetime,
                      end_dt: datetime):
    con = get_connection()
    try:
        sql = """
            SELECT
                date_trunc('month', match_datetime)::DATE AS month,
                COUNT(*)::INT AS n_matches
            FROM matches_base
            WHERE match_datetime >= ?
              AND match_datetime <= ?
        """
        params = [start_dt, end_dt]

        if countries:
            sql += " AND country IN (SELECT * FROM UNNEST(?))"
            params.append(countries)

        if comp_id is not None:
            sql += " AND comp_id = ?"
            params.append(int(comp_id))

        if team_id is not None:
            sql += " AND (home_team_id = ? OR away_team_id = ?)"
            params.extend([int(team_id), int(team_id)])

        sql += " GROUP BY 1 ORDER BY 1"

        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def heatmap_month_hourbin_avg_goals(comp_id: int | None,
                                    countries: list[str] | None,
                                    team_id: int | None = None):
    if comp_id is None:
        return []

    con = get_connection()
    try:
        sql = """
            SELECT
                (
                    CAST(strftime(match_datetime, '%Y') AS INTEGER) * 100
                    + (1 + (CAST(strftime(match_datetime, '%m') AS INTEGER) - 1) / 3)  -- 1...4
                ) AS trimes_key,
                CASE
                  WHEN extract('hour' FROM match_datetime) <= 13 THEN '<=13:59'
                  WHEN extract('hour' FROM match_datetime) BETWEEN 14 AND 16 THEN '14:00-16:59'
                  WHEN extract('hour' FROM match_datetime) BETWEEN 17 AND 19 THEN '17:00-19:59'
                  ELSE '20:00+'
                END
                 AS hour_bin,
                AVG(home_team_goals + away_team_goals)::DOUBLE AS avg_goals,
                COUNT(*)::INT AS n_matches
            FROM matches_base
            WHERE comp_id = ?
        """
        params = [int(comp_id)]

        if countries:
            sql += " AND country IN (SELECT * FROM UNNEST(?))"
            params.append(countries)

        if team_id is not None:
            sql += " AND (home_team_id = ? OR away_team_id = ?)"
            params.extend([int(team_id), int(team_id)])

        sql += " GROUP BY 1, 2 ORDER BY 1, 2"
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def parallel_coords_team_match(
        comp_id: int,
        team_ids: list[int] | None,
        start_dt,
        end_dt,
):
    con = get_connection()

    # Home team
    sql_home = """
    SELECT
        home_team_id AS team_id,
        1 AS is_home,

        CASE
            WHEN home_team_goals > away_team_goals THEN 2
            WHEN home_team_goals = away_team_goals THEN 1
            ELSE 0
        END AS result,

        home_team_total_shots AS total_shots,
        home_team_shots_on_target AS shots_on_target,
        home_team_ball_possession AS possession,
        home_team_passes_acc AS passes_acc

    FROM matches_base
    WHERE comp_id = ?
      AND match_datetime BETWEEN ? AND ?
    """

    params_home = [comp_id, start_dt, end_dt]

    if team_ids:
        sql_home += " AND home_team_id IN (SELECT * FROM UNNEST(?))"
        params_home.append(team_ids)

    # Away team
    sql_away = """
    SELECT
        away_team_id AS team_id,
        0 AS is_home,

        CASE
            WHEN away_team_goals > home_team_goals THEN 2
            WHEN away_team_goals = home_team_goals THEN 1
            ELSE 0
        END AS result,

        away_team_total_shots AS total_shots,
        away_team_shots_on_target AS shots_on_target,
        away_team_ball_possession AS possession,
        away_team_passes_acc AS passes_acc

    FROM matches_base
    WHERE comp_id = ?
      AND match_datetime BETWEEN ? AND ?
    """

    params_away = [comp_id, start_dt, end_dt]

    if team_ids:
        sql_away += " AND away_team_id IN (SELECT * FROM UNNEST(?))"
        params_away.append(team_ids)

    # Union
    sql = f"""
    {sql_home}
    UNION ALL
    {sql_away}
    """

    params = params_home + params_away

    return con.execute(sql, params).fetchdf()


def team_avg_stats_for_radar(comp_id: int, start_dt=None, end_dt=None):
    con = get_connection()
    try:
        params = [comp_id, start_dt, end_dt, comp_id, start_dt, end_dt] if start_dt and end_dt else [comp_id, comp_id]

        date_filter = "AND match_datetime BETWEEN ? AND ?" if start_dt and end_dt else ""

        sql = f"""
        WITH team_matches AS (
            SELECT
                home_team_id AS team_id,
                home_team_goals AS goals,
                home_team_total_shots AS shots,
                home_team_shots_on_target AS shots_on_goal,
                home_team_ball_possession AS possession,
                home_team_passes_acc AS pass_accuracy,
                home_team_corner_kicks AS corner_kicks
            FROM matches_base
            WHERE comp_id = ? {date_filter}

            UNION ALL

            SELECT
                away_team_id AS team_id,
                away_team_goals AS goals,
                away_team_total_shots AS shots,
                away_team_shots_on_target AS shots_on_goal,
                away_team_ball_possession AS possession,
                away_team_passes_acc AS pass_accuracy,
                away_team_corner_kicks AS corner_kicks
            FROM matches_base
            WHERE comp_id = ? {date_filter}
        )
        SELECT
            tm.team_id,
            t.name AS team_name,
            AVG(goals)::DOUBLE AS avg_goals,
            AVG(shots)::DOUBLE AS avg_shots,
            AVG(shots_on_goal)::DOUBLE AS avg_shots_on_goal,
            AVG(possession)::DOUBLE AS avg_possession,
            AVG(pass_accuracy)::DOUBLE AS avg_pass_accuracy,
            AVG(corner_kicks)::DOUBLE AS avg_corners
        FROM team_matches tm
        JOIN teams t ON t.team_id = tm.team_id
        GROUP BY tm.team_id, t.name
        ORDER BY t.name
        """
        return con.execute(sql, params).fetchdf()
    finally:
        con.close()


def goals_vs_xg_matches(
        comp_id: int,
        countries: list[str] | None,
        team_ids: list[int] | None,
        start_dt,
        end_dt,
):
    con = get_connection()
    try:
        sql = """
            SELECT
                (home_team_goals + away_team_goals) AS goals,
                total_xg
            FROM matches_base
            WHERE comp_id = ?
              AND match_datetime BETWEEN ? AND ?
              AND total_xg IS NOT NULL
        """
        params = [comp_id, start_dt, end_dt]

        if countries:
            sql += " AND country IN (SELECT * FROM UNNEST(?))"
            params.append(countries)

        if team_ids:
            sql += " AND (home_team_id IN (SELECT * FROM UNNEST(?)) OR away_team_id IN (SELECT * FROM UNNEST(?)))"
            params.extend([team_ids, team_ids])

        return con.execute(sql, params).fetchdf()
    finally:
        con.close()
