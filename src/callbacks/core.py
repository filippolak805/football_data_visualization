from __future__ import annotations

from dash import Dash, Input, Output, State, no_update, html, ALL, ctx
from datetime import datetime
from pathlib import Path
import plotly.express as px

from src.data_layer.io import get_connection
from src.data_layer.queries import list_teams_for_comp, heatmap_month_hourbin_avg_goals, parallel_coords_team_match, \
    team_avg_stats_for_radar, goals_vs_xg_matches
from src.layout.europe_map import get_europe_country_names
from src.utils.country_normalization import DATASET_COUNTRY_ALIASES, countries_with_data_from_comps_csv
from src.utils.logos import find_logo_src
from src.utils.time_axis import TimeAxis

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COMPS_CSV_PATH = Path("data/mappings/all-comps_25-10-16.csv")

# Color palette
COLOR_WIN = "#59A14F"
COLOR_DRAW = "#4C78A8"
COLOR_LOSS = "#E15759"

RADAR_COLORS = [COLOR_DRAW, "#F28E2B", COLOR_WIN]  # blue, orange, green
RESULT_COLORS = {2: COLOR_WIN, 1: COLOR_DRAW, 0: COLOR_LOSS}

PLOT_FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
PLOT_FONT_COLOR = "#263238"
PLOT_TITLE_SIZE = 16
PLOT_BASE_SIZE = 13
PLOT_BG_TRANSPARENT = "rgba(0,0,0,0)"


def apply_fig_theme(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    if title is not None:
        if "<br" in title:
            first, rest = title.split("<br", 1)
            title_text = f"<b>{first}</b><br{rest}"
        else:
            title_text = f"<b>{title}</b>"

        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.02,
                xanchor="left",
                font=dict(size=PLOT_TITLE_SIZE, color=PLOT_FONT_COLOR, family=PLOT_FONT_FAMILY),
            )
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_FONT_FAMILY, size=PLOT_BASE_SIZE, color=PLOT_FONT_COLOR),
        paper_bgcolor=PLOT_BG_TRANSPARENT,
        plot_bgcolor=PLOT_BG_TRANSPARENT,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def to_dataset_countries(ui_country: str | None) -> list[str] | None:
    if not ui_country:
        return None
    return DATASET_COUNTRY_ALIASES.get(ui_country, [ui_country])


def make_parallel_coords(df):
    if df.empty:
        fig = go.Figure()
        return apply_fig_theme(fig, "No data for current selection")

    # Sort rows so wins are drawn first (top), losses last
    df = df.sort_values("result", ascending=False)

    fig = px.parallel_coordinates(
        df,
        dimensions=[
            "result",
            "total_shots",
            "shots_on_target",
            "possession",
            "passes_acc",
        ],
        color="result",
        color_continuous_scale=[
            (0.0, COLOR_LOSS),
            (0.5, COLOR_DRAW),
            (1.0, COLOR_WIN),
        ],
        labels={
            "result": "Match result",
            "total_shots": "Total shots",
            "shots_on_target": "Shots on target",
            "possession": "Ball possession (%)",
            "passes_acc": "Pass accuracy (%)",
            "result": "Result",
        },
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=[0, 1, 2],
            ticktext=["Loss", "Draw", "Win"],
            title="Match result",
        ),
    )

    fig = apply_fig_theme(fig, "Effect of stats on match outcome")
    fig.update_layout(
        margin=dict(l=50, r=30, t=85, b=50),
        title=dict(y=0.98, pad=dict(b=8)),
    )
    return fig


def compute_win_rate(comp_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    # Get all match results for this competition + time window
    df = parallel_coords_team_match(
        comp_id=comp_id,
        team_ids=None,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    if df.empty:
        return pd.DataFrame()

    # Calculate win rate per team
    win_rate_df = (
        df.groupby("team_id")["result"]
        .apply(lambda x: (x == 2).sum() / len(x))
        .reset_index(name="win_rate")
    )

    # Map team_id -> team_name using teams table
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT team_id, name
            FROM teams
            WHERE team_id IN (SELECT * FROM UNNEST(?))
            """,
            [win_rate_df["team_id"].tolist()],
        ).fetchall()
        name_by_id = {tid: name for tid, name in rows}
    finally:
        con.close()

    win_rate_df["team_name"] = win_rate_df["team_id"].map(name_by_id)
    return win_rate_df


def build_single_team_radar(row, labels, metrics, df_full, color="#1f77b4"):
    # Normalize each metric across all 3 teams
    r = []
    hovertext = []
    for m, label in zip(metrics, labels):
        min_val = df_full[m].min()
        max_val = df_full[m].max()
        if max_val != min_val:
            norm_val = (row[m] - min_val) / (max_val - min_val)
        else:
            norm_val = 1.0
        r.append(norm_val)
        hovertext.append(f"{label}: {row[m]:.1f}")  # show raw value in hover

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=labels,
            fill="toself",
            name=row["team_name"],
            hoverinfo="text+name",
            text=hovertext,
            line=dict(color=color, width=2),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=35),
    )

    return apply_fig_theme(fig, str(row["team_name"]), height=320)


def make_goals_vs_xg_violin(df: pd.DataFrame, violin_color="#1f77b4", line_color="gray"):
    if df.empty:
        fig = go.Figure()
        return apply_fig_theme(fig, "No data for current selection", height=360)

    # Filter invalid values
    df = df[(df["goals"] >= 0) & (df["total_xg"] >= 0)]
    if df.empty:
        fig = go.Figure()
        return apply_fig_theme(fig, "No valid data for current selection", height=360)

    x_max = float(df["total_xg"].max())
    y_min = int(df["goals"].min())
    y_max = int(df["goals"].max())
    max_val = max(x_max, float(y_max))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, max_val + 0.5],
            y=[0, max_val + 0.5],
            mode="lines",
            line=dict(color=line_color, dash="dash", width=2),
            name="xG=goals",
            hoverinfo="skip",
            showlegend=False
        )
    )

    fig.add_annotation(
        x=max_val * 0.85,
        y=max_val * 0.85,
        text="xG = goals",
        showarrow=False,
        font=dict(size=12, color="gray"),
        bgcolor="rgba(255,255,255,0.6)",
    )

    # xG spread stats
    xg = df["total_xg"].astype(float)
    stats_text = (
        f"xG min={xg.min():.2f}, median={xg.median():.2f}, "
        f"max={xg.max():.2f} (n={len(df)})"
    )

    fig.add_annotation(
        x=0.05, y=1.07, xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="#6c757d"),
        align="left",
    )

    # Scatter layer
    fig.add_trace(
        go.Scattergl(
            x=df["total_xg"],
            y=df["goals"],
            mode="markers",
            marker=dict(size=3, opacity=0.12, color="#444"),
            hovertemplate="xG=%{x:.2f}<br>goals=%{y}<extra></extra>",
            showlegend=False,
        )
    )

    # Horizontal violin
    fig.add_trace(
        go.Violin(
            x=df["total_xg"],
            y=df["goals"],
            orientation="h",
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color=violin_color,
            fillcolor=violin_color,
            opacity=0.6,
            width=0.8,
            spanmode="hard",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Histogram of goal counts
    fig.add_trace(
        go.Histogram(
            y=df["goals"],
            orientation="h",
            marker=dict(color="#8FA3B2"),
            opacity=0.45,
            showlegend=False,
            xaxis="x2",
            texttemplate="%{x}",
            textposition="outside",
        )
    )

    # Axes
    fig.update_xaxes(
        title="Total xG (home + away)",
        range=[0, x_max + 0.3],
        dtick=1,
        rangemode="nonnegative"
    )

    fig.update_yaxes(
        title="Total goals (home + away)",
        tickmode="linear",
        tick0=y_min,
        dtick=1,
        range=[y_min - 0.5, y_max + 0.5],
        scaleanchor="x",
        scaleratio=1,
    )

    # Secondary x-axis for histogram
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            range=[0, 5 * df["goals"].value_counts().max()],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        )
    )

    fig.update_layout(bargap=0.15, showlegend=False)
    return apply_fig_theme(fig, "Actual goals vs expected goals (xG)", height=360)


def make_avg_goals_bar(df_avg_goals, highlight_ids_ordered: list[int] | None = None):
    """
    df_avg_goals: DataFrame with ["team_id", "team_name", "avg_goals"]
    highlight_ids_ordered: list of top 3 team_ids sorted by win rate (highest first)
    """
    if df_avg_goals.empty:
        fig = go.Figure()
        return apply_fig_theme(fig, "No data for current selection")

    # Sort bars by avg goals (DESC) so the chart is readable and comparable
    df_avg_goals = (
        df_avg_goals
        .assign(avg_goals=df_avg_goals["avg_goals"].astype(float))
        .sort_values(["avg_goals", "team_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    highlight_ids_ordered = highlight_ids_ordered or []

    # Map team_id -> color (highest win rate = blue, etc.)
    id_to_color = {tid: RADAR_COLORS[i] for i, tid in enumerate(highlight_ids_ordered) if i < len(RADAR_COLORS)}
    colors = [id_to_color.get(tid, "#d3d3d3") for tid in df_avg_goals["team_id"]]

    fig = go.Figure(
        go.Bar(
            x=df_avg_goals["team_name"],
            y=df_avg_goals["avg_goals"],
            marker_color=colors,
            text=df_avg_goals["avg_goals"].round(2),
            textposition="auto",
        )
    )

    fig.update_layout(
        xaxis_title="Team",
        yaxis_title="Avg goals per match",
        margin=dict(l=50, r=30, t=60, b=90),

        # Force x-axis category order to match the sorted dataframe order
        xaxis=dict(
            categoryorder="total descending",
        ),
    )

    return apply_fig_theme(fig, "Average goals per match")


def register_callbacks(app: Dash) -> None:
    def _axis_from_store(data) -> TimeAxis | None:
        if not data:
            return None
        return TimeAxis(
            origin=datetime.fromisoformat(data["origin"]),
            max_days=int(data["max_days"]),
        )

    @app.callback(
        Output("comp-dd", "options"),
        Input("country-dd", "value"),
        State("allowed-comp-ids", "data"),
    )
    def update_competitions(country: str | None, allowed_ids: list[int] | None):
        if not country or not allowed_ids:
            return []

        con = get_connection()
        try:
            # Clicking "United Kingdom" matches "England" and "Scotland" etc.
            countries = DATASET_COUNTRY_ALIASES.get(country, [country])

            rows = con.execute(
                """
                SELECT DISTINCT comp_id
                FROM matches_base
                WHERE country IN (SELECT * FROM UNNEST(?))
                  AND comp_id IN (SELECT * FROM UNNEST(?))
                ORDER BY comp_id
                """,
                [countries, allowed_ids],
            ).fetchall()
            comp_ids = [r[0] for r in rows]

            # Turn IDs into human labels using comps table (if available)
            # If mapping missing for some id, fallback to "comp_id=<id>"
            mapped = con.execute(
                """
                SELECT comp_id, name
                FROM comps
                WHERE comp_id IN (SELECT * FROM UNNEST(?))
                """,
                [comp_ids],
            ).fetchall()
            name_by_id = {cid: name for cid, name in mapped}

            return [{"label": name_by_id.get(cid, f"comp_id={cid}"), "value": cid} for cid in comp_ids]
        finally:
            con.close()

    @app.callback(
        Output("team-dd", "options"),
        Input("comp-dd", "value"),
    )
    def update_teams(comp_id: int | None):
        if comp_id is None:
            return []

        con = get_connection()
        try:
            rows = con.execute(
                """
                SELECT DISTINCT team_id
                FROM (
                    SELECT home_team_id AS team_id FROM matches_base WHERE comp_id = ?
                    UNION
                    SELECT away_team_id AS team_id FROM matches_base WHERE comp_id = ?
                )
                ORDER BY team_id
                """,
                [comp_id, comp_id],
            ).fetchall()
            team_ids = [r[0] for r in rows]

            mapped = con.execute(
                """
                SELECT team_id, name
                FROM teams
                WHERE team_id IN (SELECT * FROM UNNEST(?))
                """,
                [team_ids],
            ).fetchall()
            name_by_id = {tid: name for tid, name in mapped}

            return [{"label": name_by_id.get(tid, f"team_id={tid}"), "value": tid} for tid in team_ids]
        finally:
            con.close()

    from dash import ctx, no_update

    @app.callback(
        Output("selected-teams-store", "data"),
        Output("team-dd", "value"),
        Input("comp-dd", "value"),
        Input({"type": "team-card", "team_id": ALL}, "n_clicks"),
        Input("team-dd", "value"),
        State("selected-teams-store", "data"),
        prevent_initial_call=True,
    )
    def update_selected_teams(comp_id, team_card_clicks, team_dd_value, selected):
        selected = list(selected or [])
        trig = ctx.triggered_id

        # 1) league changed -> reset selection & clear dropdown
        if trig == "comp-dd":
            return [], None

        # 2) click on a team-card toggles that team
        # IMPORTANT: ignore triggers caused by re-render (many props change) or resets (value==0)
        if isinstance(trig, dict) and trig.get("type") == "team-card":
            if len(ctx.triggered) != 1:
                return no_update, no_update  # likely re-render / multiple changes
            # If the card was just re-created, n_clicks is often 0 -> not a real click
            fired_val = ctx.triggered[0].get("value", None)
            if not fired_val:  # None or 0
                return no_update, no_update
            team_id = int(trig["team_id"])

            if team_id in selected:
                selected.remove(team_id)
            else:
                if len(selected) < 3:
                    selected.append(team_id)

            # dropdown shows last selected (or None)
            dd_val = selected[-1] if selected else None
            return selected, dd_val

        # 3) dropdown interaction:
        # - selecting a team adds it (if room)
        # - clearing dropdown removes the LAST selected
        if trig == "team-dd":
            if team_dd_value is None:
                # user cleared -> pop last
                if selected:
                    selected.pop()
                return selected, (selected[-1] if selected else None)

            tid = int(team_dd_value)
            if tid not in selected:
                if len(selected) < 3:
                    selected.append(tid)
            # keep dropdown at last selected
            return selected, (selected[-1] if selected else None)

        return no_update, no_update

    @app.callback(
        Output("comp-dd", "value"),
        Input({"type": "comp-card", "comp_id": ALL}, "n_clicks"),
        State({"type": "comp-card", "comp_id": ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_comp_from_cards(n_clicks_list, ids):
        if not n_clicks_list or not ids:
            return no_update

        # Find the most recently clicked league
        max_clicks = 0
        chosen = None
        for n, idobj in zip(n_clicks_list, ids):
            if n and n > max_clicks:
                max_clicks = n
                chosen = idobj["comp_id"]

        return chosen if chosen is not None else no_update

    @app.callback(
        Output("country-dd", "options"),
        Input("time-axis-store", "data"),
    )
    def init_country_options(_):
        # Only allow countries present in the dataset (via comps mapping), normalized to NaturalEarth names
        allowed = countries_with_data_from_comps_csv(COMPS_CSV_PATH)

        return [{"label": c, "value": c} for c in allowed]

    @app.callback(
        Output("country-panel", "children"),
        Input("country-dd", "value"),
        Input("comp-dd", "value"),
        Input("team-dd", "value"),
        Input("selected-teams-store", "data"),
        Input("allowed-comp-ids", "data"),
        Input("time-window-store", "data"),
    )
    def update_country_panel(country: str | None, comp_id: int | None, team_id: int | None,
                             selected_team_ids: list[int] | None, allowed_ids: list[int] | None, win):
        if not country:
            return "Pick a country to begin."

        header = f"Country: {country}"
        if win:
            header += f" | Time: {win['start'][:10]} → {win['end'][:10]}"

        con = get_connection()
        try:
            # 1. Show leagues if no league is selected
            if comp_id is None:
                if not allowed_ids:
                    return html.Div(f"{header} | No competitions available.")

                countries = DATASET_COUNTRY_ALIASES.get(country, [country])
                comp_rows = con.execute(
                    """
                    SELECT DISTINCT c.comp_id, c.name
                    FROM comps c
                    JOIN matches_base m ON m.comp_id = c.comp_id
                    WHERE c.country IN (SELECT * FROM UNNEST(?))
                    AND c.comp_id IN (SELECT * FROM UNNEST(?))
                    ORDER BY c.name
                    """,
                    [countries, allowed_ids],
                ).fetchall()

                if not comp_rows:
                    return html.Div(f"{header} | No competitions with matches in this country.")

                # Render league buttons
                cards = []
                for cid, cname in comp_rows:
                    style = {
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                        "padding": "8px",
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "background": "#DDEBFF" if comp_id == cid else "white",
                    }
                    cards.append(html.Div(cname, id={"type": "comp-card", "comp_id": cid}, style=style))

                return html.Div(
                    children=[html.Div(header, style={"marginBottom": "10px"}), html.Div(
                        cards,
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                            "gap": "8px",
                            "maxHeight": "240px",
                            "overflowY": "auto",
                            "paddingRight": "6px",
                        },
                    )]
                )

            # 2. Show teams if a league is selected
            teams = list_teams_for_comp(int(comp_id))
            if not teams:
                return html.Div(f"{header} | Competition: {comp_id} | No teams found.")

            cards = []
            for tid, name in teams:
                logo = find_logo_src(name)
                selected_team_ids = selected_team_ids or []
                style = {
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "padding": "8px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                    "background": "#f0f8ff" if tid in selected_team_ids else "white",
                }
                children = []
                if logo:
                    children.append(html.Img(src=logo, style={"height": "28px"}))
                children.append(html.Span(name))
                cards.append(html.Div(children=children, id={"type": "team-card", "team_id": tid}, n_clicks=0,
                                      style=style))

            return html.Div(
                children=[
                    html.Div(f"{header} | Competition: {comp_id}", style={"marginBottom": "10px"}),
                    html.Div(
                        cards,
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                            "gap": "8px",
                            "maxHeight": "240px",
                            "overflowY": "auto",
                            "paddingRight": "6px",
                        },
                    ),
                ]
            )
        finally:
            con.close()

    @app.callback(
        Output("country-dd", "value"),
        Input("europe-map", "clickData"),
        prevent_initial_call=True,
    )
    def select_country_from_map(clickData):
        if not clickData:
            return no_update

        country = clickData["points"][0]["location"]
        return country

    @app.callback(
        Output("time-window-store", "data"),
        Input("time-slider", "value"),
        State("time-axis-store", "data"),
    )
    def update_time_window_store(slider_value, axis_data):
        axis = _axis_from_store(axis_data)
        if axis is None or not slider_value:
            return None

        start_day, end_day = slider_value
        start_dt = axis.day_to_dt(start_day)
        end_dt = axis.day_to_dt(end_day)

        return {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "start_day": int(start_day),
            "end_day": int(end_day),
        }

    @app.callback(
        Output("time-slider-label", "children"),
        Input("time-window-store", "data"),
    )
    def show_time_window_label(win):
        if not win:
            return "Time window: (not available)"

        start = datetime.fromisoformat(win["start"]).strftime("%Y-%m-%d")
        end = datetime.fromisoformat(win["end"]).strftime("%Y-%m-%d")

        return f"Time window: {start} → {end}"

    @app.callback(
        Output("kickoff-heatmap", "figure"),
        Input("country-dd", "value"),
        Input("comp-dd", "value"),
        Input("team-dd", "value"),
    )
    def update_kickoff_heatmap(country, comp_id, team_id):
        if comp_id is None:
            fig = go.Figure()
            return apply_fig_theme(fig, "Select a competition to see the heatmap")

        countries = to_dataset_countries(country)
        rows = heatmap_month_hourbin_avg_goals(
            comp_id=int(comp_id), countries=countries, team_id=None
        )
        df = pd.DataFrame(rows, columns=["trimes_key", "hour_bin", "avg_goals", "n_matches"])
        df = df[df["n_matches"] >= 5]

        hour_bins = ["<=13:59", "14:00-16:59", "17:00-19:59", "20:00+"]

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No data for current selection",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            return fig

        def trimes_label(k: int) -> str:
            year = k // 100
            t = k % 100  # 1...4
            m1 = (t - 1) * 3 + 1
            m3 = m1 + 2
            return f"{year}-{m1:02d}..{m3:02d}"

        # Pivot tables
        avg_mat = (
            df.pivot(index="trimes_key", columns="hour_bin", values="avg_goals")
            .reindex(columns=hour_bins)
            .sort_index()
        )
        cnt_mat = (
            df.pivot(index="trimes_key", columns="hour_bin", values="n_matches")
            .reindex(columns=hour_bins)
            .sort_index()
            .fillna(0)
            .astype(int)
        )

        # Safe writable NumPy array
        z = avg_mat.to_numpy(dtype=float).copy()
        z[cnt_mat.to_numpy() == 0] = np.nan  # grey out cells with 0 matches

        y_labels = [trimes_label(int(k)) for k in avg_mat.index]

        # Heatmap
        heat = go.Heatmap(
            z=z,
            x=hour_bins,
            y=y_labels,
            colorscale="YlOrRd",
            colorbar=dict(title="Avg goals"),
            hovertemplate=(
                "Period=%{y}<br>"
                "Month: %{y}<br>"
                "Hour bin: %{x}<br>"
            ),
            xgap=1,
            ygap=1
        )
        fig = go.Figure(data=[heat])

        # Background & layout
        fig.update_layout(
            title=(
                "<b>Avg goals by month × kick-off time</b>"
                "<br><span style='font-size:12px;color:#6c757d;'>Only showing cells with 5 matches.</span>"
            ),
            plot_bgcolor="#F0F4FA",
        )

        fig.update_layout(annotations=[])
        fig.update_xaxes(side="bottom")
        fig.update_yaxes(autorange="reversed")

        return apply_fig_theme(fig)

    @app.callback(
        Output("parallel-coords-graph", "figure"),
        Input("comp-dd", "value"),
        Input("selected-teams-store", "data"),
        Input("time-window-store", "data"),
    )
    def update_parallel_coords(comp_id, selected_team_ids, win):
        if not comp_id or not win:
            fig = go.Figure()
            return apply_fig_theme(fig, "Select competition and time window")

        start_dt = win["start"]
        end_dt = win["end"]

        team_ids = [int(x) for x in (selected_team_ids or [])]
        team_filter = team_ids if len(team_ids) > 0 else None
        df = parallel_coords_team_match(
            comp_id=int(comp_id),
            team_ids=team_filter,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        return make_parallel_coords(df)

    @app.callback(
        Output("team-radar-1", "figure"),
        Output("team-radar-2", "figure"),
        Output("team-radar-3", "figure"),
        Input("comp-dd", "value"),
        Input("selected-teams-store", "data"),
        Input("time-window-store", "data"),
    )
    def update_team_radars(comp_id, selected_team_ids, win):
        metrics = [
            "avg_shots",
            "avg_possession",
            "avg_pass_accuracy",
            "avg_corners",
            "avg_goals",
            "avg_shots_on_goal",
        ]
        labels = [
            "Shots",
            "Possession (%)",
            "Pass Accuracy (%)",
            "Corners",
            "Goals",
            "Shots on Target",
        ]

        empty_fig = go.Figure()
        empty_fig = apply_fig_theme(empty_fig, "Select competition and time window", height=320)

        if not comp_id or not win:
            return empty_fig, empty_fig, empty_fig

        start_dt = datetime.fromisoformat(win["start"])
        end_dt = datetime.fromisoformat(win["end"])

        # Full stats for normalization
        df_full = team_avg_stats_for_radar(comp_id=int(comp_id), start_dt=start_dt, end_dt=end_dt)
        if df_full.empty:
            return empty_fig, empty_fig, empty_fig

        selected_team_ids = [int(x) for x in (selected_team_ids or [])]

        if selected_team_ids:
            top3_ids = selected_team_ids[:3]
        else:
            win_rate_df = compute_win_rate(int(comp_id), start_dt.isoformat(), end_dt.isoformat())
            top3_ids = (
                win_rate_df.sort_values("win_rate", ascending=False)
                .head(3)["team_id"]
                .astype(int)
                .tolist()
            )

        df_display = pd.concat([df_full[df_full["team_id"] == tid] for tid in top3_ids], ignore_index=True)

        # Color map: top3_ids -> blue, orange, green
        color_map = {tid: RADAR_COLORS[i] for i, tid in enumerate(top3_ids)}

        figs = []
        for i in range(3):
            if i < len(df_display):
                row = df_display.iloc[i]
                figs.append(build_single_team_radar(row, labels, metrics, df_full,
                                                    color=color_map.get(row["team_id"], "#d3d3d3")))
            else:
                f = go.Figure()
                f.update_layout(title="No team")
                figs.append(f)

        return figs[0], figs[1], figs[2]

    @app.callback(
        Output("goals-xg-corr", "figure"),
        Input("country-dd", "value"),
        Input("comp-dd", "value"),
        Input("selected-teams-store", "data"),
        Input("time-window-store", "data"),
    )
    def update_goals_xg_corr(country, comp_id, selected_team_ids, win):
        if not comp_id or not win:
            fig = go.Figure()
            return apply_fig_theme(fig, "Select competition and time window", height=360)

        countries = to_dataset_countries(country)

        team_ids = [int(x) for x in (selected_team_ids or [])]
        team_filter = team_ids if team_ids else None
        df = goals_vs_xg_matches(
            comp_id=int(comp_id),
            countries=countries,
            team_ids=team_filter,
            start_dt=win["start"],
            end_dt=win["end"],
        )

        return make_goals_vs_xg_violin(df)

    @app.callback(
        Output("avg-goals-bar", "figure"),
        Input("comp-dd", "value"),
        Input("selected-teams-store", "data"),
        Input("time-window-store", "data"),
    )
    def update_avg_goals_bar(comp_id, selected_team_ids, win):
        if not comp_id or not win:
            fig = go.Figure()
            return apply_fig_theme(fig, "Select competition and time window")

        start_dt = datetime.fromisoformat(win["start"])
        end_dt = datetime.fromisoformat(win["end"])

        # Average stats per team
        df_avg = team_avg_stats_for_radar(comp_id=int(comp_id), start_dt=start_dt, end_dt=end_dt)
        if df_avg.empty:
            fig = go.Figure()
            return apply_fig_theme(fig, "No data for current selection")

        selected_team_ids = [int(x) for x in (selected_team_ids or [])]
        if selected_team_ids:
            top3_ids = selected_team_ids[:3]
        else:
            win_rate_df = compute_win_rate(int(comp_id), start_dt.isoformat(), end_dt.isoformat())
            top3_ids = (
                win_rate_df.sort_values("win_rate", ascending=False)
                .head(3)["team_id"]
                .astype(int)
                .tolist()
            )

        # Color map (top 3 highlighted, others grey)
        color_map = {tid: RADAR_COLORS[i] for i, tid in enumerate(top3_ids)}
        df_avg["color"] = df_avg["team_id"].apply(lambda tid: color_map.get(tid, "#d3d3d3"))

        # Sort by avg_goals DESC (tie-break by team_name)
        df_avg["avg_goals"] = df_avg["avg_goals"].astype(float)
        df_avg = df_avg.sort_values(["avg_goals", "team_name"], ascending=[False, True]).reset_index(drop=True)

        return make_avg_goals_bar(df_avg, highlight_ids_ordered=top3_ids)
