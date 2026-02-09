from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

# Allowed domestic comps list (IDs)
ALLOWED_COMP_IDS = [
    39, 40, 41, 42,
    61, 62,
    78, 79,
    88,
    94,
    106,
    119,
    135, 136,
    140, 141,
    144,
    179,
    203,
    207,
    218,
]


def H4_title(text: str) -> html.H4:
    return html.H4(
        text,
        style={
            "fontWeight": 600,
            "fontSize": "1.15rem",
            "marginTop": "6px",
            "marginBottom": "12px",
        },
    )


def build_layout(europe_fig, axis_store_data, slider_max, slider_marks) -> html.Div:
    return dbc.Container(
        fluid=True,
        children=[
            html.H2("Football Viz"),

            # Stores
            dcc.Store(id="time-axis-store", data=axis_store_data),
            dcc.Store(id="time-window-store", data=None),
            dcc.Store(id="allowed-comp-ids", data=ALLOWED_COMP_IDS),
            dcc.Store(id="selected-teams-store", data=[]),

            # MAIN GRID
            dbc.Row(
                className="g-3",
                children=[
                    # LEFT COLUMN: Europe map + global (not time-filtered)
                    dbc.Col(
                        width=3,
                        children=[
                            H4_title("Country selector"),
                            dcc.Graph(
                                id="europe-map",
                                figure=europe_fig,
                                style={"height": "420px"},
                            ),

                            html.Hr(),

                            # Global section wrapper (heatmap only)
                            html.Div(
                                className="section-global",
                                children=[
                                    html.Div(
                                        "Not affected by time slider.",
                                        className="section-subtitle",
                                    ),
                                    dcc.Graph(
                                        id="kickoff-heatmap",
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),

                            html.Hr(),

                            # Time-filtered chart, but intentionally placed in the left column
                            # (visually under the global heatmap)
                            html.Div(
                                className="section-timefiltered",
                                children=[
                                    html.Div(
                                        "Affected by the time slider.",
                                        className="section-subtitle",
                                    ),
                                    dcc.Graph(
                                        id="goals-xg-corr",
                                        style={
                                            "height": "360px"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # MIDDLE COLUMN: country panel + time slider + time-filtered charts
                    dbc.Col(
                        width=6,
                        children=[
                            # Time-filtered section wrapper
                            html.Div(
                                className="section-timefiltered",
                                children=[
                                    html.Div(
                                        "Affected by the time slider.",
                                        className="section-subtitle",
                                    ),

                                    H4_title("Country panel"),
                                    html.Div(
                                        id="country-panel",
                                        style={
                                            "border": "1px dashed #bbb",
                                            "padding": "10px",
                                            "borderRadius": "6px",
                                            "minHeight": "280px",
                                        },
                                    ),

                                    html.Div(style={"height": "15px"}),

                                    H4_title("Time slider"),
                                    dcc.RangeSlider(
                                        id="time-slider",
                                        min=0,
                                        max=slider_max,
                                        step=1,
                                        value=[0, slider_max],
                                        marks=slider_marks,
                                        tooltip={"always_visible": False, "placement": "bottom"},
                                        updatemode="drag",
                                    ),
                                    html.Div(id="time-slider-label", style={"marginTop": "8px"}),
                                    html.Hr(),

                                    dcc.Graph(
                                        id="parallel-coords-graph",
                                        style={"height": "480px"},
                                    ),

                                    dcc.Graph(
                                        id="avg-goals-bar",
                                        style={"height": "420px"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # RIGHT COLUMN: filters + radars (time-filtered conceptually)
                    dbc.Col(
                        width=3,
                        children=[
                            html.Div(
                                className="section-timefiltered",
                                children=[
                                    html.Div(
                                        "Affected by the time slider.",
                                        className="section-subtitle",
                                    ),

                                    H4_title("Filters"),
                                    html.Div(
                                        [
                                            html.Label("Country"),
                                            dcc.Dropdown(id="country-dd", placeholder="Select country..."),
                                            html.Br(),
                                            html.Label("Competition"),
                                            dcc.Dropdown(id="comp-dd", placeholder="Select competition..."),
                                            html.Br(),
                                            html.Label("Team"),
                                            dcc.Dropdown(id="team-dd", placeholder="Select team..."),
                                        ],
                                        style={
                                            "border": "1px solid #eee",
                                            "borderRadius": "6px",
                                            "padding": "10px",
                                            "background": "white",
                                        },
                                    ),

                                    html.Hr(),

                                    H4_title("Team radars"),
                                    dcc.Graph(id="team-radar-1", style={"height": "320px"}),
                                    dcc.Graph(id="team-radar-2", style={"height": "320px"}),
                                    dcc.Graph(id="team-radar-3", style={"height": "320px"}),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
