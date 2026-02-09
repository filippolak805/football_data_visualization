from __future__ import annotations

from dash import Dash
from pathlib import Path
from datetime import datetime, timedelta

from src.layout.layout import build_layout
from src.layout.europe_map import build_europe_map
from src.data_layer.queries import get_time_axis
from src.callbacks.core import register_callbacks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"


def build_slider_marks(origin_iso: str, max_days: int, n_marks: int = 7) -> dict:
    origin = datetime.fromisoformat(origin_iso)

    if max_days <= 0:
        return {0: {"label": origin.strftime("%Y-%m-%d"), "style": {"fontSize": "11px"}}}

    # evenly spaced indices including endpoints
    if n_marks < 2:
        n_marks = 2

    step = max(1, max_days // (n_marks - 1))
    idxs = list(range(0, max_days + 1, step))
    if idxs[-1] != max_days:
        idxs.append(max_days)

    marks = {}
    last_i = 0
    for i in idxs:
        d = (origin + timedelta(days=int(i))).date()

        style = {"fontSize": "11px", "whiteSpace": "nowrap"}
        if i == 0:
            style["transform"] = "translateX(0%)"
        elif i == max_days:
            style["transform"] = "translateX(-100%)"  # pull last label left
        else:
            style["transform"] = "translateX(-50%)"  # center intermediate labels
        marks[int(i)] = {"label": d.strftime("%Y-%m"), "style": style}
        last_i = int(i)
    marks.pop(last_i, None)

    return marks


def create_app() -> Dash:
    app = Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
        title="Football Viz",
        external_stylesheets=[
            "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        ],
    )

    # Build data for layout
    europe_fig = build_europe_map()

    axis = get_time_axis()
    axis_store_data = {
        "origin": axis.origin.isoformat(),
        "max_days": axis.max_days,
    }
    slider_max = axis.max_days

    # Layout
    slider_marks = build_slider_marks(axis_store_data["origin"], axis.max_days, n_marks=10)

    app.layout = build_layout(
        europe_fig=europe_fig,
        axis_store_data=axis_store_data,
        slider_max=slider_max,
        slider_marks=slider_marks,
    )

    register_callbacks(app)
    return app
