import plotly.express as px
import pandas as pd

from src.utils.geo import load_europe_geojson
from src.data_layer.queries import list_countries_from_matches
from src.utils.country_normalization import COUNTRY_ALIASES


def build_europe_map():
    geojson = load_europe_geojson()

    raw = list_countries_from_matches()
    available = set(COUNTRY_ALIASES.get(c, c) for c in raw)  # get countries actually present in matches

    # Build Pandas DataFrame from GeoJSON properties
    records = []
    for f in geojson["features"]:
        name = f["properties"]["ADMIN"]
        records.append({
            "country": name,
            "enabled": name in available,
        })

    df = pd.DataFrame(records)

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="country",
        featureidkey="properties.ADMIN",
        color="enabled",
        color_discrete_map={True: "#4C78A8", False: "#D3DAE3"},
        hover_name="country",
    )

    fig.update_geos(
        scope="europe",
        showland=True,
        landcolor="#f6f8fb",
        showcountries=True,
        countrycolor="black",
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",  # let paper_bgcolor show through
    )

    fig.update_geos(
        projection_scale=2.1,  # higher value = more zoomed in
        center={"lat": 47, "lon": 5}  # center Europe
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
        coloraxis_showscale=False,
        paper_bgcolor="#f6f8fb",
        plot_bgcolor = "#f6f8fb",
    )

    return fig


def get_europe_country_names():
    geojson = load_europe_geojson()
    return {f["properties"]["ADMIN"] for f in geojson["features"]}
