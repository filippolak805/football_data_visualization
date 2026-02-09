import json
from pathlib import Path

_GEOJSON_CACHE = None


def load_europe_geojson():
    global _GEOJSON_CACHE
    if _GEOJSON_CACHE is None:
        path = Path("assets/natural_earth/ne_10m_admin_0_countries.geojson")
        with open(path, "r", encoding="utf-8") as f:
            _GEOJSON_CACHE = json.load(f)
    return _GEOJSON_CACHE
