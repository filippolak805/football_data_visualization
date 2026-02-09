import geopandas as gpd
from pathlib import Path

SRC = Path("assets/natural_earth/ne_10m_admin_0_countries.shp")
DST = Path("assets/natural_earth/ne_10m_admin_0_countries.geojson")


def main():
    gdf = gpd.read_file(SRC)

    gdf = gdf[["ADMIN", "ISO_A2", "CONTINENT", "geometry"]]  # keep only necessary columns

    gdf = gdf[gdf["CONTINENT"] == "Europe"]  # Europe

    gdf.to_file(DST, driver="GeoJSON")
    print(f"Saved {DST}")


if __name__ == "__main__":
    main()
