import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
import cartopy.crs as ccrs

import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import airportsdata

# -------------------------
# CONFIG
# -------------------------
DB_PATH = "opensky.sqlite"
TABLE_NAME = "flight_phase_features_v2"

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


# LOAD DATA
con = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(f"""
SELECT estdepartureairport, estarrivalairport
FROM {TABLE_NAME}
WHERE estdepartureairport IS NOT NULL
AND estarrivalairport IS NOT NULL
""", con)

con.close()

# PRINT AIRPORT COUNTS
total_departures = len(df)
total_arrivals = len(df)

airport_counts = (
    pd.concat(
        [
            df["estdepartureairport"].value_counts().rename("departures"),
            df["estarrivalairport"].value_counts().rename("arrivals"),
        ],
        axis=1,
    )
    .fillna(0)
    .astype(int)
)
airport_counts["total"] = airport_counts["departures"] + airport_counts["arrivals"]
airport_counts = airport_counts.sort_values("total", ascending=False)

print(f"Total departures: {total_departures}")
print(f"Total arrivals: {total_arrivals}")
print("\nArrivals and departures by airport:")
print(airport_counts.to_string())

# AGGREGATE ROUTES
route_counts = (
    df.groupby(["estdepartureairport", "estarrivalairport"])
      .size()
      .reset_index(name="count")
)


# LOAD AIRPORT DATA
airports = airportsdata.load()


# TILE BACKGROUND (CartoDB Positron)

class Positron(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"

tiles = Positron()

# CREATE MAP
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=tiles.crs)

# Norway region
ax.set_extent([4, 22, 57, 71], crs=ccrs.PlateCarree())

# Basemap
ax.add_image(tiles, 6)

# Gridlines
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.4,
    color="gray",
    alpha=0.5,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 12}
gl.ylabel_style = {"size": 12}

ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))

# North arrow
ax.text(
    0.085, 0.955, "N",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color="#1f2a44",
    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    zorder=4,
)

left_blade = Polygon(
    [(0.085, 0.915), (0.066, 0.875), (0.085, 0.885)],
    closed=True,
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="#1f2a44",
    linewidth=2.5,
    zorder=4,
)
right_blade = Polygon(
    [(0.085, 0.915), (0.085, 0.885), (0.104, 0.875)],
    closed=True,
    transform=ax.transAxes,
    facecolor="#1f2a44",
    edgecolor="#1f2a44",
    linewidth=2.5,
    zorder=4,
)
ax.add_patch(left_blade)
ax.add_patch(right_blade)

route_counts["route_key"] = route_counts.apply(
    lambda r: tuple(sorted([r["estdepartureairport"], r["estarrivalairport"]])),
    axis=1
)

route_counts = (
    route_counts.groupby("route_key")["count"]
    .sum()
    .reset_index()
)

route_counts[["estdepartureairport", "estarrivalairport"]] = pd.DataFrame(
    route_counts["route_key"].tolist(),
    index=route_counts.index
)

route_counts = route_counts.sort_values("count")
max_count = route_counts["count"].max()
route_colors = [
    "#4472C4",
    "#70AD47",
    "#ED7D31",
    "#7030A0",
    "#E8AE00",
    "#D9534F",
    "#5BC0DE",
    "#1D4F17",
    "#8B5A2B",
    "#E377C2",
]


# DRAW ROUTES
for i, (_, row) in enumerate(route_counts.iterrows()):
    dep = row["estdepartureairport"]
    arr = row["estarrivalairport"]
    count = row["count"]

    if dep not in airports or arr not in airports:
        continue

    lat1 = airports[dep]["lat"]
    lon1 = airports[dep]["lon"]
    lat2 = airports[arr]["lat"]
    lon2 = airports[arr]["lon"]

    weight = count / max_count
    color = route_colors[i % len(route_colors)]

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        linewidth=0.8 + weight * 5.5,
        color=color,
        alpha=0.9,
        transform=ccrs.Geodetic(),
        zorder=2
    )

# DRAW AIRPORTS
used_airports = set(route_counts["estdepartureairport"]) | set(route_counts["estarrivalairport"])

for icao in used_airports:
    if icao not in airports:
        continue

    lat = airports[icao]["lat"]
    lon = airports[icao]["lon"]

    ax.scatter(
        lon, lat,
        s=26,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=3
    )

    ax.text(
        lon, lat,
        icao,
        fontsize=12,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=3,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
    )


# FINALIZE
plt.tight_layout()

# Save
out_path = os.path.join(OUTDIR, "norway_route_map.png")
plt.savefig(out_path, dpi=300)

plt.close()
print(f"Saved: {out_path}")
