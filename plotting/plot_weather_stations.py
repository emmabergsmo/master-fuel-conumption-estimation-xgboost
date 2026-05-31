import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# -------------------------
# CONFIG
# -------------------------
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# STATION DATA
# -------------------------
stations = [
    {"id": "SN4781", "name": "Gardermoen Sør",    "lat": 60.1883, "lon": 11.0743},
    {"id": "SN4780", "name": "Gardermoen",         "lat": 60.2065, "lon": 11.0802},
    {"id": "SN4735", "name": "Gardermoen Sørøst",  "lat": 60.1705, "lon": 11.1105},
    {"id": "SN4785", "name": "Gardermoen Plu",     "lat": 60.2105, "lon": 11.0782},
    {"id": "SN4725", "name": "E16 Gardermoen",     "lat": 60.1677, "lon": 11.1165},
]

# -------------------------
# TILE BACKGROUND (CartoDB Positron)
# -------------------------
class Positron(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"

tiles = Positron()

# -------------------------
# CREATE MAP
# -------------------------
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=tiles.crs)

ax.set_extent([11.01, 11.18, 60.144, 60.225], crs=ccrs.PlateCarree())

ax.add_image(tiles, 13)

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

ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".2f", degree_symbol="°"))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".2f", degree_symbol="°"))

# -------------------------
# NORTH ARROW
# -------------------------
ax.text(
    0.085, 0.955, "N",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color="#1f2a44",
    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
    zorder=4,
)

left_blade = Polygon(
    [(0.085, 0.925), (0.073, 0.885), (0.085, 0.895)],
    closed=True,
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="#1f2a44",
    linewidth=2.0,
    zorder=4,
)
right_blade = Polygon(
    [(0.085, 0.925), (0.085, 0.895), (0.097, 0.885)],
    closed=True,
    transform=ax.transAxes,
    facecolor="#1f2a44",
    edgecolor="#1f2a44",
    linewidth=2.0,
    zorder=4,
)
ax.add_patch(left_blade)
ax.add_patch(right_blade)

# -------------------------
# DRAW STATIONS
# -------------------------
colors = ["#ED7D31", "#70AD47", "#4472C4", "#7030A0", "#E8AE00"]

label_offsets = {
    "SN4781": (-0.0012,  0.0010, "right"),
    "SN4780": (-0.0012,  0.0010, "right"),
    "SN4735": ( 0.0012,  0.0010, "left"),
    "SN4785": ( 0.0012, -0.0014, "left"),
    "SN4725": ( 0.0012, -0.0014, "left"),
}

for station, color in zip(stations, colors):
    lon = station["lon"]
    lat = station["lat"]
    sid = station["id"]

    ax.scatter(
        lon, lat,
        s=80,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    dlon, dlat, ha = label_offsets.get(sid, (0.003, 0.003, "left"))
    ax.text(
        lon + dlon, lat + dlat,
        station["name"],
        fontsize=12,
        fontweight="bold",
        color=color,
        ha=ha,
        va="center",
        transform=ccrs.PlateCarree(),
        zorder=6,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.50,
            "boxstyle": "round,pad=0.15",
        },
        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
    )

# -------------------------
# SCALE BAR
# ~2 km bar in bottom-left, matching map style
# -------------------------
import numpy as np

# Position in axes coordinates
bar_x0 = 0.05
bar_y0 = 0.05
bar_ax_width = 0.18  # fraction of axes width

# Convert axes fraction to map units to know real-world length
x0_disp, y0_disp = ax.transAxes.transform((bar_x0, bar_y0))
x1_disp, y1_disp = ax.transAxes.transform((bar_x0 + bar_ax_width, bar_y0))

x0_map, y0_map = ax.transData.inverted().transform((x0_disp, y0_disp))
x1_map, y1_map = ax.transData.inverted().transform((x1_disp, y1_disp))

# Convert map coords (Web Mercator metres) to geographic to get km
import cartopy.geodesic as cgeo
pt0 = ccrs.PlateCarree().transform_point(x0_map, y0_map, tiles.crs)
pt1 = ccrs.PlateCarree().transform_point(x1_map, y1_map, tiles.crs)
dist_m = cgeo.Geodesic().inverse([pt0[0], pt0[1]], [pt1[0], pt1[1]])[0, 0]
dist_km = dist_m / 1000

# Round to nearest 0.5 km and rescale bar width
target_km = round(dist_km * 2) / 2
scale_factor = target_km / dist_km
bar_ax_width_scaled = bar_ax_width * scale_factor

# Draw the bar (two filled halves for classic scale bar look)
mid_x = bar_x0 + bar_ax_width_scaled / 2
half = bar_ax_width_scaled / 2
bar_height = 0.012

for i, (x_start, fc) in enumerate([(bar_x0, "#1f2a44"), (mid_x, "white")]):
    rect = plt.Rectangle(
        (x_start, bar_y0),
        half, bar_height,
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor="#1f2a44",
        linewidth=0.8,
        zorder=4,
    )
    ax.add_patch(rect)

# Labels: 0, half, full
for label, x_pos in [
    ("0", bar_x0),
    (f"{target_km/2:.1g} km", mid_x),
    (f"{target_km:.4g} km", bar_x0 + bar_ax_width_scaled),
]:
    ax.text(
        x_pos, bar_y0 + bar_height ,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#1f2a44",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        zorder=5,
    )

# -------------------------
# FINALIZE
# -------------------------
plt.tight_layout()

out_path = os.path.join(OUTDIR, "gardermoen_stations_map.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Saved: {out_path}")
