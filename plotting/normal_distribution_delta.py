import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


BASE_DIR = Path(".")
DB_PATH = BASE_DIR / "data.sqlite"


PRED_FILES = {
    "takeoff": BASE_DIR / "notebooks/xgboost/csv_files/pred_takeoff.csv",
    "climb": BASE_DIR / "notebooks/xgboost/csv_files/pred_climb.csv",
    "cruise": BASE_DIR / "notebooks/xgboost/csv_files/pred_cruise.csv",
    "descent": BASE_DIR / "notebooks/xgboost/csv_files/pred_descent.csv",
    "landing": BASE_DIR / "notebooks/xgboost/csv_files/pred_landing.csv",
}

selected_pairs = [
    "ENBR → ENGM",
    "ENGM → ENZV",
    "ENBO → ENGM",
    "ENZV → ENBR",
    "ENGM → ENCN",
    "ENHD → ENGM",
    "ENGM → ENVA",
    "ENGM → ENEV",
    "ENBR → ENVA",
]


def find_prediction_column(df: pd.DataFrame, phase_name: str) -> str:
    """
    Find the predicted fuel column in each prediction CSV.
    Tries common names first, otherwise falls back to the only numeric column
    other than flight_id.
    """
    candidates = [
        f"pred_{phase_name}",
        f"{phase_name}_pred",
        f"{phase_name}_fuel",
        "predicted_fuel",
        "pred_fuel",
        "fuel",
        "prediction",
        "pred",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = [
        c for c in df.columns
        if c != "flight_id" and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not determine prediction column for phase '{phase_name}'. "
        f"Columns found: {list(df.columns)}"
    )


conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    flight_id,
    estdepartureairport,
    estarrivalairport,
    takeoff_fuel,
    climb_fuel,
    cruise_fuel,
    descent_fuel,
    landing_fuel
FROM flight_phase_features
"""

real_df = pd.read_sql_query(query, conn)
conn.close()

real_df["real_total_fuel"] = (
    real_df["takeoff_fuel"].fillna(0)
    + real_df["climb_fuel"].fillna(0)
    + real_df["cruise_fuel"].fillna(0)
    + real_df["descent_fuel"].fillna(0)
    + real_df["landing_fuel"].fillna(0)
)

real_df = real_df[
    ["flight_id", "estdepartureairport", "estarrivalairport", "real_total_fuel"]
].copy()


pred_parts = []

for phase, path in PRED_FILES.items():
    df = pd.read_csv(path)
    pred_col = find_prediction_column(df, phase)

    df = df[["flight_id", pred_col]].copy()
    df = df.rename(columns={pred_col: f"pred_{phase}_fuel"})
    pred_parts.append(df)

pred_df = pred_parts[0]
for df in pred_parts[1:]:
    pred_df = pred_df.merge(df, on="flight_id", how="inner")

pred_phase_cols = [
    c for c in pred_df.columns
    if c.startswith("pred_") and c.endswith("_fuel")
]

pred_df["pred_total_fuel"] = pred_df[pred_phase_cols].fillna(0).sum(axis=1)
pred_df = pred_df[["flight_id", "pred_total_fuel"]].copy()


merged = real_df.merge(pred_df, on="flight_id", how="inner")

merged["dep_arr_pair"] = (
    merged["estdepartureairport"].astype(str)
    + " → "
    + merged["estarrivalairport"].astype(str)
)

merged["delta_fuel"] = merged["pred_total_fuel"] - merged["real_total_fuel"]

print(f"Matched flights: {len(merged)}")
print(f"Unique departure-arrival pairs: {merged['dep_arr_pair'].nunique()}")


delta_outdir = BASE_DIR / "delta_distribution_plots"

delta_outdir.mkdir(exist_ok=True)


delta = merged["delta_fuel"].dropna().to_numpy()

delta_mean = delta.mean()
delta_std = delta.std(ddof=1)

plt.figure(figsize=(18, 12))

plt.hist(
    delta,
    bins=30,
    density=True,
    alpha=0.5,
    label="Delta histogram",
)

if delta_std > 0:
    xmin = delta_mean - 4 * delta_std
    xmax = delta_mean + 4 * delta_std
    x = np.linspace(xmin, xmax, 1000)

    delta_pdf = norm.pdf(
        x,
        loc=delta_mean,
        scale=delta_std,
    )

    plt.plot(
        x,
        delta_pdf,
        linewidth=4,
        alpha=0.8,
        label="Normal fit",
    )

plt.axvline(
    0,
    linestyle="--",
    linewidth=4,
    label="Perfect prediction",
)

plt.xlabel("Fuel prediction error (kg)", fontsize=25)
plt.ylabel("Probability density", fontsize=25)

plt.xlim(-400, 400)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.grid(True, alpha=0.3)

plt.legend(
    fontsize=20,
    frameon=False,
)

plt.tight_layout()

outpath = delta_outdir / "all_routes_delta_distribution_1.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")

print(f"Saved: {outpath}")
plt.close()


COMMON_YMAX = 0.0069

fig, axes = plt.subplots(3, 3, figsize=(40, 30))
axes = axes.flatten()

for ax, pair_name in zip(axes, selected_pairs):
    group = merged[merged["dep_arr_pair"] == pair_name]
    delta = group["delta_fuel"].dropna().to_numpy()

    if len(delta) < 2:
        ax.set_title(pair_name, fontsize=40, pad=30)
        ax.text(
            0.5,
            0.5,
            "Not enough data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=30,
        )
        ax.axis("off")
        continue

    delta_mean = delta.mean()
    delta_std = delta.std(ddof=1)

    hist_vals, _, _ = ax.hist(
        delta,
        bins=20,
        density=True,
        alpha=0.5,
        label="Histogram",
    )

    route_ymax = hist_vals.max()

    if delta_std > 0:
        xmin = delta_mean - 4 * delta_std
        xmax = delta_mean + 4 * delta_std
        x = np.linspace(xmin, xmax, 1000)

        pdf = norm.pdf(x, loc=delta_mean, scale=delta_std)

        ax.plot(
            x,
            pdf,
            linewidth=6,
            label="Normal fit",
        )

        route_ymax = max(route_ymax, pdf.max())

    ax.axvline(
        0,
        linestyle="--",
        linewidth=6,
        label="Perfect prediction",
    )

    ax.set_title(pair_name, fontsize=40, pad=30)
    ax.set_xlim(-390, 390)

    # Use common y-axis unless route exceeds it
    if route_ymax > COMMON_YMAX:
        ax.set_ylim(0, route_ymax * 1.05)
    else:
        ax.set_ylim(0, COMMON_YMAX)

    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax.tick_params(axis="both", labelsize=30)
    ax.grid(True, alpha=0.3)

fig.supxlabel(
    "Fuel prediction error (kg)",
    fontsize=50,
    y=0.04,
)

fig.supylabel(
    "Probability density",
    fontsize=50,
    x=0.04,
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.00),
    ncol=3,
    frameon=False,
    fontsize=45,
)

fig.subplots_adjust(
    top=0.88,
    bottom=0.10,
    left=0.10,
    right=0.98,
    hspace=0.30,
    wspace=0.15,
)

outpath = delta_outdir / "delta_distribution_panel_1.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {outpath}")
