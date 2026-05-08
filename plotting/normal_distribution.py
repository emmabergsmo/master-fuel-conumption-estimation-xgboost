import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(".")
DB_PATH = BASE_DIR / "opensky_updated.sqlite"

PRED_FILES = {
    "takeoff": BASE_DIR / "xg_boost/csv_files/pred_takeoff.csv",
    "climb": BASE_DIR / "xg_boost/csv_files/pred_climb.csv",
    "cruise": BASE_DIR / "xg_boost/csv_files/pred_cruise.csv",
    "descent": BASE_DIR / "xg_boost/csv_files/pred_descent.csv",
    "landing": BASE_DIR / "xg_boost/csv_files/pred_landing.csv",
}


# -----------------------------
# Helpers
# -----------------------------
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


def plot_normal_overlay(group: pd.DataFrame, pair_name: str, outdir: Path):
    """
    Plot fitted normal curves for real vs predicted total fuel for one route pair.
    """
    real = group["real_total_fuel"].dropna().to_numpy()
    pred = group["pred_total_fuel"].dropna().to_numpy()

    if len(real) < 2 or len(pred) < 2:
        print(f"Skipping {pair_name}: not enough data points.")
        return

    real_mean = real.mean()
    real_std = real.std(ddof=1)
    pred_mean = pred.mean()
    pred_std = pred.std(ddof=1)

    if real_std == 0 or pred_std == 0:
        print(f"Skipping {pair_name}: zero variance in one of the distributions.")
        return

    # Wider x-range so the full normal curves are visible
    xmin = min(real_mean - 4 * real_std, pred_mean - 4 * pred_std)
    xmax = max(real_mean + 4 * real_std, pred_mean + 4 * pred_std)

    # Optional: prevent negative fuel on x-axis
    xmin = max(0, xmin)

    x = np.linspace(xmin, xmax, 1000)

    real_pdf = norm.pdf(x, loc=real_mean, scale=real_std)
    pred_pdf = norm.pdf(x, loc=pred_mean, scale=pred_std)

    plt.figure(figsize=(10, 6))
    plt.plot(x, real_pdf, label="Real fuel", alpha=0.5, linewidth=2)
    plt.plot(x, pred_pdf, label="Predicted fuel", alpha=0.5, linewidth=2)

    plt.fill_between(x, real_pdf, alpha=0.2)
    plt.fill_between(x, pred_pdf, alpha=0.2)

    plt.title(f"Fuel consumption distribution: {pair_name}")
    plt.xlabel("Total fuel consumption")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = (
        pair_name.replace("/", "-")
        .replace(" ", "_")
        .replace("→", "to")
    )
    outpath = outdir / f"{safe_name}_fuel_distribution.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


# -----------------------------
# Load real data from SQLite
# -----------------------------
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
FROM flight_phase_features_v2
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


# -----------------------------
# Load prediction CSVs
# -----------------------------
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


# -----------------------------
# Match real and predicted
# -----------------------------
merged = real_df.merge(pred_df, on="flight_id", how="inner")

merged["dep_arr_pair"] = (
    merged["estdepartureairport"].astype(str)
    + " → "
    + merged["estarrivalairport"].astype(str)
)

print(f"Matched flights: {len(merged)}")
print(f"Unique departure-arrival pairs: {merged['dep_arr_pair'].nunique()}")


# -----------------------------
# Plot one figure per departure-arrival pair
# -----------------------------
outdir = BASE_DIR / "fuel_distribution_plots"
outdir.mkdir(exist_ok=True)

for pair_name, group in merged.groupby("dep_arr_pair"):
    if len(group) < 2:
        continue
    plot_normal_overlay(group, pair_name, outdir)

print("Done.")