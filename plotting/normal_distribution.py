import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


BASE_DIR = Path(".")
DB_PATH = BASE_DIR / "data.sqlite"
OUTDIR = BASE_DIR / "outputs" / "fuel_distribution_plots"

PRED_FILES = {
    "takeoff": BASE_DIR / "notebooks/xgboost/csv_files/pred_takeoff.csv",
    "climb": BASE_DIR / "notebooks/xgboost/csv_files/pred_climb.csv",
    "cruise": BASE_DIR / "notebooks/xgboost/csv_files/pred_cruise.csv",
    "descent": BASE_DIR / "notebooks/xgboost/csv_files/pred_descent.csv",
    "landing": BASE_DIR / "notebooks/xgboost/csv_files/pred_landing.csv",
}

SELECTED_PAIRS = [
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


def load_real_fuel() -> pd.DataFrame:
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

    return real_df[
        [
            "flight_id",
            "estdepartureairport",
            "estarrivalairport",
            "real_total_fuel",
        ]
    ].copy()


def load_predicted_fuel() -> pd.DataFrame:
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

    return pred_df[["flight_id", "pred_total_fuel"]].copy()


def load_merged_fuel() -> pd.DataFrame:
    real_df = load_real_fuel()
    pred_df = load_predicted_fuel()

    merged = real_df.merge(pred_df, on="flight_id", how="inner")

    merged["dep_arr_pair"] = (
        merged["estdepartureairport"].astype(str)
        + " → "
        + merged["estarrivalairport"].astype(str)
    )

    print(f"Matched flights: {len(merged)}")
    print(f"Unique departure-arrival pairs: {merged['dep_arr_pair'].nunique()}")

    return merged


def plot_panel(merged: pd.DataFrame, outdir: Path) -> None:
    available_routes = set(merged["dep_arr_pair"].unique())
    missing = [r for r in SELECTED_PAIRS if r not in available_routes]

    if missing:
        print("Missing routes:")
        print(missing)

    fig, axes = plt.subplots(3, 3, figsize=(40, 30))
    axes = axes.flatten()

    for ax, pair_name in zip(axes, SELECTED_PAIRS):
        group = merged[merged["dep_arr_pair"] == pair_name]

        real = group["real_total_fuel"].dropna().to_numpy()
        pred = group["pred_total_fuel"].dropna().to_numpy()

        if len(real) < 2 or len(pred) < 2:
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

        real_mean = real.mean()
        real_std = real.std(ddof=1)

        pred_mean = pred.mean()
        pred_std = pred.std(ddof=1)

        if real_std == 0 or pred_std == 0:
            ax.set_title(pair_name, fontsize=40, pad=30)
            ax.text(
                0.5,
                0.5,
                "Zero variance",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=30,
            )
            ax.axis("off")
            continue

        xmin = min(
            real_mean - 4 * real_std,
            pred_mean - 4 * pred_std,
        )

        xmax = max(
            real_mean + 4 * real_std,
            pred_mean + 4 * pred_std,
        )

        xmin = max(0, xmin)

        x_real = np.linspace(xmin, xmax, 1000)

        # Normalize x-axis so the peak of the real curve is at x = 1
        x = x_real / real_mean

        real_pdf = norm.pdf(x_real, loc=real_mean, scale=real_std)
        pred_pdf = norm.pdf(x_real, loc=pred_mean, scale=pred_std)

        ax.plot(
            x,
            real_pdf,
            label="Real fuel",
            linewidth=6,
            alpha=0.7,
        )

        ax.plot(
            x,
            pred_pdf,
            label="Predicted fuel",
            linewidth=6,
            alpha=0.7,
        )

        ax.fill_between(x, real_pdf, alpha=0.2)
        ax.fill_between(x, pred_pdf, alpha=0.2)

        ax.axvline(
            1,
            linestyle="--",
            linewidth=5,
            label="Real fuel mean",
        )

        ax.set_xlim(0.65, 1.35)
        ax.set_ylim(0.0, 0.0049)
        ax.set_title(pair_name, fontsize=40, pad=30)

        ax.tick_params(axis="both", labelsize=30)
        ax.grid(True, alpha=0.3)

    fig.supxlabel(
        "Normalised total fuel consumption",
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

    outpath = outdir / "distribution_panel_1.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


def plot_overall(merged: pd.DataFrame, outdir: Path) -> None:
    real = merged["real_total_fuel"].dropna().to_numpy()
    pred = merged["pred_total_fuel"].dropna().to_numpy()

    real_mean = real.mean()
    real_std = real.std(ddof=1)

    pred_mean = pred.mean()
    pred_std = pred.std(ddof=1)

    xmin = max(
        0,
        min(
            real_mean - 4 * real_std,
            pred_mean - 4 * pred_std,
        )
    )

    xmax = max(
        real_mean + 4 * real_std,
        pred_mean + 4 * pred_std,
    )

    x_real = np.linspace(xmin, xmax, 1000)
    x = x_real / real_mean

    real_pdf = norm.pdf(x_real, loc=real_mean, scale=real_std)
    pred_pdf = norm.pdf(x_real, loc=pred_mean, scale=pred_std)

    plt.figure(figsize=(18, 12))

    plt.plot(
        x,
        real_pdf,
        linewidth=4,
        alpha=0.7,
        label="Real fuel",
    )

    plt.plot(
        x,
        pred_pdf,
        linewidth=4,
        alpha=0.7,
        label="Predicted fuel",
    )

    plt.fill_between(x, real_pdf, alpha=0.2)
    plt.fill_between(x, pred_pdf, alpha=0.2)

    plt.axvline(
        1,
        linestyle="--",
        linewidth=4,
        label="Real fuel mean",
    )

    plt.xlabel(
        "Normalised total fuel consumption",
        fontsize=25,
    )

    plt.ylabel(
        "Probability density",
        fontsize=25,
    )

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.grid(True, alpha=0.3)

    plt.legend(
        fontsize=20,
        frameon=False,
    )

    plt.tight_layout()

    outpath = outdir / "all_routes_fuel_distribution_1.png"

    plt.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    print(f"Saved: {outpath}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged_fuel()
    plot_panel(merged, OUTDIR)
    plot_overall(merged, OUTDIR)


if __name__ == "__main__":
    main()
