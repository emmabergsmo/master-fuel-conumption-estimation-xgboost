from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

PHASES = ["takeoff", "climb", "cruise", "descent", "landing"]

CONFIG_FILES = {
    "Config 1": BASE_DIR / "notebooks/xg_boost/csv_files/route_actual_100_predicted_vs_actual_percent.csv",
    "Config 2": BASE_DIR / "notebooks/xg_boost_physics/csv_files/route_actual_100_predicted_vs_actual_percent_config2.csv",
    "Config 3": BASE_DIR / "notebooks/xg_boost_physics_weather_time/csv_files/route_actual_100_predicted_vs_actual_percent_config3.csv",
}

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def add_route_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["total_abs_pct_error"] = (out["pred_total_vs_actual_pct"] - 100).abs()

    phase_error_cols = []
    for phase in PHASES:
        col = f"phase_abs_error_{phase}"
        out[col] = (out[f"pred_vs_actual_pct_{phase}"] - out[f"actual_pct_{phase}"]).abs()
        phase_error_cols.append(col)

    out["phase_share_mae_pct_point"] = out[phase_error_cols].mean(axis=1)
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return (values * weights).sum() / weights.sum()


def summarize_config(config_name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing route percentage CSV for {config_name}: {path}")

    df = add_route_errors(pd.read_csv(path))



    return {
        "configuration": config_name,
        "routes": len(df),
        "flights": int(df["n_flights"].sum()),
        "weighted_total_error_pct": weighted_mean(df["total_abs_pct_error"], df["n_flights"]),
        "weighted_phase_share_error_pp": weighted_mean(df["phase_share_mae_pct_point"], df["n_flights"]),
    }


def main() -> None:
    summary = pd.DataFrame(
        summarize_config(config_name, path)
        for config_name, path in CONFIG_FILES.items()
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
 

if __name__ == "__main__":
    main()
