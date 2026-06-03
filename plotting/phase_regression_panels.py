from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import (
    PHASES,
    make_phase_panel_figure,
    phase_label,
    save_figure,
)

BASE_DIR = Path(__file__).resolve().parents[1]


CONFIGS = {
    "config2": {
        "title": "Config 2",
        "path": BASE_DIR / "notebooks/xgboost_physics/csv_files",
        "true_prefix": "true_physics",
        "pred_prefix": "pred_physics",
        "file_prefix": "pred_physics",
    },
    "config3": {
        "title": "Config 3",
        "path": BASE_DIR / "notebooks/xgboost_weather_time/csv_files",
        "true_prefix": "true_physics_weather",
        "pred_prefix": "pred_physics_weather",
        "file_prefix": "pred_physics_weather",
    },
}


def load_phase_predictions(config: dict, phase: str) -> pd.DataFrame:
    path = config["path"] / f"{config['file_prefix']}_{phase}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    true_col = f"{config['true_prefix']}_{phase}"
    pred_col = f"{config['pred_prefix']}_{phase}"
    df = pd.read_csv(path)

    missing = [col for col in [true_col, pred_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    out = df[[true_col, pred_col]].rename(
        columns={true_col: "actual", pred_col: "predicted"}
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["actual"] > 0) & (out["predicted"] > 0)].copy()
    return out


def plot_config_panel(config_name: str, config: dict) -> Path:
    fig, axes = make_phase_panel_figure(figsize=(10, 12))

    for i, phase in enumerate(PHASES):
        ax = axes[i]
        df = load_phase_predictions(config, phase)

        y_test = df["actual"].to_numpy()
        y_pred = df["predicted"].to_numpy()

        ax.scatter(y_test, y_pred, alpha=0.3)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect prediction",
        )

        m, b = np.polyfit(y_test, y_pred, 1)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, m * x_line + b, label="Regression line")

        model_name = f"{phase_label(phase)} model"
        ax.set_title(f"{model_name}")
        ax.legend(fontsize=12)

    fig.supxlabel("Actual fuel (kg)")
    fig.supylabel("Predicted fuel (kg)")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    return save_figure(fig, f"phase_regression_panel_{config_name}.png")


def main() -> None:
    for config_name, config in CONFIGS.items():
        outpath = plot_config_panel(config_name, config)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
