import json
import re
from pathlib import Path

import pandas as pd

from plot_style import (
    PHASES,
    make_phase_panel_figure,
    phase_label,
    save_figure,
)

BASE_DIR = Path(__file__).resolve().parents[1]

TOP_N = 15

CONFIGS = {
    "config2": {
        "title": "Config 2",
        "path": BASE_DIR / "notebooks/xgboost_physics",
        "notebook_prefix": "{phase}_physics",
    },
    "config3": {
        "title": "Config 3",
        "path": BASE_DIR / "notebooks/xgboost_weather_time",
        "notebook_prefix": "{phase}_weather_time",
    },
}


def extract_text_plain(output: dict) -> str:
    data = output.get("data", {})
    text = data.get("text/plain", "")
    if isinstance(text, list):
        return "".join(text)
    return str(text)


def parse_importance_series(text: str) -> pd.Series:
    values = {}
    for line in text.splitlines():
        if line.startswith("dtype:"):
            break

        match = re.match(r"^(.+?)\s+([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)$", line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))

    if not values:
        raise ValueError("Could not parse feature importance values from notebook output.")

    return pd.Series(values).sort_values(ascending=False)


def load_phase_importance(config: dict, phase: str) -> pd.Series:
    notebook_name = config["notebook_prefix"].format(phase=phase)
    path = config["path"] / f"{notebook_name}.ipynb"
    if not path.exists():
        raise FileNotFoundError(f"Missing notebook: {path}")

    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "imp = pd.Series(final_model.feature_importances_" not in source:
            continue

        for output in cell.get("outputs", []):
            text = extract_text_plain(output)
            if text:
                return parse_importance_series(text)

    raise ValueError(f"Could not find stored feature importance output in {path}")


def shorten_feature_name(feature: str, phase: str) -> str:
    suffix = f"_{phase}"
    if feature.endswith(suffix):
        return feature[: -len(suffix)]
    return feature


def plot_config_panel(config_name: str, config: dict) -> Path:
    fig, axes = make_phase_panel_figure(figsize=(13, 13))

    for i, phase in enumerate(PHASES):
        ax = axes[i]
        imp = load_phase_importance(config, phase).head(TOP_N)
        labels = [shorten_feature_name(feature, phase) for feature in imp.index]

        ax.barh(labels[::-1], imp.values[::-1])
        ax.set_title(f"{phase_label(phase)} model", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)

    fig.tight_layout(rect=[0.12, 0.03, 1, 0.95])
    return save_figure(fig, f"feature_importance_panel_{config_name}.png", pad_inches=0.25)


def main() -> None:
    for config_name, config in CONFIGS.items():
        outpath = plot_config_panel(config_name, config)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
