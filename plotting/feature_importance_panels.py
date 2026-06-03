import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PHASES = ["takeoff", "climb", "cruise", "descent", "landing"]
PHASE_LABELS = {"takeoff": "Take-off"}
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
    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 1:3]),
    ]

    for i, phase in enumerate(PHASES):
        ax = axes[i]
        imp = load_phase_importance(config, phase).head(TOP_N)
        labels = [shorten_feature_name(feature, phase) for feature in imp.index]

        ax.barh(labels[::-1], imp.values[::-1])
        phase_label = PHASE_LABELS.get(phase, phase.capitalize())
        ax.set_title(f"{phase_label} model", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)

    fig.tight_layout(rect=[0.12, 0.03, 1, 0.95])

    outpath = OUTPUT_DIR / f"feature_importance_panel_{config_name}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return outpath


def main() -> None:
    for config_name, config in CONFIGS.items():
        outpath = plot_config_panel(config_name, config)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
