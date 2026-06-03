from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300
PHASES = ["takeoff", "climb", "cruise", "descent", "landing"]
PHASE_LABELS = {"takeoff": "Take-off"}


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase.capitalize())


def make_phase_panel_figure(figsize: tuple[float, float] = (10, 12)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 1:3]),
    ]
    return fig, axes


def save_figure(fig, filename: str, *, pad_inches: float = 0.15) -> Path:
    outpath = OUTPUT_DIR / filename
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return outpath
