"""Shared matplotlib styling and summary-figure export for notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Fixed palette for capstone figures
COLOR_ACTUAL = "black"
COLOR_TREATED = "C0"
COLOR_COUNTERFACTUAL = "C1"
COLOR_CQR = "C0"
COLOR_RAW = "C1"
COLOR_NOMINAL = "0.3"
COLOR_GP_MEAN = "C0"


def setup_style() -> None:
    """Apply consistent defaults for notebook summary figures."""
    import matplotlib

    try:
        get_ipython()  # type: ignore[name-defined]
    except NameError:
        matplotlib.use("Agg")

    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.5),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_summary_figure(fig: plt.Figure, stem: str, *, dpi: int = 150) -> Path:
    """Write ``results/{stem}.png`` and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{stem}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def bootstrap_notebook_paths() -> Path:
    """Ensure repo root is on ``sys.path`` when cwd is ``notebooks/``."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT
