"""Build descriptive plots for the thesis empirical dataset.

Loads btc_probability_comparison_empirical_master.parquet from the
\"Options N(d2) + Polymarket - Econometric Testing\" folder and produces:
- Overlaid histograms of prob_yes and prob_option
- Histogram of iv_hat
- Histogram of days_to_expiry

Figures are saved under Econometric Analysis/Descriptive Statistics/figures/,
and a short diagnostics summary is written to diagnostics/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script lives in Econometric Analysis/Descriptive Statistics/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parent  # Descriptive Statistics/
ECONOMETRIC_DIR = STATS_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

FIGURES_DIR = STATS_DIR / "figures"
DIAG_DIR = STATS_DIR / "diagnostics"
SUMMARY_TXT = DIAG_DIR / "descriptive_plots_summary.txt"

# Required columns for plots
REQUIRED_COLS = ["prob_yes", "prob_option", "iv_hat", "days_to_expiry"]

# Plot styling
FONT_SIZE = 10
TITLE_FONT_SIZE = 12


def load_dataset() -> pd.DataFrame:
    """Load the empirical master dataset and verify required columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def plot_probability_distributions(df: pd.DataFrame) -> Path:
    """Overlaid histograms for prob_yes and prob_option."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Drop NaNs separately for each series
    py = pd.to_numeric(df["prob_yes"], errors="coerce").dropna()
    po = pd.to_numeric(df["prob_option"], errors="coerce").dropna()

    # Use a common binning over [0,1]
    bins = 40
    ax.hist(py, bins=bins, range=(0, 1), alpha=0.5, label="Polymarket (prob_yes)")
    ax.hist(po, bins=bins, range=(0, 1), alpha=0.5, label="Option-implied (prob_option)")

    ax.set_xlabel("Probability", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.set_title("Probability distributions: Polymarket vs option-implied", fontsize=TITLE_FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.set_xlim(0, 1)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "probability_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_iv_distribution(df: pd.DataFrame) -> Path:
    """Histogram of iv_hat."""
    fig, ax = plt.subplots(figsize=(6, 4))
    iv = pd.to_numeric(df["iv_hat"], errors="coerce").dropna()

    ax.hist(iv, bins=50, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Implied volatility (iv_hat)", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.set_title("Distribution of implied volatility (iv_hat)", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "iv_hat_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_days_to_expiry_distribution(df: pd.DataFrame) -> Path:
    """Histogram of days_to_expiry."""
    fig, ax = plt.subplots(figsize=(6, 4))
    dte = pd.to_numeric(df["days_to_expiry"], errors="coerce").dropna()

    ax.hist(dte, bins=50, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Days to expiry", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.set_title("Distribution of days to expiry", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "days_to_expiry_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def main() -> None:
    # Load data and verify columns
    df = load_dataset()
    n_total = len(df)

    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Build plots
    figure_paths: list[Path] = []
    figure_paths.append(plot_probability_distributions(df))
    figure_paths.append(plot_iv_distribution(df))
    figure_paths.append(plot_days_to_expiry_distribution(df))

    # Diagnostics summary
    summary_lines = [
        "Descriptive plots summary",
        "=" * 50,
        f"Dataset path:          {DATA_PATH}",
        f"Total rows:            {n_total}",
        f"Number of figures:     {len(figure_paths)}",
        "",
        "Figures:",
    ]
    summary_lines.extend(f"  - {p.name}" for p in figure_paths)
    summary_txt = "\n".join(summary_lines)

    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)

    # Print same summary to console
    print("\n" + summary_txt)


if __name__ == "__main__":
    main()

