"""
Build visual diagnostics for the thesis empirical dataset.

Uses two datasets:
1. Global figures (overlap sample): btc_probability_comparison_empirical_master.parquet
   - probability_scatter.png, spread_distribution.png
2. Event-level figures (full Polymarket path): options_smile_event_probabilities.parquet
   - One PNG per (event_date, strike_K): event_{event_date}_strike_{strike_K}.png
   - Line = prob_yes (Polymarket), scatter = prob_option only where not null (no fill/interpolation)

Saves figures to figures/global/ and figures/events/, and a summary to diagnostics/.
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

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR = SCRIPT_DIR.parent  # Visual Diagnostics/
ECONOMETRIC_DIR = VIZ_DIR.parent
CODE_ROOT = ECONOMETRIC_DIR.parent

# Two datasets for two purposes
GLOBAL_DATA_PATH = (
    CODE_ROOT / "Actual data" / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)
EVENT_DATA_PATH = (
    CODE_ROOT / "Actual data" / "Options N(d2) + Polymarket"
    / "options_smile_event_probabilities.parquet"
)

FIGURES_GLOBAL = VIZ_DIR / "figures" / "global"
FIGURES_EVENTS = VIZ_DIR / "figures" / "events"
DIAG_DIR = VIZ_DIR / "diagnostics"
SUMMARY_TXT = DIAG_DIR / "visual_diagnostics_summary.txt"

# Columns
TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]
GLOBAL_REQUIRED = ["event_date", "prob_yes", "prob_option", "probability_spread"]
EVENT_REQUIRED = ["event_date", "strike_K", "prob_yes", "prob_option"]

# Plot settings
FONT_SIZE = 10
TITLE_FONT_SIZE = 12
ALPHA_SCATTER = 0.3
EVENT_FIG_SIZE = (8, 4)  # single-panel event-strike figure


def _resolve_time_column(df: pd.DataFrame) -> str:
    """Return the name of the time column (timestamp or bucket_5m_utc)."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Dataset must have one of: {TIME_COL_CANDIDATES}")


def load_global_dataset() -> pd.DataFrame:
    """Load empirical master for global figures only. Validate required columns."""
    if not GLOBAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Global dataset not found: {GLOBAL_DATA_PATH}")
    df = pd.read_parquet(GLOBAL_DATA_PATH)
    missing = [c for c in GLOBAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Global dataset missing columns: {missing}")
    return df


def load_event_dataset() -> tuple[pd.DataFrame, str]:
    """Load options_smile_event_probabilities for event-level figures. Returns (df, time_col)."""
    if not EVENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Event-level dataset not found: {EVENT_DATA_PATH}")
    df = pd.read_parquet(EVENT_DATA_PATH)
    missing = [c for c in EVENT_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Event-level dataset missing columns: {missing}")
    time_col = _resolve_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    return df, time_col


def plot_global_probability_scatter(df: pd.DataFrame) -> None:
    """Scatter: prob_option (x) vs prob_yes (y) with 45° reference line. Uses global dataset only."""
    valid = df.dropna(subset=["prob_yes", "prob_option"])
    if valid.empty:
        logger.warning("No valid prob_yes/prob_option pairs; skipping probability scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valid["prob_option"], valid["prob_yes"], alpha=ALPHA_SCATTER, s=8, label="Observations")
    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="45° reference")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Option-implied probability (prob_option)", fontsize=FONT_SIZE)
    ax.set_ylabel("Polymarket probability (prob_yes)", fontsize=FONT_SIZE)
    ax.set_title("Probability comparison: Polymarket vs option-implied", fontsize=TITLE_FONT_SIZE)
    ax.legend(loc="upper left", fontsize=FONT_SIZE)
    ax.set_aspect("equal")
    plt.tight_layout()
    FIGURES_GLOBAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_GLOBAL / "probability_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIGURES_GLOBAL / "probability_scatter.png")


def plot_global_spread_distribution(df: pd.DataFrame) -> None:
    """Histogram of probability_spread with vertical line at 0. Uses global dataset only."""
    spread = df["probability_spread"].dropna()
    if spread.empty:
        logger.warning("No valid probability_spread; skipping spread distribution.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spread, bins=50, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero spread")
    ax.set_xlabel("Probability spread (prob_yes − prob_option)", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.set_title("Distribution of probability spread", fontsize=TITLE_FONT_SIZE)
    ax.legend(loc="upper right", fontsize=FONT_SIZE)
    plt.tight_layout()
    FIGURES_GLOBAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_GLOBAL / "spread_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIGURES_GLOBAL / "spread_distribution.png")


def plot_event_strike_timeseries(
    df: pd.DataFrame,
    event_date: str,
    strike_k: float | int,
    time_col: str,
) -> bool:
    """
    One figure per (event_date, strike_K): single panel, line = prob_yes, scatter = prob_option
    only where not null. No interpolation, no forward/backward fill.
    Saves to event_{event_date}_strike_{strike_K}.png. Returns True if saved.
    """
    panel = df[
        (df["event_date"].astype(str) == event_date)
        & (pd.to_numeric(df["strike_K"], errors="coerce") == float(strike_k))
    ].sort_values(time_col)
    if panel.empty:
        return False

    fig, ax = plt.subplots(figsize=EVENT_FIG_SIZE)
    # Line: prob_yes (full Polymarket path for this event-strike)
    ax.plot(
        panel[time_col],
        panel["prob_yes"],
        color="C0",
        label="Polymarket probability",
        linewidth=1,
    )
    # Scatter: prob_option only where not null (gaps left where missing)
    valid = panel.dropna(subset=["prob_option"])
    if not valid.empty:
        ax.scatter(
            valid[time_col],
            valid["prob_option"],
            color="C1",
            s=14,
            alpha=0.8,
            label="Option-implied probability",
            zorder=5,
        )
    ax.set_xlabel("Time (UTC)", fontsize=FONT_SIZE)
    ax.set_ylabel("Probability", fontsize=FONT_SIZE)
    strike_int = int(float(strike_k))
    ax.set_title(f"Event {event_date} — Strike {strike_int:,}", fontsize=TITLE_FONT_SIZE)
    ax.legend(loc="best", fontsize=FONT_SIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    fig.autofmt_xdate()
    plt.tight_layout()
    FIGURES_EVENTS.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_EVENTS / f"event_{event_date}_strike_{strike_int}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return True


def main() -> None:
    # -------------------------------------------------------------------------
    # Load global dataset (empirical master) for global figures only
    # -------------------------------------------------------------------------
    df_global = load_global_dataset()
    n_global_total = len(df_global)
    n_global_with_prob = df_global["prob_option"].notna().sum()

    FIGURES_GLOBAL.mkdir(parents=True, exist_ok=True)
    plot_global_probability_scatter(df_global)
    plot_global_spread_distribution(df_global)

    # -------------------------------------------------------------------------
    # Load event-level dataset (full probability dataset) for event figures
    # -------------------------------------------------------------------------
    df_event, time_col = load_event_dataset()
    n_event_total = len(df_event)
    n_event_with_prob = df_event["prob_option"].notna().sum()
    event_dates = sorted(df_event["event_date"].astype(str).unique())
    unique_strikes = sorted(pd.to_numeric(df_event["strike_K"], errors="coerce").dropna().unique().tolist())

    # One PNG per (event_date, strike_K)
    n_event_strike_figs = 0
    for ed in event_dates:
        strikes_in_event = sorted(
            pd.to_numeric(df_event.loc[df_event["event_date"].astype(str) == ed, "strike_K"], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        for strike in strikes_in_event:
            if plot_event_strike_timeseries(df_event, ed, strike, time_col):
                n_event_strike_figs += 1

    # -------------------------------------------------------------------------
    # Diagnostics summary
    # -------------------------------------------------------------------------
    n_global_figs = 2
    n_figures_total = n_global_figs + n_event_strike_figs

    summary_lines = [
        "Visual diagnostics summary",
        "=" * 60,
        "",
        "Global figures (overlap sample):",
        f"  Dataset path:           {GLOBAL_DATA_PATH}",
        f"  Total rows:             {n_global_total}",
        f"  Rows with prob_option:  {n_global_with_prob}",
        "",
        "Event-level figures (full Polymarket path):",
        f"  Dataset path:           {EVENT_DATA_PATH}",
        f"  Total rows:             {n_event_total}",
        f"  Rows with prob_option:  {n_event_with_prob}",
        f"  Unique event_date:      {len(event_dates)}",
        f"  Unique strikes:         {len(unique_strikes)}",
        "",
        "Figures:",
        f"  Event-strike figures generated:  {n_event_strike_figs}",
        f"  Total figures (global + events): {n_figures_total}",
    ]
    summary_txt = "\n".join(summary_lines)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)

    print("\n" + summary_txt)


if __name__ == "__main__":
    main()
