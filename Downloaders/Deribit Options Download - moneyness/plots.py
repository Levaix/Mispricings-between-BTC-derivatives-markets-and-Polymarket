"""
Inspection plots: moneyness vs date, daily instrument counts, strike coverage vs date.
"""
import logging
from pathlib import Path

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


def run(df=None):
    if df is None:
        if not cfg.RAW_LONG_PARQUET.exists():
            logger.warning("No parquet at %s", cfg.RAW_LONG_PARQUET)
            return
        df = pd.read_parquet(cfg.RAW_LONG_PARQUET)
    if df.empty:
        return

    df["ts"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
    df["date"] = df["ts"].dt.date.astype(str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Scatter: moneyness (y) vs date (x), colored by volume
    dates_sorted = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    df_plot = df.copy()
    df_plot["date_idx"] = df_plot["date"].map(date_to_idx)
    fig, ax = plt.subplots(figsize=(14, 6))
    scatter = ax.scatter(
        df_plot["date_idx"],
        df_plot["moneyness"],
        c=df_plot["volume"].clip(0, df_plot["volume"].quantile(0.99)),
        s=1,
        alpha=0.5,
        cmap="YlOrRd",
    )
    ax.set_ylabel("Moneyness (strike / index)")
    ax.set_xlabel("Date")
    ax.set_title("Moneyness vs date (colored by volume)")
    plt.colorbar(scatter, ax=ax, label="Volume")
    ax.set_ylim(cfg.MONEYNESS_MIN, cfg.MONEYNESS_MAX)
    step = max(1, len(dates_sorted) // 20)
    ax.set_xticks(range(0, len(dates_sorted), step))
    ax.set_xticklabels([dates_sorted[i] for i in range(0, len(dates_sorted), step)], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "plot_moneyness_vs_date.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Daily counts: active instruments per day
    daily = df.groupby("date").agg(
        n_instruments=("instrument_name", "nunique"),
        total_volume=("volume", "sum"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(daily)), daily["n_instruments"], color="steelblue", alpha=0.8)
    ax.set_xticks(range(0, len(daily), max(1, len(daily) // 20)))
    ax.set_xticklabels(daily["date"].iloc[:: max(1, len(daily) // 20)], rotation=45, ha="right")
    ax.set_ylabel("Number of active instruments")
    ax.set_xlabel("Date")
    ax.set_title("Daily active instruments")
    plt.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "plot_daily_instrument_count.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Strike coverage vs date: min/median/max strike per day
    strike_day = df.groupby("date")["strike"].agg(["min", "median", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(strike_day)), strike_day["min"], strike_day["max"], alpha=0.3)
    ax.plot(range(len(strike_day)), strike_day["median"], color="darkblue", label="Median strike")
    ax.set_xticks(range(0, len(strike_day), max(1, len(strike_day) // 20)))
    ax.set_xticklabels(strike_day["date"].iloc[:: max(1, len(strike_day) // 20)], rotation=45, ha="right")
    ax.set_ylabel("Strike (USD)")
    ax.set_xlabel("Date")
    ax.set_title("Strike coverage per day (min / median / max)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "plot_strike_coverage_vs_date.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", cfg.REPORTS_DIR)
