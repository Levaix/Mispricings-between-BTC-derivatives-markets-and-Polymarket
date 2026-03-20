"""Top-3 most liquid instruments per expiry: daily volume evolution plot.

Loads the aligned options panel (options_long_5m.parquet), computes daily
volume per instrument and expiry, selects the three most liquid instruments
per expiry (by total volume over the window), and writes:

- CSV: reports/options_v2_qa/top3_daily_volume_by_expiry.csv
- PNG: reports/options_v2_qa/top3_daily_volume_per_expiry.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


def run() -> None:
    if not cfg.RAW_LONG_PARQUET.exists():
        logger.warning("Aligned options parquet not found at %s", cfg.RAW_LONG_PARQUET)
        return

    df = pd.read_parquet(cfg.RAW_LONG_PARQUET)
    if df.empty:
        logger.warning("Aligned options parquet is empty")
        return

    # Daily volume per instrument per expiry
    ts = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
    df["date"] = ts.dt.date.astype(str)

    daily = (
        df.groupby(["expiry_utc", "instrument_name", "date"])["volume"]
        .sum()
        .reset_index(name="daily_volume")
    )

    # Total volume per instrument within the window
    totals = (
        daily.groupby(["expiry_utc", "instrument_name"])["daily_volume"]
        .sum()
        .reset_index(name="total_volume")
    )

    # Pick top-3 instruments per expiry by total_volume
    totals = totals.sort_values(["expiry_utc", "total_volume"], ascending=[True, False])
    top3_rows = totals.groupby("expiry_utc").head(3)
    top_keys = set(zip(top3_rows["expiry_utc"], top3_rows["instrument_name"]))

    mask = list(zip(daily["expiry_utc"], daily["instrument_name"]))
    daily_top = daily[[k in top_keys for k in mask]].copy()

    # Save CSV for inspection
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.REPORTS_DIR / "top3_daily_volume_by_expiry.csv"
    daily_top.to_csv(out_csv, index=False)

    # Plot daily volume evolution
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    # Convert expiry_utc to date string for labels
    daily_top["expiry_date"] = pd.to_datetime(
        daily_top["expiry_utc"], unit="ms", utc=True
    ).dt.date.astype(str)

    # Sort by date for plotting
    daily_top = daily_top.sort_values(["date", "expiry_utc", "instrument_name"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for (exp, inst), grp in daily_top.groupby(["expiry_date", "instrument_name"]):
        ax.plot(
            grp["date"],
            grp["daily_volume"],
            marker="o",
            markersize=3,
            label=f"{inst} (exp {exp})",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily volume")
    ax.set_title("Top-3 most liquid instruments per expiry: daily volume")
    ax.tick_params(axis="x", rotation=45)

    # Avoid huge legends; limit to first ~20 entries
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 20:
        ax.legend(handles[:20], labels[:20], fontsize=7)
    else:
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_png = cfg.REPORTS_DIR / "top3_daily_volume_per_expiry.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote top-3 daily volume CSV to %s and plot to %s", out_csv, out_png)


if __name__ == "__main__":
    run()

