"""EDA module for BTC spot (Binance 5-minute data).

This script:
  - Loads the main Binance BTC spot 5-minute dataset from Actual data (read-only)
  - Constructs log returns and a rolling volatility measure
  - Produces summary tables and publication-quality figures

All outputs are written ONLY under:
  EDA/Spot/

IMPORTANT: The script NEVER writes to `Actual data/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CODE_ROOT = Path(__file__).resolve().parents[3]

ACTUAL_SPOT_DIR = CODE_ROOT / "Actual data" / "Spot Binance"

OUTPUT_ROOT = CODE_ROOT / "EDA" / "Spot"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def format_time_axis_with_month_labels(ax, timestamps: pd.Series) -> None:
    """Format x-axis with day ticks (5,10,15,20,25) and month labels.

    - Major ticks at days 5, 10, 15, 20, 25.
    - Day numbers shown at those ticks.
    - One 'Jan' label centered over the January segment,
      one 'Feb' label centered over the February segment.
      No label for March.
    """
    # Major ticks: specific month days
    locator = mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25])
    formatter = mdates.DateFormatter("%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ts = pd.to_datetime(timestamps)
    if ts.empty:
        return

    # Add month labels using axis transform (below tick labels)
    months = ts.dt.to_period("M")
    for period in months.unique():
        year = period.year
        month = period.month
        # Only label January and February
        if month not in (1, 2):
            continue
        mask = months == period
        month_ts = ts[mask]
        if month_ts.empty:
            continue
        mid = month_ts.min() + (month_ts.max() - month_ts.min()) / 2
        if month == 1:
            label = f"January {year}"
        else:
            label = f"February {year}"
        ax.text(
            mdates.date2num(mid),
            -0.12,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )


def detect_spot_file() -> Path:
    """Auto-detect the main 5m spot parquet in Actual data/Spot Binance/."""
    if not ACTUAL_SPOT_DIR.exists():
        raise FileNotFoundError(f"Spot directory not found: {ACTUAL_SPOT_DIR}")

    candidates: List[Path] = sorted(ACTUAL_SPOT_DIR.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found in {ACTUAL_SPOT_DIR} for spot EDA."
        )

    # Prefer filenames that look like 5m data, fall back to first.
    for p in candidates:
        name = p.name.lower()
        if "5m" in name or "5min" in name or "5-min" in name:
            logger.info("Detected spot 5m parquet: %s", p)
            return p

    logger.info("Using first parquet file as spot dataset: %s", candidates[0])
    return candidates[0]


def load_spot_data(path: Path) -> pd.DataFrame:
    """Load spot dataset and standardize timestamp/close columns.

    We assume a typical OHLCV schema; we detect:
      - timestamp column
      - close price column
    """
    logger.info("Loading spot dataset from %s", path)
    df = pd.read_parquet(path)

    cols_lower = {c.lower(): c for c in df.columns}

    # Detect timestamp column
    ts_candidates = [
        "timestamp",
        "timestamp_utc",
        "bucket_5m_utc",
        "time",
        "datetime",
    ]
    ts_col = None
    for key in ts_candidates:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in spot dataset; "
            f"columns available: {list(df.columns)}"
        )

    # Detect close column
    close_candidates = ["close", "spot_close", "price"]
    close_col = None
    for key in close_candidates:
        if key in cols_lower:
            close_col = cols_lower[key]
            break
    if close_col is None:
        raise KeyError(
            "Could not detect close price column in spot dataset; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={ts_col: "timestamp", close_col: "spot_close"}, inplace=True)

    # Parse timestamps as UTC and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some spot timestamps as UTC.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_spot_returns_and_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns and rolling volatility.

    Spot log returns and realized volatility are key for understanding
    the distribution and time-variation of risk in the underlying market.
    """
    df = df.copy()
    df["spot_log_return"] = np.log(df["spot_close"]).diff()

    # 1-day realized volatility from 5m returns (approx 288 5-min intervals per day)
    window = 288
    df["spot_rolling_vol_1d"] = (
        df["spot_log_return"].rolling(window=window, min_periods=window)
        .std()
        .mul(np.sqrt(window))  # scale to daily volatility
    )
    return df


def summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """Compute summary stats for spot_close and spot_log_return."""
    def _describe(series: pd.Series) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p5": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }

    rows: List[Dict[str, float]] = []
    for var in ["spot_close", "spot_log_return"]:
        stats_dict = _describe(df[var])
        stats_dict["variable"] = var
        rows.append(stats_dict)

    out_df = pd.DataFrame(rows)[
        [
            "variable",
            "count",
            "mean",
            "std",
            "min",
            "p5",
            "p25",
            "median",
            "p75",
            "p95",
            "max",
        ]
    ]
    out_df.to_csv(out_path, index=False)

    # Also save a markdown version for inclusion in the thesis text.
    md_path = out_path.with_suffix(".md")
    cols = list(out_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in out_df.iterrows():
        vals = [str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved spot summary stats to %s and %s", out_path, md_path)


def plot_spot_price_timeseries(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(df["timestamp"], df["spot_close"], linewidth=0.8)
    plt.title("BTC Spot Price (Binance, 5-minute)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Spot price (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "spot_price_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spot price timeseries to %s", out_path)


def plot_spot_return_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    returns = df["spot_log_return"].dropna()
    plt.hist(returns, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of BTC Spot Log Returns (5-minute)")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "spot_return_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spot return distribution to %s", out_path)


def plot_spot_rolling_vol(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(df["timestamp"], df["spot_rolling_vol_1d"], linewidth=0.8)
    plt.title("BTC Spot Rolling 1-Day Volatility (from 5-minute returns)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Realized daily volatility")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "spot_rolling_volatility.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spot rolling volatility to %s", out_path)


def write_diagnostics(df: pd.DataFrame, spot_path: Path) -> str:
    out_path = DIAGNOSTICS_DIR / "spot_eda_diagnostics.txt"

    lines: List[str] = []
    lines.append("Spot EDA – Diagnostics")
    lines.append("========================")
    lines.append(f"Detected spot dataset path: {spot_path}")
    lines.append(f"Row count: {len(df)}")
    lines.append(
        f"Timestamp range: {df['timestamp'].min().isoformat()} to {df['timestamp'].max().isoformat()}"
    )
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Output tables:")
    lines.append(f"  - {TABLES_DIR / 'spot_summary_stats.csv'}")
    lines.append("")
    lines.append("Output figures:")
    lines.append(f"  - {FIGURES_DIR / 'spot_price_timeseries.png'}")
    lines.append(f"  - {FIGURES_DIR / 'spot_return_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'spot_rolling_volatility.png'}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote spot diagnostics to %s", out_path)
    return text


def main() -> None:
    ensure_output_dirs()

    spot_path = detect_spot_file()
    spot_df = load_spot_data(spot_path)
    spot_df = add_spot_returns_and_vol(spot_df)

    summary_table(spot_df, TABLES_DIR / "spot_summary_stats.csv")
    plot_spot_price_timeseries(spot_df)
    plot_spot_return_distribution(spot_df)
    plot_spot_rolling_vol(spot_df)

    diag_text = write_diagnostics(spot_df, spot_path)

    # Print diagnostics to console
    print()
    print(diag_text)


if __name__ == "__main__":
    main()

