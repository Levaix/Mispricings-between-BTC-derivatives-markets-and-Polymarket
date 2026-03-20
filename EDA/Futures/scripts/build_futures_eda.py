"""EDA module for BTC perpetual futures (Binance, 5-minute) and funding.

This script:
  - Loads 5-minute perpetual futures OHLCV, mark prices, and funding rates
    from Actual data (read-only)
  - Constructs futures and mark log returns and cumulative funding
  - Optionally merges with spot to analyze basis when spot data is available
  - Produces summary tables and publication-quality figures

All outputs are written ONLY under:
  EDA/Futures/

IMPORTANT: The script NEVER writes to `Actual data/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CODE_ROOT = Path(__file__).resolve().parents[3]

FUTURES_OHLCV_PATH = (
    CODE_ROOT / "Actual data" / "Futures Binance" / "perp_ohlcv_5m.parquet"
)
MARK_PATH = CODE_ROOT / "Actual data" / "Futures Binance" / "perp_mark_5m.parquet"
FUNDING_PATH = CODE_ROOT / "Actual data" / "Futures Binance" / "funding_rate.parquet"

SPOT_DIR = CODE_ROOT / "Actual data" / "Spot Binance"

OUTPUT_ROOT = CODE_ROOT / "EDA" / "Futures"
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
    locator = mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25])
    formatter = mdates.DateFormatter("%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ts = pd.to_datetime(timestamps)
    if ts.empty:
        return

    months = ts.dt.to_period("M")
    for period in months.unique():
        year = period.year
        month = period.month
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


def load_futures_ohlcv() -> pd.DataFrame:
    """Load perpetual futures OHLCV and standardize timestamp/close columns."""
    logger.info("Loading futures OHLCV from %s", FUTURES_OHLCV_PATH)
    df = pd.read_parquet(FUTURES_OHLCV_PATH)
    cols_lower = {c.lower(): c for c in df.columns}

    ts_col = None
    for key in ["timestamp_utc", "timestamp", "time", "bucket_5m_utc"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in futures OHLCV; "
            f"columns available: {list(df.columns)}"
        )

    close_col = None
    for key in ["close", "futures_close"]:
        if key in cols_lower:
            close_col = cols_lower[key]
            break
    if close_col is None:
        raise KeyError(
            "Could not detect close column in futures OHLCV; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={ts_col: "timestamp", close_col: "close"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some futures OHLCV timestamps as UTC.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_mark() -> pd.DataFrame:
    """Load mark price dataset and standardize columns."""
    logger.info("Loading mark prices from %s", MARK_PATH)
    df = pd.read_parquet(MARK_PATH)
    cols_lower = {c.lower(): c for c in df.columns}

    ts_col = None
    for key in ["timestamp_utc", "timestamp", "time", "bucket_5m_utc"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in mark dataset; "
            f"columns available: {list(df.columns)}"
        )

    mark_col = None
    for key in ["mark_close", "mark_price", "close", "price"]:
        if key in cols_lower:
            mark_col = cols_lower[key]
            break
    if mark_col is None:
        raise KeyError(
            "Could not detect mark price column in mark dataset; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={ts_col: "timestamp", mark_col: "mark_close"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some mark timestamps as UTC.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_funding() -> pd.DataFrame:
    """Load funding rate dataset and standardize columns.

    Funding rates are informative because they measure the cost of holding
    perpetual futures relative to spot and reflect the balance of long vs short
    pressure in the derivatives market.
    """
    logger.info("Loading funding rates from %s", FUNDING_PATH)
    df = pd.read_parquet(FUNDING_PATH)
    cols_lower = {c.lower(): c for c in df.columns}

    ts_col = None
    for key in ["timestamp_utc", "timestamp", "time"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in funding dataset; "
            f"columns available: {list(df.columns)}"
        )

    fr_col = None
    for key in ["fundingrate", "funding_rate", "funding"]:
        if key in cols_lower:
            fr_col = cols_lower[key]
            break
    if fr_col is None:
        raise KeyError(
            "Could not detect funding rate column; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={ts_col: "timestamp", fr_col: "fundingRate"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some funding timestamps as UTC.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def detect_spot_file() -> Path | None:
    """Try to detect a spot 5m parquet to compute basis; returns None if not found."""
    try:
        if not SPOT_DIR.exists():
            return None
        candidates: List[Path] = sorted(SPOT_DIR.glob("*.parquet"))
        if not candidates:
            return None
        for p in candidates:
            name = p.name.lower()
            if "5m" in name or "5min" in name or "5-min" in name:
                return p
        return candidates[0]
    except Exception:
        return None


def load_spot_for_basis(path: Path) -> pd.DataFrame:
    """Load and standardize spot close for basis analysis."""
    df = pd.read_parquet(path)
    cols_lower = {c.lower(): c for c in df.columns}

    ts_col = None
    for key in ["timestamp_utc", "timestamp", "bucket_5m_utc", "time", "datetime"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in spot dataset; "
            f"columns available: {list(df.columns)}"
        )

    close_col = None
    for key in ["close", "spot_close", "price"]:
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some spot timestamps as UTC.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_futures_variables(
    fut_df: pd.DataFrame, mark_df: pd.DataFrame, funding_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add log returns and cumulative funding.

    Futures log returns help understand derivatives market dynamics,
    while fundingRate and its cumulative sum indicate persistent imbalances
    between long and short positioning.
    """
    fut = fut_df.copy()
    fut["futures_log_return"] = np.log(fut["close"]).diff()

    mark = mark_df.copy()
    mark["mark_log_return"] = np.log(mark["mark_close"]).diff()

    funding = funding_df.copy()
    funding["cumulative_funding"] = funding["fundingRate"].cumsum()

    return fut, mark, funding


def merge_spot_futures_for_basis(
    fut_df: pd.DataFrame, spot_df: pd.DataFrame
) -> pd.DataFrame:
    """Inner-merge spot and futures on timestamp and compute basis variables.

    Basis is informative because it measures the wedge between futures and spot,
    reflecting leverage, funding, and risk premia in derivatives relative to
    the underlying market.
    """
    merged = pd.merge(fut_df[["timestamp", "close"]], spot_df, on="timestamp", how="inner")
    merged.rename(columns={"close": "futures_close"}, inplace=True)
    merged["basis_abs"] = merged["futures_close"] - merged["spot_close"]
    merged["basis_rel"] = merged["futures_close"] / merged["spot_close"] - 1.0
    return merged


def summary_table(
    df: pd.DataFrame, variables: List[str], out_path: Path
) -> None:
    """Compute summary stats for selected variables."""
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
    for var in variables:
        if var not in df.columns:
            continue
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

    # Also save markdown version for thesis-ready inclusion.
    md_path = out_path.with_suffix(".md")
    cols = list(out_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in out_df.iterrows():
        vals = [str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved futures summary stats to %s and %s", out_path, md_path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_futures_and_mark_timeseries(fut_df: pd.DataFrame, mark_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(fut_df["timestamp"], fut_df["close"], label="Futures close", linewidth=0.8)
    plt.plot(mark_df["timestamp"], mark_df["mark_close"], label="Mark price", linewidth=0.8, alpha=0.8)
    plt.title("BTC Perpetual Futures and Mark Price (Binance, 5-minute)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, fut_df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "futures_and_mark_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved futures and mark timeseries to %s", out_path)


def plot_funding_timeseries(funding_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(funding_df["timestamp"], funding_df["fundingRate"], linewidth=0.8)
    plt.title("Funding Rate Over Time")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, funding_df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Funding rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "funding_rate_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved funding rate timeseries to %s", out_path)


def plot_funding_distribution(funding_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = funding_df["fundingRate"].dropna()
    plt.hist(vals, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of Funding Rates")
    plt.xlabel("Funding rate")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "funding_rate_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved funding rate distribution to %s", out_path)


def plot_cumulative_funding(funding_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(funding_df["timestamp"], funding_df["cumulative_funding"], linewidth=0.8)
    plt.title("Cumulative Funding Over Time")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, funding_df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Cumulative funding")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "cumulative_funding.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved cumulative funding timeseries to %s", out_path)


def plot_futures_return_distribution(fut_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = fut_df["futures_log_return"].dropna()
    plt.hist(vals, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of Futures Log Returns (5-minute)")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "futures_return_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved futures return distribution to %s", out_path)


def plot_basis_figures(basis_df: pd.DataFrame) -> None:
    # Spot vs futures price overlay
    plt.figure(figsize=(9, 4))
    plt.plot(basis_df["timestamp"], basis_df["spot_close"], label="Spot", linewidth=0.8)
    plt.plot(
        basis_df["timestamp"],
        basis_df["futures_close"],
        label="Futures",
        linewidth=0.8,
        alpha=0.8,
    )
    plt.title("BTC Spot vs Perpetual Futures Price")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, basis_df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "spot_vs_futures_price_overlay.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spot vs futures price overlay to %s", out_path)

    # Basis time series
    plt.figure(figsize=(9, 4))
    plt.plot(basis_df["timestamp"], basis_df["basis_abs"], linewidth=0.8)
    plt.title("Absolute Basis Over Time (Futures - Spot)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, basis_df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Basis (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "basis_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved basis timeseries to %s", out_path)

    # Basis distribution
    plt.figure(figsize=(6, 4))
    vals = basis_df["basis_abs"].dropna()
    plt.hist(vals, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of Absolute Basis (Futures - Spot)")
    plt.xlabel("Basis (USD)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "basis_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved basis distribution to %s", out_path)

    # Scatter of spot vs futures returns
    plt.figure(figsize=(6, 4))
    # Construct returns from merged series
    spot_ret = np.log(basis_df["spot_close"]).diff()
    fut_ret = np.log(basis_df["futures_close"]).diff()
    mask = spot_ret.notna() & fut_ret.notna()
    plt.scatter(spot_ret[mask], fut_ret[mask], s=5, alpha=0.5)
    plt.title("Spot vs Futures Log Returns (5-minute)")
    plt.xlabel("Spot log return")
    plt.ylabel("Futures log return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "spot_vs_futures_returns_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spot vs futures returns scatter to %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def write_diagnostics(
    fut_df: pd.DataFrame,
    mark_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    spot_loaded: bool,
    spot_path: Path | None,
    basis_df: pd.DataFrame | None,
) -> str:
    out_path = DIAGNOSTICS_DIR / "futures_eda_diagnostics.txt"

    lines: List[str] = []
    lines.append("Futures EDA – Diagnostics")
    lines.append("==========================")
    lines.append(f"Futures OHLCV path: {FUTURES_OHLCV_PATH}")
    lines.append(f"Mark dataset path:  {MARK_PATH}")
    lines.append(f"Funding dataset path: {FUNDING_PATH}")
    lines.append("")
    lines.append(f"Futures OHLCV rows: {len(fut_df)}")
    lines.append(f"Mark rows:          {len(mark_df)}")
    lines.append(f"Funding rows:       {len(funding_df)}")
    lines.append("")
    lines.append("Timestamp ranges:")
    lines.append(
        f"  Futures: {fut_df['timestamp'].min().isoformat()} to {fut_df['timestamp'].max().isoformat()}"
    )
    lines.append(
        f"  Mark:    {mark_df['timestamp'].min().isoformat()} to {mark_df['timestamp'].max().isoformat()}"
    )
    lines.append(
        f"  Funding: {funding_df['timestamp'].min().isoformat()} to {funding_df['timestamp'].max().isoformat()}"
    )
    lines.append("")
    lines.append(f"Spot dataset loaded for basis: {spot_loaded}")
    if spot_loaded and spot_path is not None:
        lines.append(f"  Spot path: {spot_path}")
    if basis_df is not None:
        lines.append(f"Merged spot/futures rows (basis): {len(basis_df)}")
        lines.append(
            f"Basis timestamp range: {basis_df['timestamp'].min().isoformat()} to {basis_df['timestamp'].max().isoformat()}"
        )
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Output tables:")
    lines.append(f"  - {TABLES_DIR / 'futures_summary_stats.csv'}")
    lines.append(f"  - {TABLES_DIR / 'futures_basis_summary_stats.csv'} (if basis available)")
    lines.append("")
    lines.append("Output figures:")
    lines.append(f"  - {FIGURES_DIR / 'futures_and_mark_timeseries.png'}")
    lines.append(f"  - {FIGURES_DIR / 'funding_rate_timeseries.png'}")
    lines.append(f"  - {FIGURES_DIR / 'funding_rate_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'cumulative_funding.png'}")
    lines.append(f"  - {FIGURES_DIR / 'futures_return_distribution.png'}")
    lines.append(
        f"  - {FIGURES_DIR / 'spot_vs_futures_price_overlay.png'} (if basis available)"
    )
    lines.append(f"  - {FIGURES_DIR / 'basis_timeseries.png'} (if basis available)")
    lines.append(f"  - {FIGURES_DIR / 'basis_distribution.png'} (if basis available)")
    lines.append(
        f"  - {FIGURES_DIR / 'spot_vs_futures_returns_scatter.png'} (if basis available)"
    )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote futures diagnostics to %s", out_path)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    fut_df = load_futures_ohlcv()
    mark_df = load_mark()
    funding_df = load_funding()

    fut_df, mark_df, funding_df = add_futures_variables(fut_df, mark_df, funding_df)

    # Summary tables
    main_vars = [
        "close",
        "futures_log_return",
        "mark_close",
        "mark_log_return",
        "fundingRate",
        "cumulative_funding",
    ]
    summary_table(
        fut_df.join(mark_df.set_index("timestamp"), on="timestamp", rsuffix="_mark")
        .join(funding_df.set_index("timestamp"), on="timestamp", rsuffix="_fund"),
        variables=main_vars,
        out_path=TABLES_DIR / "futures_summary_stats.csv",
    )

    # Figures (core futures-only)
    plot_futures_and_mark_timeseries(fut_df, mark_df)
    plot_funding_timeseries(funding_df)
    plot_funding_distribution(funding_df)
    plot_cumulative_funding(funding_df)
    plot_futures_return_distribution(fut_df)

    # Optional basis analysis
    basis_df: pd.DataFrame | None = None
    spot_loaded = False
    spot_path: Path | None = None
    try:
        spot_path = detect_spot_file()
        if spot_path is not None:
            spot_df = load_spot_for_basis(spot_path)
            basis_df = merge_spot_futures_for_basis(fut_df, spot_df)
            spot_loaded = True

            # Basis summary table
            basis_vars = ["basis_abs", "basis_rel"]
            summary_table(
                basis_df, variables=basis_vars, out_path=TABLES_DIR / "futures_basis_summary_stats.csv"
            )

            # Basis figures
            plot_basis_figures(basis_df)
        else:
            logger.warning("No spot parquet detected; skipping basis analysis.")
    except Exception as exc:
        logger.warning("Basis analysis skipped due to error: %s", exc)
        basis_df = None
        spot_loaded = False

    diag_text = write_diagnostics(
        fut_df, mark_df, funding_df, spot_loaded, spot_path, basis_df
    )

    # Print diagnostics
    print()
    print(diag_text)


if __name__ == "__main__":
    main()

