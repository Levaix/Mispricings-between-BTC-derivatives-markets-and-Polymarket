"""Cross-market Exploratory Data Analysis (EDA) module.

This script provides a purely exploratory, thesis-ready overview of how
Polymarket probabilities, option-implied probabilities, BTC spot prices,
and futures returns interact around the same underlying Bitcoin market.

The focus is on:
  - documenting the structure of the empirical master dataset that links
    Polymarket and option-implied probabilities;
  - adding BTC spot and futures returns on a common 5-minute grid; and
  - visualising cross-market relationships (scatter plots, distributions,
    and simple time-series examples).

The module is explicitly NON-econometric: it does *not* run regressions
or formal statistical tests. Instead, it prepares the ground for later
econometric work by showing how key variables behave and co-move.

All inputs are read-only from `Actual data/` or pipeline outputs.
All outputs are written ONLY under:

    EDA/Cross Market/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and folder structure
# ---------------------------------------------------------------------------

CODE_ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

SPOT_PATH = CODE_ROOT / "Actual data" / "Spot Binance" / "spot_5m.parquet"
FUTURES_PATH = (
    CODE_ROOT / "Actual data" / "Futures Binance" / "perp_ohlcv_5m.parquet"
)

EVENTS_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Manual News Event Identification"
    / "outputs"
    / "btc_news_events_final.csv"
)

OUTPUT_ROOT = CODE_ROOT / "EDA" / "Cross Market"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"


def ensure_output_dirs() -> None:
    """Create EDA/Cross Market directories if they do not yet exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------

def _detect_timestamp_column(cols: Sequence[str]) -> str:
    """Detect a plausible timestamp column name."""
    # Include variations seen across datasets (e.g. spot_5m uses timestamp_utc)
    candidates = ["timestamp", "timestamp_utc", "bucket_5m_utc", "time", "datetime"]
    lowered = {c.lower(): c for c in cols}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    raise KeyError(
        "Could not detect timestamp column; available columns are: "
        f"{list(cols)}"
    )


def _detect_close_column(cols: Sequence[str]) -> str:
    """Detect a plausible close price column name."""
    candidates = ["close", "price", "close_price"]
    lowered = {c.lower(): c for c in cols}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    raise KeyError(
        "Could not detect close price column; available columns are: "
        f"{list(cols)}"
    )


def load_master() -> pd.DataFrame:
    """Load the empirical probability comparison master dataset.

    The master dataset is expected to already contain both Polymarket
    probabilities and option-implied probabilities, as well as basic
    option characteristics such as moneyness and maturity.
    """
    logger.info("Loading master probability comparison dataset from %s", MASTER_PATH)
    df = pd.read_parquet(MASTER_PATH)

    ts_col = _detect_timestamp_column(df.columns)
    df = df.copy()
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Construct probability changes and spreads
    if "prob_yes" not in df.columns:
        raise KeyError(
            "Master dataset is missing 'prob_yes' (Polymarket probability)."
        )
    if "prob_option" not in df.columns:
        raise KeyError(
            "Master dataset is missing 'prob_option' (option-implied probability)."
        )
    if "probability_spread" not in df.columns:
        df["probability_spread"] = df["prob_yes"] - df["prob_option"]

    df["d_prob_yes"] = df["prob_yes"].diff()
    df["d_prob_option"] = df["prob_option"].diff()
    df["abs_spread"] = df["probability_spread"].abs()

    return df


def load_spot_returns() -> pd.DataFrame:
    """Load spot 5-minute data and compute log returns."""
    logger.info("Loading BTC spot data from %s", SPOT_PATH)
    df = pd.read_parquet(SPOT_PATH)
    ts_col = _detect_timestamp_column(df.columns)
    close_col = _detect_close_column(df.columns)

    df = df[[ts_col, close_col]].copy()
    df.rename(columns={ts_col: "timestamp", close_col: "spot_close"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df["spot_return"] = np.log(df["spot_close"]).diff()
    return df[["timestamp", "spot_close", "spot_return"]]


def load_futures_returns() -> pd.DataFrame:
    """Load futures 5-minute data and compute log returns."""
    logger.info("Loading BTC perpetual futures data from %s", FUTURES_PATH)
    df = pd.read_parquet(FUTURES_PATH)
    ts_col = _detect_timestamp_column(df.columns)
    close_col = _detect_close_column(df.columns)

    df = df[[ts_col, close_col]].copy()
    df.rename(columns={ts_col: "timestamp", close_col: "futures_close"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df["futures_return"] = np.log(df["futures_close"]).diff()
    return df[["timestamp", "futures_close", "futures_return"]]


def load_news_events() -> pd.DataFrame:
    """Load manually curated news events (for overlays only)."""
    logger.info("Loading news events from %s", EVENTS_PATH)
    events = pd.read_csv(EVENTS_PATH)
    if "event_timestamp_utc" not in events.columns:
        raise KeyError(
            "Expected column 'event_timestamp_utc' not found in news events dataset."
        )
    events["event_timestamp_utc"] = pd.to_datetime(
        events["event_timestamp_utc"], utc=True, errors="raise"
    )
    return events


def merge_cross_market() -> Dict[str, pd.DataFrame]:
    """Load all components and build a merged cross-market dataset."""
    master = load_master()
    spot = load_spot_returns()
    futures = load_futures_returns()
    events = load_news_events()

    df = (
        master.merge(spot, on="timestamp", how="left")
        .merge(futures, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    logger.info(
        "Merged cross-market dataset: %d rows (master=%d, spot=%d, futures=%d)",
        len(df),
        len(master),
        len(spot),
        len(futures),
    )
    return {"cross": df, "events": events}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _summary_with_percentiles(series: pd.Series) -> pd.Series:
    """Return summary statistics with selected percentiles."""
    s = series.dropna()
    if s.empty:
        return pd.Series(
            {
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p5": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
        )
    return pd.Series(
        {
            "count": s.count(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "p5": s.quantile(0.05),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
            "max": s.max(),
        }
    )


def save_table_with_markdown(
    df: pd.DataFrame, csv_path: Path, md_path: Path, index: bool = True
) -> None:
    """Save a table both as CSV and as a simple markdown file."""
    df.to_csv(csv_path, index=index)
    try:
        md_text = df.to_markdown(index=index)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to render markdown for %s: %s", csv_path.name, exc)
        md_text = df.to_csv(index=index)
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Saved table to %s and %s", csv_path, md_path)


def build_cross_market_summary(df: pd.DataFrame) -> Path:
    """Summary statistics for key cross-market variables."""
    vars_of_interest = [
        "prob_yes",
        "prob_option",
        "probability_spread",
        "abs_spread",
        "spot_return",
        "futures_return",
        "iv_hat",
        "moneyness",
        "tau_years",
    ]
    summary = {}
    for v in vars_of_interest:
        if v not in df.columns:
            logger.warning("Variable %s not found in merged dataset; filling with NaN.", v)
            summary[v] = _summary_with_percentiles(pd.Series(dtype=float))
        else:
            summary[v] = _summary_with_percentiles(df[v])

    out = pd.DataFrame(summary).T
    csv_path = TABLES_DIR / "cross_market_summary.csv"
    md_path = TABLES_DIR / "cross_market_summary.md"
    save_table_with_markdown(out, csv_path, md_path)
    return csv_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_polymarket_vs_option(df: pd.DataFrame) -> Path:
    """Scatter plot: option-implied vs Polymarket probabilities.

    This visualises how closely the two probability measures align.
    A tight cloud around the 45-degree line indicates agreement,
    whereas systematic deviations highlight disagreement.
    """
    plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    ax.scatter(
        df["prob_yes"],
        df["prob_option"],
        alpha=0.4,
        s=10,
        color="tab:blue",
        edgecolor="none",
    )

    # 45-degree reference line
    both = pd.concat([df["prob_yes"], df["prob_option"]]).dropna()
    if not both.empty:
        lo, hi = both.min(), both.max()
        line = np.linspace(lo, hi, 100)
        ax.plot(line, line, "k--", linewidth=1.0, label="45° line")

    ax.set_xlabel("Polymarket probability (prob_yes)")
    ax.set_ylabel("Option-implied probability (prob_option)")
    ax.set_title("Polymarket vs Option-Implied Probabilities")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "polymarket_vs_option_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved polymarket vs option scatter to %s", out_path)
    return out_path


def plot_spread_distributions(df: pd.DataFrame) -> List[Path]:
    """Histograms of raw and absolute spreads."""
    paths: List[Path] = []

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(df["probability_spread"].dropna(), bins=60, color="tab:blue", alpha=0.8)
    ax.set_title("Distribution of Probability Spread")
    ax.set_xlabel("probability_spread (Polymarket - option)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out1 = FIGURES_DIR / "spread_distribution.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    paths.append(out1)

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(df["abs_spread"].dropna(), bins=60, color="tab:orange", alpha=0.8)
    ax.set_title("Distribution of Absolute Probability Spread")
    ax.set_xlabel("abs_spread = |probability_spread|")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = FIGURES_DIR / "absolute_spread_distribution.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    paths.append(out2)

    logger.info("Saved spread distribution figures to %s and %s", out1, out2)
    return paths


def plot_spread_vs_moneyness(df: pd.DataFrame) -> Path:
    """Scatter: abs_spread vs moneyness.

    Spreads may depend on moneyness because deep in- or out-of-the-money
    options carry different liquidity and information content than
    near-the-money options.
    """
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    sub = df[["moneyness", "abs_spread"]].dropna()
    ax.scatter(sub["moneyness"], sub["abs_spread"], s=10, alpha=0.4, color="tab:green")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Absolute probability spread (abs_spread)")
    ax.set_title("Absolute Spread vs Moneyness")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "spread_vs_moneyness.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spread vs moneyness to %s", out_path)
    return out_path


def plot_spread_vs_maturity(df: pd.DataFrame) -> Path:
    """Scatter: abs_spread vs time-to-maturity.

    Spreads can also vary with maturity: short-dated contracts react more
    to short-term information, whereas long-dated contracts embed longer-
    horizon views about Bitcoin.
    """
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    sub = df[["tau_years", "abs_spread"]].dropna()
    ax.scatter(sub["tau_years"], sub["abs_spread"], s=10, alpha=0.4, color="tab:red")
    ax.set_xlabel("Time to maturity (tau_years)")
    ax.set_ylabel("Absolute probability spread (abs_spread)")
    ax.set_title("Absolute Spread vs Maturity")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "spread_vs_maturity.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved spread vs maturity to %s", out_path)
    return out_path


def plot_return_vs_prob_changes(
    df: pd.DataFrame,
    ret_col: str,
    label_prefix: str,
    out_name: str,
) -> Path:
    """Helper to plot returns vs probability changes.

    BTC spot and futures returns may drive updates in both Polymarket
    and option-implied probabilities if markets respond quickly to the
    same underlying news and order flow.
    """
    plt.figure(figsize=(10, 4))

    sub = df[[ret_col, "d_prob_yes", "d_prob_option"]].dropna()

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(
        sub[ret_col],
        sub["d_prob_yes"],
        s=8,
        alpha=0.4,
        color="tab:blue",
    )
    ax1.set_xlabel(f"{label_prefix} return")
    ax1.set_ylabel("Δ prob_yes (Polymarket)")
    ax1.set_title(f"{label_prefix} return vs Δ prob_yes")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(
        sub[ret_col],
        sub["d_prob_option"],
        s=8,
        alpha=0.4,
        color="tab:orange",
    )
    ax2.set_xlabel(f"{label_prefix} return")
    ax2.set_ylabel("Δ prob_option (options)")
    ax2.set_title(f"{label_prefix} return vs Δ prob_option")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved %s to %s", out_name, out_path)
    return out_path


def plot_example_event_timeseries(df: pd.DataFrame) -> Path:
    """Time-series example for one representative event_date.

    We show Polymarket probabilities, option-implied probabilities, and
    the resulting spread for a single event_date, illustrating how the
    two markets evolve jointly.
    """
    if "event_date" in df.columns:
        # Use the middle event_date by construction if available
        dates = np.sort(df["event_date"].dropna().unique())
        if len(dates) == 0:
            sub = df
            chosen = None
        else:
            chosen = dates[len(dates) // 2]
            sub = df[df["event_date"] == chosen]
    else:
        sub = df
        chosen = None

    if sub.empty:
        logger.warning("No data available for example event time-series plot.")
        return FIGURES_DIR / "example_event_probability_timeseries.png"

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.plot(sub["timestamp"], sub["prob_yes"], label="prob_yes (Polymarket)")
    ax.plot(sub["timestamp"], sub["prob_option"], label="prob_option (options)")
    ax.plot(
        sub["timestamp"],
        sub["probability_spread"],
        label="probability_spread",
        linestyle="--",
    )

    if chosen is not None:
        title_suffix = f" – event_date={pd.to_datetime(chosen).date().isoformat()}"
    else:
        title_suffix = ""
    ax.set_title(f"Example Event Probability Time Series{title_suffix}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "example_event_probability_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved example event probability time-series to %s", out_path)
    return out_path


def plot_btc_price_with_news_events(
    cross_df: pd.DataFrame, events: pd.DataFrame
) -> Path:
    """Spot BTC price over time with vertical news-event overlays.

    This figure links manually curated news shocks to the underlying
    spot price path, visually illustrating which parts of the BTC price
    trajectory are associated with major news events.
    """
    if "spot_close" not in cross_df.columns:
        logger.warning(
            "spot_close not available in merged dataset; cannot plot BTC price."
        )
        return FIGURES_DIR / "btc_price_with_news_events.png"

    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    sub = cross_df.dropna(subset=["spot_close"])
    ax.plot(sub["timestamp"], sub["spot_close"], label="BTC spot close", color="black")

    # Overlay news events within the plotted time range
    tmin, tmax = sub["timestamp"].min(), sub["timestamp"].max()
    mask = (events["event_timestamp_utc"] >= tmin) & (
        events["event_timestamp_utc"] <= tmax
    )
    ev_in_range = events[mask]
    for _, row in ev_in_range.iterrows():
        ts = row["event_timestamp_utc"]
        cat = row.get("category", "event")
        color = "tab:blue" if cat == "crypto_bitcoin" else "tab:orange"
        ax.axvline(ts, color=color, linestyle="--", alpha=0.7)

    ax.set_title("BTC Spot Price with News Event Overlays")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("BTC spot price")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "btc_price_with_news_events.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved BTC price with news events to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def write_diagnostics(
    cross_df: pd.DataFrame,
    events: pd.DataFrame,
    created_tables: List[Path],
    created_figures: List[Path],
) -> Path:
    """Write a concise diagnostics file for the cross-market EDA."""
    out_path = DIAGNOSTICS_DIR / "cross_market_eda_diagnostics.txt"

    n_rows = len(cross_df)
    ts_min = cross_df["timestamp"].min()
    ts_max = cross_df["timestamp"].max()
    n_events = len(events)

    lines: List[str] = []
    lines.append("Cross-Market EDA – Diagnostics")
    lines.append("================================")
    lines.append(f"Master dataset path: {MASTER_PATH}")
    lines.append(f"Spot dataset path:   {SPOT_PATH}")
    lines.append(f"Futures dataset path:{FUTURES_PATH}")
    lines.append(f"News events path:    {EVENTS_PATH}")
    lines.append("")
    lines.append(f"Rows in merged cross-market dataset: {n_rows}")
    lines.append(f"Timestamp range (UTC): {ts_min.isoformat()} to {ts_max.isoformat()}")
    lines.append(f"Number of news events: {n_events}")
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Tables created:")
    for p in created_tables:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append("Figures created:")
    for p in created_figures:
        lines.append(f"  - {p}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote diagnostics to %s", out_path)

    # Echo to console for quick inspection
    print("\n".join(lines))

    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Cross-Market EDA pipeline.

    The goal is to build intuition about how Polymarket probabilities,
    option-implied probabilities, BTC spot prices, and futures returns
    co-move, without imposing any specific econometric model.
    """
    ensure_output_dirs()
    merged = merge_cross_market()
    cross_df = merged["cross"]
    events = merged["events"]

    # Tables
    created_tables: List[Path] = []
    created_tables.append(build_cross_market_summary(cross_df))

    # Figures
    created_figures: List[Path] = []
    created_figures.append(plot_polymarket_vs_option(cross_df))
    created_figures.extend(plot_spread_distributions(cross_df))
    created_figures.append(plot_spread_vs_moneyness(cross_df))
    created_figures.append(plot_spread_vs_maturity(cross_df))
    created_figures.append(
        plot_return_vs_prob_changes(
            cross_df,
            ret_col="spot_return",
            label_prefix="BTC spot",
            out_name="spot_return_vs_prob_change.png",
        )
    )
    created_figures.append(
        plot_return_vs_prob_changes(
            cross_df,
            ret_col="futures_return",
            label_prefix="Futures",
            out_name="futures_return_vs_prob_change.png",
        )
    )
    created_figures.append(plot_example_event_timeseries(cross_df))
    created_figures.append(plot_btc_price_with_news_events(cross_df, events))

    # Diagnostics
    write_diagnostics(cross_df, events, created_tables, created_figures)


if __name__ == "__main__":
    main()

