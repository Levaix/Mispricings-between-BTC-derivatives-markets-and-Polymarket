"""EDA module for Polymarket BTC probability data.

This script:
  - Loads Polymarket contracts metadata and 5-minute probability panels (long and wide)
    from Actual data (read-only)
  - Explores prediction market probabilities, strike levels, event dates,
    and the panel structure of the data
  - Produces summary tables and publication-quality figures

All outputs are written ONLY under:
  EDA/Polymarket/

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

CONTRACTS_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Polymarket"
    / "polymarket_contracts_metadata.parquet"
)
PROB_LONG_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Polymarket"
    / "polymarket_probabilities_5m_long.parquet"
)
PROB_WIDE_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Polymarket"
    / "polymarket_probabilities_5m_wide.parquet"
)

OUTPUT_ROOT = CODE_ROOT / "EDA" / "Polymarket"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_contracts() -> pd.DataFrame:
    logger.info("Loading Polymarket contracts metadata from %s", CONTRACTS_PATH)
    return pd.read_parquet(CONTRACTS_PATH)


def load_prob_long() -> pd.DataFrame:
    logger.info("Loading Polymarket probabilities (long) from %s", PROB_LONG_PATH)
    df = pd.read_parquet(PROB_LONG_PATH)
    cols_lower = {c.lower(): c for c in df.columns}

    # Detect timestamp, event_date, strike_K, prob_yes
    ts_col = None
    for key in ["timestamp", "timestamp_utc", "bucket_5m_utc", "time", "datetime"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in Polymarket long dataset; "
            f"columns available: {list(df.columns)}"
        )

    # Event date: in this dataset, expiry_date/expiry_utc represent the
    # contract resolution date, which we treat as event_date.
    event_date_col = None
    for key in ["event_date", "eventdate", "date", "expiry_date", "expiry_utc"]:
        if key in cols_lower:
            event_date_col = cols_lower[key]
            break
    if event_date_col is None:
        raise KeyError(
            "Could not detect event_date / expiry_date column in Polymarket long dataset; "
            f"columns available: {list(df.columns)}"
        )

    strike_col = None
    for key in ["strike_k", "strike", "k"]:
        if key in cols_lower:
            strike_col = cols_lower[key]
            break
    if strike_col is None:
        raise KeyError(
            "Could not detect strike_K column in Polymarket long dataset; "
            f"columns available: {list(df.columns)}"
        )

    prob_col = None
    for key in ["prob_yes", "probability", "p_yes"]:
        if key in cols_lower:
            prob_col = cols_lower[key]
            break
    if prob_col is None:
        raise KeyError(
            "Could not detect prob_yes column in Polymarket long dataset; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(
        columns={
            ts_col: "timestamp",
            event_date_col: "event_date",
            strike_col: "strike_K",
            prob_col: "prob_yes",
        },
        inplace=True,
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some Polymarket long timestamps as UTC.")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if df["event_date"].isna().any():
        raise ValueError("Failed to parse some event_date values.")

    # Ensure probabilities are in [0,1]
    df["prob_yes"] = df["prob_yes"].clip(lower=0.0, upper=1.0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_prob_wide() -> pd.DataFrame:
    logger.info("Loading Polymarket probabilities (wide) from %s", PROB_WIDE_PATH)
    df = pd.read_parquet(PROB_WIDE_PATH)
    cols_lower = {c.lower(): c for c in df.columns}

    ts_col = None
    for key in ["timestamp", "timestamp_utc", "bucket_5m_utc", "time", "datetime"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in Polymarket wide dataset; "
            f"columns available: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some Polymarket wide timestamps as UTC.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clip all probability-like columns to [0,1]
    prob_cols = [c for c in df.columns if c != "timestamp"]
    df[prob_cols] = df[prob_cols].clip(lower=0.0, upper=1.0)
    return df


# ---------------------------------------------------------------------------
# Summary and market structure tables
# ---------------------------------------------------------------------------

def summary_stats_prob(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Summary stats for prob_yes, strike_K, event_date."""
    # Convert event_date to ordinal for numeric summary
    df_num = df.copy()
    df_num["event_date_num"] = df_num["event_date"].astype("int64") // 10**9

    variables = ["prob_yes", "strike_K", "event_date_num"]

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
        stats_dict = _describe(df_num[var])
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
    out_df.to_csv(out_csv, index=False)

    # Markdown version
    md_path = out_csv.with_suffix(".md")
    cols = list(out_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in out_df.iterrows():
        vals = [str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "Saved Polymarket probability summary stats to %s and %s", out_csv, md_path
    )
    return out_df


def market_structure_table(contracts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Compute basic market structure from contracts metadata."""
    # Heuristically detect identifiers
    cols_lower = {c.lower(): c for c in contracts_df.columns}

    market_col = None
    for key in ["market_id", "event_id", "market_slug", "event_slug"]:
        if key in cols_lower:
            market_col = cols_lower[key]
            break

    strike_col = None
    for key in ["strike_k", "strike", "k"]:
        if key in cols_lower:
            strike_col = cols_lower[key]
            break

    event_date_col = None
    for key in ["event_date", "resolution_date", "expiry"]:
        if key in cols_lower:
            event_date_col = cols_lower[key]
            break

    df = contracts_df.copy()
    n_markets = df[market_col].nunique() if market_col else np.nan
    n_strikes = df[strike_col].nunique() if strike_col else np.nan
    if event_date_col:
        df[event_date_col] = pd.to_datetime(df[event_date_col], errors="coerce")
        n_event_dates = df[event_date_col].nunique()
    else:
        n_event_dates = np.nan

    out_df = pd.DataFrame(
        {
            "metric": [
                "n_markets",
                "n_strikes",
                "n_event_dates",
            ],
            "value": [n_markets, n_strikes, n_event_dates],
        }
    )
    out_df.to_csv(out_csv, index=False)

    md_path = out_csv.with_suffix(".md")
    lines = []
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for _, r in out_df.iterrows():
        lines.append(f"| {r['metric']} | {r['value']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved Polymarket market structure to %s and %s", out_csv, md_path)
    return out_df


def strike_distribution_table(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    counts = df["strike_K"].value_counts().sort_index()
    out_df = counts.rename_axis("strike_K").reset_index(name="count")
    out_df.to_csv(out_csv, index=False)

    md_path = out_csv.with_suffix(".md")
    lines = []
    lines.append("| strike_K | count |")
    lines.append("| --- | --- |")
    for _, r in out_df.iterrows():
        lines.append(f"| {r['strike_K']} | {r['count']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved Polymarket strike distribution to %s and %s", out_csv, md_path)
    return out_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def format_time_axis_with_month_labels(ax, timestamps: pd.Series) -> None:
    """Format x-axis with day ticks (5,10,15,20,25) and month labels.

    - Major ticks at days 5, 10, 15, 20, 25.
    - Day numbers shown at those ticks.
    - One 'January YYYY' and 'February YYYY' style label centered
      over each month segment when present.
    """
    locator = mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25])
    formatter = mdates.DateFormatter("%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ts = pd.to_datetime(timestamps)
    if ts.empty:
        return

    # Support both Series and DatetimeIndex
    if isinstance(ts, pd.DatetimeIndex):
        months = ts.to_period("M")
    else:
        months = ts.dt.to_period("M")
    for period in months.unique():
        year = period.year
        month = period.month
        mask = months == period
        month_ts = ts[mask]
        if month_ts.empty:
            continue
        mid = month_ts.min() + (month_ts.max() - month_ts.min()) / 2
        label = mid.strftime("%B %Y")
        ax.text(
            mdates.date2num(mid),
            -0.12,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )


def plot_probability_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["prob_yes"].dropna()
    plt.hist(vals, bins=50, density=True, alpha=0.7)
    plt.title("Distribution of Polymarket Probabilities (prob_yes)")
    plt.xlabel("Probability (prob_yes)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "probability_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved probability distribution to %s", out_path)


def plot_probability_timeseries_example(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    # Pick up to 3 representative strikes
    unique_strikes = np.sort(df["strike_K"].unique())
    if len(unique_strikes) == 0:
        logger.warning("No strikes found for probability timeseries example.")
        return
    chosen = unique_strikes[:: max(1, len(unique_strikes) // 3)][:3]
    for k in chosen:
        sub = df[df["strike_K"] == k]
        plt.plot(sub["timestamp"], sub["prob_yes"], linewidth=0.8, label=f"K={k}")

    plt.title("Polymarket Probability Time Series (example strikes)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Probability (prob_yes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "probability_timeseries_example.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved probability timeseries example to %s", out_path)


def plot_probability_vs_strike(df: pd.DataFrame) -> None:
    # Choose representative timestamp (median)
    ts_vals = df["timestamp"].unique()
    if len(ts_vals) == 0:
        logger.warning("No timestamps available for probability_vs_strike.")
        return
    ts_vals = np.sort(ts_vals)
    chosen_ts = ts_vals[len(ts_vals) // 2]
    sub = df[df["timestamp"] == chosen_ts]

    plt.figure(figsize=(6, 4))
    plt.scatter(sub["strike_K"], sub["prob_yes"], s=8, alpha=0.7)
    plt.title(f"Probability vs Strike\n{chosen_ts.isoformat()}")
    plt.xlabel("Strike_K")
    plt.ylabel("Probability (prob_yes)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "probability_vs_strike.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved probability vs strike figure to %s", out_path)


def plot_strike_distribution_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["strike_K"].dropna()
    plt.hist(vals, bins=50, density=False, alpha=0.7)
    plt.title("Strike Distribution (Polymarket strikes_K)")
    plt.xlabel("Strike_K")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "strike_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved strike distribution histogram to %s", out_path)


def plot_event_date_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["event_date"].dropna().dt.normalize()
    counts = vals.value_counts().sort_index()
    plt.bar(counts.index, counts.values, width=0.8)
    plt.title("Event Date Distribution (Polymarket)")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, counts.index)
    plt.xlabel("Event date")
    plt.ylabel("Number of observations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "event_date_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved event date distribution to %s", out_path)


def plot_probability_heatmap(prob_wide_df: pd.DataFrame) -> None:
    if prob_wide_df.shape[1] <= 1:
        logger.warning("Wide Polymarket dataset has no strike columns for heatmap.")
        return
    # Use values matrix; rows=time, cols=strike columns
    strikes = [c for c in prob_wide_df.columns if c != "timestamp"]
    data = prob_wide_df[strikes].T.values  # shape: (n_strikes, n_times)

    plt.figure(figsize=(9, 4))
    im = plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, label="Probability")
    plt.title("Polymarket Probability Heatmap (strikes x time)")
    plt.ylabel("Strike index")
    plt.xlabel("Time index")
    plt.tight_layout()
    out_path = FIGURES_DIR / "probability_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved probability heatmap to %s", out_path)


def plot_example_market_evolution(df: pd.DataFrame) -> None:
    # Choose representative event_date
    if df["event_date"].nunique() == 0:
        logger.warning("No event_date values for example market evolution.")
        return
    dates = np.sort(df["event_date"].unique())
    chosen_date = pd.to_datetime(dates[len(dates) // 2])
    sub = df[df["event_date"] == chosen_date]

    plt.figure(figsize=(9, 4))
    # Plot a few strikes for the chosen event_date
    unique_strikes = np.sort(sub["strike_K"].unique())
    chosen = unique_strikes[:: max(1, len(unique_strikes) // 4)][:4]
    for k in chosen:
        s2 = sub[sub["strike_K"] == k]
        plt.plot(s2["timestamp"], s2["prob_yes"], linewidth=0.8, label=f"K={k}")

    plt.title(f"Example Market Evolution – event_date={chosen_date.date().isoformat()}")
    ax = plt.gca()
    format_time_axis_with_month_labels(ax, sub["timestamp"])
    # Place "Time" below the month label (January 2026); extra labelpad moves it down
    ax.set_xlabel("Time", labelpad=48)
    plt.ylabel("Probability (prob_yes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "example_market_evolution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved example market evolution figure to %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostics and markdown summary
# ---------------------------------------------------------------------------

def write_diagnostics(
    contracts_df: pd.DataFrame,
    prob_long_df: pd.DataFrame,
    prob_wide_df: pd.DataFrame,
) -> str:
    out_path = DIAGNOSTICS_DIR / "polymarket_eda_diagnostics.txt"

    prob_min = float(prob_long_df["prob_yes"].min())
    prob_max = float(prob_long_df["prob_yes"].max())

    n_markets = contracts_df.shape[0]
    n_strikes = prob_long_df["strike_K"].nunique()
    n_event_dates = prob_long_df["event_date"].nunique()

    lines: List[str] = []
    lines.append("Polymarket EDA – Diagnostics")
    lines.append("================================")
    lines.append(f"Contracts metadata path: {CONTRACTS_PATH}")
    lines.append(f"Probabilities long path: {PROB_LONG_PATH}")
    lines.append(f"Probabilities wide path: {PROB_WIDE_PATH}")
    lines.append("")
    lines.append(f"Contracts rows: {len(contracts_df)}")
    lines.append(f"Probabilities long rows: {len(prob_long_df)}")
    lines.append(f"Probabilities wide rows: {len(prob_wide_df)}")
    lines.append("")
    lines.append(
        f"Timestamp range (long): {prob_long_df['timestamp'].min().isoformat()} "
        f"to {prob_long_df['timestamp'].max().isoformat()}"
    )
    lines.append(
        f"Timestamp range (wide): {prob_wide_df['timestamp'].min().isoformat()} "
        f"to {prob_wide_df['timestamp'].max().isoformat()}"
    )
    lines.append("")
    lines.append(f"Number of markets (contracts rows): {n_markets}")
    lines.append(f"Number of strikes: {n_strikes}")
    lines.append(f"Number of event dates: {n_event_dates}")
    lines.append(f"Probability min/max (prob_yes): {prob_min:.6f} / {prob_max:.6f}")
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Tables created:")
    lines.append(f"  - {TABLES_DIR / 'polymarket_probability_summary.csv'}")
    lines.append(f"  - {TABLES_DIR / 'polymarket_probability_summary.md'}")
    lines.append(f"  - {TABLES_DIR / 'polymarket_market_structure.csv'}")
    lines.append(f"  - {TABLES_DIR / 'polymarket_market_structure.md'}")
    lines.append(f"  - {TABLES_DIR / 'strike_distribution.csv'}")
    lines.append(f"  - {TABLES_DIR / 'strike_distribution.md'}")
    lines.append(f"  - {TABLES_DIR / 'polymarket_eda_summary.md'}")
    lines.append("")
    lines.append("Figures created:")
    lines.append(f"  - {FIGURES_DIR / 'probability_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'probability_timeseries_example.png'}")
    lines.append(f"  - {FIGURES_DIR / 'probability_vs_strike.png'}")
    lines.append(f"  - {FIGURES_DIR / 'strike_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'event_date_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'probability_heatmap.png'}")
    lines.append(f"  - {FIGURES_DIR / 'example_market_evolution.png'}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote Polymarket diagnostics to %s", out_path)
    return text


def write_markdown_summary(
    contracts_df: pd.DataFrame,
    prob_long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    out_path = TABLES_DIR / "polymarket_eda_summary.md"

    n_markets = contracts_df.shape[0]
    n_strikes = prob_long_df["strike_K"].nunique()
    n_event_dates = prob_long_df["event_date"].nunique()

    ts_min, ts_max = prob_long_df["timestamp"].min(), prob_long_df["timestamp"].max()

    prob_stats = summary_df[summary_df["variable"] == "prob_yes"].iloc[0].to_dict()

    lines: List[str] = []
    lines.append("# Polymarket EDA – Summary")
    lines.append("")
    lines.append("This module summarizes the Polymarket BTC prediction markets used")
    lines.append("in the thesis. Prediction market prices are interpreted as")
    lines.append("probabilities of specific BTC outcomes (e.g. crossing a strike).")
    lines.append("")
    lines.append("## Dataset description")
    lines.append("")
    lines.append(
        "- **Contracts metadata**: `Actual data/Polymarket/polymarket_contracts_metadata.parquet`"
    )
    lines.append(
        "- **Probabilities (long)**: `Actual data/Polymarket/polymarket_probabilities_5m_long.parquet`"
    )
    lines.append(
        "- **Probabilities (wide)**: `Actual data/Polymarket/polymarket_probabilities_5m_wide.parquet`"
    )
    lines.append(f"- **Number of markets (rows in metadata)**: {n_markets}")
    lines.append(f"- **Number of strikes**: {n_strikes}")
    lines.append(f"- **Number of event dates**: {n_event_dates}")
    lines.append(
        f"- **Sample period (5-minute probabilities)**: {ts_min.isoformat()} to {ts_max.isoformat()}"
    )
    lines.append("")
    lines.append("## Prediction market probabilities")
    lines.append("")
    lines.append(
        "- Prediction market probabilities (`prob_yes`) represent the market-implied"
    )
    lines.append(
        "  probability that a given BTC outcome (e.g. price above a strike on"
    )
    lines.append("  an event date) will occur. They reflect the beliefs and risk")
    lines.append("  preferences of traders who buy and sell contracts on Polymarket.")
    lines.append("")
    lines.append("## Key probability statistics")
    lines.append("")
    lines.append(
        f"- **Mean prob_yes**: {prob_stats['mean']:.4f}, **std**: {prob_stats['std']:.4f}"
    )
    lines.append(
        f"- **5th–95th percentiles**: {prob_stats['p5']:.4f} – {prob_stats['p95']:.4f}"
    )
    lines.append(
        f"- **Range**: {prob_stats['min']:.4f} – {prob_stats['max']:.4f}"
    )
    lines.append("")
    lines.append("## Event dates and strikes")
    lines.append("")
    lines.append(
        "- `event_date` corresponds to the underlying BTC event date or contract "
    )
    lines.append("  settlement/expiry date; the probabilities describe outcomes on")
    lines.append("  these dates.")
    lines.append(
        "- `strike_K` corresponds to BTC price thresholds used in the contracts "
    )
    lines.append("  (e.g. \"BTC above 80k on event_date\").")
    lines.append("")
    lines.append("## Key figures produced")
    lines.append("")
    lines.append("- Probability distribution (`probability_distribution.png`)")
    lines.append(
        "- Probability time series for representative strikes (`probability_timeseries_example.png`)"
    )
    lines.append(
        "- Probability vs strike cross-section (`probability_vs_strike.png`)"
    )
    lines.append("- Strike and event-date distributions")
    lines.append(
        "- Probability heatmap across strikes and time (`probability_heatmap.png`)"
    )
    lines.append(
        "- Example market evolution for one event date (`example_market_evolution.png`)"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote Polymarket EDA markdown summary to %s", out_path)


def main() -> None:
    ensure_output_dirs()

    contracts_df = load_contracts()
    prob_long_df = load_prob_long()
    prob_wide_df = load_prob_wide()

    summary_df = summary_stats_prob(
        prob_long_df, TABLES_DIR / "polymarket_probability_summary.csv"
    )
    market_structure_table(
        contracts_df, TABLES_DIR / "polymarket_market_structure.csv"
    )
    strike_distribution_table(
        prob_long_df, TABLES_DIR / "strike_distribution.csv"
    )

    # Figures
    plot_probability_distribution(prob_long_df)
    plot_probability_timeseries_example(prob_long_df)
    plot_probability_vs_strike(prob_long_df)
    plot_strike_distribution_figure(prob_long_df)
    plot_event_date_distribution(prob_long_df)
    plot_probability_heatmap(prob_wide_df)
    plot_example_market_evolution(prob_long_df)

    # Text outputs
    write_markdown_summary(contracts_df, prob_long_df, summary_df)
    diag_text = write_diagnostics(contracts_df, prob_long_df, prob_wide_df)

    print()
    print(diag_text)


if __name__ == "__main__":
    main()

