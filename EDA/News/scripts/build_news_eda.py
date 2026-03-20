"""EDA module for manually curated Bitcoin-related news events.

This script provides a transparent, thesis-ready overview of the
`btc_news_events_final.csv` dataset that underpins the news event
study. The emphasis is on:

- documenting the structure of the event dataset,
- summarising event timing and category composition, and
- visualising how events are distributed over time.

Events were *manually curated* to ensure that only clearly
identifiable, economically meaningful shocks are included. The
`event_timestamp_utc` column is crucial: it anchors the event-study
window that is used later when aligning market data in pre- and
post-event intervals. Separating `crypto_bitcoin` from `macro`
events is useful because these categories can trigger very different
mechanisms (crypto-specific news versus broad risk or liquidity
shocks) even when they occur on the same underlying asset.

The module is **purely exploratory** and deliberately does *not*
run any regressions or econometric tests.

All outputs are written ONLY under:

    EDA/News/

The input dataset is read in a strictly read-only fashion from:

    pipeline/Manual News Event Identification/outputs/btc_news_events_final.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and folder structure
# ---------------------------------------------------------------------------

CODE_ROOT = Path(__file__).resolve().parents[3]

# The manual news events live in Actual data (read-only). We only ever read
# from this location; all outputs stay under EDA/News/.
EVENTS_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Manual News Event Identification"
    / "outputs"
    / "btc_news_events_final.csv"
)

OUTPUT_ROOT = CODE_ROOT / "EDA" / "News"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"


def ensure_output_dirs() -> None:
    """Create EDA/News directories if they do not yet exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------

def load_events() -> pd.DataFrame:
    """Load manually curated news events and construct exploratory fields.

    The function:
    - reads the CSV from the pipeline output (read-only),
    - parses `event_timestamp_utc` as timezone-aware UTC datetimes,
    - parses `event_date` as a date-like field,
    - sorts events chronologically, and
    - creates simple timing variables used in the EDA:
      `event_hour` and `event_weekday`.
    """
    logger.info("Loading news events from %s", EVENTS_PATH)
    df = pd.read_csv(EVENTS_PATH)

    # Parse timestamps and dates
    if "event_timestamp_utc" not in df.columns:
        raise KeyError(
            "Expected column 'event_timestamp_utc' not found in news events dataset."
        )
    if "event_date" not in df.columns:
        raise KeyError(
            "Expected column 'event_date' not found in news events dataset."
        )

    df["event_timestamp_utc"] = pd.to_datetime(
        df["event_timestamp_utc"], utc=True, errors="raise"
    )
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    # Sort chronologically by event time
    df = df.sort_values("event_timestamp_utc").reset_index(drop=True)

    # Exploratory timing variables
    df["event_hour"] = df["event_timestamp_utc"].dt.hour
    df["event_weekday"] = df["event_timestamp_utc"].dt.day_name()

    logger.info("Loaded %d events.", len(df))
    return df


# ---------------------------------------------------------------------------
# Helpers for tables
# ---------------------------------------------------------------------------

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


def build_event_hour_summary(df: pd.DataFrame) -> Path:
    """Summary statistics for event_hour."""
    desc = df["event_hour"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    ).to_frame(name="event_hour")
    csv_path = TABLES_DIR / "news_event_hour_summary.csv"
    md_path = TABLES_DIR / "news_event_hour_summary.md"
    save_table_with_markdown(desc, csv_path, md_path)
    return csv_path


def build_category_counts(df: pd.DataFrame) -> Path:
    """Counts by category (`crypto_bitcoin` vs `macro`)."""
    counts = df["category"].value_counts().rename_axis("category").to_frame("count")
    csv_path = TABLES_DIR / "news_category_counts.csv"
    md_path = TABLES_DIR / "news_category_counts.md"
    save_table_with_markdown(counts, csv_path, md_path)
    return csv_path


def build_source_counts(df: pd.DataFrame) -> Path:
    """Counts by news source (e.g. CoinDesk, Bloomberg)."""
    counts = df["source"].value_counts().rename_axis("source").to_frame("count")
    csv_path = TABLES_DIR / "news_source_counts.csv"
    md_path = TABLES_DIR / "news_source_counts.md"
    save_table_with_markdown(counts, csv_path, md_path)
    return csv_path


def build_event_timeline_table(df: pd.DataFrame) -> Path:
    """Create a clean event timeline table."""
    cols = [
        "event_id",
        "event_timestamp_utc",
        "event_date",
        "category",
        "source",
        "event_label",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Event dataset is missing required columns for timeline table: {missing}"
        )

    timeline = df[cols].copy()
    csv_path = TABLES_DIR / "news_event_timeline.csv"
    md_path = TABLES_DIR / "news_event_timeline.md"
    save_table_with_markdown(timeline, csv_path, md_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_event_timeline(df: pd.DataFrame) -> Path:
    """Plot a simple event timeline with colour-coded categories.

    Each event is represented as a point on the time axis. The
    `event_timestamp_utc` is the anchor used later in the event study
    when defining pre- and post-event windows for market variables.
    """
    if df.empty:
        logger.warning("No events available for event timeline plot.")
        return FIGURES_DIR / "news_event_timeline.png"

    plt.figure(figsize=(9, 2.8))
    ax = plt.gca()

    # Map categories to vertical positions and colours
    cat_to_y = {"crypto_bitcoin": 1.0, "macro": 0.0}
    cat_to_color = {"crypto_bitcoin": "tab:blue", "macro": "tab:orange"}

    for cat, group in df.groupby("category"):
        y_val = cat_to_y.get(cat, 0.5)
        color = cat_to_color.get(cat, "grey")
        ax.scatter(
            group["event_timestamp_utc"],
            np.full(len(group), y_val),
            label=cat,
            color=color,
            s=50,
            edgecolor="black",
            alpha=0.9,
            zorder=3,
        )

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["macro", "crypto_bitcoin"])

    ax.set_title("News Event Timeline (UTC)")
    ax.set_xlabel("Event timestamp (UTC)")
    ax.set_ylabel("Category")

    # Modest date formatting – typically only a handful of events
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()

    out_path = FIGURES_DIR / "news_event_timeline.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved event timeline plot to %s", out_path)
    return out_path


def plot_category_distribution(df: pd.DataFrame) -> Path:
    """Bar chart for number of events by category."""
    counts = df["category"].value_counts().sort_index()

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.bar(counts.index, counts.values, color=["tab:blue", "tab:orange"])
    ax.set_title("News Events by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of events")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    out_path = FIGURES_DIR / "news_category_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved category distribution plot to %s", out_path)
    return out_path


def plot_event_hour_distribution(df: pd.DataFrame) -> Path:
    """Histogram of event_hour to show intraday timing."""
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(df["event_hour"], bins=range(0, 25), color="tab:blue", alpha=0.8)
    ax.set_title("News Event Timing – Hour of Day (UTC)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Number of events")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    out_path = FIGURES_DIR / "news_event_hour_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved event hour distribution plot to %s", out_path)
    return out_path


def plot_weekday_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of events by weekday."""
    # Ensure a stable Monday–Sunday ordering
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    counts = (
        df["event_weekday"]
        .value_counts()
        .reindex(weekday_order)
        .fillna(0)
        .astype(int)
    )

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(counts.index, counts.values, color="tab:purple")
    ax.set_title("News Events by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Number of events")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "news_event_weekday_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved weekday distribution plot to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Diagnostics and markdown summary
# ---------------------------------------------------------------------------

def write_diagnostics(df: pd.DataFrame, created_tables: List[Path], created_figures: List[Path]) -> Path:
    """Write a concise diagnostics file for the news EDA."""
    out_path = DIAGNOSTICS_DIR / "news_eda_diagnostics.txt"

    n_rows = len(df)
    ts_min = df["event_timestamp_utc"].min()
    ts_max = df["event_timestamp_utc"].max()

    n_crypto = int((df["category"] == "crypto_bitcoin").sum())
    n_macro = int((df["category"] == "macro").sum())
    n_sources = df["source"].nunique()

    lines: List[str] = []
    lines.append("News Events EDA – Diagnostics")
    lines.append("================================")
    lines.append(f"Dataset path: {EVENTS_PATH}")
    lines.append(f"Row count: {n_rows}")
    lines.append(
        f"Timestamp range (UTC): {ts_min.isoformat()} to {ts_max.isoformat()}"
    )
    lines.append("")
    lines.append(f"Number of crypto_bitcoin events: {n_crypto}")
    lines.append(f"Number of macro events: {n_macro}")
    lines.append(f"Number of distinct sources: {n_sources}")
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

    # Also echo a short summary to the console
    print("\n".join(lines))

    return out_path


def write_markdown_summary(df: pd.DataFrame) -> Path:
    """High-level markdown summary for the thesis."""
    out_path = TABLES_DIR / "news_eda_summary.md"

    n_events = len(df)
    date_min = df["event_date"].min()
    date_max = df["event_date"].max()

    cat_counts = df["category"].value_counts()
    n_crypto = int(cat_counts.get("crypto_bitcoin", 0))
    n_macro = int(cat_counts.get("macro", 0))

    src_counts = df["source"].value_counts()
    n_sources = src_counts.shape[0]

    lines: List[str] = []
    lines.append("# News Events EDA – Summary")
    lines.append("")
    lines.append("This module provides exploratory analysis for the manually curated ")
    lines.append("Bitcoin-related news events dataset used in the news event study.")
    lines.append("")
    lines.append("## Dataset description")
    lines.append("")
    lines.append(
        f"- **Number of events**: {n_events} (each row is a distinct news shock)."
    )
    lines.append(
        f"- **Event date range**: {date_min} to {date_max} (event_date, in UTC calendar terms)."
    )
    lines.append(
        "- **Key columns**: `event_id`, `event_timestamp_utc`, `event_date`, "
        "`category`, `source`, `event_label`, `headline`, `url`, "
        "`short_reason_for_inclusion`."
    )
    lines.append("")
    lines.append("Events were **manually curated** from high-quality news sources ")
    lines.append(
        "(primarily CoinDesk for crypto-specific stories and Bloomberg for macroeconomic "
        "announcements). This manual step ensures that only clearly identifiable, "
        "market-relevant shocks enter the subsequent event study."
    )
    lines.append("")
    lines.append("## Category breakdown")
    lines.append("")
    lines.append(
        f"- **crypto_bitcoin events**: {n_crypto} (crypto-specific news on Bitcoin and the wider crypto market)."
    )
    lines.append(
        f"- **macro events**: {n_macro} (macro announcements that affect global risk and liquidity conditions)."
    )
    lines.append("")
    lines.append(
        "Distinguishing `crypto_bitcoin` from `macro` events is empirically useful: "
        "crypto news tends to move expectations about the crypto ecosystem directly, "
        "whereas macro news primarily affects Bitcoin via risk appetite, funding "
        "conditions, and cross-asset flows."
    )
    lines.append("")
    lines.append("## Source breakdown")
    lines.append("")
    lines.append(f"- **Number of distinct sources**: {n_sources}.")
    for src, cnt in src_counts.items():
        lines.append(f"- **{src}**: {int(cnt)} events.")
    lines.append("")
    lines.append(
        "Using a small set of reputable outlets keeps the event definition consistent "
        "and mitigates concerns about noise from low-quality news sources."
    )
    lines.append("")
    lines.append("## Role of event timestamps in the event study")
    lines.append("")
    lines.append(
        "The `event_timestamp_utc` field records the time at which the news item "
        "became public in Coordinated Universal Time. In the event study, this "
        "timestamp serves as the **anchor point** for constructing symmetric "
        "pre- and post-event windows (e.g. \\[-12h, +12h\\]). Market data "
        "from Bitcoin futures, options-implied probabilities, and Polymarket "
        "predictions are aligned relative to this timestamp to quantify how "
        "probability spreads and prices respond to the news shock."
    )
    lines.append("")
    lines.append("## Figures produced")
    lines.append("")
    lines.append(
        "- **news_event_timeline.png** – location of each event on the UTC time line, "
        "colour-coded by category."
    )
    lines.append(
        "- **news_category_distribution.png** – bar chart of the number of "
        "`crypto_bitcoin` versus `macro` events."
    )
    lines.append(
        "- **news_event_hour_distribution.png** – histogram of `event_hour`, showing "
        "when in the day news tends to be released."
    )
    lines.append(
        "- **news_event_weekday_distribution.png** – counts of events by weekday."
    )
    lines.append("")
    lines.append(
        "Together, these tables and figures document the structure of the news "
        "dataset and provide context for interpreting the econometric results of "
        "the subsequent event study."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown summary to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full News EDA pipeline."""
    ensure_output_dirs()
    df = load_events()

    # Tables
    created_tables: List[Path] = []
    created_tables.append(build_event_hour_summary(df))
    created_tables.append(build_category_counts(df))
    created_tables.append(build_source_counts(df))
    created_tables.append(build_event_timeline_table(df))

    # Figures
    created_figures: List[Path] = []
    created_figures.append(plot_event_timeline(df))
    created_figures.append(plot_category_distribution(df))
    created_figures.append(plot_event_hour_distribution(df))
    created_figures.append(plot_weekday_distribution(df))

    # Diagnostics and markdown overview
    created_tables.append(write_markdown_summary(df))
    write_diagnostics(df, created_tables, created_figures)


if __name__ == "__main__":
    main()

