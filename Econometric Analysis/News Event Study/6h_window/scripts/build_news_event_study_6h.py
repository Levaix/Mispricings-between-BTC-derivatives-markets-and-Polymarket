"""News Event Study (6-hour robustness window).

Goal
-----
Replicate the main 12-hour news event study using a *symmetric 6-hour*
event window around each news shock. This is a robustness check of the
baseline design.

Main mispricing metric
-----------------------
The primary metric is ``abs_spread`` (absolute probability spread), which
captures the *magnitude* of disagreement between markets regardless of
direction. The signed ``probability_spread`` is still summarized
descriptively, but the main statistical tests focus on ``abs_spread``.

Data
-----
1) Probability dataset (READ-ONLY – NEVER written to):
   Actual data/Options N(d2) + Polymarket - Econometric Testing/
     btc_probability_comparison_empirical_master.parquet

   Required columns:
     - timestamp  (provided as bucket_5m_utc in the file)
     - event_date
     - strike_K
     - probability_spread
     - abs_spread

2) Event dataset (READ-ONLY):
   Actual data/Manual News Event Identification/outputs/btc_news_events_final.csv

   Required columns:
     - event_id
     - event_timestamp_utc
     - event_date
     - category
     - event_label
     - source

All outputs are written ONLY under:
  Econometric Analysis/News Event Study/6h_window/

This script MUST NEVER modify, overwrite, or save any files inside:
  Actual data/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

# Script lives under:
#   Code/Econometric Analysis/News Event Study/6h_window/scripts/
# so the project root `Code/` is four levels up.
CODE_ROOT = Path(__file__).resolve().parents[4]

PROBABILITY_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

EVENTS_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Manual News Event Identification"
    / "outputs"
    / "btc_news_events_final.csv"
)

OUTPUT_ROOT = (
    CODE_ROOT
    / "Econometric Analysis"
    / "News Event Study"
    / "6h_window"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"

# 6-hour symmetric event window
EVENT_WINDOW_HOURS = 6.0


@dataclass
class TestResult:
    test_name: str
    group_1: str
    group_2: str
    mean_group_1: float
    mean_group_2: float
    difference_in_means: float
    t_statistic: float
    p_value: float
    n_group_1: int
    n_group_2: int


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_probability_data() -> pd.DataFrame:
    """Load probability comparison panel from parquet (read-only).

    The underlying file stores the timestamp as ``bucket_5m_utc``; this is
    renamed to ``timestamp`` for the event-study logic.
    """
    logger.info("Loading probability dataset from %s", PROBABILITY_PATH)
    df = pd.read_parquet(PROBABILITY_PATH)
    required_cols = {
        "bucket_5m_utc",
        "event_date",
        "strike_K",
        "probability_spread",
        "abs_spread",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Probability dataset missing required columns: {missing}")

    df = df.copy()
    df.rename(columns={"bucket_5m_utc": "timestamp"}, inplace=True)
    return df


def load_event_data() -> pd.DataFrame:
    """Load manually identified news events from CSV (read-only)."""
    logger.info("Loading event dataset from %s", EVENTS_PATH)
    df = pd.read_csv(EVENTS_PATH)
    required_cols = {
        "event_id",
        "event_timestamp_utc",
        "event_date",
        "category",
        "event_label",
        "source",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Event dataset missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Time handling and event window assignment
# ---------------------------------------------------------------------------

def parse_timestamps(
    prob_df: pd.DataFrame, events_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse timestamps as timezone-aware UTC datetimes."""
    # Probability dataset timestamp
    ts = pd.to_datetime(prob_df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Failed to parse some timestamps in probability dataset.")
    prob_df = prob_df.copy()
    prob_df["timestamp"] = ts

    # Event timestamp
    ev_ts = pd.to_datetime(events_df["event_timestamp_utc"], utc=True, errors="coerce")
    if ev_ts.isna().any():
        raise ValueError("Failed to parse some event_timestamp_utc values.")
    events_df = events_df.copy()
    events_df["event_timestamp_utc"] = ev_ts

    return prob_df, events_df


def assign_event_windows(
    prob_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each probability observation to normal/pre_event/post_event regimes.

    6-hour symmetric window per event:
      - pre_event:  [t - 6h, t)
      - post_event: [t, t + 6h]

    Rules:
      - If within any pre_event window, label pre_event.
      - If within any post_event window, label post_event.
      - If an observation falls into multiple overlapping windows,
        assign it to the event with minimum absolute time distance.
      - Outside all windows -> regime='normal'.
    """
    df = prob_df.copy()
    df["regime"] = "normal"
    df["event_id"] = pd.NA
    df["event_category"] = pd.NA
    df["event_time_hours"] = np.nan

    ts = df["timestamp"]
    best_abs_delta = pd.Series(np.inf, index=df.index, dtype="float64")

    for _, ev in events_df.iterrows():
        ev_id = ev["event_id"]
        ev_cat = ev["category"]
        ev_time = ev["event_timestamp_utc"]

        # Time difference in hours (prob - event)
        delta = (ts - ev_time).dt.total_seconds() / 3600.0
        abs_delta = delta.abs()

        in_window = abs_delta <= EVENT_WINDOW_HOURS
        if not in_window.any():
            continue

        better = in_window & (abs_delta < best_abs_delta)
        if not better.any():
            continue

        best_abs_delta.loc[better] = abs_delta[better]
        df.loc[better, "event_id"] = ev_id
        df.loc[better, "event_category"] = ev_cat
        df.loc[better, "event_time_hours"] = delta[better]

    has_event = df["event_id"].notna()
    df.loc[has_event & (df["event_time_hours"] < 0), "regime"] = "pre_event"
    df.loc[has_event & (df["event_time_hours"] >= 0), "regime"] = "post_event"

    return df


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------

def compute_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats by regime for probability_spread and abs_spread."""
    regimes = ["normal", "pre_event", "post_event"]
    df = df.copy()
    df["regime"] = pd.Categorical(df["regime"], categories=regimes, ordered=True)

    def _summary(group: pd.Series) -> Dict[str, float]:
        return {
            "mean": group.mean(),
            "median": group.median(),
            "std": group.std(),
            "count": int(group.count()),
        }

    rows: List[Dict[str, object]] = []
    for regime, sub in df.groupby("regime", observed=True):
        if regime is None:
            continue
        for col in ["probability_spread", "abs_spread"]:
            stats_dict = _summary(sub[col])
            rows.append(
                {
                    "regime": regime,
                    "variable": col,
                    **stats_dict,
                }
            )
    return pd.DataFrame(rows)


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats by event category (crypto_bitcoin/macro) and regime (pre/post)."""
    window_df = df[df["regime"].isin(["pre_event", "post_event"])].copy()
    window_df = window_df[window_df["event_category"].notna()]

    def _summary(group: pd.Series) -> Dict[str, float]:
        return {
            "mean": group.mean(),
            "median": group.median(),
            "std": group.std(),
            "count": int(group.count()),
        }

    rows: List[Dict[str, object]] = []
    for (cat, regime), sub in window_df.groupby(
        ["event_category", "regime"], observed=True
    ):
        for col in ["probability_spread", "abs_spread"]:
            stats_dict = _summary(sub[col])
            rows.append(
                {
                    "event_category": cat,
                    "regime": regime,
                    "variable": col,
                    **stats_dict,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests (difference-in-means t-tests on abs_spread)
# ---------------------------------------------------------------------------

def _ttest_wrapper(
    data: pd.DataFrame,
    mask1: pd.Series,
    mask2: pd.Series,
    test_name: str,
    group_1: str,
    group_2: str,
) -> TestResult:
    """Run Welch t-test on abs_spread between two masks."""
    x = data.loc[mask1, "abs_spread"].dropna()
    y = data.loc[mask2, "abs_spread"].dropna()

    if len(x) == 0 or len(y) == 0:
        return TestResult(
            test_name=test_name,
            group_1=group_1,
            group_2=group_2,
            mean_group_1=float("nan"),
            mean_group_2=float("nan"),
            difference_in_means=float("nan"),
            t_statistic=float("nan"),
            p_value=float("nan"),
            n_group_1=len(x),
            n_group_2=len(y),
        )

    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    return TestResult(
        test_name=test_name,
        group_1=group_1,
        group_2=group_2,
        mean_group_1=float(x.mean()),
        mean_group_2=float(y.mean()),
        difference_in_means=float(x.mean() - y.mean()),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        n_group_1=len(x),
        n_group_2=len(y),
    )


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run pre-specified t-tests on abs_spread."""
    res: List[TestResult] = []

    normal = df["regime"] == "normal"
    pre = df["regime"] == "pre_event"
    post = df["regime"] == "post_event"

    crypto_post = post & (df["event_category"] == "crypto_bitcoin")
    macro_post = post & (df["event_category"] == "macro")

    # Test A: pre_event vs normal
    res.append(
        _ttest_wrapper(
            df,
            pre,
            normal,
            test_name="A_pre_vs_normal_6h",
            group_1="pre_event",
            group_2="normal",
        )
    )

    # Test B: post_event vs normal
    res.append(
        _ttest_wrapper(
            df,
            post,
            normal,
            test_name="B_post_vs_normal_6h",
            group_1="post_event",
            group_2="normal",
        )
    )

    # Test C: post_event vs pre_event
    res.append(
        _ttest_wrapper(
            df,
            post,
            pre,
            test_name="C_post_vs_pre_6h",
            group_1="post_event",
            group_2="pre_event",
        )
    )

    # Test D: crypto post_event vs macro post_event
    res.append(
        _ttest_wrapper(
            df,
            crypto_post,
            macro_post,
            test_name="D_crypto_post_vs_macro_post_6h",
            group_1="crypto_bitcoin_post",
            group_2="macro_post",
        )
    )

    return pd.DataFrame([r.__dict__ for r in res])


# ---------------------------------------------------------------------------
# Event-level summary
# ---------------------------------------------------------------------------

def compute_event_level_summary(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pre/post means by event_id for the 6-hour window."""
    window_df = df[df["regime"].isin(["pre_event", "post_event"])].copy()
    if window_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for ev_id, sub in window_df.groupby("event_id"):
        if pd.isna(ev_id):
            continue
        pre = sub[sub["regime"] == "pre_event"]
        post = sub[sub["regime"] == "post_event"]

        row: Dict[str, object] = {
            "event_id": ev_id,
            "mean_prob_spread_pre": pre["probability_spread"].mean(),
            "mean_prob_spread_post": post["probability_spread"].mean(),
            "mean_abs_spread_pre": pre["abs_spread"].mean(),
            "mean_abs_spread_post": post["abs_spread"].mean(),
        }
        row["abs_spread_post_minus_pre"] = (
            row["mean_abs_spread_post"] - row["mean_abs_spread_pre"]
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    meta = events_df[
        ["event_id", "category", "event_label", "source", "event_timestamp_utc"]
    ].drop_duplicates("event_id")
    out = out.merge(meta, on="event_id", how="left")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_event_time_abs_spread(df: pd.DataFrame) -> None:
    """Figure 1: Average abs_spread by event time (1-hour bins, ±6h)."""
    window_df = df[df["regime"].isin(["pre_event", "post_event"])].copy()
    window_df = window_df.dropna(subset=["event_time_hours", "abs_spread"])
    if window_df.empty:
        logger.warning("No event-window observations for Figure 1 (6h).")
        return

    bins = np.arange(-EVENT_WINDOW_HOURS, EVENT_WINDOW_HOURS + 1, 1.0)
    labels = (bins[:-1] + bins[1:]) / 2.0
    window_df["event_time_bin"] = pd.cut(
        window_df["event_time_hours"], bins=bins, labels=labels, include_lowest=True
    )

    grp = window_df.groupby("event_time_bin", observed=True)["abs_spread"].mean()
    x = grp.index.astype(float)
    y = grp.values

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker="o")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Event time (hours, binned)")
    plt.ylabel("Mean |probability spread|")
    plt.title("Average absolute probability spread by event time (±6h window)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "event_time_abs_spread_6h.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved Figure 1 (6h) to %s", out_path)


def plot_abs_spread_by_regime(df: pd.DataFrame) -> None:
    """Figure 2: Boxplot of abs_spread by regime (6-hour window study)."""
    regimes = ["normal", "pre_event", "post_event"]
    data = [df.loc[df["regime"] == r, "abs_spread"].dropna() for r in regimes]
    if all(len(x) == 0 for x in data):
        logger.warning("No data for Figure 2 boxplot (6h).")
        return

    plt.figure(figsize=(6, 4.8))
    plt.boxplot(data, labels=regimes, showfliers=False)
    plt.ylabel("|probability spread|")
    plt.title("Distribution of absolute probability spread by regime (6h window)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "abs_spread_by_regime_boxplot_6h.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved Figure 2 (6h) to %s", out_path)


def plot_abs_spread_by_category(df: pd.DataFrame) -> None:
    """Figure 3: Bar chart of mean abs_spread by category and regime (6h)."""
    window_df = df[df["regime"].isin(["pre_event", "post_event"])].copy()
    window_df = window_df.dropna(subset=["event_category"])
    if window_df.empty:
        logger.warning("No event-window data for Figure 3 (6h).")
        return

    grp = (
        window_df.groupby(["event_category", "regime"], observed=True)["abs_spread"]
        .mean()
        .reset_index()
    )

    categories = ["crypto_bitcoin", "macro"]
    regimes = ["pre_event", "post_event"]

    means = np.full((len(categories), len(regimes)), np.nan)
    for i, cat in enumerate(categories):
        for j, reg in enumerate(regimes):
            val = grp.loc[
                (grp["event_category"] == cat) & (grp["regime"] == reg), "abs_spread"
            ]
            if not val.empty:
                means[i, j] = float(val.iloc[0])

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(6, 4.5))
    plt.bar(x - width / 2, means[:, 0], width, label="pre_event")
    plt.bar(x + width / 2, means[:, 1], width, label="post_event")
    plt.xticks(x, categories, rotation=0)
    plt.ylabel("Mean |probability spread|")
    plt.title(
        "Mean absolute probability spread\nby category and regime (6h window)"
    )
    plt.legend()
    plt.tight_layout()
    out_path = FIGURES_DIR / "abs_spread_by_category_6h.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved Figure 3 (6h) to %s", out_path)


# ---------------------------------------------------------------------------
# Text summaries and diagnostics
# ---------------------------------------------------------------------------

def write_summary_text(
    df: pd.DataFrame,
    tests_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> None:
    """Write high-level summary for the 6-hour robustness study."""
    out_path = TABLES_DIR / "news_event_study_summary_6h.txt"

    n_events = len(events_df)
    n_crypto = int((events_df["category"] == "crypto_bitcoin").sum())
    n_macro = int((events_df["category"] == "macro").sum())

    counts = df["regime"].value_counts(dropna=False)
    n_normal = int(counts.get("normal", 0))
    n_pre = int(counts.get("pre_event", 0))
    n_post = int(counts.get("post_event", 0))

    regime_means = (
        df.groupby("regime", observed=True)["abs_spread"].mean().to_dict()
    )

    lines: List[str] = []
    lines.append("News Event Study – 6-hour Window Summary")
    lines.append("============================================")
    lines.append("")
    lines.append(f"Number of events: {n_events}")
    lines.append(f"  crypto_bitcoin: {n_crypto}")
    lines.append(f"  macro:          {n_macro}")
    lines.append("")
    lines.append("Row counts by regime:")
    lines.append(f"  normal:    {n_normal}")
    lines.append(f"  pre_event: {n_pre}")
    lines.append(f"  post_event:{n_post}")
    lines.append("")
    lines.append("Mean |probability spread| by regime:")
    for reg in ["normal", "pre_event", "post_event"]:
        val = regime_means.get(reg, float("nan"))
        lines.append(f"  {reg:9s}: {val:.6f}" if pd.notna(val) else f"  {reg:9s}: NA")
    lines.append("")
    lines.append(
        "Key t-tests on abs_spread (difference in means, Welch t-tests, 6h window):"
    )
    for _, row in tests_df.iterrows():
        lines.append(
            f"- {row['test_name']}: {row['group_1']} vs {row['group_2']}, "
            f"diff = {row['difference_in_means']:.6f}, "
            f"t = {row['t_statistic']:.3f}, p = {row['p_value']:.3g}, "
            f"n1 = {int(row['n_group_1'])}, n2 = {int(row['n_group_2'])}"
        )
    lines.append("")
    lines.append("Outputs created:")
    lines.append(f"  - {TABLES_DIR / 'news_event_regime_summary_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_category_summary_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_tests_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_level_summary_6h.csv'}")
    lines.append(f"  - {FIGURES_DIR / 'event_time_abs_spread_6h.png'}")
    lines.append(f"  - {FIGURES_DIR / 'abs_spread_by_regime_boxplot_6h.png'}")
    lines.append(f"  - {FIGURES_DIR / 'abs_spread_by_category_6h.png'}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote 6h summary text to %s", out_path)


def write_diagnostics(
    prob_df: pd.DataFrame,
    events_df: pd.DataFrame,
    df_with_regimes: pd.DataFrame,
) -> str:
    """Write diagnostics file for the 6-hour robustness study."""
    out_path = DIAGNOSTICS_DIR / "news_event_study_diagnostics_6h.txt"

    counts = df_with_regimes["regime"].value_counts(dropna=False)
    n_normal = int(counts.get("normal", 0))
    n_pre = int(counts.get("pre_event", 0))
    n_post = int(counts.get("post_event", 0))

    lines: List[str] = []
    lines.append("News Event Study – 6-hour Window Diagnostics")
    lines.append("============================================")
    lines.append(f"Probability dataset path: {PROBABILITY_PATH}")
    lines.append(f"Event dataset path:       {EVENTS_PATH}")
    lines.append("")
    lines.append(f"Total probability rows loaded: {len(prob_df)}")
    lines.append(f"Total event rows loaded:       {len(events_df)}")
    lines.append("")
    lines.append("Rows by regime after window assignment:")
    lines.append(f"  normal:    {n_normal}")
    lines.append(f"  pre_event: {n_pre}")
    lines.append(f"  post_event:{n_post}")
    lines.append("")
    lines.append("Timestamp parsing:")
    lines.append("  All probability timestamps parsed as UTC.")
    lines.append("  All event timestamps parsed as UTC.")
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Outputs created:")
    lines.append(f"  - {TABLES_DIR / 'news_event_regime_summary_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_category_summary_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_tests_6h.csv'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_level_summary_6h.csv'}")
    lines.append(f"  - {FIGURES_DIR / 'event_time_abs_spread_6h.png'}")
    lines.append(f"  - {FIGURES_DIR / 'abs_spread_by_regime_boxplot_6h.png'}")
    lines.append(f"  - {FIGURES_DIR / 'abs_spread_by_category_6h.png'}")
    lines.append(f"  - {TABLES_DIR / 'news_event_study_summary_6h.txt'}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote 6h diagnostics to %s", out_path)
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    # Load datasets (read-only)
    prob_df = load_probability_data()
    events_df = load_event_data()

    # Parse timestamps as UTC
    prob_df, events_df = parse_timestamps(prob_df, events_df)

    # Assign regimes and event-time variable using the 6-hour window
    df = assign_event_windows(prob_df, events_df)

    # Descriptive tables
    regime_summary = compute_regime_summary(df)
    regime_summary.to_csv(
        TABLES_DIR / "news_event_regime_summary_6h.csv", index=False
    )

    category_summary = compute_category_summary(df)
    category_summary.to_csv(
        TABLES_DIR / "news_event_category_summary_6h.csv", index=False
    )

    # Statistical tests
    tests_df = run_tests(df)
    tests_df.to_csv(TABLES_DIR / "news_event_tests_6h.csv", index=False)

    # Event-level summary
    event_level_summary = compute_event_level_summary(df, events_df)
    event_level_summary.to_csv(
        TABLES_DIR / "news_event_level_summary_6h.csv", index=False
    )

    # Figures
    plot_event_time_abs_spread(df)
    plot_abs_spread_by_regime(df)
    plot_abs_spread_by_category(df)

    # Text summary and diagnostics
    write_summary_text(df, tests_df, events_df)
    diag_text = write_diagnostics(prob_df, events_df, df)

    # Also print diagnostics to console
    print()
    print(diag_text)


if __name__ == "__main__":
    main()

