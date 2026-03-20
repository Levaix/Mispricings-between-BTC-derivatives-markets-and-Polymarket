"""Granger causality tests between BTC futures and probability changes.

Goal:
  Examine predictive relationships between:
    - BTC futures returns
    - Changes in option-implied probabilities
    - Changes in prediction market probabilities

Datasets (READ-ONLY – NEVER modified or written to):
  1) Probability master (overlap sample):
     Actual data/Options N(d2) + Polymarket - Econometric Testing/
       btc_probability_comparison_empirical_master.parquet

  2) Futures OHLCV:
     Actual data/Futures Binance/perp_ohlcv_5m.parquet

All outputs (tables, diagnostics) are written ONLY under:
  Econometric Analysis/Granger Causality Tests/
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (read-only inputs, write-only outputs)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GC_DIR = SCRIPT_DIR.parent  # Granger Causality Tests/
ECONOMETRIC_DIR = GC_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

PROB_DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)
FUTURES_DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Futures Binance"
    / "perp_ohlcv_5m.parquet"
)

TABLES_DIR = GC_DIR / "tables"
DIAG_DIR = GC_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "granger_causality_results.csv"
SUMMARY_TXT = TABLES_DIR / "granger_causality_summary.txt"
DIAG_TXT = DIAG_DIR / "granger_diagnostics.txt"

TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]

REQUIRED_PROB_COLS = ["prob_yes", "prob_option", "event_date", "strike_K"]
REQUIRED_FUTURES_COLS = ["timestamp_utc", "close"]


# ---------------------------------------------------------------------------
# Loading and alignment (read-only)
# ---------------------------------------------------------------------------

def load_probability_dataset() -> pd.DataFrame:
    """Load probability master and ensure required columns exist (read-only)."""
    if not PROB_DATA_PATH.exists():
        raise FileNotFoundError(f"Probability dataset not found: {PROB_DATA_PATH}")
    df = pd.read_parquet(PROB_DATA_PATH)
    missing = [c for c in REQUIRED_PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Probability dataset missing columns: {missing}")

    # Resolve time column (timestamp or bucket_5m_utc)
    time_col = None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(
            f"Probability dataset must contain one of time columns: {TIME_COL_CANDIDATES}"
        )
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "timestamp"})
    return df


def load_futures_dataset() -> pd.DataFrame:
    """Load futures OHLCV dataset and ensure required columns exist (read-only)."""
    if not FUTURES_DATA_PATH.exists():
        raise FileNotFoundError(f"Futures dataset not found: {FUTURES_DATA_PATH}")
    df = pd.read_parquet(FUTURES_DATA_PATH)
    missing = [c for c in REQUIRED_FUTURES_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Futures dataset missing columns: {missing}")
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.rename(columns={"timestamp_utc": "timestamp"})
    return df


def align_and_merge(prob_df: pd.DataFrame, fut_df: pd.DataFrame) -> pd.DataFrame:
    """Align probability and futures datasets on timestamp via inner merge.

Original parquet files are never modified; all work is in-memory.
    """
    prob = prob_df.copy()
    fut = fut_df.copy()
    merged = prob.merge(fut[["timestamp", "close"]], on="timestamp", how="inner")
    merged = merged.sort_values(["event_date", "strike_K", "timestamp"])
    return merged


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def construct_futures_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log futures returns.

futures_return = log(close_t / close_{t-1})
    """
    df = df.copy()
    df = df.sort_values("timestamp")
    close = pd.to_numeric(df["close"], errors="coerce")
    log_price = np.log(close)
    df["futures_return"] = log_price.diff()
    return df


def construct_probability_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute probability changes within (event_date, strike_K) series.

Within each (event_date, strike_K) series:
  d_prob_yes    = prob_yes_t - prob_yes_{t-1}
  d_prob_option = prob_option_t - prob_option_{t-1}

We then drop rows where these differences are missing.
    """
    df = df.copy()
    df["prob_yes"] = pd.to_numeric(df["prob_yes"], errors="coerce")
    df["prob_option"] = pd.to_numeric(df["prob_option"], errors="coerce")

    df = df.sort_values(["event_date", "strike_K", "timestamp"])
    grouped = df.groupby(["event_date", "strike_K"], group_keys=False)

    df["d_prob_yes"] = grouped["prob_yes"].diff()
    df["d_prob_option"] = grouped["prob_option"].diff()

    df = df.dropna(subset=["d_prob_yes", "d_prob_option"])
    return df


def collapse_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse panel to a single time series via cross-sectional averaging.

Granger causality in `statsmodels` operates on single time series, not
panels. To respect this requirement, we average the probability changes
across all (event_date, strike_K) at each timestamp:

  avg_d_prob_yes_t    = mean_i d_prob_yes_{t,i}
  avg_d_prob_option_t = mean_i d_prob_option_{t,i}

This retains the time dimension while aggregating cross-sectional detail.
    """
    df = df.copy()
    # Include futures_return (same value at each timestamp) so the collapsed ts has it
    agg = {
        "d_prob_yes": "mean",
        "d_prob_option": "mean",
        "futures_return": "first",
    }
    ts = (
        df.groupby("timestamp", as_index=False)
        .agg(agg)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Rename to explicit average-series names used in tests
    ts = ts.rename(
        columns={
            "d_prob_yes": "avg_d_prob_yes",
            "d_prob_option": "avg_d_prob_option",
        }
    )
    return ts


# ---------------------------------------------------------------------------
# Granger causality tests
# ---------------------------------------------------------------------------

def run_granger_tests(
    y: pd.Series,
    x: pd.Series,
    max_lag: int,
) -> List[Dict[str, float]]:
    """Run Granger causality tests of x → y for lags up to max_lag.

Returns a list of dicts with lag, F-statistic, and p-value (using ssr_ftest).
    """
    data = pd.DataFrame({"y": y.values, "x": x.values}).dropna()
    results: List[Dict[str, float]] = []

    # statsmodels expects array with columns [y, x] for x → y test
    arr = data[["y", "x"]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        gc_res = grangercausalitytests(arr, maxlag=max_lag, verbose=False)

    for lag, res in gc_res.items():
        # Newer statsmodels: per-lag result is (test_dict, ...); older: test_dict directly
        test_dict = res[0] if isinstance(res, tuple) else res
        # 'ssr_ftest' entry: (F-statistic, p-value, df_denom, df_num)
        f_stat, p_val, _, _ = test_dict["ssr_ftest"]
        results.append({"lag": lag, "F_statistic": float(f_stat), "p_value": float(p_val)})
    return results


def build_results_table(
    ts: pd.DataFrame,
    max_lag: int = 2,
) -> pd.DataFrame:
    """Run all four Granger directions and assemble a results table.

Directions:
  1) futures → options       (futures_return → avg_d_prob_option)
  2) futures → prediction_markets (futures_return → avg_d_prob_yes)
  3) options → prediction_markets (avg_d_prob_option → avg_d_prob_yes)
  4) prediction_markets → options (avg_d_prob_yes → avg_d_prob_option)
    """
    rows: List[Dict[str, object]] = []

    yx_specs: List[Tuple[str, str, str]] = [
        ("futures → options", "avg_d_prob_option", "futures_return"),
        ("futures → prediction_markets", "avg_d_prob_yes", "futures_return"),
        ("options → prediction_markets", "avg_d_prob_yes", "avg_d_prob_option"),
        ("prediction_markets → options", "avg_d_prob_option", "avg_d_prob_yes"),
    ]

    for direction, y_col, x_col in yx_specs:
        res_list = run_granger_tests(ts[y_col], ts[x_col], max_lag=max_lag)
        for entry in res_list:
            rows.append(
                {
                    "direction": direction,
                    "lag": entry["lag"],
                    "F_statistic": entry["F_statistic"],
                    "p_value": entry["p_value"],
                }
            )

    return pd.DataFrame(rows)


def build_summary_text(ts: pd.DataFrame) -> str:
    """Short summary for the Granger causality module."""
    n_obs = len(ts)
    ts_min = ts["timestamp"].min()
    ts_max = ts["timestamp"].max()

    lines: List[str] = []
    lines.append("Granger causality tests – futures and probability changes")
    lines.append("=" * 72)
    lines.append(f"Probability dataset: {PROB_DATA_PATH}")
    lines.append(f"Futures dataset:     {FUTURES_DATA_PATH}")
    lines.append(f"Observations used in time series: {n_obs}")
    lines.append(f"Timestamp range: {ts_min} to {ts_max}")
    lines.append("")
    lines.append(
        "Note: The original panel (event_date, strike_K) is collapsed to a single "
        "time series by averaging probability changes across contracts at each "
        "timestamp. This is required because standard Granger causality tests "
        "operate on univariate (or multivariate) time series, not panels."
    )
    return "\n".join(lines)


def build_diagnostics_text(ts: pd.DataFrame, ok: bool) -> str:
    """Diagnostics for the Granger causality step."""
    n_obs = len(ts)
    ts_min = ts["timestamp"].min()
    ts_max = ts["timestamp"].max()

    lines: List[str] = []
    lines.append("Granger causality diagnostics")
    lines.append("=" * 40)
    lines.append(f"Rows used after aggregation: {n_obs}")
    lines.append(f"Timestamp range: {ts_min} to {ts_max}")
    lines.append(f"Tests ran successfully: {'YES' if ok else 'NO'}")
    lines.append("")
    lines.append("Outputs created:")
    lines.append(f"  - {RESULTS_CSV.name}")
    lines.append(f"  - {SUMMARY_TXT.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure output folder structure exists (NO writes to Actual data/)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets (read-only)
    prob_df = load_probability_dataset()
    fut_df = load_futures_dataset()

    # Align and merge
    merged = align_and_merge(prob_df, fut_df)
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between probability and futures datasets.")

    # Construct futures returns
    merged = construct_futures_returns(merged)

    # Construct probability changes within (event_date, strike_K)
    merged = construct_probability_diffs(merged)

    # Collapse to time series via cross-sectional averaging at each timestamp
    ts = collapse_to_timeseries(merged)

    # Run Granger causality tests (lags 1 and 2)
    results_df = build_results_table(ts, max_lag=2)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Summary
    summary_txt = build_summary_text(ts)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Diagnostics
    diag_txt = build_diagnostics_text(ts, ok=True)
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

