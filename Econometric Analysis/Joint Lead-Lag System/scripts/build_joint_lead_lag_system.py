"""Joint Lead–Lag Information Flow System for BTC futures, options, and prediction markets.

Goal:
  Test joint lead–lag relationships between:
    - BTC futures returns
    - Option-implied probabilities
    - Polymarket prediction market probabilities

Datasets (READ-ONLY):
  1) Probability master (overlap sample):
     Actual data/Options N(d2) + Polymarket - Econometric Testing/
       btc_probability_comparison_empirical_master.parquet

  2) Futures OHLCV:
     Actual data/Futures Binance/perp_ohlcv_5m.parquet

IMPORTANT DATA SAFETY:
  This script NEVER writes to or modifies anything under `Actual data/`.
  All outputs are written only to:
    Econometric Analysis/Joint Lead-Lag System/{tables,figures,diagnostics}
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
JOINT_DIR = SCRIPT_DIR.parent  # Joint Lead-Lag System/
ECONOMETRIC_DIR = JOINT_DIR.parent  # Econometric Analysis/
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

TABLES_DIR = JOINT_DIR / "tables"
FIGURES_DIR = JOINT_DIR / "figures"
DIAG_DIR = JOINT_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "joint_lead_lag_results.csv"
SUMMARY_TXT = TABLES_DIR / "joint_lead_lag_summary.txt"
FIG_PNG = FIGURES_DIR / "joint_information_flow_coefficients.png"
DIAG_TXT = DIAG_DIR / "joint_lead_lag_diagnostics.txt"

TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]

REQUIRED_PROB_COLS = ["prob_yes", "prob_option", "event_date", "strike_K"]
REQUIRED_FUTURES_COLS = ["timestamp_utc", "close"]


# ---------------------------------------------------------------------------
# Load and align datasets (READ ONLY)
# ---------------------------------------------------------------------------

def load_probability_dataset() -> pd.DataFrame:
    """Load probability master and ensure required columns exist."""
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
    """Load futures OHLCV dataset and ensure required columns exist."""
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

All modifications happen in memory; the original parquet files are untouched.
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
    """Compute log futures returns and their lag.

Returns are computed at the timestamp level for the underlying futures close
series and then attached to all probability rows at that timestamp.
    """
    df = df.copy()
    df = df.sort_values("timestamp")
    close = pd.to_numeric(df["close"], errors="coerce")
    log_price = np.log(close)
    fut_ret = log_price.diff()
    df["futures_return"] = fut_ret
    df["futures_return_lag1"] = df["futures_return"].shift(1)
    return df


def construct_probability_diffs_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute probability changes and their lags within (event_date, strike_K).

Within each (event_date, strike_K) series, we sort by timestamp and compute:
  d_prob_yes    = prob_yes_t - prob_yes_{t-1}
  d_prob_option = prob_option_t - prob_option_{t-1}
  d_prob_yes_lag1, d_prob_option_lag1 as one-period lags.

This stays strictly within each event and strike; there is no mixing across
different events or strikes.
    """
    df = df.copy()
    df["prob_yes"] = pd.to_numeric(df["prob_yes"], errors="coerce")
    df["prob_option"] = pd.to_numeric(df["prob_option"], errors="coerce")

    df = df.sort_values(["event_date", "strike_K", "timestamp"])
    grouped = df.groupby(["event_date", "strike_K"], group_keys=False)

    df["d_prob_yes"] = grouped["prob_yes"].diff()
    df["d_prob_option"] = grouped["prob_option"].diff()

    df["d_prob_yes_lag1"] = grouped["d_prob_yes"].shift(1)
    df["d_prob_option_lag1"] = grouped["d_prob_option"].shift(1)

    # Drop rows with missing required variables
    df = df.dropna(
        subset=[
            "d_prob_yes",
            "d_prob_option",
            "d_prob_yes_lag1",
            "d_prob_option_lag1",
            "futures_return_lag1",
            "event_date",
        ]
    )
    return df


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

def run_clustered_ols(
    y: pd.Series, X: pd.DataFrame, clusters: pd.Series
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with cluster-robust SEs clustered by event_date."""
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters.astype(str)})
    return res


def build_results_table(
    res1: sm.regression.linear_model.RegressionResultsWrapper,
    res2: sm.regression.linear_model.RegressionResultsWrapper,
    n_clusters: int,
) -> pd.DataFrame:
    """Build combined results table for MODEL 1 and MODEL 2."""
    rows = []
    for reg_name, res in [
        ("MODEL1_pm", res1),
        ("MODEL2_opt", res2),
    ]:
        params = res.params
        se = res.bse
        tvals = res.tvalues
        pvals = res.pvalues
        ci = res.conf_int(alpha=0.05)
        for term in params.index:
            coef = float(params[term])
            std_err = float(se[term])
            t_stat = float(tvals[term])
            p_val = float(pvals[term])
            ci_lower = float(ci.loc[term, 0])
            ci_upper = float(ci.loc[term, 1])
            rows.append(
                {
                    "model": reg_name,
                    "term": term,
                    "coefficient": coef,
                    "std_error_clustered": std_err,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "r_squared": float(res.rsquared),
                    "observations": int(res.nobs),
                    "clusters": int(n_clusters),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )
    return pd.DataFrame(rows)


def build_summary_text(
    res1: sm.regression.linear_model.RegressionResultsWrapper,
    res2: sm.regression.linear_model.RegressionResultsWrapper,
    n_obs: int,
    n_clusters: int,
) -> str:
    """Readable summary of joint lead–lag system."""
    # MODEL 1 coefficients
    alpha_1 = float(res1.params["const"])
    beta1 = float(res1.params["futures_return_lag1"])
    beta2 = float(res1.params["d_prob_option_lag1"])
    r2_1 = float(res1.rsquared)

    # MODEL 2 coefficients
    alpha_2 = float(res2.params["const"])
    gamma1 = float(res2.params["futures_return_lag1"])
    gamma2 = float(res2.params["d_prob_yes_lag1"])
    r2_2 = float(res2.rsquared)

    lines: list[str] = []
    lines.append("Joint Lead–Lag Information Flow System")
    lines.append("=" * 72)
    lines.append(f"Probability dataset: {PROB_DATA_PATH}")
    lines.append(f"Futures dataset:     {FUTURES_DATA_PATH}")
    lines.append(f"Rows after merge & filtering: {n_obs}")
    lines.append(f"Event clusters (event_date):  {n_clusters}")
    lines.append("")
    lines.append("MODEL 1: What moves prediction markets?")
    lines.append("  d_prob_yes_t = alpha + beta1 * futures_return_lag1 + beta2 * d_prob_option_lag1 + error")
    lines.append(f"  alpha: {alpha_1: .6f}")
    lines.append(f"  beta1 (futures → PM):           {beta1: .6f}")
    lines.append(f"  beta2 (options → PM):           {beta2: .6f}")
    lines.append(f"  R^2: {r2_1: .6f}")
    lines.append("")
    lines.append("MODEL 2: What moves options?")
    lines.append("  d_prob_option_t = alpha + gamma1 * futures_return_lag1 + gamma2 * d_prob_yes_lag1 + error")
    lines.append(f"  alpha: {alpha_2: .6f}")
    lines.append(f"  gamma1 (futures → options):     {gamma1: .6f}")
    lines.append(f"  gamma2 (PM → options):          {gamma2: .6f}")
    lines.append(f"  R^2: {r2_2: .6f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- beta1/gamma1 capture how futures returns feed into PM and options, "
        "while beta2/gamma2 capture cross-effects between PM and options."
    )
    lines.append(
        "- Clustering by event_date accounts for within-event dependence while "
        "treating different events as independent clusters."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_coefficient_comparison(results: pd.DataFrame) -> None:
    """Bar plot of lead–lag coefficients with 95% CI error bars.

We focus on the key lead–lag terms:
  - MODEL1: futures_return_lag1 (beta1), d_prob_option_lag1 (beta2)
  - MODEL2: futures_return_lag1 (gamma1), d_prob_yes_lag1 (gamma2)
    """
    if results.empty:
        logger.warning("No joint lead–lag results to plot.")
        return

    # Filter to slope terms only (exclude intercepts)
    slope_terms = results[results["term"] != "const"].copy()

    # Sort by model, then term for a stable ordering
    slope_terms = slope_terms.sort_values(["model", "term"]).reset_index(drop=True)

    betas = slope_terms["coefficient"].values
    ses = slope_terms["std_error_clustered"].values
    labels = [f"{m}:{t}" for m, t in zip(slope_terms["model"], slope_terms["term"])]

    x = np.arange(len(betas))
    fig, ax = plt.subplots(figsize=(max(8, len(betas) * 0.8), 4))
    ax.bar(x, betas, yerr=1.96 * ses, capsize=4, alpha=0.8, color="C0")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient", fontsize=10)
    ax.set_title("Joint information flow: key lead–lag coefficients (95% CIs)", fontsize=12)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def build_diagnostics_text(
    n_merged: int,
    n_used: int,
    n_clusters: int,
    ok: bool,
) -> str:
    """Short diagnostics summary for joint lead–lag system."""
    lines = [
        "Joint lead–lag diagnostics",
        "=" * 40,
        f"Probability dataset:          {PROB_DATA_PATH}",
        f"Futures dataset:              {FUTURES_DATA_PATH}",
        f"Rows after merge:             {n_merged}",
        f"Rows used in regressions:     {n_used}",
        f"Unique event_date clusters:   {n_clusters}",
        f"Regressions ran successfully: {'YES' if ok else 'NO'}",
        "",
        "Outputs created:",
        f"  - {RESULTS_CSV.name}",
        f"  - {SUMMARY_TXT.name}",
        f"  - {FIG_PNG.name}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure output folder structure exists (NO writes to Actual data/)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets (read-only)
    prob_df = load_probability_dataset()
    fut_df = load_futures_dataset()

    # Align and merge
    merged = align_and_merge(prob_df, fut_df)
    n_merged = len(merged)
    if n_merged == 0:
        raise RuntimeError("No overlapping timestamps between probability and futures datasets.")

    # Construct futures returns and lags
    merged = construct_futures_returns(merged)

    # Construct probability differences and lags within (event_date, strike_K)
    merged = construct_probability_diffs_and_lags(merged)
    n_used = len(merged)
    if n_used == 0:
        raise RuntimeError("No observations left after constructing differences and lags.")

    clusters = merged["event_date"].astype(str)
    n_clusters = clusters.nunique()

    # MODEL 1: d_prob_yes_t on futures_return_lag1 and d_prob_option_lag1
    y1 = merged["d_prob_yes"]
    X1 = merged[["futures_return_lag1", "d_prob_option_lag1"]]
    res1 = run_clustered_ols(y1, X1, clusters)

    # MODEL 2: d_prob_option_t on futures_return_lag1 and d_prob_yes_lag1
    y2 = merged["d_prob_option"]
    X2 = merged[["futures_return_lag1", "d_prob_yes_lag1"]]
    res2 = run_clustered_ols(y2, X2, clusters)

    # Results table
    results_df = build_results_table(res1, res2, n_clusters=n_clusters)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Summary text
    summary_txt = build_summary_text(res1, res2, n_obs=n_used, n_clusters=n_clusters)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Visualization
    plot_coefficient_comparison(results_df)

    # Diagnostics
    diag_txt = build_diagnostics_text(
        n_merged=n_merged,
        n_used=n_used,
        n_clusters=n_clusters,
        ok=True,
    )
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

