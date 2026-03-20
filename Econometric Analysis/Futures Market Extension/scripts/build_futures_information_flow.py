"""Futures Market Extension – Information Flow Tests.

Tests whether BTC futures returns predict changes in option-implied
probabilities and prediction market probabilities using the empirical
master dataset and Binance futures data.

Datasets:
- Probability: Actual data/Options N(d2) + Polymarket - Econometric Testing/
    btc_probability_comparison_empirical_master.parquet
- Futures:     Actual data/Futures Binance/perp_ohlcv_5m.parquet

Regressions:
  A) Futures → Prediction Markets
     d_prob_yes_t = alpha_A + beta_A * futures_return_lag1 + error_A

  B) Futures → Options
     d_prob_option_t = alpha_B + beta_B * futures_return_lag1 + error_B

Cluster-robust standard errors are grouped by event_date.
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
# Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FUTURES_EXT_DIR = SCRIPT_DIR.parent  # Futures Market Extension/
ECONOMETRIC_DIR = FUTURES_EXT_DIR.parent  # Econometric Analysis/
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

TABLES_DIR = FUTURES_EXT_DIR / "tables"
FIGURES_DIR = FUTURES_EXT_DIR / "figures"
DIAG_DIR = FUTURES_EXT_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "futures_information_flow_results.csv"
SUMMARY_TXT = TABLES_DIR / "futures_information_flow_summary.txt"
FIG_PM_PNG = FIGURES_DIR / "futures_prediction_market_response.png"
FIG_OPT_PNG = FIGURES_DIR / "futures_option_response.png"
DIAG_TXT = DIAG_DIR / "futures_extension_diagnostics.txt"

TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]

REQUIRED_PROB_COLS = [
    "prob_yes",
    "prob_option",
    "event_date",
    "strike_K",
]
REQUIRED_FUTURES_COLS = [
    "timestamp_utc",
    "close",
]


# ---------------------------------------------------------------------------
# Loading and alignment
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
    """Align probability and futures datasets on timestamp via inner merge."""
    prob = prob_df.copy()
    fut = fut_df.copy()
    merged = prob.merge(fut[["timestamp", "close"]], on="timestamp", how="inner")
    merged = merged.sort_values(["event_date", "strike_K", "timestamp"])
    return merged


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def construct_futures_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log futures returns and their lag (global futures series).

We treat the futures close price as a single time series across all events
and strikes. Returns are computed at the timestamp level, then merged back
onto all probability rows at that timestamp.
    """
    df = df.copy()
    # Compute log return at each timestamp for the underlying futures close
    df = df.sort_values("timestamp")
    close = pd.to_numeric(df["close"], errors="coerce")
    log_price = np.log(close)
    fut_ret = log_price.diff()
    df["futures_return"] = fut_ret
    df["futures_return_lag1"] = df["futures_return"].shift(1)
    return df


def construct_probability_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute d_prob_yes, d_prob_option within each (event_date, strike_K) series.

All differences are constructed within each event/strike so that we do not
mix dynamics across different events or strikes.
    """
    df = df.copy()
    df["prob_yes"] = pd.to_numeric(df["prob_yes"], errors="coerce")
    df["prob_option"] = pd.to_numeric(df["prob_option"], errors="coerce")

    grouped = df.groupby(["event_date", "strike_K"], group_keys=False)
    df["d_prob_yes"] = grouped["prob_yes"].diff()
    df["d_prob_option"] = grouped["prob_option"].diff()

    # Drop rows where key inputs are missing
    df = df.dropna(
        subset=[
            "d_prob_yes",
            "d_prob_option",
            "futures_return_lag1",
            "event_date",
        ]
    )
    return df


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

def run_clustered_ols(
    y: pd.Series, x: pd.Series, clusters: pd.Series
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with cluster-robust SEs clustered by event_date."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters.astype(str)})
    return res


def build_results_table(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
    n_clusters: int,
) -> pd.DataFrame:
    """Construct combined results table for regressions A and B.

All regressions are clustered by event_date, so the same number of clusters
applies to both A and B; we pass it in explicitly from the calling context.
    """
    rows = []
    for reg_name, res in [
        ("A_futures_lead_pm", res_A),
        ("B_futures_lead_options", res_B),
    ]:
        params = res.params
        se = res.bse
        tvals = res.tvalues
        pvals = res.pvalues
        ci = res.conf_int(alpha=0.05)
        for term in ["const", "futures_return_lag1"]:
            coef = float(params[term])
            std_err = float(se[term])
            t_stat = float(tvals[term])
            p_val = float(pvals[term])
            ci_lower = float(ci.loc[term, 0])
            ci_upper = float(ci.loc[term, 1])
            rows.append(
                {
                    "regression": reg_name,
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


def clusters_from_res(res: sm.regression.linear_model.RegressionResultsWrapper) -> int:
    """Extract number of clusters from clustered covariance structure if present."""
    # statsmodels stores cluster counts in the 'cov_kwds' or in robust covariance results
    cov_kwds = getattr(res, "cov_kwds", {}) or {}
    n_groups = cov_kwds.get("groups", None)
    # If groups vector is not stored, fall back to NaN
    if isinstance(n_groups, (int, np.integer)):
        return n_groups
    return np.nan


def build_summary_text(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
    n_obs: int,
    n_clusters: int,
) -> str:
    """Readable summary of futures information flow regressions."""
    alpha_A = float(res_A.params["const"])
    beta_A = float(res_A.params["futures_return_lag1"])
    r2_A = float(res_A.rsquared)

    alpha_B = float(res_B.params["const"])
    beta_B = float(res_B.params["futures_return_lag1"])
    r2_B = float(res_B.rsquared)

    lines: list[str] = []
    lines.append("Futures Market Extension – Information Flow")
    lines.append("=" * 72)
    lines.append(f"Probability dataset: {PROB_DATA_PATH}")
    lines.append(f"Futures dataset:     {FUTURES_DATA_PATH}")
    lines.append(f"Rows after merge & filtering: {n_obs}")
    lines.append(f"Event clusters (event_date):  {n_clusters}")
    lines.append("")
    lines.append("Regression A (Futures → Prediction Markets):")
    lines.append("  d_prob_yes_t = alpha_A + beta_A * futures_return_lag1 + error_A")
    lines.append(f"  alpha_A: {alpha_A: .6f}")
    lines.append(f"  beta_A : {beta_A: .6f}")
    lines.append(f"  R^2:    {r2_A: .6f}")
    lines.append("")
    lines.append("Regression B (Futures → Options):")
    lines.append("  d_prob_option_t = alpha_B + beta_B * futures_return_lag1 + error_B")
    lines.append(f"  alpha_B: {alpha_B: .6f}")
    lines.append(f"  beta_B : {beta_B: .6f}")
    lines.append(f"  R^2:    {r2_B: .6f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- If beta_A > 0 and statistically significant, futures returns tend to "
        "precede and predict changes in Polymarket probabilities."
    )
    lines.append(
        "- If beta_B > 0 and statistically significant, futures returns tend to "
        "precede and predict changes in option-implied probabilities."
    )
    lines.append(
        "- Clustering by event_date allows for correlated shocks within events "
        "while treating different events as independent clusters."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_response_scatter(
    x: pd.Series,
    y: pd.Series,
    res: sm.regression.linear_model.RegressionResultsWrapper,
    out_path: Path,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Scatter plot of y vs x with fitted regression line.

We use a simple scatter with strong transparency for readability. For dense
data, this behaves similarly to a low-resolution hexbin.
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.15, s=10, color="C0")

    # Fitted line
    x_min, x_max = float(x.min()), float(x.max())
    x_line = np.linspace(x_min, x_max, 200)
    alpha_hat = float(res.params["const"])
    beta_hat = float(res.params["futures_return_lag1"])
    y_fit = alpha_hat + beta_hat * x_line
    ax.plot(x_line, y_fit, color="red", linewidth=2, label="Fitted regression line")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Zero response")

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def build_diagnostics_text(
    n_merged: int,
    n_used: int,
    n_clusters: int,
    ok: bool,
) -> str:
    """Short diagnostics summary for futures extension step."""
    lines = [
        "Futures market extension diagnostics",
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
        f"  - {FIG_PM_PNG.name}",
        f"  - {FIG_OPT_PNG.name}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure folder structure exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    prob_df = load_probability_dataset()
    fut_df = load_futures_dataset()

    # Align and merge
    merged = align_and_merge(prob_df, fut_df)
    n_merged = len(merged)
    if n_merged == 0:
        raise RuntimeError("No overlapping timestamps between probability and futures datasets.")

    # Construct features: futures returns and lag
    merged = construct_futures_returns(merged)

    # Probability differences within (event_date, strike_K)
    merged = construct_probability_diffs(merged)
    n_used = len(merged)
    if n_used == 0:
        raise RuntimeError("No observations left after constructing returns and differences.")

    # Prepare for regressions
    clusters = merged["event_date"].astype(str)
    n_clusters = clusters.nunique()

    # Regression A: Futures → Prediction Markets
    y_A = merged["d_prob_yes"]
    x_A = merged["futures_return_lag1"]
    res_A = run_clustered_ols(y_A, x_A, clusters)

    # Regression B: Futures → Options
    y_B = merged["d_prob_option"]
    x_B = merged["futures_return_lag1"]
    res_B = run_clustered_ols(y_B, x_B, clusters)

    # Results table
    results_df = build_results_table(res_A, res_B, n_clusters=n_clusters)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Summary text
    summary_txt = build_summary_text(res_A, res_B, n_obs=n_used, n_clusters=n_clusters)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Visualizations
    plot_response_scatter(
        x=merged["futures_return_lag1"],
        y=merged["d_prob_yes"],
        res=res_A,
        out_path=FIG_PM_PNG,
        x_label="Lagged futures log return",
        y_label="Change in Polymarket probability (d_prob_yes)",
        title="Futures returns and Polymarket probability changes",
    )
    plot_response_scatter(
        x=merged["futures_return_lag1"],
        y=merged["d_prob_option"],
        res=res_B,
        out_path=FIG_OPT_PNG,
        x_label="Lagged futures log return",
        y_label="Change in option-implied probability (d_prob_option)",
        title="Futures returns and option-implied probability changes",
    )

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

