"""Spread Dynamics step for the thesis.

Uses the empirical master dataset (overlap sample):
  btc_probability_comparison_empirical_master.parquet
from:
  Actual data/Options N(d2) + Polymarket - Econometric Testing

Goal:
  Study the dynamics of the probability spread
    probability_spread = prob_yes - prob_option
  via an AR(1)-type regression:
    probability_spread_t = alpha + rho * spread_lag1 + error_t

Lags are constructed strictly within each (event_date, strike_K) series so that
we do not mix information across different events or strikes. Cluster-robust
standard errors are computed by event_date to allow arbitrary correlation
within each event.
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
# Script lives in Econometric Analysis/Spread Dynamics/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
SPREAD_DYN_DIR = SCRIPT_DIR.parent  # Spread Dynamics/
ECONOMETRIC_DIR = SPREAD_DYN_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = SPREAD_DYN_DIR / "tables"
FIGURES_DIR = SPREAD_DYN_DIR / "figures"
DIAG_DIR = SPREAD_DYN_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "spread_dynamics_results.csv"
SUMMARY_TXT = TABLES_DIR / "spread_dynamics_summary.txt"
FIG_PNG = FIGURES_DIR / "spread_dynamics_fit.png"
DIAG_TXT = DIAG_DIR / "spread_dynamics_diagnostics.txt"

TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]
REQUIRED_BASE_COLS = ["event_date", "strike_K", "probability_spread"]


def load_dataset() -> pd.DataFrame:
    """Load empirical master and ensure required columns exist.

We require event_date, strike_K, probability_spread and at least one of
timestamp or bucket_5m_utc as the time axis.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if not any(col in df.columns for col in TIME_COL_CANDIDATES):
        raise ValueError(
            f"Dataset must contain a time column; expected one of {TIME_COL_CANDIDATES}."
        )
    return df


def resolve_time_column(df: pd.DataFrame) -> str:
    """Return the name of the time column (timestamp or bucket_5m_utc)."""
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    # Should not happen if load_dataset already validated
    raise ValueError(f"No valid time column found; expected one of {TIME_COL_CANDIDATES}.")


def construct_lagged_spread(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Construct spread_lag1 within each (event_date, strike_K) series.

We sort by (event_date, strike_K, time_col), then for each combination we
shift probability_spread by one period. Rows where the lag is missing are
dropped so the regression sample only contains well-defined spread_lag1.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Sort before grouping so the lag is time-ordered within each series
    df = df.sort_values(["event_date", "strike_K", time_col])

    # Coerce spread to numeric
    df["probability_spread"] = pd.to_numeric(df["probability_spread"], errors="coerce")

    # Lag within each (event_date, strike_K)
    df["spread_lag1"] = (
        df.groupby(["event_date", "strike_K"])["probability_spread"].shift(1)
    )

    # Drop rows where lag is missing or current spread is missing
    df = df.dropna(subset=["probability_spread", "spread_lag1", "event_date"])
    return df


def run_spread_dynamics_regression(
    sample: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with cluster-robust SEs clustered by event_date.

Regression:
  probability_spread_t = alpha + rho * spread_lag1 + error_t

Clustering by event_date allows arbitrary correlation of residuals within
each event (e.g. across strikes and timestamps for the same macro event),
while assuming independence across different events.
    """
    y = sample["probability_spread"]
    X = sm.add_constant(sample["spread_lag1"])
    groups = sample["event_date"].astype(str)

    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res


def build_results_table(res: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """Construct tidy results for alpha (const) and rho (spread_lag1)."""
    params = res.params
    se = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    ci = res.conf_int(alpha=0.05)

    rows = []
    for term in ["const", "spread_lag1"]:
        coef = float(params[term])
        std_err = float(se[term])
        t_stat = float(tvals[term])
        p_val = float(pvals[term])
        ci_lower = float(ci.loc[term, 0])
        ci_upper = float(ci.loc[term, 1])
        rows.append(
            {
                "term": term,
                "coefficient": coef,
                "std_error_clustered": std_err,
                "t_stat": t_stat,
                "p_value": p_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def run_hypothesis_tests(
    res: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict[str, dict[str, float]]:
    """Run H0: alpha = 0, H0: rho = 0, H0: rho = 1."""
    tests: dict[str, dict[str, float]] = {}

    # H0: alpha (const) = 0
    t_alpha = res.t_test("const = 0")
    tests["alpha_eq_0"] = {
        "stat": float(t_alpha.statistic),
        "p_value": float(t_alpha.pvalue),
    }

    # H0: rho (spread_lag1) = 0
    t_rho_0 = res.t_test("spread_lag1 = 0")
    tests["rho_eq_0"] = {
        "stat": float(t_rho_0.statistic),
        "p_value": float(t_rho_0.pvalue),
    }

    # H0: rho (spread_lag1) = 1
    t_rho_1 = res.t_test("spread_lag1 = 1")
    tests["rho_eq_1"] = {
        "stat": float(t_rho_1.statistic),
        "p_value": float(t_rho_1.pvalue),
    }

    return tests


def plot_spread_dynamics(sample: pd.DataFrame, res: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """Hexbin plot of (spread_lag1, probability_spread) with fitted line and axes at 0.

rho (the coefficient on spread_lag1) measures persistence of spreads:
- If rho is close to 0, spreads mean-revert quickly.
- If rho is close to 1, spreads are highly persistent over time.
    """
    x = pd.to_numeric(sample["spread_lag1"], errors="coerce")
    y = pd.to_numeric(sample["probability_spread"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(6, 6))

    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")

    # Fitted regression line y = alpha + rho * x over observed x-range
    x_min, x_max = float(x.min()), float(x.max())
    x_line = np.linspace(x_min, x_max, 200)
    alpha_hat = float(res.params["const"])
    rho_hat = float(res.params["spread_lag1"])
    y_fit = alpha_hat + rho_hat * x_line
    ax.plot(x_line, y_fit, color="red", linewidth=2, label="Fitted regression line")

    # Axes at zero
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero spread")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Lagged spread (spread_lag1)", fontsize=10)
    ax.set_ylabel("Current spread (probability_spread)", fontsize=10)
    ax.set_title("Spread dynamics: current vs lagged spread", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


def build_summary_text(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    tests: dict[str, dict[str, float]],
    n_obs: int,
    n_clusters: int,
    n_series: int,
) -> str:
    """Readable summary of spread dynamics regression."""
    alpha_hat = float(res.params["const"])
    rho_hat = float(res.params["spread_lag1"])
    r2 = float(res.rsquared)
    r2_adj = float(res.rsquared_adj)

    lines: list[str] = []
    lines.append("Spread dynamics regression: spread_t = alpha + rho * spread_lag1 + error_t")
    lines.append("=" * 78)
    lines.append(f"Dataset:                        {DATA_PATH}")
    lines.append(f"Observations (after lag):       {n_obs}")
    lines.append(f"Event clusters (event_date):    {n_clusters}")
    lines.append(f"(event_date, strike_K) series:  {n_series}")
    lines.append("")
    lines.append("Coefficients (cluster-robust SE by event_date):")
    lines.append(f"  alpha (const):      {alpha_hat: .6f}")
    lines.append(f"  rho (spread_lag1):  {rho_hat: .6f}")
    lines.append("")
    lines.append(f"R-squared:            {r2: .6f}")
    lines.append(f"Adjusted R-squared:   {r2_adj: .6f}")
    lines.append("")
    lines.append("Hypothesis tests:")
    lines.append("  H0: alpha = 0")
    lines.append(f"    t-statistic: {tests['alpha_eq_0']['stat']:.6f}")
    lines.append(f"    p-value:     {tests['alpha_eq_0']['p_value']:.6g}")
    lines.append("  H0: rho = 0")
    lines.append(f"    t-statistic: {tests['rho_eq_0']['stat']:.6f}")
    lines.append(f"    p-value:     {tests['rho_eq_0']['p_value']:.6g}")
    lines.append("  H0: rho = 1")
    lines.append(f"    t-statistic: {tests['rho_eq_1']['stat']:.6f}")
    lines.append(f"    p-value:     {tests['rho_eq_1']['p_value']:.6g}")
    lines.append("")
    lines.append("Economic interpretation:")
    lines.append(
        "- rho measures the persistence of probability spreads. Values close to 1 "
        "indicate highly persistent deviations between Polymarket and options, "
        "whereas values near 0 indicate rapid mean reversion."
    )
    lines.append(
        "- Clustering by event_date allows dependence within an event (across "
        "strikes and timestamps) while treating different events as independent "
        "clusters."
    )
    return "\n".join(lines)


def build_diagnostics_text(
    total_rows: int,
    n_used: int,
    n_clusters: int,
    n_series: int,
    ok: bool,
) -> str:
    """Short diagnostics summary for spread dynamics step."""
    lines = [
        "Spread dynamics diagnostics",
        "=" * 40,
        f"Dataset path:                            {DATA_PATH}",
        f"Total input rows:                        {total_rows}",
        f"Rows used after lag construction:        {n_used}",
        f"Unique event_date clusters:              {n_clusters}",
        f"Unique (event_date, strike_K) series:    {n_series}",
        "Lag constructed within (event_date, strike_K): YES",
        f"Regression ran successfully:             {'YES' if ok else 'NO'}",
        "",
        "Outputs created:",
        f"  - {RESULTS_CSV.name}",
        f"  - {SUMMARY_TXT.name}",
        f"  - {FIG_PNG.name}",
    ]
    return "\n".join(lines)


def main() -> None:
    # Ensure folder structure exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_dataset()
    total_rows = len(df)
    time_col = resolve_time_column(df)
    df_lagged = construct_lagged_spread(df, time_col)
    n_used = len(df_lagged)
    if n_used == 0:
        raise RuntimeError("No observations left after constructing lagged spreads.")

    n_clusters = df_lagged["event_date"].astype(str).nunique()
    n_series = (
        df_lagged[["event_date", "strike_K"]].drop_duplicates().shape[0]
    )

    # Run regression
    res = run_spread_dynamics_regression(df_lagged)

    # Results table
    results_df = build_results_table(res)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Hypothesis tests
    tests = run_hypothesis_tests(res)

    # Summary text
    summary_txt = build_summary_text(
        res,
        tests=tests,
        n_obs=n_used,
        n_clusters=n_clusters,
        n_series=n_series,
    )
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Plot
    plot_spread_dynamics(df_lagged, res)

    # Diagnostics
    diag_txt = build_diagnostics_text(
        total_rows=total_rows,
        n_used=n_used,
        n_clusters=n_clusters,
        n_series=n_series,
        ok=True,
    )
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

