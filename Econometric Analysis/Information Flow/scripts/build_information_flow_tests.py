"""Information Flow (Lead–Lag Tests) for the thesis empirical dataset.

Uses the empirical master dataset (overlap sample):
  btc_probability_comparison_empirical_master.parquet
from:
  Actual data/Options N(d2) + Polymarket - Econometric Testing

For each (event_date, strike_K) time series, we construct first differences
of Polymarket probabilities (prob_yes) and option-implied probabilities
(prob_option), plus one-period lags of those differences:

  d_prob_yes_t    = prob_yes_t - prob_yes_{t-1}
  d_prob_option_t = prob_option_t - prob_option_{t-1}

  d_prob_yes_lag1    = d_prob_yes_{t-1}
  d_prob_option_lag1 = d_prob_option_{t-1}

All differences and lags are built strictly within each (event_date, strike_K)
series, never across different events or strikes.

Lead–lag regressions:

  A) Options → Prediction Markets
     d_prob_yes_t = alpha_A + beta_A * d_prob_option_lag1 + error_A

  B) Prediction Markets → Options
     d_prob_option_t = alpha_B + beta_B * d_prob_yes_lag1 + error_B

Cluster-robust standard errors are computed by event_date, allowing
dependence within an event while treating different events as independent
clusters.
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
# Script lives in Econometric Analysis/Information Flow/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
INFO_DIR = SCRIPT_DIR.parent  # Information Flow/
ECONOMETRIC_DIR = INFO_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = INFO_DIR / "tables"
FIGURES_DIR = INFO_DIR / "figures"
DIAG_DIR = INFO_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "information_flow_results.csv"
SUMMARY_TXT = TABLES_DIR / "information_flow_summary.txt"
FIG_PNG = FIGURES_DIR / "information_flow_coefficients.png"
DIAG_TXT = DIAG_DIR / "information_flow_diagnostics.txt"

TIME_COL_CANDIDATES = ["timestamp", "bucket_5m_utc"]
REQUIRED_COLS = ["event_date", "strike_K", "prob_yes", "prob_option"]


def load_dataset() -> pd.DataFrame:
    """Load empirical master and ensure required columns exist."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
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
    raise ValueError(f"No valid time column found; expected one of {TIME_COL_CANDIDATES}.")


def construct_differences_and_lags(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Construct first differences and one-period lags within (event_date, strike_K).

Implementation:
- Sort by (event_date, strike_K, time_col)
- Within each (event_date, strike_K):
  * d_prob_yes    = prob_yes_t - prob_yes_{t-1}
  * d_prob_option = prob_option_t - prob_option_{t-1}
  * d_prob_yes_lag1    = lag of d_prob_yes
  * d_prob_option_lag1 = lag of d_prob_option
- Drop rows with missing lags or differences.
"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(["event_date", "strike_K", time_col])

    # Coerce to numeric for differencing
    df["prob_yes"] = pd.to_numeric(df["prob_yes"], errors="coerce")
    df["prob_option"] = pd.to_numeric(df["prob_option"], errors="coerce")

    grouped = df.groupby(["event_date", "strike_K"], group_keys=False)

    df["d_prob_yes"] = grouped["prob_yes"].diff()
    df["d_prob_option"] = grouped["prob_option"].diff()

    df["d_prob_yes_lag1"] = grouped["d_prob_yes"].shift(1)
    df["d_prob_option_lag1"] = grouped["d_prob_option"].shift(1)

    # Drop rows where required quantities are missing
    df = df.dropna(
        subset=[
            "d_prob_yes",
            "d_prob_option",
            "d_prob_yes_lag1",
            "d_prob_option_lag1",
            "event_date",
        ]
    )
    return df


def run_regression(
    y: pd.Series,
    x: pd.Series,
    clusters: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with cluster-robust SEs clustered by event_date."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters.astype(str)})
    return res


def build_results_table(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """Construct combined results table for regressions A and B."""
    rows = []
    for reg_name, res in [
        ("A_options_lead_pm", res_A),
        ("B_pm_lead_options", res_B),
    ]:
        params = res.params
        se = res.bse
        tvals = res.tvalues
        pvals = res.pvalues
        ci = res.conf_int(alpha=0.05)
        for term in ["const", params.index[1]]:  # const and the single slope
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
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )
    return pd.DataFrame(rows)


def run_hypothesis_tests(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run H0: alpha = 0 and H0: beta = 0/1 for both regressions."""
    tests: dict[str, dict[str, dict[str, float]]] = {}

    slope_A_name = res_A.params.index[1]
    slope_B_name = res_B.params.index[1]

    # Regression A: d_prob_yes_t on d_prob_option_lag1
    tests["A"] = {}
    t_alpha_A = res_A.t_test("const = 0")
    t_beta_A_0 = res_A.t_test(f"{slope_A_name} = 0")
    tests["A"]["alpha_eq_0"] = {
        "stat": float(t_alpha_A.statistic),
        "p_value": float(t_alpha_A.pvalue),
    }
    tests["A"]["beta_eq_0"] = {
        "stat": float(t_beta_A_0.statistic),
        "p_value": float(t_beta_A_0.pvalue),
    }

    # Regression B: d_prob_option_t on d_prob_yes_lag1
    tests["B"] = {}
    t_alpha_B = res_B.t_test("const = 0")
    t_beta_B_0 = res_B.t_test(f"{slope_B_name} = 0")
    tests["B"]["alpha_eq_0"] = {
        "stat": float(t_alpha_B.statistic),
        "p_value": float(t_alpha_B.pvalue),
    }
    tests["B"]["beta_eq_0"] = {
        "stat": float(t_beta_B_0.statistic),
        "p_value": float(t_beta_B_0.pvalue),
    }

    return tests


def build_summary_text(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
    tests: dict[str, dict[str, dict[str, float]]],
    n_obs: int,
    n_clusters: int,
    n_series: int,
) -> str:
    """Readable summary of information flow regressions."""
    slope_A_name = res_A.params.index[1]
    slope_B_name = res_B.params.index[1]

    alpha_A = float(res_A.params["const"])
    beta_A = float(res_A.params[slope_A_name])
    r2_A = float(res_A.rsquared)
    r2adj_A = float(res_A.rsquared_adj)

    alpha_B = float(res_B.params["const"])
    beta_B = float(res_B.params[slope_B_name])
    r2_B = float(res_B.rsquared)
    r2adj_B = float(res_B.rsquared_adj)

    lines: list[str] = []
    lines.append("Information Flow (Lead–Lag Tests)")
    lines.append("=" * 72)
    lines.append(f"Dataset:                        {DATA_PATH}")
    lines.append(f"Observations after differencing: {n_obs}")
    lines.append(f"Event clusters (event_date):    {n_clusters}")
    lines.append(f"(event_date, strike_K) series:  {n_series}")
    lines.append("")
    lines.append("Regression A (Options → Prediction Markets):")
    lines.append("  d_prob_yes_t = alpha_A + beta_A * d_prob_option_lag1 + error_A")
    lines.append(f"  alpha_A: {alpha_A: .6f}")
    lines.append(f"  beta_A : {beta_A: .6f}")
    lines.append(f"  R^2 / adj R^2: {r2_A: .6f} / {r2adj_A: .6f}")
    lines.append("  Tests:")
    lines.append(
        f"    H0: alpha_A = 0  -> t = {tests['A']['alpha_eq_0']['stat']:.6f}, "
        f"p = {tests['A']['alpha_eq_0']['p_value']:.6g}"
    )
    lines.append(
        f"    H0: beta_A  = 0  -> t = {tests['A']['beta_eq_0']['stat']:.6f}, "
        f"p = {tests['A']['beta_eq_0']['p_value']:.6g}"
    )
    lines.append("")
    lines.append("Regression B (Prediction Markets → Options):")
    lines.append("  d_prob_option_t = alpha_B + beta_B * d_prob_yes_lag1 + error_B")
    lines.append(f"  alpha_B: {alpha_B: .6f}")
    lines.append(f"  beta_B : {beta_B: .6f}")
    lines.append(f"  R^2 / adj R^2: {r2_B: .6f} / {r2adj_B: .6f}")
    lines.append("  Tests:")
    lines.append(
        f"    H0: alpha_B = 0  -> t = {tests['B']['alpha_eq_0']['stat']:.6f}, "
        f"p = {tests['B']['alpha_eq_0']['p_value']:.6g}"
    )
    lines.append(
        f"    H0: beta_B  = 0  -> t = {tests['B']['beta_eq_0']['stat']:.6f}, "
        f"p = {tests['B']['beta_eq_0']['p_value']:.6g}"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- If beta_A > 0 and statistically significant, changes in option-implied "
        "probabilities tend to precede and predict changes in Polymarket "
        "probabilities (options lead prediction markets)."
    )
    lines.append(
        "- If beta_B > 0 and statistically significant, changes in Polymarket "
        "probabilities tend to precede and predict changes in option-implied "
        "probabilities (prediction markets lead options)."
    )
    lines.append(
        "- Clustering by event_date accounts for within-event correlation in "
        "shocks across strikes and timestamps, while treating events as "
        "independent clusters."
    )
    return "\n".join(lines)


def plot_information_flow_bars(
    res_A: sm.regression.linear_model.RegressionResultsWrapper,
    res_B: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    """Bar chart comparing lead coefficients beta_A and beta_B with 95% CIs."""
    slope_A_name = res_A.params.index[1]
    slope_B_name = res_B.params.index[1]

    beta_A = float(res_A.params[slope_A_name])
    se_A = float(res_A.bse[slope_A_name])

    beta_B = float(res_B.params[slope_B_name])
    se_B = float(res_B.bse[slope_B_name])

    # 95% CI ≈ beta ± 1.96 * SE for plotting error bars
    betas = np.array([beta_A, beta_B])
    ses = np.array([se_A, se_B])
    labels = ["Options → PM", "PM → Options"]

    x = np.arange(len(betas))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, betas, yerr=1.96 * ses, capsize=5, color=["C1", "C0"], alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lead coefficient (beta)", fontsize=10)
    ax.set_title("Information flow: lead coefficients with 95% CIs", fontsize=12)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


def build_diagnostics_text(
    total_rows: int,
    n_used: int,
    n_clusters: int,
    n_series: int,
    ok: bool,
) -> str:
    """Short diagnostics summary for information flow step."""
    lines = [
        "Information flow diagnostics",
        "=" * 40,
        f"Dataset path:                            {DATA_PATH}",
        f"Total input rows:                        {total_rows}",
        f"Rows used after differencing/lags:       {n_used}",
        f"Unique event_date clusters:              {n_clusters}",
        f"Unique (event_date, strike_K) series:    {n_series}",
        "Lags built within (event_date, strike_K): YES",
        f"Regressions ran successfully:            {'YES' if ok else 'NO'}",
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
    df_d = construct_differences_and_lags(df, time_col)
    n_used = len(df_d)

    if n_used == 0:
        raise RuntimeError("No observations left after constructing differences and lags.")

    n_clusters = df_d["event_date"].astype(str).nunique()
    n_series = df_d[["event_date", "strike_K"]].drop_duplicates().shape[0]

    groups = df_d["event_date"].astype(str)

    # Regression A: Options → Prediction Markets
    y_A = df_d["d_prob_yes"]
    x_A = df_d["d_prob_option_lag1"]
    res_A = run_regression(y_A, x_A, groups)

    # Regression B: Prediction Markets → Options
    y_B = df_d["d_prob_option"]
    x_B = df_d["d_prob_yes_lag1"]
    res_B = run_regression(y_B, x_B, groups)

    # Results table
    results_df = build_results_table(res_A, res_B)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Hypothesis tests (alpha=0 and beta=0 for both A and B)
    tests = run_hypothesis_tests(res_A, res_B)

    # Summary text
    summary_txt = build_summary_text(
        res_A,
        res_B,
        tests=tests,
        n_obs=n_used,
        n_clusters=n_clusters,
        n_series=n_series,
    )
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Visualization: bar chart of lead coefficients with 95% CI error bars
    plot_information_flow_bars(res_A, res_B)

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

