"""Baseline regression step for the thesis.

Uses the empirical master dataset (overlap sample):
  btc_probability_comparison_empirical_master.parquet
from:
  Actual data/Options N(d2) + Polymarket - Econometric Testing

Baseline OLS regression:
  prob_yes = alpha + beta * prob_option + error

Key features:
- Cluster-robust standard errors clustered by event_date
- Regression results table (CSV)
- Text summary
- Hexbin + fitted line + 45-degree line plot
- Short diagnostics file
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
# Script lives in Econometric Analysis/Baseline Regression/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent  # Baseline Regression/
ECONOMETRIC_DIR = BASELINE_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = BASELINE_DIR / "tables"
FIGURES_DIR = BASELINE_DIR / "figures"
DIAG_DIR = BASELINE_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "baseline_regression_results.csv"
SUMMARY_TXT = TABLES_DIR / "baseline_regression_summary.txt"
FIG_PNG = FIGURES_DIR / "baseline_regression_fit.png"
DIAG_TXT = DIAG_DIR / "baseline_regression_diagnostics.txt"

REQUIRED_COLS = ["prob_yes", "prob_option", "event_date"]


def load_dataset() -> pd.DataFrame:
    """Load empirical master and ensure required columns exist."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def prepare_regression_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing prob_yes, prob_option, or event_date."""
    sample = df.copy()
    sample = sample.dropna(subset=["prob_yes", "prob_option", "event_date"])
    # Coerce to numeric where needed
    sample["prob_yes"] = pd.to_numeric(sample["prob_yes"], errors="coerce")
    sample["prob_option"] = pd.to_numeric(sample["prob_option"], errors="coerce")
    sample = sample.dropna(subset=["prob_yes", "prob_option"])
    return sample


def run_baseline_regression(sample: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS with cluster-robust SEs clustered by event_date.

Clustering by event_date allows arbitrary correlation of errors within each
event (e.g. multiple timestamps and strikes for a given Fed meeting), while
assuming independence across events. This is appropriate because event-level
shocks may affect all observations within the same event_date.
    """
    y = sample["prob_yes"]
    X = sm.add_constant(sample["prob_option"])
    # Use event_date string as cluster label
    groups = sample["event_date"].astype(str)
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res


def build_results_table(res: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """Construct a tidy results table for alpha (const) and beta (prob_option)."""
    params = res.params
    se = res.bse  # cluster-robust SE because we specified cov_type="cluster"
    tvals = res.tvalues
    pvals = res.pvalues
    ci = res.conf_int(alpha=0.05)

    rows = []
    for term in ["const", "prob_option"]:
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


def run_hypothesis_tests(res: sm.regression.linear_model.RegressionResultsWrapper) -> dict[str, dict[str, float]]:
    """Run H0: alpha = 0 and H0: beta = 1, return dict with stats and p-values."""
    tests: dict[str, dict[str, float]] = {}

    # H0: alpha (const) = 0
    t_alpha = res.t_test("const = 0")
    tests["alpha_eq_0"] = {
        "stat": float(t_alpha.statistic),
        "p_value": float(t_alpha.pvalue),
    }

    # H0: beta (prob_option) = 1
    t_beta = res.t_test("prob_option = 1")
    tests["beta_eq_1"] = {
        "stat": float(t_beta.statistic),
        "p_value": float(t_beta.pvalue),
    }
    return tests


def plot_regression_hexbin(sample: pd.DataFrame, res: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """Hexbin plot of (prob_option, prob_yes) with fitted line and 45-degree line.

The 45-degree line represents the benchmark prob_yes = prob_option. Deviations
from this line show how prediction market probabilities differ from option-
implied probabilities. The fitted regression line summarizes the average
relationship between the two (alpha and beta).
    """
    x = pd.to_numeric(sample["prob_option"], errors="coerce")
    y = pd.to_numeric(sample["prob_yes"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Hexbin for density
    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")

    # Fitted regression line
    x_line = np.linspace(0, 1, 200)
    alpha_hat = float(res.params["const"])
    beta_hat = float(res.params["prob_option"])
    y_fit = alpha_hat + beta_hat * x_line
    ax.plot(x_line, y_fit, color="red", linewidth=2, label="Fitted regression line")

    # 45-degree reference line
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, label="45° reference line")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Option-implied probability (prob_option)", fontsize=10)
    ax.set_ylabel("Polymarket probability (prob_yes)", fontsize=10)
    ax.set_title("Baseline regression: Polymarket vs option-implied probabilities", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


def build_text_summary(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    tests: dict[str, dict[str, float]],
    n_obs: int,
    n_clusters: int,
) -> str:
    """Create a human-readable regression summary for the thesis."""
    alpha_hat = float(res.params["const"])
    beta_hat = float(res.params["prob_option"])
    r2 = float(res.rsquared)
    r2_adj = float(res.rsquared_adj)

    lines: list[str] = []
    lines.append("Baseline regression: prob_yes = alpha + beta * prob_option + error")
    lines.append("=" * 72)
    lines.append(f"Dataset:       {DATA_PATH}")
    lines.append(f"Observations:  {n_obs}")
    lines.append(f"Event clusters (event_date): {n_clusters}")
    lines.append("")
    lines.append("Coefficients (cluster-robust SE by event_date):")
    lines.append(f"  alpha (const):      {alpha_hat: .6f}")
    lines.append(f"  beta (prob_option): {beta_hat: .6f}")
    lines.append("")
    lines.append(f"R-squared:            {r2: .6f}")
    lines.append(f"Adjusted R-squared:   {r2_adj: .6f}")
    lines.append("")
    lines.append("Hypothesis tests:")
    lines.append("  H0: alpha = 0")
    lines.append(f"    t-statistic: {tests['alpha_eq_0']['stat']:.6f}")
    lines.append(f"    p-value:     {tests['alpha_eq_0']['p_value']:.6g}")
    lines.append("  H0: beta = 1")
    lines.append(f"    t-statistic: {tests['beta_eq_1']['stat']:.6f}")
    lines.append(f"    p-value:     {tests['beta_eq_1']['p_value']:.6g}")
    lines.append("")
    lines.append("Economic interpretation:")
    lines.append(
        "- alpha measures systematic bias in Polymarket probabilities when "
        "option-implied probabilities are zero."
    )
    lines.append(
        "- beta measures how sensitive Polymarket probabilities are to changes "
        "in option-implied probabilities. beta = 1 and alpha = 0 would mean "
        "perfect one-to-one alignment."
    )
    lines.append(
        "- The 45-degree line in the plot represents this benchmark; the "
        "fitted regression line shows deviations from it."
    )
    return "\n".join(lines)


def build_diagnostics_text(
    total_rows: int,
    n_reg_rows: int,
    n_clusters: int,
    sample: pd.DataFrame,
    regression_ok: bool,
) -> str:
    """Short diagnostics summary for verification."""
    min_py = float(sample["prob_yes"].min()) if not sample.empty else float("nan")
    max_py = float(sample["prob_yes"].max()) if not sample.empty else float("nan")
    min_po = float(sample["prob_option"].min()) if not sample.empty else float("nan")
    max_po = float(sample["prob_option"].max()) if not sample.empty else float("nan")
    event_missing = sample["event_date"].isna().any()

    lines = [
        "Baseline regression diagnostics",
        "=" * 40,
        f"Dataset path:                          {DATA_PATH}",
        f"Total input rows:                      {total_rows}",
        f"Rows used in regression sample:        {n_reg_rows}",
        f"Unique event_date clusters (regress.): {n_clusters}",
        "",
        "Regression sample ranges:",
        f"  prob_yes   min/max: {min_py:.6f} / {max_py:.6f}",
        f"  prob_option min/max: {min_po:.6f} / {max_po:.6f}",
        "",
        f"event_date missing in regression sample: {'YES' if event_missing else 'NO'}",
        f"Regression ran successfully:            {'YES' if regression_ok else 'NO'}",
        "",
        "Outputs created:",
        f"  - {RESULTS_CSV.name}",
        f"  - {SUMMARY_TXT.name}",
        f"  - {FIG_PNG.name}",
    ]
    return "\n".join(lines)


def main() -> None:
    # Ensure folders exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_dataset()
    total_rows = len(df)
    sample = prepare_regression_sample(df)
    n_reg_rows = len(sample)
    n_clusters = sample["event_date"].astype(str).nunique()

    if n_reg_rows == 0:
        raise RuntimeError("No observations available for regression after dropping missing values.")

    # Run regression with cluster-robust SEs
    res = run_baseline_regression(sample)

    # Build and save results table
    results_table = build_results_table(res)
    results_table.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Hypothesis tests
    tests = run_hypothesis_tests(res)

    # Text summary
    summary_txt = build_text_summary(res, tests, n_obs=n_reg_rows, n_clusters=n_clusters)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)

    # Print key results to console
    print("\n" + summary_txt)

    # Plot (hexbin + fitted line + 45-degree line)
    plot_regression_hexbin(sample, res)

    # Diagnostics
    diag_txt = build_diagnostics_text(
        total_rows=total_rows,
        n_reg_rows=n_reg_rows,
        n_clusters=n_clusters,
        sample=sample,
        regression_ok=True,
    )
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

