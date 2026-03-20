"""Spread Significance Test for the thesis empirical dataset.

Uses the empirical master dataset (overlap sample):
  btc_probability_comparison_empirical_master.parquet
from:
  Actual data/Options N(d2) + Polymarket - Econometric Testing

The goal is to test whether the average probability spread
  probability_spread = prob_yes - prob_option
is statistically different from zero once we account for correlation
within events. Event-level clustering is handled via cluster-robust
standard errors grouped by event_date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script lives in Econometric Analysis/Spread Significance/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
SPREAD_DIR = SCRIPT_DIR.parent  # Spread Significance/
ECONOMETRIC_DIR = SPREAD_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = SPREAD_DIR / "tables"
FIGURES_DIR = SPREAD_DIR / "figures"
DIAG_DIR = SPREAD_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "spread_significance_results.csv"
SUMMARY_TXT = TABLES_DIR / "spread_significance_summary.txt"
FIG_PNG = FIGURES_DIR / "spread_distribution_with_mean.png"
DIAG_TXT = DIAG_DIR / "spread_significance_diagnostics.txt"

REQUIRED_COLS = ["probability_spread", "event_date"]


def load_dataset() -> pd.DataFrame:
    """Load empirical master dataset and ensure required columns exist."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def prepare_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing spread or event_date; coerce spread to numeric."""
    sample = df.copy()
    sample = sample.dropna(subset=["probability_spread", "event_date"])
    sample["probability_spread"] = pd.to_numeric(
        sample["probability_spread"], errors="coerce"
    )
    sample = sample.dropna(subset=["probability_spread"])
    return sample


def descriptive_spread_stats(sample: pd.DataFrame) -> dict[str, float]:
    """Compute basic descriptive statistics for probability_spread."""
    s = sample["probability_spread"]
    mean_spread = float(s.mean())
    median_spread = float(s.median())
    std_spread = float(s.std(ddof=1))
    n = int(s.count())
    se_mean = float(std_spread / np.sqrt(n)) if n > 0 else float("nan")
    return {
        "mean_spread": mean_spread,
        "median_spread": median_spread,
        "std_spread": std_spread,
        "se_mean": se_mean,
        "observations": n,
    }


def clustered_mean_test(sample: pd.DataFrame) -> dict[str, float]:
    """Run H0: mean(probability_spread) = 0 using cluster-robust SEs by event_date.

We implement this as an intercept-only OLS regression:
  probability_spread_i = mu + u_i
and test H0: mu = 0. Cluster-robust standard errors are computed with
clusters defined by event_date, allowing arbitrary correlation within
each event and treating events as independent clusters.
    """
    spread = sample["probability_spread"]
    # Regress spread on a constant only. Use a plain numpy column of ones so
    # the parameter vector has a single unnamed element (index 0).
    X = np.ones((len(spread), 1))
    model = sm.OLS(spread, X)
    # Cluster-robust by event_date
    groups = sample["event_date"].astype(str)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Only one parameter (the mean); it is at position 0
    mu_hat = float(res.params.iloc[0])
    se_cluster = float(res.bse.iloc[0])
    t_stat = float(res.tvalues.iloc[0])
    p_val = float(res.pvalues.iloc[0])
    ci_low, ci_high = res.conf_int().iloc[0].tolist()
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    n_obs = int(res.nobs)
    n_clusters = int(groups.nunique())

    return {
        "mean_spread": mu_hat,
        "clustered_std_error": se_cluster,
        "t_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "observations": n_obs,
        "event_clusters": n_clusters,
    }


def plot_spread_histogram(sample: pd.DataFrame) -> None:
    """Histogram of probability_spread with vertical lines at 0 and mean spread."""
    spread = sample["probability_spread"].dropna()
    if spread.empty:
        logger.warning("No valid probability_spread values for histogram.")
        return

    mean_spread = float(spread.mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spread, bins=50, alpha=0.7, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Zero spread")
    ax.axvline(
        mean_spread,
        color="red",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean spread ({mean_spread:.4f})",
    )
    ax.set_xlabel("Probability spread (prob_yes − prob_option)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of probability spread with mean", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


def build_results_table(
    desc_stats: dict[str, float], clustered_test: dict[str, float]
) -> pd.DataFrame:
    """Combine descriptive stats and clustered test into a tidy table."""
    rows = [
        {"metric": "mean_spread", "value": desc_stats["mean_spread"]},
        {"metric": "median_spread", "value": desc_stats["median_spread"]},
        {"metric": "std_spread", "value": desc_stats["std_spread"]},
        {"metric": "standard_error_mean", "value": desc_stats["se_mean"]},
        {"metric": "clustered_std_error", "value": clustered_test["clustered_std_error"]},
        {"metric": "t_statistic", "value": clustered_test["t_statistic"]},
        {"metric": "p_value", "value": clustered_test["p_value"]},
        {"metric": "ci_lower", "value": clustered_test["ci_lower"]},
        {"metric": "ci_upper", "value": clustered_test["ci_upper"]},
        {"metric": "observations", "value": clustered_test["observations"]},
        {"metric": "event_clusters", "value": clustered_test["event_clusters"]},
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_summary_text(
    desc_stats: dict[str, float], clustered_test: dict[str, float]
) -> str:
    """Human-readable summary of the spread significance test."""
    lines: list[str] = []
    lines.append("Spread Significance Test")
    lines.append("=" * 60)
    lines.append(f"Dataset: {DATA_PATH}")
    lines.append("")
    lines.append("Descriptive statistics for probability_spread:")
    lines.append(f"  Mean:    {desc_stats['mean_spread']:.6f}")
    lines.append(f"  Median:  {desc_stats['median_spread']:.6f}")
    lines.append(f"  Std dev: {desc_stats['std_spread']:.6f}")
    lines.append(f"  Std error of mean (i.i.d.): {desc_stats['se_mean']:.6f}")
    lines.append(f"  Observations: {desc_stats['observations']}")
    lines.append("")
    lines.append("Cluster-robust one-sample t-test (H0: mean spread = 0):")
    lines.append(f"  Mean spread (mû):     {clustered_test['mean_spread']:.6f}")
    lines.append(
        f"  Clustered std error:   {clustered_test['clustered_std_error']:.6f}"
    )
    lines.append(f"  t-statistic:           {clustered_test['t_statistic']:.6f}")
    lines.append(f"  p-value:               {clustered_test['p_value']:.6g}")
    lines.append(
        f"  95% CI for mean:       [{clustered_test['ci_lower']:.6f}, {clustered_test['ci_upper']:.6f}]"
    )
    lines.append("")
    lines.append(
        f"Event clusters (event_date): {clustered_test['event_clusters']}"
    )
    lines.append(
        "Note: Standard errors and test statistics are clustered by event_date\n"
        "to allow arbitrary correlation of spreads within each event."
    )
    return "\n".join(lines)


def build_diagnostics_text(
    total_rows: int,
    n_used: int,
    n_clusters: int,
    ok: bool,
) -> str:
    """Short diagnostics summary for verification."""
    lines = [
        "Spread significance diagnostics",
        "=" * 40,
        f"Dataset path:                 {DATA_PATH}",
        f"Rows loaded:                  {total_rows}",
        f"Rows used in test:            {n_used}",
        f"Unique event_date clusters:   {n_clusters}",
        f"Test executed successfully:   {'YES' if ok else 'NO'}",
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
    sample = prepare_sample(df)
    n_used = len(sample)
    if n_used == 0:
        raise RuntimeError("No observations available after dropping missing values.")

    # Descriptive statistics (unclustered)
    desc_stats = descriptive_spread_stats(sample)

    # Clustered one-sample t-test via intercept-only OLS
    clustered_test = clustered_mean_test(sample)

    # Save results table
    results_df = build_results_table(desc_stats, clustered_test)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Summary text
    summary_txt = build_summary_text(desc_stats, clustered_test)
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved %s", SUMMARY_TXT)
    print("\n" + summary_txt)

    # Histogram visualization
    plot_spread_histogram(sample)

    # Diagnostics
    n_clusters = clustered_test["event_clusters"]
    diag_txt = build_diagnostics_text(
        total_rows=total_rows,
        n_used=n_used,
        n_clusters=n_clusters,
        ok=True,
    )
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

