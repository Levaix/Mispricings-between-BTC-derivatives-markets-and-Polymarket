"""Robustness Analysis for the thesis empirical dataset.

Uses the empirical master dataset (overlap sample):
  btc_probability_comparison_empirical_master.parquet
from:
  Actual data/Options N(d2) + Polymarket - Econometric Testing

For a range of subsamples, we re-estimate the baseline regression:

  prob_yes = alpha + beta * prob_option + error

Subgroups:
  A) Moneyness:
     - deep_OTM:     moneyness < 0.9
     - near_ATM:     0.9 ≤ moneyness ≤ 1.1
     - ITM:          moneyness > 1.1

  B) Maturity (days_to_expiry):
     - short_maturity:   days_to_expiry ≤ 2
     - medium_maturity:  2 < days_to_expiry ≤ 4
     - long_maturity:    days_to_expiry > 4

  C) Probability level (prob_option):
     - low_probability:  prob_option < 0.2
     - mid_probability:  0.2 ≤ prob_option ≤ 0.8
     - high_probability: prob_option > 0.8

All regressions use cluster-robust standard errors clustered by event_date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
# Script lives in Econometric Analysis/Robustness Analysis/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
ROBUST_DIR = SCRIPT_DIR.parent  # Robustness Analysis/
ECONOMETRIC_DIR = ROBUST_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = ROBUST_DIR / "tables"
FIGURES_DIR = ROBUST_DIR / "figures"
DIAG_DIR = ROBUST_DIR / "diagnostics"

RESULTS_CSV = TABLES_DIR / "robustness_results.csv"
FIG_PNG = FIGURES_DIR / "robustness_beta_comparison.png"
DIAG_TXT = DIAG_DIR / "robustness_diagnostics.txt"

REQUIRED_COLS = [
    "prob_yes",
    "prob_option",
    "event_date",
    "moneyness",
    "days_to_expiry",
]  # mid_probability is optional


def load_dataset() -> pd.DataFrame:
    """Load dataset and ensure required columns exist."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def prepare_base_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare base sample: drop rows with missing prob_yes/prob_option/event_date."""
    sample = df.copy()
    sample = sample.dropna(subset=["prob_yes", "prob_option", "event_date"])
    sample["prob_yes"] = pd.to_numeric(sample["prob_yes"], errors="coerce")
    sample["prob_option"] = pd.to_numeric(sample["prob_option"], errors="coerce")
    sample["moneyness"] = pd.to_numeric(sample["moneyness"], errors="coerce")
    sample["days_to_expiry"] = pd.to_numeric(sample["days_to_expiry"], errors="coerce")
    sample = sample.dropna(subset=["prob_yes", "prob_option", "moneyness", "days_to_expiry"])
    return sample


def define_groups(df: pd.DataFrame) -> Dict[str, List[Tuple[str, pd.Series]]]:
    """Define subgroup indicators for moneyness, maturity, and probability level."""
    groups: Dict[str, List[Tuple[str, pd.Series]]] = {}

    # A) Moneyness groups
    m = df["moneyness"]
    groups["moneyness"] = [
        ("deep_OTM", m < 0.9),
        ("near_ATM", (m >= 0.9) & (m <= 1.1)),
        ("ITM", m > 1.1),
    ]

    # B) Maturity groups (days_to_expiry)
    dte = df["days_to_expiry"]
    groups["maturity"] = [
        ("short_maturity", dte <= 2),
        ("medium_maturity", (dte > 2) & (dte <= 4)),
        ("long_maturity", dte > 4),
    ]

    # C) Probability level groups (prob_option)
    po = df["prob_option"]
    groups["probability_level"] = [
        ("low_probability", po < 0.2),
        ("mid_probability", (po >= 0.2) & (po <= 0.8)),
        ("high_probability", po > 0.8),
    ]

    return groups


def run_group_regression(
    df: pd.DataFrame,
    mask: pd.Series,
    min_obs: int = 30,
    min_clusters: int = 5,
) -> Dict[str, float] | None:
    """Run baseline regression for a given subgroup mask.

prob_yes = alpha + beta * prob_option + error

Cluster-robust standard errors are clustered by event_date. Returns a dict
with alpha, beta, std_error (for beta), t_stat, p_value, r_squared, n_obs,
or None if the group has too few observations or clusters.
    """
    sub = df[mask].copy()
    n_obs = len(sub)
    if n_obs < min_obs:
        return None

    n_clusters = sub["event_date"].astype(str).nunique()
    if n_clusters < min_clusters:
        return None

    y = sub["prob_yes"]
    X = sm.add_constant(sub["prob_option"])
    groups = sub["event_date"].astype(str)

    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    alpha_hat = float(res.params["const"])
    beta_hat = float(res.params["prob_option"])
    se_beta = float(res.bse["prob_option"])
    t_beta = float(res.tvalues["prob_option"])
    p_beta = float(res.pvalues["prob_option"])
    r2 = float(res.rsquared)

    return {
        "alpha": alpha_hat,
        "beta": beta_hat,
        "std_error": se_beta,
        "t_stat": t_beta,
        "p_value": p_beta,
        "r_squared": r2,
        "observations": n_obs,
    }


def build_results_table(
    df: pd.DataFrame,
    group_defs: Dict[str, List[Tuple[str, pd.Series]]],
) -> pd.DataFrame:
    """Run regressions for all subgroups and assemble results table."""
    rows: List[Dict[str, object]] = []

    for gtype, groups in group_defs.items():
        for gname, mask in groups:
            res = run_group_regression(df, mask)
            if res is None:
                logger.info(
                    "Skipping group %s/%s (insufficient data or clusters).",
                    gtype,
                    gname,
                )
                continue
            row = {
                "group_type": gtype,
                "group_name": gname,
                "alpha": res["alpha"],
                "beta": res["beta"],
                "std_error": res["std_error"],
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "r_squared": res["r_squared"],
                "observations": res["observations"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def plot_beta_comparison(results: pd.DataFrame) -> None:
    """Plot beta coefficients across all groups with 95% CI error bars."""
    if results.empty:
        logger.warning("No robustness results to plot.")
        return

    # Sort for a consistent visual ordering
    results = results.sort_values(["group_type", "group_name"]).reset_index(drop=True)

    betas = results["beta"].values
    ses = results["std_error"].values
    labels = [f"{gt}:{gn}" for gt, gn in zip(results["group_type"], results["group_name"])]

    x = np.arange(len(betas))
    fig, ax = plt.subplots(figsize=(max(8, len(betas) * 0.6), 4))
    ax.bar(x, betas, yerr=1.96 * ses, capsize=4, alpha=0.8, color="C0")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("beta (coefficient on prob_option)", fontsize=10)
    ax.set_title("Robustness: beta across moneyness, maturity, and probability groups", fontsize=12)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", FIG_PNG)


def build_diagnostics_text(
    total_rows: int,
    results: pd.DataFrame,
) -> str:
    """Short diagnostics summary for robustness analysis."""
    lines: List[str] = []
    lines.append("Robustness analysis diagnostics")
    lines.append("=" * 40)
    lines.append(f"Dataset path:            {DATA_PATH}")
    lines.append(f"Total rows:              {total_rows}")
    lines.append(f"Number of regressions run: {len(results)}")
    lines.append("")
    lines.append("Rows used per subgroup:")
    if results.empty:
        lines.append("  (no regressions ran; all groups filtered out)")
    else:
        for _, row in results.iterrows():
            lines.append(
                f"  {row['group_type']}/{row['group_name']}: "
                f"observations = {int(row['observations'])}"
            )
    lines.append("")
    lines.append("Regression execution:    " + ("SUCCESS" if not results.empty else "NO REGRESSIONS"))
    lines.append("")
    lines.append("Outputs created:")
    lines.append(f"  - {RESULTS_CSV.name}")
    lines.append(f"  - {FIG_PNG.name}")
    return "\n".join(lines)


def main() -> None:
    # Ensure folder structure exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_dataset()
    total_rows = len(df)
    base_sample = prepare_base_sample(df)

    # Define subgroup masks
    group_defs = define_groups(base_sample)

    # Run regressions
    results_df = build_results_table(base_sample, group_defs)
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Saved %s", RESULTS_CSV)

    # Plot beta comparison
    plot_beta_comparison(results_df)

    # Diagnostics
    diag_txt = build_diagnostics_text(total_rows=total_rows, results=results_df)
    DIAG_TXT.write_text(diag_txt, encoding="utf-8")
    logger.info("Saved %s", DIAG_TXT)
    print("\n" + diag_txt)


if __name__ == "__main__":
    main()

