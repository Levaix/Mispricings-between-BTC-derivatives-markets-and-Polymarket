"""Compute correlations between Polymarket and option-implied probabilities.

Loads btc_probability_comparison_empirical_master.parquet from the
\"Options N(d2) + Polymarket - Econometric Testing\" folder and computes:
- Pearson correlation between prob_yes and prob_option
- Spearman rank correlation between prob_yes and prob_option

Results are written to:
Econometric Analysis/Descriptive Statistics/tables/probability_correlations.csv
and also printed to the console.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script lives in Econometric Analysis/Descriptive Statistics/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parent  # Descriptive Statistics/
ECONOMETRIC_DIR = STATS_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

DATA_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options N(d2) + Polymarket - Econometric Testing"
    / "btc_probability_comparison_empirical_master.parquet"
)

TABLES_DIR = STATS_DIR / "tables"
FIGURES_DIR = STATS_DIR / "figures"
DIAG_DIR = STATS_DIR / "diagnostics"

OUTPUT_CSV = TABLES_DIR / "probability_correlations.csv"

REQUIRED_COLS = ["prob_yes", "prob_option"]


def load_dataset() -> pd.DataFrame:
    """Load dataset and ensure required columns exist."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def compute_correlation(
    x: pd.Series, y: pd.Series, method: str
) -> tuple[float, float, int]:
    """Compute correlation coefficient, p-value, and n using scipy."""
    sx = pd.to_numeric(x, errors="coerce")
    sy = pd.to_numeric(y, errors="coerce")
    valid = sx.notna() & sy.notna()
    sx = sx[valid]
    sy = sy[valid]
    n = int(valid.sum())
    if n == 0:
        return float("nan"), float("nan"), 0

    if method == "pearson":
        r, p = pearsonr(sx, sy)
    elif method == "spearman":
        r, p = spearmanr(sx, sy)
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(r), float(p), n


def main() -> None:
    # Ensure base folders exist (for consistency with other descriptive scripts)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    # Compute correlations
    pearson_r, pearson_p, n_pearson = compute_correlation(
        df["prob_yes"], df["prob_option"], method="pearson"
    )
    spearman_r, spearman_p, n_spearman = compute_correlation(
        df["prob_yes"], df["prob_option"], method="spearman"
    )

    # Build results table
    rows = [
        {
            "correlation_type": "Pearson",
            "coefficient": pearson_r,
            "p_value": pearson_p,
            "observations": n_pearson,
        },
        {
            "correlation_type": "Spearman",
            "coefficient": spearman_r,
            "p_value": spearman_p,
            "observations": n_spearman,
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved %s", OUTPUT_CSV)

    # Print nicely to console
    print("\nProbability correlations between prob_yes and prob_option")
    print("=========================================================")
    print(f"Dataset: {DATA_PATH}")
    print("")
    for row in rows:
        ctype = row["correlation_type"]
        coeff = row["coefficient"]
        pval = row["p_value"]
        n = row["observations"]
        print(f"{ctype}:")
        print(f"  coefficient : {coeff:.6f}" if n > 0 else "  coefficient : N/A")
        print(f"  p-value     : {pval:.6g}" if n > 0 else "  p-value     : N/A")
        print(f"  observations: {n}")
        print("")


if __name__ == "__main__":
    main()

