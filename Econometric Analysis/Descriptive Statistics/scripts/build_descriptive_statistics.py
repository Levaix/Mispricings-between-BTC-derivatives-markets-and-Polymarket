"""
Build descriptive statistics for the thesis empirical dataset.

Loads btc_probability_comparison_empirical_master.parquet, computes count, mean,
median, std, min, p5, p25, p75, p95, max for selected variables, and saves
tables (CSV), diagnostics (text summary), and prints to console.
Rows = variables, columns = statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Script lives in Econometric Analysis/Descriptive Statistics/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parent  # Descriptive Statistics/
ECONOMETRIC_DIR = STATS_DIR.parent  # Econometric Analysis/
CODE_ROOT = ECONOMETRIC_DIR.parent  # Code/

# Dataset: Actual data --> Options N(d2) + Polymarket - Econometric Testing
CANDIDATE_PATHS = [
    CODE_ROOT / "Actual data" / "Options N(d2) + Polymarket - Econometric Testing" / "btc_probability_comparison_empirical_master.parquet",
    CODE_ROOT / "Actual data" / "Option Probabilities" / "Options N(d2) + Polymarket" / "btc_probability_comparison_empirical_master.parquet",
    CODE_ROOT / "Actual data" / "btc_probability_comparison_empirical_master.parquet",
]

OUTPUT_TABLES = STATS_DIR / "tables"
OUTPUT_FIGURES = STATS_DIR / "figures"
OUTPUT_DIAGNOSTICS = STATS_DIR / "diagnostics"

TABLE_CSV = OUTPUT_TABLES / "descriptive_statistics_table.csv"
SUMMARY_TXT = OUTPUT_DIAGNOSTICS / "descriptive_statistics_summary.txt"

# Variables for descriptive statistics (order preserved; only those present in data are used)
VARIABLES = [
    "prob_yes",
    "prob_option",
    "probability_spread",
    "abs_spread",
    "iv_hat",
    "tau_years",
    "moneyness",
    "days_to_expiry",
]

# Statistics to compute (rows = variables, columns = these)
STAT_NAMES = ["count", "mean", "median", "std", "min", "p5", "p25", "p75", "p95", "max"]


def _resolve_data_path() -> Path:
    """Return first existing path for the empirical master parquet."""
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "btc_probability_comparison_empirical_master.parquet not found. Tried:\n  " +
        "\n  ".join(str(p) for p in CANDIDATE_PATHS)
    )


def compute_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for selected variables.
    Returns DataFrame with rows = variables, columns = statistics.
    """
    available = [v for v in VARIABLES if v in df.columns]
    missing = [v for v in VARIABLES if v not in df.columns]
    if missing:
        logger.warning("Variables not in dataset (skipped): %s", missing)
    if not available:
        raise ValueError("None of the requested variables found in the dataset.")

    rows = []
    for var in available:
        s = pd.to_numeric(df[var], errors="coerce").dropna()
        count = int(s.count())
        if count == 0:
            row = {stat: None for stat in STAT_NAMES}
            row["count"] = 0
        else:
            row = {
                "count": count,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()) if count > 1 else 0.0,
                "min": float(s.min()),
                "p5": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
        row["variable"] = var
        rows.append(row)

    # Build table: rows = variables, columns = statistics
    result = pd.DataFrame(rows)
    result = result[["variable"] + STAT_NAMES]
    result = result.set_index("variable")
    return result


def format_summary_txt(table: pd.DataFrame, n_total: int, data_path: str) -> str:
    """Produce a human-readable text summary for diagnostics."""
    lines = [
        "Descriptive statistics – thesis empirical dataset",
        "=" * 60,
        "",
        f"Dataset: {data_path}",
        f"Total rows in dataset: {n_total}",
        "",
        "Statistics (rows = variables, columns = statistics):",
        "-" * 60,
        "",
    ]
    # Simple aligned text table (format numbers; count may be int)
    def _fmt(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        if isinstance(x, (int, float)):
            return f"{x:.6g}" if isinstance(x, float) else str(int(x))
        return str(x)
    table_str = table.to_string(formatters={c: _fmt for c in table.columns})
    lines.append(table_str)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    # Resolve input path
    data_path = _resolve_data_path()
    logger.info("Loading dataset: %s", data_path)

    df = pd.read_parquet(data_path)
    n_total = len(df)

    # Compute descriptive statistics (rows = variables, columns = statistics)
    table = compute_descriptive_table(df)

    # Create output directories
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

    # Save CSV (variables as index = rows, statistics = columns)
    table.to_csv(TABLE_CSV)
    logger.info("Saved table: %s", TABLE_CSV)

    # Save formatted text summary
    summary_txt = format_summary_txt(table, n_total, str(data_path))
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    logger.info("Saved summary: %s", SUMMARY_TXT)

    # Print to console
    print("\n" + summary_txt)


if __name__ == "__main__":
    main()
