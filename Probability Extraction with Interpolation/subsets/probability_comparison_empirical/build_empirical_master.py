"""
Build the empirical comparison master dataset for thesis.

Reads options_smile_event_probabilities.parquet, applies filters and creates
derived variables, then saves a clean subset to Actual data/ for all later
empirical tests (Polymarket vs Deribit option-implied probabilities).

Does not modify the source probability dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths: script lives in analysis/subsets/probability_comparison_empirical/
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent.parent.parent  # Code/
SOURCE_PARQUET = CODE_ROOT / "Actual data" / "Option Probabilities" / "options_smile_event_probabilities.parquet"
OUTPUT_PARQUET = CODE_ROOT / "Actual data" / "btc_probability_comparison_empirical_master.parquet"
DIAG_FILE = SCRIPT_DIR / "dataset_diagnostics.txt"

# Columns to keep if they exist (after adding derived vars).
# "timestamp" may not exist in source; bucket_5m_utc is the time column.
KEEP_COLUMNS = [
    "timestamp",
    "bucket_5m_utc",
    "event_date",
    "strike_K",
    "chosen_option_expiry",
    "prob_yes",
    "prob_option",
    "probability_spread",
    "abs_spread",
    "mid_probability",
    "uncertainty_term",
    "spot_snapshot",
    "iv_hat",
    "tau_years",
    "days_to_expiry",
    "risk_free_rate",
    "d2",
    "moneyness",
    "log_moneyness",
    "fit_success",
    "out_of_range",
    "no_snapshot_fit",
    "n_unique_strikes",
]


def main() -> None:
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(f"Source not found: {SOURCE_PARQUET}")

    logger.info("Loading source: %s", SOURCE_PARQUET)
    df = pd.read_parquet(SOURCE_PARQUET)
    n_input = len(df)
    df = df.copy()

    # ---- Filters (all must be satisfied) ----
    # prob_option not null
    mask = df["prob_option"].notna()
    # iv_hat > 0 and iv_hat <= 3
    iv = pd.to_numeric(df["iv_hat"], errors="coerce")
    mask = mask & (iv > 0) & (iv <= 3)
    # tau_years > 0
    tau = pd.to_numeric(df["tau_years"], errors="coerce")
    mask = mask & (tau > 0)
    # prob_yes in [0, 1]
    py = pd.to_numeric(df["prob_yes"], errors="coerce")
    mask = mask & (py >= 0) & (py <= 1)
    # prob_option in [0, 1]
    po = pd.to_numeric(df["prob_option"], errors="coerce")
    mask = mask & (po >= 0) & (po <= 1)
    # spot_snapshot > 0, strike_K > 0
    spot = pd.to_numeric(df["spot_snapshot"], errors="coerce")
    strike = pd.to_numeric(df["strike_K"], errors="coerce")
    mask = mask & (spot > 0) & (strike > 0)

    df = df.loc[mask].copy()
    n_output = len(df)
    retained_share = n_output / n_input if n_input else 0.0

    # ---- Derived variables ----
    df["probability_spread"] = df["prob_yes"].astype(float) - df["prob_option"].astype(float)
    df["abs_spread"] = df["probability_spread"].abs()
    df["mid_probability"] = (df["prob_yes"].astype(float) + df["prob_option"].astype(float)) / 2.0
    df["uncertainty_term"] = df["mid_probability"] * (1.0 - df["mid_probability"])
    df["moneyness"] = df["spot_snapshot"].astype(float) / df["strike_K"].astype(float)
    df["log_moneyness"] = np.log(df["moneyness"])
    df["days_to_expiry"] = df["tau_years"].astype(float) * 365.0

    # ---- Keep only requested columns that exist ----
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        logger.info("Columns not in source (omitted): %s", missing)
    df = df[available].copy()

    # ---- Sort: timestamp (or bucket_5m_utc), event_date, strike_K ----
    time_col = "timestamp" if "timestamp" in df.columns else "bucket_5m_utc"
    sort_cols = [c for c in [time_col, "event_date", "strike_K"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # ---- Save ----
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Saved master dataset: %s (%d rows)", OUTPUT_PARQUET, n_output)

    # ---- Diagnostics ----
    prob_spread = df["probability_spread"]
    abs_sp = df["abs_spread"]
    diag_lines = [
        "Empirical comparison master dataset — diagnostics",
        "=" * 50,
        f"Source file:          {SOURCE_PARQUET.name}",
        f"Output file:          {OUTPUT_PARQUET.name}",
        f"Input row count:      {n_input}",
        f"Output row count:     {n_output}",
        f"Retained share:       {retained_share:.4f} ({100*retained_share:.2f}%)",
        "",
        "probability_spread:",
        f"  mean   = {prob_spread.mean():.6f}",
        f"  median = {prob_spread.median():.6f}",
        f"  std    = {prob_spread.std():.6f}",
        "",
        "abs_spread:",
        f"  mean   = {abs_sp.mean():.6f}",
        f"  median = {abs_sp.median():.6f}",
        f"  std    = {abs_sp.std():.6f}",
        "",
        f"prob_yes   min = {df['prob_yes'].min():.6f}  max = {df['prob_yes'].max():.6f}",
        f"prob_option min = {df['prob_option'].min():.6f}  max = {df['prob_option'].max():.6f}",
        f"iv_hat     min = {df['iv_hat'].min():.6f}  max = {df['iv_hat'].max():.6f}",
        f"tau_years  min = {df['tau_years'].min():.6f}  max = {df['tau_years'].max():.6f}",
        "",
        f"Unique event_date:   {df['event_date'].nunique()}",
        f"Unique timestamps:    {df[time_col].nunique()}",
        f"Unique strikes:      {df['strike_K'].nunique()}",
    ]
    text = "\n".join(diag_lines)
    DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIAG_FILE.write_text(text, encoding="utf-8")
    logger.info("Saved diagnostics: %s", DIAG_FILE)

    # Print same to console
    print("\n" + text)


if __name__ == "__main__":
    main()
