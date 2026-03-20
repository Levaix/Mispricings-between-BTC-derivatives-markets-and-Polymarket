"""
Build the final option-implied probability dataset from Method 4 output.

Reads options_smile_event_iv.parquet (Method 4), computes N(d2) as option-implied
probability, adds prob_option and probability_spread, writes to Actual data/Option Probabilities/
and diagnostics to analysis/probability_dataset_diagnostics/.

Does not modify Method 4 output or other existing datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    HAS_NORM = True
except ImportError:
    HAS_NORM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
# Method 4 output (analysis folder)
INPUT_PARQUET = SCRIPT_DIR.parent / "3h_window_staleness120_min3_gap10k" / "options_smile_event_iv.parquet"
# Output under Actual data
ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = ROOT / "Actual data" / "Option Probabilities"
OUTPUT_PARQUET = OUTPUT_DIR / "options_smile_event_probabilities.parquet"
OUTPUT_CSV = OUTPUT_DIR / "options_smile_event_probabilities.csv"
# Diagnostics
DIAG_DIR = SCRIPT_DIR
DIAG_PLOTS = DIAG_DIR / "plots"
SUMMARY_JSON = DIAG_DIR / "probability_dataset_summary.json"
SPREAD_JSON = DIAG_DIR / "spread_distribution.json"

RISK_FREE_RATE = 0.037
EXPIRY_HOUR_UTC = 8

# Spread buckets: (label, low_incl, high_incl or None for open-ended)
# <-0.20: strict < -0.20; -0.20 to -0.10: [-0.20, -0.10); ...; >0.20: strict > 0.20
SPREAD_BUCKETS = [
    ("<-0.20", None, -0.20, "strict_lt"),   # x < -0.20
    ("-0.20 to -0.10", -0.20, -0.10, "low_incl"),  # -0.20 <= x < -0.10
    ("-0.10 to -0.05", -0.10, -0.05, "low_incl"),
    ("-0.05 to -0.02", -0.05, -0.02, "low_incl"),
    ("-0.02 to 0.02", -0.02, 0.02, "both_incl"),
    ("0.02 to 0.05", 0.02, 0.05, "high_incl"),
    ("0.05 to 0.10", 0.05, 0.10, "high_incl"),
    ("0.10 to 0.20", 0.10, 0.20, "high_incl"),
    (">0.20", 0.20, None, "strict_gt"),
]


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    if not HAS_NORM:
        raise RuntimeError("scipy.stats.norm required for N(d2)")
    return norm.cdf(x)


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Method 4 input not found: {INPUT_PARQUET}")

    logger.info("Loading Method 4 output: %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    n_input = len(df)
    df = df.copy()

    # Ensure datetime UTC
    df["bucket_5m_utc"] = pd.to_datetime(df["bucket_5m_utc"], utc=True)
    # Expiry at 08:00 UTC on chosen_option_expiry
    expiry_dt = pd.to_datetime(
        df["chosen_option_expiry"].astype(str) + f" {EXPIRY_HOUR_UTC:02d}:00:00",
        utc=True
    )
    df["expiry_datetime_utc"] = expiry_dt
    delta = (df["expiry_datetime_utc"] - df["bucket_5m_utc"]).dt.total_seconds()
    tau_seconds = np.maximum(delta.values, 0.0)
    tau_years = tau_seconds / (365.0 * 24 * 60 * 60)
    df["tau_years"] = tau_years
    df["risk_free_rate"] = RISK_FREE_RATE

    S = df["spot_snapshot"].astype(float)
    K = df["strike_K"].astype(float)
    sigma = df["iv_hat"].astype(float)
    tau = df["tau_years"].values

    # Valid-IV mask for computing d2 / prob_option
    valid_iv = (
        np.isfinite(sigma) & (sigma > 0) & (sigma <= 3) &
        (S > 0) & (K > 0) & (tau > 0)
    )
    # Counts for diagnostics (overlapping)
    count_iv_le0 = int((np.isfinite(sigma) & (sigma <= 0)).sum())
    count_iv_gt3 = int((np.isfinite(sigma) & (sigma > 3)).sum())
    count_tau_le0 = int((tau <= 0).sum())
    count_missing_spot = int((~(S > 0) | ~np.isfinite(S)).sum())
    count_missing_strike = int((~(K > 0) | ~np.isfinite(K)).sum())

    # d2 = (ln(S/K) + (r - 0.5*sigma^2)*tau) / (sigma * sqrt(tau))
    ln_SK = np.log(S / K)
    r = RISK_FREE_RATE
    d2 = np.full(len(df), np.nan, dtype=float)
    where = valid_iv
    sqrt_tau = np.sqrt(tau)
    sqrt_tau = np.where(sqrt_tau > 0, sqrt_tau, np.nan)
    d2[where] = (
        (ln_SK.values[where] + (r - 0.5 * sigma.values[where] ** 2) * tau[where])
        / (sigma.values[where] * np.sqrt(tau[where]))
    )
    df["d2"] = d2

    # prob_option = N(d2)
    prob_option = np.full(len(df), np.nan, dtype=float)
    if HAS_NORM:
        prob_option[where] = _norm_cdf(d2[where])
    df["prob_option"] = prob_option

    # probability_spread = prob_yes - prob_option (only when both valid)
    prob_yes = pd.to_numeric(df["prob_yes"], errors="coerce")
    valid_prob_yes = prob_yes.notna() & np.isfinite(prob_yes)
    valid_both = valid_iv & valid_prob_yes.values & np.isfinite(prob_option)
    probability_spread = np.full(len(df), np.nan, dtype=float)
    probability_spread[valid_both] = (prob_yes.values[valid_both] - prob_option[valid_both])
    df["probability_spread"] = probability_spread

    # Overwrite tau_years with NaN where invalid for consistency
    df.loc[~valid_iv, "tau_years"] = np.nan

    # Validation
    assert len(df) == n_input, "Output row count must equal input row count"
    if np.isfinite(prob_option).any():
        p = prob_option[np.isfinite(prob_option)]
        assert (p >= 0).all() and (p <= 1).all(), "prob_option must be in [0,1] when not null"
    assert (df["tau_years"].dropna() >= 0).all(), "tau_years must be non-negative"

    # Save output (parquet first; diagnostics next; CSV last so timeout still yields diagnostics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s (%d rows)", OUTPUT_PARQUET, len(df))

    # --- Summary JSON ---
    rows_with_prob = int(np.isfinite(prob_option).sum())
    tau_valid = df["tau_years"].dropna()
    prob_valid = df["prob_option"].dropna()
    spread_valid = df["probability_spread"].dropna()

    summary = {
        "input_path": str(INPUT_PARQUET),
        "output_path": str(OUTPUT_PARQUET),
        "total_rows": n_input,
        "valid_iv_rows": int(valid_iv.sum()),
        "rows_with_prob_option": rows_with_prob,
        "pct_rows_with_prob_option": round(100.0 * rows_with_prob / n_input, 4) if n_input else 0,
        "tau_years": {
            "min": float(tau_valid.min()) if len(tau_valid) else None,
            "max": float(tau_valid.max()) if len(tau_valid) else None,
            "mean": float(tau_valid.mean()) if len(tau_valid) else None,
            "median": float(tau_valid.median()) if len(tau_valid) else None,
        },
        "prob_option": {
            "min": float(prob_valid.min()) if len(prob_valid) else None,
            "max": float(prob_valid.max()) if len(prob_valid) else None,
            "mean": float(prob_valid.mean()) if len(prob_valid) else None,
            "median": float(prob_valid.median()) if len(prob_valid) else None,
        },
        "filter_counts": {
            "iv_hat_le_0": count_iv_le0,
            "iv_hat_gt_3": count_iv_gt3,
            "tau_years_le_0": count_tau_le0,
            "missing_or_invalid_spot": count_missing_spot,
            "missing_or_invalid_strike": count_missing_strike,
        },
        "probability_spread": {
            "min": float(spread_valid.min()) if len(spread_valid) else None,
            "max": float(spread_valid.max()) if len(spread_valid) else None,
            "mean": float(spread_valid.mean()) if len(spread_valid) else None,
            "median": float(spread_valid.median()) if len(spread_valid) else None,
            "p1": float(spread_valid.quantile(0.01)) if len(spread_valid) else None,
            "p5": float(spread_valid.quantile(0.05)) if len(spread_valid) else None,
            "p25": float(spread_valid.quantile(0.25)) if len(spread_valid) else None,
            "p50": float(spread_valid.quantile(0.50)) if len(spread_valid) else None,
            "p75": float(spread_valid.quantile(0.75)) if len(spread_valid) else None,
            "p95": float(spread_valid.quantile(0.95)) if len(spread_valid) else None,
            "p99": float(spread_valid.quantile(0.99)) if len(spread_valid) else None,
        },
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", SUMMARY_JSON)

    # --- Spread distribution JSON ---
    n_spread = len(spread_valid)
    bucket_list = []
    for item in SPREAD_BUCKETS:
        label = item[0]
        low, high = item[1], item[2]
        kind = item[3] if len(item) > 3 else "low_incl"
        if kind == "strict_lt":
            mask = spread_valid < high
        elif kind == "strict_gt":
            mask = spread_valid > low
        elif kind == "both_incl":
            mask = (spread_valid >= low) & (spread_valid <= high)
        elif kind == "low_incl":
            mask = (spread_valid >= low) & (spread_valid < high)
        else:
            mask = (spread_valid > low) & (spread_valid <= high)
        cnt = int(mask.sum())
        pct = round(100.0 * cnt / n_spread, 4) if n_spread else 0
        bucket_list.append({"bucket": label, "count": cnt, "pct": pct})
    abs_thresholds = {
        "abs_spread_gt_0_01": int((spread_valid.abs() > 0.01).sum()),
        "abs_spread_gt_0_02": int((spread_valid.abs() > 0.02).sum()),
        "abs_spread_gt_0_05": int((spread_valid.abs() > 0.05).sum()),
        "abs_spread_gt_0_10": int((spread_valid.abs() > 0.10).sum()),
    }
    spread_dist = {
        "n_valid_spread": n_spread,
        "bucket_distribution": bucket_list,
        "absolute_spread_thresholds": abs_thresholds,
    }
    SPREAD_JSON.write_text(json.dumps(spread_dist, indent=2), encoding="utf-8")
    logger.info("Wrote %s", SPREAD_JSON)

    # --- Plots ---
    DIAG_PLOTS.mkdir(parents=True, exist_ok=True)
    if HAS_MPL:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(prob_valid, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
            ax.set_xlabel("prob_option (N(d2))")
            ax.set_ylabel("count")
            ax.set_title("Distribution of option-implied probability")
            fig.tight_layout()
            fig.savefig(DIAG_PLOTS / "hist_prob_option.png", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", DIAG_PLOTS / "hist_prob_option.png")
        except Exception as e:
            logger.warning("hist_prob_option: %s", e)
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(spread_valid, bins=50, color="coral", alpha=0.8, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("probability_spread (prob_yes - prob_option)")
            ax.set_ylabel("count")
            ax.set_title("Distribution of probability spread")
            fig.tight_layout()
            fig.savefig(DIAG_PLOTS / "hist_probability_spread.png", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", DIAG_PLOTS / "hist_probability_spread.png")
        except Exception as e:
            logger.warning("hist_probability_spread: %s", e)
        try:
            df_plot = df.copy()
            df_plot["date"] = pd.to_datetime(df_plot["bucket_5m_utc"]).dt.date
            daily = df_plot.groupby("date")["probability_spread"].agg(["mean", "count"]).reset_index()
            daily = daily[daily["count"] > 0]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(daily["date"], daily["mean"], marker="o", markersize=3)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("date")
            ax.set_ylabel("mean probability_spread")
            ax.set_title("Average probability spread by day")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(DIAG_PLOTS / "timeseries_mean_spread_by_day.png", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", DIAG_PLOTS / "timeseries_mean_spread_by_day.png")
        except Exception as e:
            logger.warning("timeseries_mean_spread: %s", e)

    # Optional CSV (can be slow for large datasets)
    try:
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info("Wrote %s", OUTPUT_CSV)
    except Exception as e:
        logger.warning("CSV write skipped: %s", e)

    print("--- Summary ---")
    print(f"Input rows: {n_input}, Output rows: {len(df)} (match: {len(df) == n_input})")
    print(f"Valid IV rows: {valid_iv.sum()}, Rows with prob_option: {rows_with_prob}")
    print(f"prob_option range: [{prob_valid.min():.4f}, {prob_valid.max():.4f}]")
    print(f"probability_spread range: [{spread_valid.min():.4f}, {spread_valid.max():.4f}]")


if __name__ == "__main__":
    main()
