#!/usr/bin/env python3
"""
Full sanity audit of cleaned Step 3 outputs.
Uses: step3_events_clean.csv, step3_probabilities_5m_clean.parquet
Run: python scripts/run_step3_full_sanity_audit.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np


def main():
    window_dir = ROOT / "outputs" / "window=2026-01-01_2026-02-15"
    step3_dir = window_dir / "step3"

    events_path = step3_dir / "step3_events_clean.csv"
    probs_path = step3_dir / "step3_probabilities_5m_clean.parquet"

    if not events_path.exists():
        print("ERROR: step3_events_clean.csv not found at", events_path)
        sys.exit(1)
    if not probs_path.exists():
        print("ERROR: step3_probabilities_5m_clean.parquet not found at", probs_path)
        sys.exit(1)

    events = pd.read_csv(events_path)
    prob = pd.read_parquet(probs_path)

    # Normalize column names (Step 3 writes P_opt_bs, sigma_iv, S_t)
    if "P_opt_bs" in prob.columns and "probability_opt" not in prob.columns:
        prob["probability_opt"] = prob["P_opt_bs"]
    if "sigma_iv" in prob.columns and "sigma" not in prob.columns:
        prob["sigma"] = prob["sigma_iv"]
    if "S_t" in prob.columns and "index_close" not in prob.columns:
        prob["index_close"] = prob["S_t"]

    required = ["event_id", "open_time", "tau_years", "sigma", "probability_opt", "strike_K", "expiry_ms", "index_close"]
    missing = [c for c in required if c not in prob.columns]
    if missing:
        print("ERROR: Probabilities table missing required columns:", missing)
    else:
        print("All required columns present.")

    n_bars_expected = 13248  # Jan 1 - Feb 15 5m

    # ----- SECTION 1 — Structural Integrity -----
    print("\n" + "=" * 50)
    print("SECTION 1 — Structural Integrity")
    print("=" * 50)
    n_events = events["event_id"].nunique()
    dup_events = events["event_id"].duplicated().sum()
    print("Events table: n_events=%d, duplicates in event_id=%d" % (n_events, int(dup_events)))
    miss_expiry = events["expiry_ms"].isna().sum() if "expiry_ms" in events.columns else 0
    miss_strike = events["strike_K"].isna().sum() if "strike_K" in events.columns else 0
    print("Events missing expiry: %d, missing strike: %d" % (miss_expiry, miss_strike))

    n_rows = len(prob)
    dup_prob = prob.duplicated(subset=["event_id", "open_time"]).sum()
    print("Probabilities table: n_rows=%d, duplicates (event_id, open_time)=%d" % (n_rows, int(dup_prob)))
    print("Probabilities columns:", list(prob.columns))

    # ----- SECTION 2 — Probability Validity -----
    print("\n" + "=" * 50)
    print("SECTION 2 — Probability Validity")
    print("=" * 50)
    p = prob["probability_opt"]
    out_lo = (p < -0.001).sum()
    out_hi = (p > 1.001).sum()
    print("probability < -0.001: %d" % out_lo)
    print("probability > 1.001: %d" % out_hi)
    print("min(probability): %s" % p.min())
    print("max(probability): %s" % p.max())
    print("median(probability): %s" % p.median())
    print("NaNs: sigma=%d, probability_opt=%d, tau_years=%d" % (
        prob["sigma"].isna().sum(), prob["probability_opt"].isna().sum(), prob["tau_years"].isna().sum()))

    probability_out_of_bounds_count = int(out_lo + out_hi)
    prob_min = float(p.min()) if p.notna().any() else np.nan
    prob_max = float(p.max()) if p.notna().any() else np.nan
    prob_median = float(p.median()) if p.notna().any() else np.nan

    # ----- SECTION 3 — Time to Expiry -----
    print("\n" + "=" * 50)
    print("SECTION 3 — Time to Expiry Sanity")
    print("=" * 50)
    negative_tau_count = (prob["tau_years"] < 0).sum()
    if negative_tau_count > 0:
        print("FLAG: tau < 0 count = %d" % negative_tau_count)
    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid].sort_values("open_time")
        tau = sub["tau_years"].dropna()
        if len(tau):
            print("%s: tau_years min=%.6f, max=%.6f" % (eid, tau.min(), tau.max()))
            if tau.min() < 0:
                print("  FLAG: negative tau")

    # ----- SECTION 4 — Sigma Sanity -----
    print("\n" + "=" * 50)
    print("SECTION 4 — Sigma Sanity")
    print("=" * 50)
    sigma_extreme_count = 0
    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid]
        sig = sub["sigma"].dropna()
        if len(sig) == 0:
            print("%s: no sigma" % eid)
            continue
        med = sig.median()
        p10 = sig.quantile(0.1)
        p90 = sig.quantile(0.9)
        mn = sig.min()
        mx = sig.max()
        n_neg = (sub["sigma"] < 0).sum()
        n_gt3 = (sub["sigma"] > 3).sum()
        sigma_extreme_count += n_neg + n_gt3
        flags = []
        if n_neg: flags.append("sigma<0")
        if n_gt3: flags.append("sigma>3")
        if med < 0.01: flags.append("median_sigma<0.01")
        print("%s: median=%.4f, p10=%.4f, p90=%.4f, min=%.4f, max=%.4f  %s" % (eid, med, p10, p90, mn, mx, " FLAG: " + ", ".join(flags) if flags else ""))

    # ----- SECTION 5 — Monotonicity -----
    print("\n" + "=" * 50)
    print("SECTION 5 — Monotonicity Across Strikes")
    print("=" * 50)
    mono_checks = 0
    mono_violations = 0
    expiries = prob["expiry_ms"].unique()
    for t in prob["open_time"].drop_duplicates():
        for exp_ms in expiries:
            row = prob[(prob["open_time"] == t) & (prob["expiry_ms"] == exp_ms)]
            if len(row) < 3:
                continue
            ser = row.set_index("strike_K")["probability_opt"].dropna()
            vals = ser.reindex([65000, 70000, 72000]).dropna()
            if len(vals) != 3:
                continue
            p65, p70, p72 = vals.iloc[0], vals.iloc[1], vals.iloc[2]
            mono_checks += 1
            if p65 < p70 - 1e-6 or p70 < p72 - 1e-6:
                mono_violations += 1
    mono_rate = 100.0 * mono_violations / mono_checks if mono_checks else 0
    print("number of checks: %d" % mono_checks)
    print("violations: %d" % mono_violations)
    print("violation %%: %.2f" % mono_rate)

    # ----- SECTION 6 — Spot Consistency (correlation delta_P vs index return) -----
    print("\n" + "=" * 50)
    print("SECTION 6 — Spot Consistency (delta_P vs index return)")
    print("=" * 50)
    prob_sorted = prob.sort_values(["event_id", "open_time"])
    prob_sorted["index_return"] = prob_sorted.groupby("event_id")["index_close"].transform(lambda x: x.pct_change())
    prob_sorted["delta_P"] = prob_sorted.groupby("event_id")["probability_opt"].transform(lambda x: x.diff())
    for eid in prob["event_id"].unique():
        sub = prob_sorted[(prob_sorted["event_id"] == eid)][["delta_P", "index_return"]].dropna()
        if len(sub) < 10:
            print("%s: insufficient data" % eid)
            continue
        corr = sub["delta_P"].corr(sub["index_return"])
        print("%s: corr(delta_P, index_return) = %.4f" % (eid, corr))

    # ----- SECTION 7 — Coverage -----
    print("\n" + "=" * 50)
    print("SECTION 7 — Coverage")
    print("=" * 50)
    lowest_coverage = 100.0
    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid]
        n_non_nan = sub["probability_opt"].notna().sum()
        cov = 100.0 * n_non_nan / n_bars_expected if n_bars_expected else 0
        if cov < lowest_coverage:
            lowest_coverage = cov
        flag = " FLAG: coverage < 25%%" if cov < 25 else ""
        print("%s: n_bars_expected=%d, n_non_nan_probability=%d, coverage %%=%.2f%s" % (eid, n_bars_expected, n_non_nan, cov, flag))

    # ----- SECTION 8 — Final Summary Block -----
    print("\n" + "=" * 60)
    print("STEP 3 FULL SANITY AUDIT")
    print("=" * 60)
    print("n_events: %d" % n_events)
    print("n_rows_total: %d" % n_rows)
    print("probability_min: %s" % prob_min)
    print("probability_max: %s" % prob_max)
    print("probability_median: %s" % prob_median)
    print("probability_out_of_bounds_count: %d" % probability_out_of_bounds_count)
    print("negative_tau_count: %d" % int(negative_tau_count))
    print("sigma_extreme_count: %d" % sigma_extreme_count)
    print("monotonicity_violation_rate: %.2f" % mono_rate)
    print("lowest_event_coverage_pct: %.2f" % lowest_coverage)
    print("=" * 60)

    print("\nRun this:")
    print("python scripts/run_step3_full_sanity_audit.py")


if __name__ == "__main__":
    main()
