#!/usr/bin/env python3
"""
Step 3: Build derivatives-implied benchmark probability P^OPT for prototype event set.
Event set: expiries 2026-02-20 08:00, 2026-02-27 08:00, 2026-03-27 08:00 UTC; K in {65000, 70000, 72000}.
Method 1 (BS): Infer IV from near-K option, then P^OPT_t = N(d2). Method 2 (optional): digital from call spread.

Run: python scripts/run_step3_derivatives_probability.py --window_dir "outputs/window=2026-01-01_2026-02-15" [--overwrite] [--method bs|bs+spread]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
except ImportError:
    norm = None
    brentq = None


# Default event set (expiry UTC strings and strikes)
DEFAULT_EXPIRY_UTC = [
    "2026-02-20T08:00:00.000Z",
    "2026-02-27T08:00:00.000Z",
    "2026-03-27T08:00:00.000Z",
]
DEFAULT_STRIKES = [65000, 70000, 72000]

# ACT/365: seconds per year
SEC_PER_YEAR = 365.25 * 24 * 3600
MS_PER_YEAR = SEC_PER_YEAR * 1000

# IV bounds (skip if outside)
SIGMA_MIN, SIGMA_MAX = 0.05, 3.0
FFILL_BARS = 6


def _utc_str_to_ms(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _ms_to_utc_iso(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def bs_call_price(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_put_price(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def p_above_k_lognormal(S: float, K: float, tau: float, sigma: float) -> float:
    """P(S_T > K) = N(d2) under BS/lognormal."""
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    sqrt_tau = np.sqrt(tau)
    d2 = (np.log(S / K) - 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    return float(norm.cdf(d2))


def invert_iv_call(price: float, S: float, K: float, tau: float, sigma_lo: float = 0.001, sigma_hi: float = 5.0):
    if price <= 0 or S <= 0 or K <= 0 or tau <= 0:
        return np.nan
    no_arb_lo = max(0, S - K)
    if price < no_arb_lo - 1e-6 or price > S + 1e-6:
        return np.nan
    def obj(sig):
        return bs_call_price(S, K, tau, sig) - price
    try:
        sigma = brentq(obj, sigma_lo, sigma_hi)
        return sigma
    except Exception:
        return np.nan


def invert_iv_put(price: float, S: float, K: float, tau: float, sigma_lo: float = 0.001, sigma_hi: float = 5.0):
    if price <= 0 or S <= 0 or K <= 0 or tau <= 0:
        return np.nan
    no_arb_lo = max(0, K - S)
    if price < no_arb_lo - 1e-6 or price > K + 1e-6:
        return np.nan
    def obj(sig):
        return bs_put_price(S, K, tau, sig) - price
    try:
        sigma = brentq(obj, sigma_lo, sigma_hi)
        return sigma
    except Exception:
        return np.nan


def build_events_from_options(options: pd.DataFrame) -> pd.DataFrame:
    """Build event set: (expiry_ms, strike_K) with best instrument per event."""
    if options.empty or "expiry" not in options.columns:
        return pd.DataFrame()
    options = options.copy()
    if "expiry_utc" not in options.columns and "expiry" in options.columns:
        options["expiry_utc"] = options["expiry"].astype("int64").map(_ms_to_utc_iso)
    if "call_put" not in options.columns and "instrument_name" in options.columns:
        options["call_put"] = options["instrument_name"].str.extract(r"-([CP])$", expand=False)

    expiry_ms_list = [_utc_str_to_ms(e) for e in DEFAULT_EXPIRY_UTC]
    rows = []
    for exp_ms in expiry_ms_list:
        exp_utc = _ms_to_utc_iso(exp_ms)
        sub = options[options["expiry"].astype("int64") == exp_ms]
        if sub.empty:
            sub = options[options["expiry_utc"] == exp_utc]
        if sub.empty and options["expiry"].notna().any():
            exp_sec = exp_ms // 1000
            sub = options[options["expiry"].astype("int64") == exp_sec]
        if sub.empty:
            continue
        for K in DEFAULT_STRIKES:
            inst_sub = sub[(sub["strike"] >= K - 2000) & (sub["strike"] <= K + 2000)]
            if inst_sub.empty:
                continue
            inst_sub = inst_sub.drop_duplicates(subset=["instrument_name"])
            call_pref = inst_sub[inst_sub["call_put"] == "C"]
            if not call_pref.empty:
                best = call_pref.iloc[(call_pref["strike"] - K).abs().argmin()]
            else:
                best = inst_sub.iloc[(inst_sub["strike"] - K).abs().argmin()]
            instrument = best["instrument_name"]
            strike_used = float(best["strike"])
            opt_type = "CALL" if best.get("call_put") == "C" else "PUT"
            event_id = "%s_%d" % (exp_utc[:10].replace("-", ""), int(K))
            rows.append({
                "event_id": event_id,
                "expiry_utc": exp_utc,
                "expiry_ms": exp_ms,
                "strike_K": K,
                "instrument_name": instrument,
                "strike_used": strike_used,
                "option_type_used": opt_type,
                "method_bs": True,
                "method_spread": False,
            })
    return pd.DataFrame(rows)


def compute_probabilities_for_event(
    event_row: pd.Series,
    master: pd.DataFrame,
    options: pd.DataFrame,
    method_spread: bool,
) -> pd.DataFrame:
    """For one event, compute P_opt_bs (and optionally P_opt_spread) for each 5m bar."""
    event_id = event_row["event_id"]
    expiry_ms = int(event_row["expiry_ms"])
    K = float(event_row["strike_K"])
    instrument = event_row["instrument_name"]
    opt_type = event_row["option_type_used"]
    strike_used = float(event_row["strike_used"])

    opt_cols = ["open_time", "close", "index_close"]
    if "volume" in options.columns:
        opt_cols.append("volume")
    opt_sub = options[options["instrument_name"] == instrument][opt_cols].copy()
    opt_sub = opt_sub.rename(columns={"close": "option_close", "index_close": "opt_index_close"})
    m = master[["open_time", "index_close"]].copy()
    m = m.rename(columns={"index_close": "S_t"})
    df = m.merge(opt_sub, on="open_time", how="left")
    df["event_id"] = event_id
    df["expiry_ms"] = expiry_ms
    df["strike_K"] = K
    df["tau_years"] = (expiry_ms - df["open_time"]) / MS_PER_YEAR
    df["option_instrument_used"] = instrument
    df["option_type_used"] = opt_type
    df["option_price_used"] = df["option_close"]
    df["option_volume"] = df["volume"] if "volume" in df.columns else np.nan
    df["sigma_iv"] = np.nan
    df["sigma_ffill_used"] = 0
    df["P_opt_bs"] = np.nan

    S_arr = df["S_t"].values
    price_arr = df["option_price_used"].values
    tau_arr = df["tau_years"].values
    n = len(df)

    for i in range(n):
        S, price, tau = S_arr[i], price_arr[i], tau_arr[i]
        if pd.isna(S) or S <= 0 or pd.isna(price) or price <= 0 or pd.isna(tau) or tau <= 0:
            continue
        if opt_type == "CALL":
            sigma = invert_iv_call(price, S, strike_used, tau)
        else:
            sigma = invert_iv_put(price, S, strike_used, tau)
        if np.isnan(sigma) or sigma < SIGMA_MIN or sigma > SIGMA_MAX:
            continue
        df.iloc[i, df.columns.get_loc("sigma_iv")] = sigma
        df.iloc[i, df.columns.get_loc("P_opt_bs")] = p_above_k_lognormal(S, K, tau, sigma)

    # Optional forward-fill sigma from PAST bars (up to 6 bars) to bridge sparse trading
    sigma_col = df["sigma_iv"].values.copy()
    p_col = df["P_opt_bs"].values.copy()
    for i in range(n):
        if not np.isnan(sigma_col[i]):
            continue
        for j in range(1, min(FFILL_BARS + 1, i + 1)):
            if i - j < 0:
                break
            if not np.isnan(sigma_col[i - j]):
                sigma_col[i] = sigma_col[i - j]
                df.iloc[i, df.columns.get_loc("sigma_ffill_used")] = 1
                S, tau = S_arr[i], tau_arr[i]
                if not (pd.isna(S) or S <= 0 or pd.isna(tau) or tau <= 0):
                    p_col[i] = p_above_k_lognormal(S, K, tau, sigma_col[i])
                break

    df["sigma_iv"] = sigma_col
    df["P_opt_bs"] = p_col
    df = df.drop(columns=["option_close", "opt_index_close", "volume"], errors="ignore")
    return df


def main():
    parser = argparse.ArgumentParser(description="Step 3: Derivatives-implied probability P^OPT")
    parser.add_argument("--window_dir", type=str, default="outputs/window=2026-01-01_2026-02-15")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", type=str, default="bs", choices=["bs", "bs+spread"])
    parser.add_argument("--events_config", type=str, default=None)
    args = parser.parse_args()

    if norm is None or brentq is None:
        print("ERROR: scipy (norm, brentq) required. Install scipy.")
        sys.exit(1)

    window_dir = Path(args.window_dir)
    if not window_dir.is_absolute():
        window_dir = ROOT / window_dir
    if not window_dir.exists():
        print("ERROR: window_dir not found:", window_dir)
        sys.exit(1)

    step3_dir = window_dir / "step3"
    tables_dir = step3_dir / "step3_summary_tables"
    figures_dir = step3_dir / "step3_figures"
    if step3_dir.exists() and not args.overwrite:
        print("step3 output exists and --overwrite not set. Exiting.")
        sys.exit(0)
    step3_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    master_path = window_dir / "master_panel_5m.parquet"
    opt_path = window_dir / "options_panel_long_5m_liquid_aligned.parquet"
    if not master_path.exists():
        master_path = next(window_dir.glob("master_panel*.parquet"), None)
    if not opt_path.exists():
        opt_path = next((f for f in window_dir.glob("options_panel*liquid*.parquet")), None)
    if not master_path or not master_path.exists():
        print("ERROR: master panel not found")
        sys.exit(1)
    if not opt_path or not opt_path.exists():
        print("ERROR: options panel not found")
        sys.exit(1)

    master = pd.read_parquet(master_path)
    options = pd.read_parquet(opt_path)
    if "open_time_utc" not in master.columns and "open_time" in master.columns:
        master["open_time_utc"] = master["open_time"].astype("int64").map(_ms_to_utc_iso)

    events_df = build_events_from_options(options)
    if events_df.empty:
        print("ERROR: no events built from options")
        sys.exit(1)
    events_df.to_csv(step3_dir / "step3_events.csv", index=False)

    prob_dfs = []
    for _, ev in events_df.iterrows():
        prob_dfs.append(compute_probabilities_for_event(ev, master, options, method_spread=args.method == "bs+spread"))
    prob = pd.concat(prob_dfs, ignore_index=True)
    prob["open_time_utc"] = prob["open_time"].astype("int64").map(_ms_to_utc_iso)
    prob.to_parquet(step3_dir / "step3_probabilities_5m.parquet", index=False)

    n_bars_expected = len(master)
    n_events = len(events_df)

    # Summary tables
    iv_rows = []
    prob_summary_rows = []
    inst_usage = prob.groupby("option_instrument_used").size().reset_index(name="usage_count")
    inst_usage.to_csv(tables_dir / "instrument_usage_counts.csv", index=False)

    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid]
        n_sig = sub["sigma_iv"].notna().sum()
        n_p = sub["P_opt_bs"].notna().sum()
        pct_sig = 100.0 * n_sig / n_bars_expected if n_bars_expected else 0
        pct_p = 100.0 * n_p / n_bars_expected if n_bars_expected else 0
        sig_vals = sub["sigma_iv"].dropna()
        iv_rows.append({
            "event_id": eid,
            "n_bars_expected": n_bars_expected,
            "n_sigma_non_nan": int(n_sig),
            "pct_sigma": round(pct_sig, 2),
            "n_p_non_nan": int(n_p),
            "pct_p": round(pct_p, 2),
            "median_sigma": sig_vals.median() if len(sig_vals) else np.nan,
            "p10_sigma": sig_vals.quantile(0.1) if len(sig_vals) else np.nan,
            "p90_sigma": sig_vals.quantile(0.9) if len(sig_vals) else np.nan,
        })
        p_vals = sub["P_opt_bs"].dropna()
        prob_summary_rows.append({
            "event_id": eid,
            "mean_P": p_vals.mean() if len(p_vals) else np.nan,
            "std_P": p_vals.std() if len(p_vals) else np.nan,
            "min_P": p_vals.min() if len(p_vals) else np.nan,
            "max_P": p_vals.max() if len(p_vals) else np.nan,
        })
    pd.DataFrame(iv_rows).to_csv(tables_dir / "iv_coverage_by_event.csv", index=False)
    pd.DataFrame(prob_summary_rows).to_csv(tables_dir / "probability_summary_by_event.csv", index=False)

    # Figures: for each event (at least top 3) P_opt_bs, sigma_iv, option_price time series; then sigma hist
    event_ids_plot = list(prob["event_id"].unique())[: max(3, n_events)]
    for eid in event_ids_plot:
        sub = prob[prob["event_id"] == eid].sort_values("open_time")
        if sub.empty:
            continue
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # P_opt_bs
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["open_time"], sub["P_opt_bs"], alpha=0.8)
        ax.set_xlabel("open_time (ms)")
        ax.set_ylabel("P_opt_bs")
        ax.set_title("P^OPT (BS) — %s" % eid)
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(figures_dir / ("P_opt_%s.png" % eid.replace("-", "_")), dpi=150, bbox_inches="tight")
        plt.close()
        # sigma_iv
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["open_time"], sub["sigma_iv"], alpha=0.8)
        ax.set_xlabel("open_time (ms)")
        ax.set_ylabel("sigma_iv")
        ax.set_title("IV (BS) — %s" % eid)
        fig.savefig(figures_dir / ("sigma_iv_%s.png" % eid.replace("-", "_")), dpi=150, bbox_inches="tight")
        plt.close()
        # option_price_used
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["open_time"], sub["option_price_used"], alpha=0.8)
        ax.set_xlabel("open_time (ms)")
        ax.set_ylabel("option_price_used")
        ax.set_title("Option price used — %s" % eid)
        fig.savefig(figures_dir / ("option_price_%s.png" % eid.replace("-", "_")), dpi=150, bbox_inches="tight")
        plt.close()
    # Sigma distribution across all events
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(prob["sigma_iv"].dropna(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("sigma_iv")
    ax.set_ylabel("count")
    ax.set_title("Distribution of sigma_iv (all events)")
    fig.savefig(figures_dir / "sigma_iv_hist_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sanity checks
    violations = ((prob["P_opt_bs"] < -0.001) | (prob["P_opt_bs"] > 1.001)).sum()
    prob_clipped = prob["P_opt_bs"].clip(0, 1)

    # Monotonicity: 10 random timestamps, P(65k) >= P(70k) >= P(72k) at same expiry
    rng = np.random.default_rng(42)
    expiries = prob["expiry_ms"].unique()
    mono_checks = 0
    mono_violations = 0
    for _ in range(10):
        t = prob["open_time"].drop_duplicates().sample(1, random_state=rng).iloc[0]
        for exp_ms in expiries:
            row = prob[(prob["open_time"] == t) & (prob["expiry_ms"] == exp_ms)]
            if len(row) < 3:
                continue
            ser = row.set_index("strike_K")["P_opt_bs"].dropna()
            vals = ser.reindex([65000, 70000, 72000]).dropna()
            if len(vals) != 3:
                continue
            p65, p70, p72 = vals.iloc[0], vals.iloc[1], vals.iloc[2]
            mono_checks += 1
            if p65 < p70 - 1e-6 or p70 < p72 - 1e-6:
                mono_violations += 1
    mono_rate = 100.0 * mono_violations / mono_checks if mono_checks else 0

    # Instrument liquidity by event
    liquidity_rows = []
    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid]
        if "option_volume" not in sub.columns:
            vol = pd.Series(0, index=sub.index)
        else:
            vol = sub["option_volume"].fillna(0)
        nz = (vol > 0).sum()
        share_nz = 100.0 * nz / len(sub) if len(sub) else 0
        med_nz = vol[vol > 0].median() if (vol > 0).any() else np.nan
        liquidity_rows.append({"event_id": eid, "share_bars_volume_gt0": round(share_nz, 2), "median_volume_nonzero": med_nz})
    pd.DataFrame(liquidity_rows).to_csv(tables_dir / "instrument_liquidity_by_event.csv", index=False)

    # Print STEP 3 SANITY CHECK BLOCK
    print("\n" + "=" * 60)
    print("STEP 3 SANITY CHECK BLOCK")
    print("=" * 60)
    print("\nA) Coverage (per event)")
    for r in iv_rows:
        status = "PASS" if r["pct_p"] >= 20 else "WARN"
        print("  %s: n_bars_expected=%s, n_sigma_non_nan=%s, pct_sigma=%.2f%%, n_p_non_nan=%s, pct_p=%.2f%%, median_sigma=%.4f, p10/p90_sigma=%.4f/%.4f  [%s]" % (
            r["event_id"], r["n_bars_expected"], r["n_sigma_non_nan"], r["pct_sigma"], r["n_p_non_nan"], r["pct_p"],
            r["median_sigma"] if not np.isnan(r["median_sigma"]) else 0,
            r["p10_sigma"] if not np.isnan(r.get("p10_sigma", np.nan)) else 0,
            r["p90_sigma"] if not np.isnan(r.get("p90_sigma", np.nan)) else 0,
            status,
        ))
    print("\nB) Probability bounds")
    print("  Violations (P < -0.001 or P > 1.001): %d (must be 0 or negligible)" % violations)
    print("\nC) Monotonicity (P(65k) >= P(70k) >= P(72k) at same t, expiry)")
    print("  Checks: %d, Violations: %d, Rate: %.2f%%  %s" % (mono_checks, mono_violations, mono_rate, "WARN" if mono_rate > 5 else "OK"))
    print("\nD) Time-to-expiry")
    for eid in prob["event_id"].unique():
        sub = prob[prob["event_id"] == eid]
        tau = sub["tau_years"].dropna()
        if len(tau):
            print("  %s: tau_years min=%.6f, max=%.6f" % (eid, tau.min(), tau.max()))
    print("\nE) Instrument liquidity (share bars with volume>0, median vol nonzero)")
    for r in liquidity_rows:
        print("  %s: share_vol_gt0=%.2f%%, median_vol_nz=%s" % (r["event_id"], r["share_bars_volume_gt0"], r["median_volume_nonzero"]))
    print("\nF) Output paths")
    print("  events: %s" % (step3_dir / "step3_events.csv"))
    print("  probabilities: %s" % (step3_dir / "step3_probabilities_5m.parquet"))
    print("  tables: %s" % tables_dir)
    print("  figures: %s" % figures_dir)
    print("=" * 60)

    # Top lines of step3_events.csv and sample iv_coverage for main event
    print("\nFirst lines of step3_events.csv:")
    print(events_df.head().to_string())
    main_ev = "20260227_70000" if "20260227_70000" in events_df["event_id"].values else events_df["event_id"].iloc[0]
    iv_main = pd.DataFrame(iv_rows)
    if main_ev in iv_main["event_id"].values:
        print("\niv_coverage_by_event.csv sample (main event %s):" % main_ev)
        print(iv_main[iv_main["event_id"] == main_ev].to_string())


if __name__ == "__main__":
    main()
