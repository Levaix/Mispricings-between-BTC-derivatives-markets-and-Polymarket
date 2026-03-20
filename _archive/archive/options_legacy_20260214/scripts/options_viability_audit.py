#!/usr/bin/env python3
"""
Options viability audit: validate Deribit options dataset for Oct–Dec 2025 analysis.
Checks Risk 1 (instrument universe / candles in window) and Risk 2 (ATM selection realistic).
Saves: options_viability_report.json, options_selected_instruments_summary.parquet,
       options_viability_top_instruments.csv.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.time import get_start_end_ms, ms_to_utc_iso

# Window
START_MS, END_MS = get_start_end_ms()
START_UTC = "2025-10-01T00:00:00Z"
END_UTC = "2025-12-31T23:55:00Z"

# Paths
DATA_ROOT = Path(config.DATA_ROOT)
OPTIONS_SELECTED = Path(config.OPTIONS_SELECTED_INSTRUMENTS_PATH)
OPTIONS_PANEL = ROOT / "outputs" / "options_panel_long_5m.parquet"
MASTER_PANEL = ROOT / "outputs" / "master_panel_5m.parquet"
OUTPUT_DIR = ROOT / "outputs"
REPORT_JSON = OUTPUT_DIR / "options_viability_report.json"
SUMMARY_PARQUET = OUTPUT_DIR / "options_selected_instruments_summary.parquet"
TOP_CSV = OUTPUT_DIR / "options_viability_top_instruments.csv"

# Thresholds
FAIL_CREATION_AFTER_END_PCT = 50.0
FAIL_EXPIRY_2026_AND_CREATION_2026_PCT = 50.0
FAIL_MIN_INSTRUMENTS_WITH_500_BARS = 10
FAIL_INDEX_NAN_PCT = 5.0
FAIL_MEDIAN_STRIKE_DISTANCE_PCT = 0.12
FAIL_STRIKE_DISTANCE_15_PCT = 0.15
FALLBACK_ATM = 95000.0
FALLBACK_TOLERANCE_PCT = 5.0  # if median index ~95k and strikes cluster 95k, warn


def _ms_to_utc_iso(ms: int) -> str:
    if ms is None or (isinstance(ms, float) and (ms != ms or ms <= 0)):
        return ""
    return ms_to_utc_iso(int(ms))


def run_risk1_check_1a(selected: list, report: dict) -> bool:
    """Selected instruments expiry/creation distribution. Returns True if PASS."""
    if not selected:
        report["risk1_check1a"] = {"pass": False, "reason": "selected count is 0"}
        return False

    n = len(selected)
    expiries = []
    creations = []
    creation_missing = 0
    creation_after_end = 0
    expiry_before_start = 0

    for inv in selected:
        exp = inv.get("expiration_timestamp") or inv.get("expiry")
        if exp is not None:
            exp = int(exp)
            expiries.append(exp)
            if exp < START_MS:
                expiry_before_start += 1
        cr = inv.get("creation_timestamp")
        if cr is None or cr == 0 or (isinstance(cr, float) and (cr != cr or cr <= 0)):
            creation_missing += 1
        else:
            cr = int(cr)
            creations.append(cr)
            if cr > END_MS:
                creation_after_end += 1

    # Histogram by expiry year-month
    hist = defaultdict(int)
    for ms in expiries:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        hist[key] += 1
    expiry_histogram = dict(sorted(hist.items()))

    min_exp = min(expiries) if expiries else None
    max_exp = max(expiries) if expiries else None
    median_exp = int(pd.Series(expiries).median()) if expiries else None
    min_cr = min(creations) if creations else None
    max_cr = max(creations) if creations else None
    median_cr = int(pd.Series(creations).median()) if creations else None

    share_creation_missing = (creation_missing / n) * 100.0
    share_creation_after_end = (creation_after_end / n) * 100.0 if n else 0
    share_expiry_before_start = (expiry_before_start / n) * 100.0 if n else 0

    # FAIL conditions
    fail = False
    reasons = []
    if share_creation_after_end > FAIL_CREATION_AFTER_END_PCT:
        fail = True
        reasons.append(f">{FAIL_CREATION_AFTER_END_PCT}% creation_timestamp after 2025-12-31")
    if n == 0:
        fail = True
        reasons.append("selected count is 0")

    # 2026+ expiry AND creation in 2026+
    if expiries and creations:
        exp_2026_plus = sum(1 for e in expiries if datetime.fromtimestamp(e / 1000.0, tz=timezone.utc).year >= 2026)
        cr_2026_plus = sum(1 for c in creations if datetime.fromtimestamp(c / 1000.0, tz=timezone.utc).year >= 2026)
        if (exp_2026_plus / n) * 100 > FAIL_EXPIRY_2026_AND_CREATION_2026_PCT and (cr_2026_plus / n) * 100 > FAIL_EXPIRY_2026_AND_CREATION_2026_PCT:
            fail = True
            reasons.append(">50% expiries in 2026+ and creation also in 2026+")

    report["risk1_check1a"] = {
        "pass": not fail,
        "reasons": reasons if fail else None,
        "count_selected": n,
        "expiry_min_utc": _ms_to_utc_iso(min_exp) if min_exp else None,
        "expiry_median_utc": _ms_to_utc_iso(median_exp) if median_exp else None,
        "expiry_max_utc": _ms_to_utc_iso(max_exp) if max_exp else None,
        "expiry_histogram_year_month": expiry_histogram,
        "creation_min_utc": _ms_to_utc_iso(min_cr) if min_cr else None,
        "creation_median_utc": _ms_to_utc_iso(median_cr) if median_cr else None,
        "creation_max_utc": _ms_to_utc_iso(max_cr) if max_cr else None,
        "share_creation_missing_or_zero_pct": round(share_creation_missing, 2),
        "share_creation_after_end_pct": round(share_creation_after_end, 2),
        "share_expiry_before_start_pct": round(share_expiry_before_start, 2),
    }
    return not fail


def run_risk1_check_1b(panel_path: Path, report: dict, top_table: list) -> bool:
    """Actual candles in window. Returns True if PASS."""
    if not panel_path.exists():
        report["risk1_check1b"] = {"pass": False, "reason": "options_panel_long_5m.parquet not found"}
        return False

    # Load only needed columns
    try:
        schema_names = pq.read_schema(panel_path).names
    except Exception:
        schema_names = ["instrument_name", "open_time", "volume"]
    base_cols = ["instrument_name", "open_time", "volume"]
    optional = [c for c in ["expiry", "strike", "call_put"] if c in schema_names]
    cols = base_cols + optional
    df = pd.read_parquet(panel_path, columns=cols)
    total_rows = len(df)
    if total_rows == 0:
        report["risk1_check1b"] = {"pass": False, "reason": "total rows == 0", "total_rows": 0}
        return False

    unique_instruments = df["instrument_name"].nunique()
    df["open_time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    earliest_utc = df["open_time_utc"].min()
    latest_utc = df["open_time_utc"].max()

    grp = df.groupby("instrument_name").agg(
        n_bars=("open_time", "nunique"),
        start_utc=("open_time_utc", "min"),
        end_utc=("open_time_utc", "max"),
        median_volume=("volume", "median"),
    )
    grp["zero_volume_share"] = df.groupby("instrument_name")["volume"].apply(lambda s: (s.fillna(0) == 0).mean())
    grp = grp.reset_index()

    n_with_500_bars = (grp["n_bars"] > 500).sum()
    most_zero_volume = grp["zero_volume_share"].median()

    fail = False
    reasons = []
    if total_rows == 0:
        fail = True
        reasons.append("total rows == 0")
    if n_with_500_bars < FAIL_MIN_INSTRUMENTS_WITH_500_BARS:
        fail = True
        reasons.append(f"<{FAIL_MIN_INSTRUMENTS_WITH_500_BARS} instruments with n_bars > 500")
    if most_zero_volume >= 0.99:
        fail = True
        reasons.append("most instruments have ~100% zero volume")

    # Top 15 by n_bars (use same df; add expiry/strike/call_put if present)
    for c in ["expiry", "strike", "call_put"]:
        if c not in df.columns:
            df[c] = pd.NA if c != "call_put" else ""
    grp_full = df.groupby("instrument_name").agg(
        n_bars=("open_time", "nunique"),
        start_utc=("open_time_utc", "min"),
        end_utc=("open_time_utc", "max"),
        median_volume=("volume", "median"),
        zero_volume_share=("volume", lambda s: (s.fillna(0) == 0).mean()),
    ).reset_index()
    if "expiry" in df.columns and df["expiry"].notna().any():
        grp_full["expiry"] = grp_full["instrument_name"].map(df.groupby("instrument_name")["expiry"].first())
    else:
        grp_full["expiry"] = pd.NA
    if "strike" in df.columns and df["strike"].notna().any():
        grp_full["strike"] = grp_full["instrument_name"].map(df.groupby("instrument_name")["strike"].first())
    else:
        grp_full["strike"] = pd.NA
    if "call_put" in df.columns:
        grp_full["call_put"] = grp_full["instrument_name"].map(df.groupby("instrument_name")["call_put"].first()).fillna("")
    else:
        grp_full["call_put"] = ""
    grp_full["expiry_utc"] = grp_full["expiry"].apply(lambda x: _ms_to_utc_iso(x) if pd.notna(x) and x else "")
    top15 = grp_full.nlargest(15, "n_bars")
    for _, r in top15.iterrows():
        top_table.append({
            "instrument_name": r["instrument_name"],
            "expiry_utc": r.get("expiry_utc", ""),
            "strike": r.get("strike"),
            "call_put": r.get("call_put", ""),
            "n_bars": int(r["n_bars"]),
            "start_utc": str(r["start_utc"]) if pd.notna(r.get("start_utc")) else "",
            "end_utc": str(r["end_utc"]) if pd.notna(r.get("end_utc")) else "",
            "median_volume": float(r["median_volume"]) if pd.notna(r.get("median_volume")) else None,
            "zero_volume_share": float(r["zero_volume_share"]) if pd.notna(r.get("zero_volume_share")) else None,
        })

    report["risk1_check1b"] = {
        "pass": not fail,
        "reasons": reasons if fail else None,
        "total_rows": int(total_rows),
        "unique_instruments": int(unique_instruments),
        "earliest_open_time_utc": earliest_utc,
        "latest_open_time_utc": latest_utc,
        "n_instruments_with_500_plus_bars": int(n_with_500_bars),
        "median_zero_volume_share_across_instruments": round(float(most_zero_volume), 4),
    }
    return not fail


def run_risk2_check_2a(master_path: Path, report: dict) -> bool:
    """Underlying index availability. Returns True if PASS."""
    if not master_path.exists():
        report["risk2_check2a"] = {"pass": False, "reason": "master_panel_5m.parquet not found"}
        return False

    df = pd.read_parquet(master_path, columns=["open_time", "index_close"])
    if "index_close" not in df.columns:
        report["risk2_check2a"] = {"pass": False, "reason": "index_close column missing"}
        return False

    n = len(df)
    nan_count = df["index_close"].isna().sum()
    share_nan = (nan_count / n) * 100.0 if n else 100.0
    valid = df["index_close"].dropna()
    index_min = float(valid.min()) if len(valid) else None
    index_median = float(valid.median()) if len(valid) else None
    index_max = float(valid.max()) if len(valid) else None

    fail = share_nan > FAIL_INDEX_NAN_PCT
    report["risk2_check2a"] = {
        "pass": not fail,
        "reason": f"index_close NaN share {share_nan:.2f}% > {FAIL_INDEX_NAN_PCT}%" if fail else None,
        "index_close_nan_share_pct": round(share_nan, 2),
        "index_close_min": index_min,
        "index_close_median": index_median,
        "index_close_max": index_max,
    }
    return not fail


def run_risk2_check_2b(panel_path: Path, master_path: Path, selected: list, report: dict) -> bool:
    """Strike proximity sanity and fallback 95k detection. Returns True if PASS."""
    if not panel_path.exists() or not master_path.exists():
        report["risk2_check2b"] = {"pass": False, "reason": "missing panel or master"}
        return False

    master = pd.read_parquet(master_path, columns=["open_time", "index_close"])
    first_valid = master[master["index_close"].notna()].head(1)
    if first_valid.empty:
        report["risk2_check2b"] = {"pass": False, "reason": "no non-NaN index_close in master"}
        return False

    t0 = int(first_valid["open_time"].iloc[0])
    atm_proxy = float(first_valid["index_close"].iloc[0])

    # Strikes from selected instruments (authoritative)
    strikes_by_name = {inv.get("instrument_name"): float(inv.get("strike", 0)) for inv in selected if inv.get("strike") is not None}
    if not strikes_by_name:
        report["risk2_check2b"] = {"pass": False, "reason": "no strikes in selected_instruments"}
        return False

    distances = []
    worst = []
    for name, strike in strikes_by_name.items():
        if atm_proxy <= 0:
            continue
        d = abs(strike - atm_proxy) / atm_proxy
        distances.append(d)
        worst.append((name, strike, d))

    worst.sort(key=lambda x: x[2], reverse=True)
    worst_10 = [{"instrument_name": n, "strike": s, "strike_distance_pct": round(d * 100, 2)} for n, s, d in worst[:10]]

    median_dist_pct = (pd.Series(distances).median() * 100) if distances else None
    p90_dist_pct = (pd.Series(distances).quantile(0.9) * 100) if distances else None

    fail = False
    reasons = []
    if median_dist_pct is not None and (median_dist_pct / 100.0) > FAIL_MEDIAN_STRIKE_DISTANCE_PCT:
        fail = True
        reasons.append(f"median strike_distance_pct > {FAIL_MEDIAN_STRIKE_DISTANCE_PCT*100}%")
    n_above_15 = sum(1 for d in distances if d > FAIL_STRIKE_DISTANCE_15_PCT)
    if n_above_15 > len(distances) * 0.2:
        fail = True
        reasons.append(f"many instruments exceed {FAIL_STRIKE_DISTANCE_15_PCT*100}%")

    # Fallback 95k detection: median index over window vs 95k, and median strike
    master_full = pd.read_parquet(master_path, columns=["index_close"])
    median_index_window = float(master_full["index_close"].median())
    median_strike = pd.Series(list(strikes_by_name.values())).median()
    fallback_warn = False
    if abs(median_index_window - FALLBACK_ATM) / FALLBACK_ATM < FALLBACK_TOLERANCE_PCT / 100.0 and abs(median_strike - FALLBACK_ATM) / FALLBACK_ATM < FALLBACK_TOLERANCE_PCT / 100.0:
        fallback_warn = True
    elif abs(median_index_window - FALLBACK_ATM) > 5000 and abs(median_strike - FALLBACK_ATM) < 2000:
        fallback_warn = True  # index far from 95k but strikes cluster 95k

    report["risk2_check2b"] = {
        "pass": not fail,
        "reasons": reasons if fail else None,
        "atm_proxy_at_first_bar": atm_proxy,
        "first_bar_open_time_utc": _ms_to_utc_iso(t0),
        "median_strike_distance_pct": round(median_dist_pct, 2) if median_dist_pct is not None else None,
        "p90_strike_distance_pct": round(p90_dist_pct, 2) if p90_dist_pct is not None else None,
        "worst_10_strike_distance": worst_10,
        "fallback_95k_likely_warning": fallback_warn,
    }
    return not fail


def build_selected_summary_df(selected: list) -> pd.DataFrame:
    rows = []
    for inv in selected:
        exp = inv.get("expiration_timestamp") or inv.get("expiry")
        cr = inv.get("creation_timestamp")
        rows.append({
            "instrument_name": inv.get("instrument_name"),
            "creation_timestamp": int(cr) if cr is not None and cr != 0 else None,
            "expiration_timestamp": int(exp) if exp is not None else None,
            "strike": float(inv.get("strike")) if inv.get("strike") is not None else None,
            "call_put": str(inv.get("option_type", inv.get("call_put", ""))).upper(),
            "creation_utc": _ms_to_utc_iso(cr) if cr else "",
            "expiry_utc": _ms_to_utc_iso(exp) if exp else "",
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"window_start_utc": START_UTC, "window_end_utc": END_UTC}

    # Load selected
    if not OPTIONS_SELECTED.exists():
        print("ERROR: selected_instruments.json not found")
        report["risk1_check1a"] = {"pass": False, "reason": "selected_instruments.json not found"}
        report["risk1"] = "FAIL"
        report["risk2"] = "SKIP"
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        return

    with open(OPTIONS_SELECTED, "r") as f:
        selected = json.load(f)
    if not isinstance(selected, list):
        selected = []

    # Risk 1
    r1a = run_risk1_check_1a(selected, report)
    top_table = []
    r1b = run_risk1_check_1b(OPTIONS_PANEL, report, top_table)
    risk1_pass = r1a and r1b
    report["risk1"] = "PASS" if risk1_pass else "FAIL"

    # Risk 2
    r2a = run_risk2_check_2a(MASTER_PANEL, report)
    r2b = run_risk2_check_2b(OPTIONS_PANEL, MASTER_PANEL, selected, report)
    risk2_pass = r2a and r2b
    report["risk2"] = "PASS" if risk2_pass else "FAIL"

    # Save report
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Save selected summary parquet
    summary_df = build_selected_summary_df(selected)
    summary_df.to_parquet(SUMMARY_PARQUET, index=False)

    # Save top 15 CSV
    if top_table:
        pd.DataFrame(top_table).to_csv(TOP_CSV, index=False)

    # Console output
    print("\n" + "=" * 60)
    print("OPTIONS VIABILITY AUDIT — PASS/FAIL SUMMARY")
    print("=" * 60)
    print(f"Risk 1 (Deribit instrument universe / candles in window): {report['risk1']}")
    print(f"  Check 1A (selected instruments expiry/creation): {'PASS' if r1a else 'FAIL'}")
    print(f"  Check 1B (actual candles in window):             {'PASS' if r1b else 'FAIL'}")
    print(f"Risk 2 (ATM selection realistic):                   {report['risk2']}")
    print(f"  Check 2A (index_close availability):              {'PASS' if r2a else 'FAIL'}")
    print(f"  Check 2B (strike proximity / fallback):          {'PASS' if r2b else 'FAIL'}")
    print("=" * 60)

    # Top 15 table
    print("\nTop 15 instruments by n_bars:")
    print("-" * 100)
    if top_table:
        tbl = pd.DataFrame(top_table)
        tbl["median_volume"] = tbl["median_volume"].apply(lambda x: f"{x:.4f}" if x is not None else "")
        tbl["zero_volume_share"] = tbl["zero_volume_share"].apply(lambda x: f"{x:.2%}" if x is not None else "")
        print(tbl.to_string(index=False))
    else:
        print("(none)")
    print("-" * 100)

    # Index min/median/max
    if "risk2_check2a" in report and report["risk2_check2a"].get("index_close_median") is not None:
        c2a = report["risk2_check2a"]
        print("\nindex_close over window: min =", c2a.get("index_close_min"), " median =", c2a.get("index_close_median"), " max =", c2a.get("index_close_max"))

    # Expiry histogram
    if "risk1_check1a" in report and report["risk1_check1a"].get("expiry_histogram_year_month"):
        print("\nExpiries by year-month (counts):")
        for k, v in report["risk1_check1a"]["expiry_histogram_year_month"].items():
            print(f"  {k}: {v}")

    # Report JSON pass/fail
    print("\noptions_viability_report.json: Risk1 =", report["risk1"], ", Risk2 =", report["risk2"])

    # 5 bullet takeaways
    print("\n--- 5 takeaways ---")
    takeaway = []
    takeaway.append(f"1. Selected instruments: {report.get('risk1_check1a', {}).get('count_selected', 0)}; Risk 1 overall: {report['risk1']}.")
    takeaway.append(f"2. Options panel: {report.get('risk1_check1b', {}).get('total_rows', 0)} rows, {report.get('risk1_check1b', {}).get('unique_instruments', 0)} unique instruments; Risk 1B: {'PASS' if r1b else 'FAIL'}.")
    takeaway.append(f"3. index_close: {report.get('risk2_check2a', {}).get('index_close_nan_share_pct')}% NaN; min/median/max as above; Risk 2A: {'PASS' if r2a else 'FAIL'}.")
    takeaway.append(f"4. Strike proximity: median distance {report.get('risk2_check2b', {}).get('median_strike_distance_pct')}%; Risk 2B: {'PASS' if r2b else 'FAIL'}.")
    takeaway.append(f"5. Fallback 95k warning: {report.get('risk2_check2b', {}).get('fallback_95k_likely_warning', False)}.")
    for t in takeaway:
        print(t)

    print("\nSaved:", REPORT_JSON, SUMMARY_PARQUET, TOP_CSV)


if __name__ == "__main__":
    main()
