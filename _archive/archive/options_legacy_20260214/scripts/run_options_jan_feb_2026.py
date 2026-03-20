#!/usr/bin/env python3
"""
Jan–Feb 2026 options backfill experiment (additive, separate from Oct–Dec 2025).
Downloads Deribit BTC options for 2026-01-01 to 2026-02-15, builds long panel, runs viability audit,
and compares liquidity vs Oct–Dec 2025.
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from deribit_client import RateLimiter, get_index_price, list_instruments
from utils.time import get_start_end_ms, window_output_suffix
from downloaders.options import (
    _expiry_bucket,
    build_selected_instruments,
    candles_result_to_rows,
    fetch_candles_chunk,
)
from pipeline.step1_qa_alignment import (
    _ms_to_utc_iso_column,
    options_coverage_summary,
)
from utils.storage import add_utc_columns, ensure_dir, is_chunk_done, save_done_chunk
from utils.time import utc_to_ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test window (override config)
START_DT_TEST = datetime(2026, 1, 1, 0, 0, 0)
END_DT_TEST = datetime(2026, 2, 15, 23, 55, 0)
START_MS_TEST = utc_to_ms(START_DT_TEST)
END_MS_TEST = utc_to_ms(END_DT_TEST)

# Separate output directories
DATA_ROOT = Path(config.DATA_ROOT)
OPTIONS_DATA_DIR = DATA_ROOT / "options_jan_feb_2026"
OPTIONS_KLINES_DIR = OPTIONS_DATA_DIR / "klines"
OPTIONS_EXCHANGE_INFO_DIR = OPTIONS_DATA_DIR
SELECTED_INSTRUMENTS_PATH = OPTIONS_DATA_DIR / "selected_instruments.json"
# Use same window output folder as main pipeline (single output tree)
_start_ms, _end_ms = get_start_end_ms()
OUTPUT_DIR = ROOT / "outputs" / window_output_suffix(_start_ms, _end_ms)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse config constants
ATM_PERCENT_RANGE = config.ATM_PERCENT_RANGE
MAX_STRIKES_PER_SIDE = config.MAX_STRIKES_PER_SIDE
OPTIONS_CANDLE_CHUNK_DAYS = config.OPTIONS_CANDLE_CHUNK_DAYS
OPTIONS_RESOLUTION_MINUTES = config.OPTIONS_RESOLUTION_MINUTES
BAR_MS = 5 * 60 * 1000


def get_atm_reference_test() -> float:
    """ATM proxy: try Deribit index first, then fallback."""
    limiter = RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    try:
        idx = get_index_price(config.DERIBIT_INDEX_NAME, rate_limiter=limiter)
        if idx and "index_price" in idx:
            atm = float(idx["index_price"])
            logger.info("ATM reference from Deribit index: %s", atm)
            return atm
    except Exception as e:
        logger.warning("Deribit index price failed: %s", e)
    fallback = 95000.0
    logger.warning("Using fallback ATM: %s", fallback)
    return fallback


def download_instruments_test(rate_limiter: RateLimiter) -> List[Dict[str, Any]]:
    """Fetch BTC option instruments from Deribit (active + expired), save to test dir."""
    active = list_instruments(currency="BTC", kind="option", expired=False, rate_limiter=rate_limiter)
    expired = list_instruments(currency="BTC", kind="option", expired=True, rate_limiter=rate_limiter)
    by_name = {inv.get("instrument_name"): inv for inv in (active + expired) if inv.get("instrument_name")}
    instruments = list(by_name.values())
    ensure_dir(str(OPTIONS_EXCHANGE_INFO_DIR))
    raw_path = OPTIONS_EXCHANGE_INFO_DIR / "deribit_instruments.json"
    with open(raw_path, "w") as f:
        json.dump(instruments, f, indent=2)
    logger.info("Saved %s Deribit option instruments to %s", len(instruments), raw_path)
    return instruments


def load_selected_instruments_test() -> List[Dict[str, Any]]:
    """Load previously saved selected instruments if present."""
    if not SELECTED_INSTRUMENTS_PATH.exists():
        return []
    with open(SELECTED_INSTRUMENTS_PATH, "r") as f:
        return json.load(f)


def download_options_klines_for_instrument_test(
    instrument_info: Dict[str, Any],
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> int:
    """Download 5m candles for one option; write to test dir. Returns row count."""
    instrument_name = instrument_info.get("instrument_name")
    if not instrument_name:
        return 0
    manifest_key = f"options_jan_feb_2026_{instrument_name}"
    if is_chunk_done(manifest_key, "full"):
        return 0
    chunk_ms = OPTIONS_CANDLE_CHUNK_DAYS * 24 * 60 * 60 * 1000
    all_rows = []
    current = start_ms
    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        result = fetch_candles_chunk(instrument_name, current, chunk_end, limiter)
        rows = candles_result_to_rows(result, instrument_name)
        all_rows.extend(rows)
        current = chunk_end

    if not all_rows:
        save_done_chunk(manifest_key, "full")
        return 0

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df = df[df[col].notna() & (df[col] > 0)]
    if df.empty:
        save_done_chunk(manifest_key, "full")
        return 0

    df["instrument_name"] = instrument_name
    df["expiry"] = int(instrument_info.get("expiration_timestamp", 0))
    df["strike"] = float(instrument_info.get("strike", 0))
    df["call_put"] = str(instrument_info.get("option_type", "")).upper()
    df["source"] = "deribit"
    df = add_utc_columns(df)

    safe_name = instrument_name.replace(":", "_")
    base_path = OPTIONS_KLINES_DIR / f"instrument={safe_name}"
    df["_year"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.year
    df["_month"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.month
    total = 0
    for (y, m), group in df.groupby(["_year", "_month"]):
        out_dir = base_path / f"year={y}" / f"month={m:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "klines_5m.parquet"
        group = group.drop(columns=["_year", "_month"])
        group.to_parquet(out_file, index=False)
        total += len(group)
    save_done_chunk(manifest_key, "full")
    logger.info("Wrote %s options klines %s rows to %s", instrument_name, total, base_path)
    return total


def load_options_panel_test() -> pd.DataFrame:
    """Load all option parquet files from test dir and concatenate."""
    dfs = []
    if not OPTIONS_KLINES_DIR.exists():
        return pd.DataFrame()
    for instrument_dir in sorted(OPTIONS_KLINES_DIR.iterdir()):
        if not instrument_dir.is_dir():
            continue
        for yd in instrument_dir.glob("year=2026"):
            if not yd.is_dir():
                continue
            for m in (1, 2):
                md = yd / f"month={m:02d}"
                if not md.exists():
                    continue
                for f in md.glob("*.parquet"):
                    try:
                        dfs.append(pd.read_parquet(f))
                    except Exception as e:
                        logger.debug("Skip %s: %s", f, e)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df = df[df["open_time"].notna()]
    df = df[(df["open_time"] >= START_MS_TEST) & (df["open_time"] <= END_MS_TEST)]
    if "instrument_name" in df.columns and "open_time" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["instrument_name", "open_time"], keep="last")
        logger.info("Loaded options: %s rows (after dedup: %s)", n_before, len(df))
    return df


def build_options_panel_test(options: pd.DataFrame, atm_proxy: float) -> pd.DataFrame:
    """Build options long panel with derived columns (no master panel needed)."""
    if options.empty:
        return options
    opt = options.copy()
    # Add expiry_bucket
    if "expiry" in opt.columns:
        opt["expiry_bucket"] = opt["expiry"].astype("int64").map(_expiry_bucket)
    else:
        opt["expiry_bucket"] = ""
    # Moneyness proxy using constant ATM (for liquidity check, precision less critical)
    opt["moneyness_proxy"] = opt["strike"] / atm_proxy
    # Human-readable date columns
    opt["open_time_utc"] = _ms_to_utc_iso_column(opt["open_time"])
    if "expiry" in opt.columns:
        opt["expiry_utc"] = _ms_to_utc_iso_column(opt["expiry"])
    return opt


def run_viability_audit_test(options: pd.DataFrame) -> Dict[str, Any]:
    """Run simplified viability audit (no master panel needed)."""
    if options.empty:
        return {"pass": False, "reason": "options empty"}
    total_rows = len(options)
    unique_instruments = options["instrument_name"].nunique()
    earliest = options["open_time"].min()
    latest = options["open_time"].max()

    grp = options.groupby("instrument_name").agg(
        n_bars=("open_time", "nunique"),
        median_volume=("volume", "median"),
    )
    grp["zero_volume_share"] = options.groupby("instrument_name")["volume"].apply(lambda s: (s.fillna(0) == 0).mean())
    grp = grp.reset_index()
    n_with_500_bars = (grp["n_bars"] > 500).sum()
    median_zero_vol = grp["zero_volume_share"].median()

    fail = False
    reasons = []
    if total_rows == 0:
        fail = True
        reasons.append("total rows == 0")
    if n_with_500_bars < 10:
        fail = True
        reasons.append("<10 instruments with n_bars > 500")
    if median_zero_vol >= 0.99:
        fail = True
        reasons.append("most instruments have ~100% zero volume")

    return {
        "pass": not fail,
        "reasons": reasons if fail else None,
        "total_rows": int(total_rows),
        "unique_instruments": int(unique_instruments),
        "earliest_open_time_utc": pd.to_datetime(earliest, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "latest_open_time_utc": pd.to_datetime(latest, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "n_instruments_with_500_plus_bars": int(n_with_500_bars),
        "median_zero_volume_share": float(median_zero_vol),
    }


def compare_with_oct_dec() -> Dict[str, Any]:
    """Load Oct–Dec 2025 coverage summary and compare stats."""
    oct_dec_coverage = ROOT / "outputs" / "options_coverage_summary.parquet"
    if not oct_dec_coverage.exists():
        return {"available": False}
    df = pd.read_parquet(oct_dec_coverage)
    if df.empty:
        return {"available": False}
    return {
        "available": True,
        "median_zero_volume_share": float(df["zero_volume_share"].median()),
        "median_n_bars": float(df["n_bars"].median()),
        "n_instruments": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Jan–Feb 2026 options backfill experiment")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing selected_instruments.json")
    args = parser.parse_args()

    logger.info("Jan–Feb 2026 options backfill experiment")
    logger.info("Window: %s – %s", START_DT_TEST.isoformat() + "Z", END_DT_TEST.isoformat() + "Z")
    logger.info("Data dir: %s", OPTIONS_DATA_DIR)
    logger.info("Output dir: %s", OUTPUT_DIR)

    limiter = RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)

    # 1. Download instruments
    logger.info("--- Step 1: Fetch Deribit instruments ---")
    instruments = download_instruments_test(limiter)
    logger.info("Fetched %s instruments", len(instruments))

    # 2. Build selected list (or load if exists)
    logger.info("--- Step 2: Select near-ATM instruments ---")
    selected = load_selected_instruments_test()
    if not selected or args.overwrite:
        atm_proxy = get_atm_reference_test()
        selected = build_selected_instruments(instruments, START_MS_TEST, END_MS_TEST, atm_price=atm_proxy)
        # Save to test dir
        with open(SELECTED_INSTRUMENTS_PATH, "w") as f:
            json.dump(selected, f, indent=2)
        logger.info("Selected %s instruments (ATM proxy: %s)", len(selected), atm_proxy)
    else:
        logger.info("Loaded %s selected instruments from %s", len(selected), SELECTED_INSTRUMENTS_PATH)
        # Get ATM from first instrument or Deribit
        atm_proxy = get_atm_reference_test()

    # Expiry histogram
    hist = defaultdict(int)
    for inv in selected:
        exp = inv.get("expiration_timestamp")
        if exp:
            try:
                dt = datetime.fromtimestamp(int(exp) / 1000.0, tz=timezone.utc)
                key = f"{dt.year}-{dt.month:02d}"
                hist[key] += 1
            except (ValueError, TypeError):
                pass
    logger.info("Expiry distribution:")
    for k, v in sorted(hist.items()):
        logger.info("  %s: %s instruments", k, v)

    # 3. Download klines
    logger.info("--- Step 3: Download 5m candles ---")
    total_rows = 0
    for inv in selected:
        total_rows += download_options_klines_for_instrument_test(inv, START_MS_TEST, END_MS_TEST, limiter)
    logger.info("Total candle rows written: %s", total_rows)

    # 4. Build options panel
    logger.info("--- Step 4: Build options long panel ---")
    options_raw = load_options_panel_test()
    if options_raw.empty:
        logger.error("No options data loaded")
        return
    options_panel = build_options_panel_test(options_raw, atm_proxy)
    options_panel.to_parquet(OUTPUT_DIR / "options_panel_long_5m.parquet", index=False)
    logger.info("Saved options panel: %s rows", len(options_panel))

    # 5. Coverage summary
    logger.info("--- Step 5: Compute coverage summary ---")
    # Build minimal master grid for coverage (just time index)
    master_times = pd.DataFrame({"open_time": range(START_MS_TEST, END_MS_TEST + BAR_MS, BAR_MS)})
    coverage = options_coverage_summary(options_panel, master_times)
    if not coverage.empty:
        coverage.to_parquet(OUTPUT_DIR / "options_coverage_summary.parquet", index=False)
        logger.info("Coverage summary: %s instruments", len(coverage))

    # 6. Viability audit
    logger.info("--- Step 6: Run viability audit ---")
    audit = run_viability_audit_test(options_panel)
    audit["window_start_utc"] = START_DT_TEST.isoformat() + "Z"
    audit["window_end_utc"] = END_DT_TEST.isoformat() + "Z"
    with open(OUTPUT_DIR / "options_viability_report.json", "w") as f:
        json.dump(audit, f, indent=2)

    # 7. Top instruments by volume and nonzero bars
    logger.info("--- Step 7: Top instruments ---")
    grp = options_panel.groupby("instrument_name").agg(
        total_volume=("volume", "sum"),
        n_nonzero_bars=("volume", lambda s: (s.fillna(0) > 0).sum()),
        n_bars=("open_time", "nunique"),
    ).reset_index()
    top_volume = grp.nlargest(15, "total_volume")
    top_nonzero = grp.nlargest(15, "n_nonzero_bars")
    top_volume.to_csv(OUTPUT_DIR / "options_viability_top_by_volume.csv", index=False)
    top_nonzero.to_csv(OUTPUT_DIR / "options_viability_top_by_nonzero_bars.csv", index=False)

    # 8. Compare with Oct–Dec
    logger.info("--- Step 8: Compare with Oct–Dec 2025 ---")
    oct_dec = compare_with_oct_dec()

    # Print summary
    print("\n" + "=" * 70)
    print("JAN–FEB 2026 OPTIONS BACKFILL — SUMMARY")
    print("=" * 70)
    print(f"Selected instruments: {len(selected)}")
    print(f"Total candle rows written: {total_rows}")
    print(f"Options panel rows: {len(options_panel)}")
    print(f"Unique instruments in panel: {options_panel['instrument_name'].nunique()}")
    print("\nExpiry distribution:")
    for k, v in sorted(hist.items()):
        print(f"  {k}: {v}")
    print(f"\nMedian zero_volume_share: {audit['median_zero_volume_share']:.2%}")
    print(f"Viability audit: {'PASS' if audit['pass'] else 'FAIL'}")
    if audit.get("reasons"):
        print(f"  Reasons: {', '.join(audit['reasons'])}")

    print("\nTop 15 by total volume:")
    print(top_volume[["instrument_name", "total_volume", "n_nonzero_bars", "n_bars"]].to_string(index=False))
    print("\nTop 15 by nonzero-volume bars:")
    print(top_nonzero[["instrument_name", "n_nonzero_bars", "total_volume", "n_bars"]].to_string(index=False))

    if oct_dec.get("available"):
        print("\nComparison vs Oct–Dec 2025:")
        print(f"  Oct–Dec median zero_volume_share: {oct_dec['median_zero_volume_share']:.2%}")
        print(f"  Jan–Feb median zero_volume_share: {audit['median_zero_volume_share']:.2%}")
        print(f"  Oct–Dec instruments: {oct_dec['n_instruments']}")
        print(f"  Jan–Feb instruments: {audit['unique_instruments']}")
    else:
        print("\n(Oct–Dec 2025 comparison data not available)")

    print("\n" + "=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
