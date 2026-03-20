#!/usr/bin/env python3
"""
Verify that downloaded Deribit options were trading (existed) during Oct–Dec 2025.
Checks: (1) instrument creation/expiry overlap the window; (2) kline open_time falls in window.
Run: python scripts/verify_options_oct_dec_2025.py
"""
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from utils.time import get_start_end_ms, ms_to_utc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_selected() -> List[Dict[str, Any]]:
    path = config.OPTIONS_SELECTED_INSTRUMENTS_PATH
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def check_instrument_window(
    inv: Dict[str, Any],
    start_ms: int,
    end_ms: int,
) -> Tuple[bool, str]:
    """
    Return (ok, message). ok if creation <= end and expiration >= start (was live in window).
    """
    name = inv.get("instrument_name", "?")
    creation = inv.get("creation_timestamp")
    expiration = inv.get("expiration_timestamp")
    if creation is None:
        creation = 0
    if expiration is None:
        return False, f"{name}: missing expiration_timestamp"
    creation = int(creation)
    expiration = int(expiration)
    # Was trading during [start_ms, end_ms]?
    ok = creation <= end_ms and expiration >= start_ms
    creation_d = ms_to_utc(creation).strftime("%Y-%m-%d") if creation else "n/a"
    expiration_d = ms_to_utc(expiration).strftime("%Y-%m-%d")
    msg = f"created {creation_d} expired {expiration_d} -> {'in window' if ok else 'NOT in window'}"
    return ok, msg


def get_kline_time_range(instrument_name: str) -> Tuple[int, int, int]:
    """
    Return (min_open_time_ms, max_open_time_ms, total_rows) for that instrument's parquet files.
    """
    safe = instrument_name.replace(":", "_")
    base = os.path.join(config.OPTIONS_KLINES_PATH, f"instrument={safe}")
    if not os.path.isdir(base):
        return 0, 0, 0
    total = 0
    min_ts = 0
    max_ts = 0
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.endswith(".parquet"):
                continue
            try:
                df = pd.read_parquet(os.path.join(root, f))
                if "open_time" not in df.columns or df.empty:
                    continue
                total += len(df)
                mn = int(df["open_time"].min())
                mx = int(df["open_time"].max())
                if min_ts == 0 or mn < min_ts:
                    min_ts = mn
                if mx > max_ts:
                    max_ts = mx
            except Exception:
                pass
    return min_ts, max_ts, total


def main() -> None:
    start_ms, end_ms = get_start_end_ms()
    start_d = ms_to_utc(start_ms).strftime("%Y-%m-%d %H:%M")
    end_d = ms_to_utc(end_ms).strftime("%Y-%m-%d %H:%M")
    logger.info("Target window: %s to %s UTC", start_d, end_d)

    selected = load_selected()
    if not selected:
        logger.warning("No selected instruments found at %s. Run the backfill first.", config.OPTIONS_SELECTED_INSTRUMENTS_PATH)
        return

    logger.info("Loaded %s selected instruments", len(selected))

    # 1) Metadata check: creation/expiry overlap window
    ok_count = 0
    fail_instruments = []
    for inv in selected:
        ok, msg = check_instrument_window(inv, start_ms, end_ms)
        if ok:
            ok_count += 1
        else:
            fail_instruments.append((inv.get("instrument_name"), msg))
    logger.info("--- Instrument window check (trading life overlaps Oct–Dec 2025) ---")
    logger.info("Passed: %s / %s", ok_count, len(selected))
    if fail_instruments:
        for name, msg in fail_instruments[:10]:
            logger.warning("  %s: %s", name, msg)
        if len(fail_instruments) > 10:
            logger.warning("  ... and %s more", len(fail_instruments) - 10)

    # 2) Kline data check: open_time overlaps or falls in window
    logger.info("--- Kline data time range (open_time in Oct–Dec 2025) ---")
    in_window_count = 0
    overlaps_window_count = 0
    out_of_window = []
    for inv in selected:
        name = inv.get("instrument_name", "?")
        min_ts, max_ts, total = get_kline_time_range(name)
        if total == 0:
            continue
        all_in_window = min_ts >= start_ms and max_ts <= end_ms
        has_overlap = not (max_ts < start_ms or min_ts > end_ms)
        if all_in_window:
            in_window_count += 1
        if has_overlap:
            overlaps_window_count += 1
        if not all_in_window:
            min_d = ms_to_utc(min_ts).strftime("%Y-%m-%d") if min_ts else "n/a"
            max_d = ms_to_utc(max_ts).strftime("%Y-%m-%d") if max_ts else "n/a"
            out_of_window.append((name, min_d, max_d, total))
    instruments_with_data = sum(1 for inv in selected if get_kline_time_range(inv.get("instrument_name", ""))[2] > 0)
    logger.info("Instruments with kline data: %s", instruments_with_data)
    logger.info("Instruments with at least some open_time in Oct–Dec 2025: %s", overlaps_window_count)
    logger.info("Instruments with all open_time strictly in [Oct 1 – Dec 31 2025]: %s", in_window_count)
    if out_of_window:
        logger.info("Instruments with some data outside window (expected if contract traded before/after):")
        for name, min_d, max_d, total in out_of_window[:15]:
            logger.info("  %s: open_time range %s – %s (%s rows)", name, min_d, max_d, total)
        if len(out_of_window) > 15:
            logger.info("  ... and %s more", len(out_of_window) - 15)

    # 3) Summary
    logger.info("--- Summary ---")
    if ok_count == len(selected) and overlaps_window_count == instruments_with_data:
        logger.info("All selected instruments were trading in Oct–Dec 2025; all kline data overlaps the window.")
    elif ok_count == len(selected):
        logger.info("All instruments pass metadata check (trading life overlaps window). Some kline files extend before/after Oct–Dec 2025 (normal if contract existed earlier/later).")
    else:
        logger.warning("Some instruments did not have trading life overlapping Oct–Dec 2025; review list above.")


if __name__ == "__main__":
    main()
