#!/usr/bin/env python3
"""
Deribit options v2: quote-based pipeline (best bid/ask -> 5m aligned -> IV from mid).
Run from repo root. Supports:
  --live       Fetch current snapshot for selected instruments and append to raw store.
  --ingest PATH  Load raw quotes from external parquet (e.g. Tardis historical) and align.
  --start / --end  Window (UTC) for alignment; default from config.
Deribit does not provide historical order book; use --live on a schedule to build history,
or --ingest with a pre-downloaded quote file.
"""
import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from utils.time import get_start_end_ms, parse_window_start_end, utc_to_ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pipeline modules
from data_pipeline_v2_deribit_quotes.align_quotes import align_raw_to_5m, load_raw_quotes
from data_pipeline_v2_deribit_quotes.compute_iv import compute_mid_and_iv
from data_pipeline_v2_deribit_quotes.coverage_report import build_coverage_report, write_coverage_report
from data_pipeline_v2_deribit_quotes.fetch_quotes import (
    append_raw_snapshots,
    fetch_snapshots_for_instruments,
    load_instruments_deribit,
)
from data_pipeline_v2_deribit_quotes.qa_filters import apply_quote_filters


def main() -> int:
    parser = argparse.ArgumentParser(description="Deribit options v2: quote-based 5m panel + IV")
    parser.add_argument("--live", action="store_true", help="Fetch current snapshot and append to raw store")
    parser.add_argument("--ingest", type=str, default=None, help="Path to external raw quotes parquet (or dir)")
    parser.add_argument("--start", type=str, default=None, help="Window start UTC e.g. 2026-01-01 00:00:00")
    parser.add_argument("--end", type=str, default=None, help="Window end UTC e.g. 2026-02-15 23:55:00")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite aligned/IV outputs")
    parser.add_argument("--no-spread-filter", action="store_true", help="Disable max spread filter")
    args = parser.parse_args()

    if not args.live and not args.ingest:
        parser.error("Specify --live and/or --ingest to run.")

    # Window
    if args.start and args.end:
        start_dt, end_dt = parse_window_start_end(args.start, args.end)
        start_ms = utc_to_ms(start_dt)
        end_ms = utc_to_ms(end_dt)
    else:
        start_ms, end_ms = get_start_end_ms()
    logger.info("Window: %s – %s ms", start_ms, end_ms)

    root = Path(config.DERIBIT_QUOTES_V2_ROOT)
    raw_dir = root / getattr(config, "DERIBIT_QUOTES_RAW_SUBDIR", "raw")
    aligned_dir = root / getattr(config, "DERIBIT_QUOTES_5M_SUBDIR", "aligned_5m")
    raw_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # Load raw quotes
    if args.live:
        instruments = load_instruments_deribit()
        if not instruments:
            logger.error("No instruments loaded (check OPTIONS_SELECTED_INSTRUMENTS_PATH or Deribit API)")
            return 1
        snapshot = fetch_snapshots_for_instruments(instruments)
        if snapshot.empty:
            logger.error("No quote snapshots returned")
            return 1
        append_raw_snapshots(snapshot, raw_dir)
        raw_df = load_raw_quotes(raw_dir)
    elif args.ingest:
        raw_df = load_raw_quotes(Path(args.ingest))
    else:
        raw_df = load_raw_quotes(raw_dir)

    if raw_df.empty:
        logger.error("No raw quotes to align (run --live first or provide --ingest)")
        return 1

    # Ensure required columns (map common names)
    if "open_time" not in raw_df.columns and "timestamp" in raw_df.columns:
        raw_df = raw_df.rename(columns={"timestamp": "open_time"})
    if "open_time" not in raw_df.columns and "timestamp_utc" in raw_df.columns:
        raw_df["open_time"] = pd.to_datetime(raw_df["timestamp_utc"], utc=True).astype("int64") // 10**6
    for c in ["best_bid_price", "best_ask_price", "instrument_name"]:
        if c not in raw_df.columns:
            logger.error("Raw quotes missing column: %s", c)
            return 1
    if "strike" not in raw_df.columns:
        raw_df["strike"] = 0
    if "expiry" not in raw_df.columns and "expiration_timestamp" in raw_df.columns:
        raw_df["expiry"] = raw_df["expiration_timestamp"]
    if "expiry" not in raw_df.columns:
        raw_df["expiry"] = 0
    if "call_put" not in raw_df.columns and "option_type" in raw_df.columns:
        raw_df["call_put"] = raw_df["option_type"].astype(str).str.upper().str[:1]
    if "call_put" not in raw_df.columns:
        raw_df["call_put"] = ""
    if "index_price" not in raw_df.columns:
        raw_df["index_price"] = float("nan")

    # Sanity filters
    max_spread = None if args.no_spread_filter else getattr(config, "DERIBIT_QUOTE_MAX_SPREAD_PCT", None)
    raw_df = apply_quote_filters(raw_df, max_spread_pct=max_spread)

    # Align to 5m grid
    aligned = align_raw_to_5m(raw_df, start_ms=start_ms, end_ms=end_ms)
    if aligned.empty:
        logger.error("Aligned panel is empty")
        return 1

    # Mid + IV
    aligned = compute_mid_and_iv(aligned)

    # Outputs
    quotes_path = aligned_dir / "deribit_quotes_5m.parquet"
    iv_path = aligned_dir / "deribit_iv_5m.parquet"
    if quotes_path.exists() and not args.overwrite:
        logger.info("Skip write (exists): %s", quotes_path)
    else:
        quote_cols = [c for c in ["open_time", "instrument_name", "strike", "expiry", "call_put",
                                  "best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size",
                                  "index_price", "mid_price"] if c in aligned.columns]
        aligned[quote_cols].to_parquet(quotes_path, index=False)
        logger.info("Wrote %s", quotes_path)
    if iv_path.exists() and not args.overwrite:
        logger.info("Skip write (exists): %s", iv_path)
    else:
        iv_cols = [c for c in ["open_time", "instrument_name", "strike", "expiry", "call_put",
                               "mid_price", "iv_mid", "index_price"] if c in aligned.columns]
        aligned[iv_cols].to_parquet(iv_path, index=False)
        logger.info("Wrote %s", iv_path)

    # Coverage report
    report = build_coverage_report(aligned, start_ms, end_ms)
    write_coverage_report(report, aligned_dir)
    logger.info("Done. Quotes: %s  IV: %s", quotes_path, iv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
