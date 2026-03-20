#!/usr/bin/env python3
"""
Data backfill: window from config (default Jan–Feb 2026), 5m, BTC only.
Spot (BTCUSDT), USD-M Futures (perp + mark/index + funding; OI and continuous off by default), Options (optional).
Output: ./data (partitioned Parquet), restartable with checkpointing.
Run: python scripts/backfill_binance_oct_dec_2025.py [--with-oi] [--with-continuous] [--with-options]
"""
import argparse
import logging
import os
import sys

# Project root = parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from binance_client import get as api_get
from downloaders import spot, futures
try:
    from downloaders import options
except ImportError:
    options = None  # options downloader removed; use pipeline.options_v2_deribit_moneyness for Deribit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sanity_check_server_time() -> None:
    """Optional: log Binance server time for sanity check (UTC)."""
    try:
        data = api_get(config.SPOT_BASE, "api/v3/time")
        server_ms = data.get("serverTime")
        if server_ms:
            from utils.time import ms_to_utc
            logger.info("Binance server time (UTC): %s", ms_to_utc(server_ms))
    except Exception as e:
        logger.warning("Server time check skipped: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Spot + Futures (and optionally OI, continuous, Options)")
    parser.add_argument("--with-oi", action="store_true", help="Download open interest (Binance OI limited to last 30d)")
    parser.add_argument("--with-continuous", action="store_true", help="Download continuous quarterly klines")
    parser.add_argument("--with-options", action="store_true", help="Download Deribit options (main run uses liquid file instead)")
    args = parser.parse_args()

    logger.info("Backfill %s -> %s (5m, UTC)", config.START_UTC, config.END_UTC)
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    os.makedirs(config.MANIFEST_DIR, exist_ok=True)
    sanity_check_server_time()

    api_key = os.environ.get("BINANCE_API_KEY") or None
    summary = {"spot_klines": 0, "futures": {}, "options_exchange_info": False, "options_klines": 0, "errors": []}

    # --- Spot ---
    try:
        summary["spot_klines"] = spot.download_spot_klines(api_key)
        logger.info("Spot klines: %s rows", summary["spot_klines"])
    except Exception as e:
        logger.exception("Spot download failed")
        summary["errors"].append(("spot", str(e)))

    # --- Futures (OI and continuous off by default) ---
    try:
        summary["futures"] = futures.download_all_futures(
            api_key,
            skip_oi=not args.with_oi,
            skip_continuous=not args.with_continuous,
        )
        for k, v in summary["futures"].items():
            logger.info("Futures %s: %s rows", k, v)
    except Exception as e:
        logger.exception("Futures download failed")
        summary["errors"].append(("futures", str(e)))

    # --- Options (Deribit; skip by default for main run) ---
    if args.with_options:
        if options is None:
            raise RuntimeError(
                "Options downloader not available. Use pipeline.options_v2_deribit_moneyness.history_trades_probe "
                "for Deribit trade-level data, or restore downloaders/options.py from _archive if needed."
            )
        try:
            exchange_info, options_rows = options.download_all_options(api_key=api_key)
            summary["options_exchange_info"] = bool(exchange_info.get("optionSymbols") or exchange_info.get("deribit_instruments"))
            summary["options_klines"] = options_rows
            logger.info("Options (Deribit) instruments: %s", len(exchange_info.get("optionSymbols") or exchange_info.get("deribit_instruments") or []))
            logger.info("Options klines: %s rows", options_rows)
            try:
                stats = options.get_options_summary_stats()
                logger.info("Options expiries per bucket: %s", stats.get("expiries_per_bucket", {}))
                logger.info("Options instruments selected: %s", stats.get("instruments_total", 0))
            except Exception:
                pass
        except Exception as e:
            logger.exception("Options download failed")
            summary["errors"].append(("options", str(e)))
    else:
        logger.info("Options download skipped (use --with-options to download Deribit options)")

    # --- Summary ---
    logger.info("--- Summary ---")
    logger.info("Spot klines: %s", summary["spot_klines"])
    logger.info("Futures: %s", summary["futures"])
    logger.info("Options: %s", "yes" if summary["options_klines"] else "no")
    if summary["errors"]:
        logger.warning("Errors: %s", summary["errors"])
    logger.info("Data root: %s", config.DATA_ROOT)
    logger.info("Done.")


if __name__ == "__main__":
    main()
