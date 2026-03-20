#!/usr/bin/env python3
"""
Main pipeline runner: backfill (Spot + Futures) then Step 1 QA + alignment.
Default window: Jan 1 – Feb 15, 2026 (UTC, 5m). Options pipeline archived.
Outputs go to ./outputs/window=YYYY-MM-DD_YYYY-MM-DD/

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --window_name jan_feb_2026 --overwrite
  python scripts/run_pipeline.py --no-backfill
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from pipeline.step1_qa_alignment import run as step1_run
from utils.time import get_start_end_ms, parse_window_start_end, utc_to_ms, window_output_suffix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline: backfill + Step 1 QA alignment")
    parser.add_argument("--start", type=str, default=None, help="Window start UTC, e.g. '2026-01-01 00:00:00'")
    parser.add_argument("--end", type=str, default=None, help="Window end UTC, e.g. '2026-02-15 23:55:00'")
    parser.add_argument("--window_name", type=str, default=None, help="Label e.g. jan_feb_2026")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Step 1 outputs")
    parser.add_argument("--no-backfill", action="store_true", help="Skip backfill (use existing data)")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: config.DATA_ROOT)")
    args = parser.parse_args()

    if args.start and args.end:
        start_dt, end_dt = parse_window_start_end(args.start, args.end)
        start_ms = utc_to_ms(start_dt)
        end_ms = utc_to_ms(end_dt)
        logger.info("Window from CLI: %s – %s", args.start, args.end)
    else:
        start_ms, end_ms = get_start_end_ms()
        logger.info("Window from config: %s – %s", config.START_UTC, config.END_UTC)

    suffix = window_output_suffix(start_ms, end_ms)
    output_dir = ROOT / "outputs" / suffix
    data_root = args.data_root or Path(config.DATA_ROOT)

    # Options pipeline archived; no options data
    options_liquid_path = None

    if not args.no_backfill:
        logger.info("Running backfill (Spot + Futures; OI/continuous/options skipped by default)...")
        cmd = [sys.executable, str(SCRIPT_DIR / "backfill_binance_oct_dec_2025.py")]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    else:
        logger.info("Skipping backfill (--no-backfill).")

    logger.info("Step 1: QA + alignment -> %s", output_dir)
    step1_run(
        data_root=data_root,
        output_dir=output_dir,
        start_ms=int(start_ms),
        end_ms=int(end_ms),
        overwrite=args.overwrite,
        options_liquid_path=options_liquid_path,
    )
    logger.info("Pipeline done. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
