#!/usr/bin/env python3
"""
Run Step 1 (Data QA + alignment) of the thesis pipeline.
Builds master 5m panel, QA report, options coverage, and sanity plots.
Outputs go to ./outputs/window=YYYY-MM-DD_YYYY-MM-DD/ by default.

Usage:
  python scripts/run_step1_qa_alignment.py [--overwrite]
  python scripts/run_step1_qa_alignment.py --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --options-liquid-path outputs/window=2026-01-01_2026-02-15/options_panel_long_5m_liquid.parquet
"""
import argparse
import logging
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
    parser = argparse.ArgumentParser(description="Step 1: Data QA + alignment")
    parser.add_argument("--start", type=str, default=None, help="Window start UTC, e.g. '2026-01-01 00:00:00'")
    parser.add_argument("--end", type=str, default=None, help="Window end UTC, e.g. '2026-02-15 23:55:00'")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root (default: config.DATA_ROOT)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: ./outputs/window=START_END)")
    parser.add_argument("--options-liquid-path", type=Path, default=None, help="Use pre-built liquid options parquet (e.g. jan_feb_2026)")
    args = parser.parse_args()

    if args.start and args.end:
        start_dt, end_dt = parse_window_start_end(args.start, args.end)
        start_ms = int(utc_to_ms(start_dt))
        end_ms = int(utc_to_ms(end_dt))
        logger.info("Window from CLI: %s – %s", args.start, args.end)
    else:
        start_ms, end_ms = get_start_end_ms()
        start_ms, end_ms = int(start_ms), int(end_ms)
        logger.info("Window from config: %s – %s", config.START_UTC, config.END_UTC)

    data_root = args.data_root or Path(config.DATA_ROOT)
    output_dir = args.output_dir or (ROOT / "outputs" / window_output_suffix(start_ms, end_ms))
    options_liquid_path = None
    if args.options_liquid_path:
        p = args.options_liquid_path if args.options_liquid_path.is_absolute() else ROOT / args.options_liquid_path
        if p.exists():
            options_liquid_path = p
            logger.info("Using options liquid file: %s", options_liquid_path)
        else:
            logger.warning("Options liquid path not found: %s", p)

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", output_dir)
    step1_run(
        data_root=data_root,
        output_dir=output_dir,
        start_ms=start_ms,
        end_ms=end_ms,
        overwrite=args.overwrite,
        options_liquid_path=options_liquid_path,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
