"""
Main entry: discover -> index -> universe -> fetch -> merge -> QA report -> plots.
Run from repo root: python -m pipeline.options_v2_deribit_moneyness.run [--force-refresh]
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.options_v2_deribit_moneyness import config as cfg
from pipeline.options_v2_deribit_moneyness.merge_qa import run as merge_run
from pipeline.options_v2_deribit_moneyness.qa_report import run as qa_report_run
from pipeline.options_v2_deribit_moneyness.plots import run as plots_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Options v2 pipeline: moneyness-based Deribit BTC options")
    parser.add_argument("--force-refresh", action="store_true", help="Re-run from scratch (ignore caches)")
    args = parser.parse_args()

    logger.info("Window: %s to %s", cfg.START_UTC, cfg.END_UTC)
    logger.info("Moneyness band: %.2f - %.2f", cfg.MONEYNESS_MIN, cfg.MONEYNESS_MAX)
    logger.info("Max expiry horizon: %s days", getattr(cfg, "MAX_EXPIRY_HORIZON_DAYS", 180))

    df = merge_run(force_refresh=args.force_refresh)

    if df is not None and not df.empty:
        qa_report_run(df)
        plots_run(df)
    else:
        logger.warning("No data after merge; skipping QA and plots. Load existing parquet to run QA/plots only.")

    logger.info("Done. Data: %s  Reports: %s", cfg.DATA_DIR, cfg.REPORTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
