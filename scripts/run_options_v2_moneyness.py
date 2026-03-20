#!/usr/bin/env python3
"""
Run the options v2 (moneyness-based) pipeline from repo root.
  python scripts/run_options_v2_moneyness.py [--force-refresh]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.options_v2_deribit_moneyness.run import main

if __name__ == "__main__":
    sys.exit(main())
