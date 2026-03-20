#!/usr/bin/env python3
"""
List spot klines partitions and row counts. Use to verify which date range
you have on disk (e.g. after backfill) and why spot might be missing in Step 1.

Run: python scripts/inspect_spot_partitions.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import pandas as pd

def main():
    base = Path(config.SPOT_PATH)
    if not base.exists():
        print(f"Spot path does not exist: {base}")
        return
    print("Spot klines on disk (partition -> rows, open_time range)")
    print("-" * 60)
    total = 0
    by_partition = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            files = list(month_dir.glob("*.parquet"))
            if not files:
                continue
            dfs = [pd.read_parquet(f) for f in files]
            combined = pd.concat(dfs, ignore_index=True)
            if "open_time" not in combined.columns:
                continue
            n = len(combined)
            total += n
            t_min = combined["open_time"].min()
            t_max = combined["open_time"].max()
            from utils.time import ms_to_utc_iso
            by_partition.append((year_dir.name, month_dir.name, n, ms_to_utc_iso(int(t_min)), ms_to_utc_iso(int(t_max))))
    for y, m, n, t0, t1 in by_partition:
        print(f"  {y}/{m}: {n} rows  {t0} -> {t1}")
    print("-" * 60)
    print(f"  TOTAL: {total} rows")
    expected_full_window = 13248  # Jan 1 00:00 - Feb 15 23:55, 5m
    if total < expected_full_window:
        print(f"  Expected for full Jan 1 - Feb 15 2026: {expected_full_window} rows")
        print(f"  Missing: {expected_full_window - total} rows (likely later chunks not yet downloaded or not yet available from Binance).")

if __name__ == "__main__":
    main()
