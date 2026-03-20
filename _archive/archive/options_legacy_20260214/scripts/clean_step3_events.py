#!/usr/bin/env python3
"""
Remove Step 3 events with zero IV coverage (pct_sigma == 0 or n_sigma_non_nan == 0).
Reads step3_events.csv, step3_probabilities_5m.parquet, iv_coverage_by_event.csv;
writes step3_events_clean.csv and step3_probabilities_5m_clean.parquet.
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Remove zero-IV events from Step 3 outputs")
    parser.add_argument("--window_dir", type=str, default="outputs/window=2026-01-01_2026-02-15")
    args = parser.parse_args()

    window_dir = Path(args.window_dir)
    if not window_dir.is_absolute():
        window_dir = ROOT / window_dir
    step3_dir = window_dir / "step3"
    tables_dir = step3_dir / "step3_summary_tables"

    events_path = step3_dir / "step3_events.csv"
    probs_path = step3_dir / "step3_probabilities_5m.parquet"
    iv_path = tables_dir / "iv_coverage_by_event.csv"

    if not events_path.exists() or not probs_path.exists() or not iv_path.exists():
        print("ERROR: Missing one of step3_events.csv, step3_probabilities_5m.parquet, iv_coverage_by_event.csv")
        sys.exit(1)

    events = pd.read_csv(events_path)
    probs = pd.read_parquet(probs_path)
    iv = pd.read_csv(iv_path)

    before = len(events["event_id"].unique())

    # Events to remove: pct_sigma == 0 OR n_sigma_non_nan == 0
    zero_iv = (iv["pct_sigma"].fillna(0) == 0) | (iv["n_sigma_non_nan"].fillna(0) == 0)
    remove = iv.loc[zero_iv, "event_id"].tolist()
    keep_ids = iv.loc[~zero_iv, "event_id"].tolist()

    events_clean = events[events["event_id"].isin(keep_ids)]
    probs_clean = probs[probs["event_id"].isin(keep_ids)]

    after = len(events_clean["event_id"].unique())

    events_clean.to_csv(step3_dir / "step3_events_clean.csv", index=False)
    probs_clean.to_parquet(step3_dir / "step3_probabilities_5m_clean.parquet", index=False)

    print("Events before: %d" % before)
    print("Events after: %d" % after)
    print("Removed event_ids: %s" % remove)


if __name__ == "__main__":
    main()
