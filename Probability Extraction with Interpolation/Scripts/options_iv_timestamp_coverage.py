"""Compute IV timestamp coverage diagnostics for options_smile_event_iv.parquet.

Reads analysis/options_smile_event_iv.parquet, computes coverage across 5-minute
timestamps and by day, writes analysis/options_iv_timestamp_coverage.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = ANALYSIS_DIR / "options_smile_event_iv.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "options_iv_timestamp_coverage.json"


def main() -> None:
    # 1) Load the dataset
    df = pd.read_parquet(INPUT_PATH, columns=["bucket_5m_utc", "iv_hat"])
    df["bucket_5m_utc"] = pd.to_datetime(df["bucket_5m_utc"], utc=True)

    total_rows = len(df)
    usable_iv = df["iv_hat"].notna() & np.isfinite(df["iv_hat"])
    usable_iv_rows = int(usable_iv.sum())
    pct_rows_with_iv = (100.0 * usable_iv_rows / total_rows) if total_rows else 0.0

    # 2) Timestamp diagnostics
    total_unique_timestamps = df["bucket_5m_utc"].nunique()
    ts_with_any_iv = df.loc[usable_iv, "bucket_5m_utc"].unique()
    timestamps_with_iv = len(ts_with_any_iv)
    timestamps_without_iv = total_unique_timestamps - timestamps_with_iv
    pct_timestamps_with_iv = (
        100.0 * timestamps_with_iv / total_unique_timestamps
    ) if total_unique_timestamps else 0.0

    # 3) IV density per timestamp: for each timestamp, count rows with usable IV
    per_ts = df.groupby("bucket_5m_utc")["iv_hat"].apply(
        lambda s: (s.notna() & np.isfinite(s)).sum()
    )
    avg_usable_iv_per_timestamp = float(per_ts.mean()) if len(per_ts) else 0.0
    median_usable_iv_per_timestamp = float(per_ts.median()) if len(per_ts) else 0.0
    max_usable_iv_per_timestamp = int(per_ts.max()) if len(per_ts) else 0

    # 4) Daily coverage: for each calendar day, unique timestamps and how many have at least one IV
    df["date"] = df["bucket_5m_utc"].dt.date
    day_ts_total = df.groupby("date")["bucket_5m_utc"].nunique()
    day_ts_with_iv = df.loc[usable_iv].groupby("date")["bucket_5m_utc"].nunique()
    daily_coverage = []
    for d in sorted(df["date"].unique()):
        ts_tot = day_ts_total.get(d, 0)
        ts_iv = day_ts_with_iv.get(d, 0)
        pct = (100.0 * ts_iv / ts_tot) if ts_tot else 0.0
        daily_coverage.append({
            "date": str(d),
            "timestamps": int(ts_tot),
            "timestamps_with_iv": int(ts_iv),
            "pct": round(pct, 4),
        })

    out = {
        "total_rows": total_rows,
        "usable_iv_rows": usable_iv_rows,
        "pct_rows_with_iv": round(pct_rows_with_iv, 4),
        "timestamp_diagnostics": {
            "total_unique_timestamps": total_unique_timestamps,
            "timestamps_with_iv": timestamps_with_iv,
            "timestamps_without_iv": timestamps_without_iv,
            "pct_timestamps_with_iv": round(pct_timestamps_with_iv, 4),
        },
        "iv_density_per_timestamp": {
            "avg_usable_iv_per_timestamp": round(avg_usable_iv_per_timestamp, 4),
            "median_usable_iv_per_timestamp": round(median_usable_iv_per_timestamp, 4),
            "max_usable_iv_per_timestamp": max_usable_iv_per_timestamp,
        },
        "daily_coverage": daily_coverage,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Print key metrics
    print("pct_rows_with_iv:", pct_rows_with_iv)
    print("pct_timestamps_with_iv:", pct_timestamps_with_iv)
    print("avg_usable_iv_per_timestamp:", avg_usable_iv_per_timestamp)


if __name__ == "__main__":
    main()
