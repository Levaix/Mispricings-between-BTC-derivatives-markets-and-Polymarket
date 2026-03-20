"""
Diagnostics script: distribution of iv_hat in options_smile_event_iv (Method 4).
Reads parquet, keeps finite iv_hat, computes summary stats and range buckets, writes JSON.
Does not modify any existing datasets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# Method 4 output
INPUT_PATH = SCRIPT_DIR / "3h_window_staleness120_min3_gap10k" / "options_smile_event_iv.parquet"
OUTPUT_PATH = SCRIPT_DIR / "iv_hat_diagnostics.json"

# Buckets: (label, low_excl, high_incl) for (low_excl, high_incl]; first bucket is iv_hat <= 0
BUCKETS = [
    ("<=0", None, 0),
    ("0-0.05", 0, 0.05),
    ("0.05-0.1", 0.05, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.5", 0.2, 0.5),
    ("0.5-1", 0.5, 1),
    ("1-2", 1, 2),
    ("2-3", 2, 3),
    (">3", 3, None),
]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH, columns=["iv_hat"])
    usable = df["iv_hat"].notna() & np.isfinite(df["iv_hat"])
    iv = df.loc[usable, "iv_hat"].astype(float)
    n = len(iv)
    if n == 0:
        summary_stats = {"count": 0}
        bucket_counts = []
        print("No usable iv_hat values.")
    else:
        summary_stats = {
            "count": int(n),
            "min": float(iv.min()),
            "max": float(iv.max()),
            "mean": float(iv.mean()),
            "median": float(iv.median()),
            "p1": float(iv.quantile(0.01)),
            "p5": float(iv.quantile(0.05)),
            "p25": float(iv.quantile(0.25)),
            "p50": float(iv.quantile(0.50)),
            "p75": float(iv.quantile(0.75)),
            "p95": float(iv.quantile(0.95)),
            "p99": float(iv.quantile(0.99)),
        }
        bucket_counts = []
        for label, low, high in BUCKETS:
            if low is None and high is not None:
                mask = iv <= high
            elif low is not None and high is None:
                mask = iv > low
            else:
                mask = (iv > low) & (iv <= high)
            cnt = int(mask.sum())
            pct = round(100.0 * cnt / n, 4) if n else 0.0
            bucket_counts.append({"bucket": label, "count": cnt, "pct": pct})

    out = {
        "input_path": str(INPUT_PATH),
        "total_rows": int(len(df)),
        "usable_iv_count": int(n),
        "summary_statistics": summary_stats,
        "bucket_distribution": bucket_counts,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")
    print("--- Console summary ---")
    print(f"Total rows: {len(df)}, Usable iv_hat: {n}")
    if n:
        print(f"Min: {summary_stats['min']:.4f}, Max: {summary_stats['max']:.4f}, Mean: {summary_stats['mean']:.4f}, Median: {summary_stats['median']:.4f}")
        print("Percentiles: p1={:.4f} p5={:.4f} p25={:.4f} p50={:.4f} p75={:.4f} p95={:.4f} p99={:.4f}".format(
            summary_stats["p1"], summary_stats["p5"], summary_stats["p25"], summary_stats["p50"],
            summary_stats["p75"], summary_stats["p95"], summary_stats["p99"]))
        print("Buckets:")
        for b in bucket_counts:
            print(f"  {b['bucket']}: count={b['count']}, pct={b['pct']}%")


if __name__ == "__main__":
    main()
