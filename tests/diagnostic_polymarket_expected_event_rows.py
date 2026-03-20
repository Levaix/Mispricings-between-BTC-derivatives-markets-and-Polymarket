"""
Diagnostic: theoretical number of event rows before any options smile matching.

Uses:
- Actual data/Polymarket/polymarket_contracts_metadata.parquet (event dates, strikes)
- Actual data/Options Deribit/options_5m_agg/agg_5m_merged.parquet (unique timestamps)

Polymarket daily markets expire at 17:00 UTC. For each event date Z we count options
timestamps t with t < Z 17:00 UTC, then event rows for Z = (timestamps before expiry) * (unique strikes for Z).
Outputs saved under tests/.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
POLYMARKET_META = ROOT / "Actual data" / "Polymarket" / "polymarket_contracts_metadata.parquet"
OPTIONS_5M = ROOT / "Actual data" / "Options Deribit" / "options_5m_agg" / "agg_5m_merged.parquet"
OUT_DIR = ROOT / "tests"
POLYMARKET_DAILY_EXPIRY_HOUR_UTC = 17


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load Polymarket metadata
    if not POLYMARKET_META.exists():
        raise FileNotFoundError(f"Polymarket metadata not found: {POLYMARKET_META}")
    meta = pd.read_parquet(POLYMARKET_META)
    if "expiry_date" not in meta.columns:
        raise ValueError("Metadata must have 'expiry_date'")
    strike_col = "strike_K" if "strike_K" in meta.columns else "strike"
    if strike_col not in meta.columns:
        raise ValueError(f"Metadata must have 'strike_K' or 'strike'")
    meta = meta.copy()
    meta["expiry_date"] = meta["expiry_date"].astype(str)
    meta["strike_K"] = pd.to_numeric(meta[strike_col], errors="coerce")
    meta = meta.dropna(subset=["strike_K", "expiry_date"])

    # 2) Unique event dates and unique strikes per event date
    unique_event_dates = meta["expiry_date"].unique().tolist()
    n_unique_event_dates = len(unique_event_dates)
    strikes_per_date = meta.groupby("expiry_date")["strike_K"].nunique()
    avg_strikes_per_day = float(strikes_per_date.mean())

    # 3) Load options timestamps; for each event date Z, count t where t < Z 17:00 UTC
    if not OPTIONS_5M.exists():
        raise FileNotFoundError(f"Options parquet not found: {OPTIONS_5M}")
    options = pd.read_parquet(OPTIONS_5M, columns=["bucket_5m_utc"])
    options["bucket_5m_utc"] = pd.to_datetime(options["bucket_5m_utc"], utc=True)
    unique_buckets = options["bucket_5m_utc"].drop_duplicates().sort_values()

    total_expected_event_rows = 0
    timestamps_per_date = []

    for z in unique_event_dates:
        expiry_cutoff = pd.Timestamp(f"{z} {POLYMARKET_DAILY_EXPIRY_HOUR_UTC:02d}:00:00", tz="UTC")
        before_expiry = unique_buckets[unique_buckets < expiry_cutoff]
        n_t = len(before_expiry)
        n_strikes = int(strikes_per_date[z])
        rows_z = n_t * n_strikes
        total_expected_event_rows += rows_z
        timestamps_per_date.append(n_t)

    n_timestamps_used = sum(timestamps_per_date)  # total (t, Z) pairs
    avg_markets_active_per_timestamp = (
        total_expected_event_rows / n_timestamps_used if n_timestamps_used else 0.0
    )

    result = {
        "total_expected_event_rows": int(total_expected_event_rows),
        "n_unique_event_dates": n_unique_event_dates,
        "avg_strikes_per_day": round(avg_strikes_per_day, 2),
        "avg_markets_active_per_timestamp": round(avg_markets_active_per_timestamp, 2),
        "n_unique_options_timestamps": int(len(unique_buckets)),
        "n_timestamp_event_date_pairs": int(n_timestamps_used),
    }

    out_path = OUT_DIR / "diagnostic_polymarket_expected_event_rows.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Diagnostic output:", out_path)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
