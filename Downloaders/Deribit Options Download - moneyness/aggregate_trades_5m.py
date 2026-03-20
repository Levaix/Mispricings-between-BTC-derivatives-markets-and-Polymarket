"""Aggregate Deribit BTC options trade-level daily files into 5-minute trade bars.

Reads from history_trades/date=YYYY-MM-DD/trades.parquet (unchanged).
Writes to a new output folder (e.g. options_5m_agg/) with partitioned parquet,
merged parquet, and a QA report. No empty windows, no filling, no full grid.

Usage:
  python -m pipeline.options_v2_deribit_moneyness.aggregate_trades_5m \\
    --input_dir data/options_v2_deribit_moneyness/history_trades \\
    --output_dir data/options_v2_deribit_moneyness/options_5m_agg \\
    --start_date 2026-01-01 --end_date 2026-02-28

Loader (from code):
  from pipeline.options_v2_deribit_moneyness.aggregate_trades_5m import load_agg_5m
  df = load_agg_5m("data/options_v2_deribit_moneyness/options_5m_agg")
  # Or by date range without merged file:
  df = load_agg_5m(output_dir, start_date="2026-01-01", end_date="2026-01-31", use_merged=False)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["instrument_name", "timestamp", "price", "iv"]
BUCKET_MINUTES = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate trade-level data to 5m bars")
    p.add_argument("--input_dir", type=str, required=True, help="Raw daily trade parquet root (e.g. history_trades)")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder for aggregated parquet and QA")
    p.add_argument("--start_date", type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end_date", type=str, required=True, help="End date YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Reprocess days that already have output")
    return p.parse_args()


def _date_range(start: str, end: str) -> list[str]:
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    d = start_d
    while d <= end_d:
        out.append(d.strftime("%Y-%m-%d"))
        d = (datetime.combine(d, datetime.min.time()) + pd.Timedelta(days=1)).date()
    return out


def _validate_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
        return False
    return True


def _validate_and_clean(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Row-level validation; drop invalid rows and update stats."""
    raw_count = len(df)
    stats["raw_rows"] = raw_count

    # Null instrument_name or timestamp
    bad_null = df["instrument_name"].isna() | df["timestamp"].isna()
    n_null = bad_null.sum()
    if n_null:
        stats["dropped_null_instrument_or_timestamp"] = int(n_null)
    df = df.loc[~bad_null]

    # Timestamp conversion
    try:
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    except Exception as e:
        logger.warning("Timestamp conversion failed: %s", e)
        stats["dropped_bad_timestamp"] = len(df)
        return pd.DataFrame()
    bad_ts = df["timestamp_utc"].isna()
    n_bad_ts = bad_ts.sum()
    if n_bad_ts:
        stats["dropped_bad_timestamp"] = int(n_bad_ts)
    df = df.loc[~bad_ts]

    # price >= 0, iv >= 0
    bad_price = (df["price"] < 0) | df["price"].isna()
    bad_iv = (df["iv"] < 0) | df["iv"].isna()
    n_bad_price = bad_price.sum()
    n_bad_iv = bad_iv.sum()
    if n_bad_price:
        stats["dropped_bad_price"] = int(n_bad_price)
    if n_bad_iv:
        stats["dropped_bad_iv"] = int(n_bad_iv)
    df = df.loc[~bad_price & ~bad_iv]

    stats["rows_kept"] = len(df)
    return df


def _floor_5m(ts: pd.Series) -> pd.Series:
    """Floor timestamps to 5-minute boundary (UTC)."""
    return ts.dt.floor(f"{BUCKET_MINUTES}min")


def _aggregate_one_day(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (instrument_name, bucket_5m_utc) and compute OHLCV + VWAP, iv_vwap, etc."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["bucket_5m_utc"] = _floor_5m(df["timestamp_utc"])
    # Weight for VWAP: amount if present and > 0, else contracts
    if "amount" in df.columns and "contracts" in df.columns:
        amount_ok = df["amount"].fillna(0) > 0
        df["_weight"] = df["amount"].where(amount_ok, df["contracts"].fillna(0))
    elif "amount" in df.columns:
        df["_weight"] = df["amount"].fillna(0)
    elif "contracts" in df.columns:
        df["_weight"] = df["contracts"].fillna(0)
    else:
        df["_weight"] = 1.0

    # Sort by timestamp within each group for first/last
    df = df.sort_values("timestamp_utc")

    def ohlc_vwap(g: pd.DataFrame) -> pd.Series:
        price = g["price"]
        iv = g["iv"]
        wt = g["_weight"]
        total_w = wt.sum()
        if total_w and total_w > 0:
            vwap_price = (price * wt).sum() / total_w
            iv_vwap = (iv * wt).sum() / total_w
        else:
            vwap_price = price.mean()
            iv_vwap = iv.mean()
        index_price = g["index_price"] if "index_price" in g.columns else pd.Series(dtype=float)
        return pd.Series({
            "open_price": price.iloc[0],
            "close_price": price.iloc[-1],
            "high_price": price.max(),
            "low_price": price.min(),
            "vwap_price": vwap_price,
            "iv_vwap": iv_vwap,
            "trade_count": len(g),
            "amount_sum": g["amount"].sum() if "amount" in g.columns else None,
            "contracts_sum": g["contracts"].sum() if "contracts" in g.columns else None,
            "index_price_last": index_price.iloc[-1] if len(index_price) and index_price.notna().any() else None,
            "timestamp_first": g["timestamp_utc"].iloc[0],
            "timestamp_last": g["timestamp_utc"].iloc[-1],
        })

    agg = df.groupby(["instrument_name", "bucket_5m_utc"], group_keys=False).apply(ohlc_vwap, include_groups=False)
    if agg.empty:
        return pd.DataFrame()
    # apply returns DataFrame with MultiIndex (instrument_name, bucket_5m_utc); reset to columns
    agg = agg.reset_index()
    return agg


def _validate_aggregation(agg: pd.DataFrame) -> None:
    """Assert uniqueness and 5-min alignment."""
    if agg.empty:
        return
    key = agg[["instrument_name", "bucket_5m_utc"]]
    dupes = key.duplicated()
    if dupes.any():
        raise AssertionError(f"Duplicate (instrument_name, bucket_5m_utc): {dupes.sum()} rows")
    # Check 5-min alignment: bucket_5m_utc minute % 5 == 0, second == 0
    minutes = agg["bucket_5m_utc"].dt.minute
    seconds = agg["bucket_5m_utc"].dt.second
    bad = (minutes % BUCKET_MINUTES != 0) | (seconds != 0)
    if bad.any():
        raise AssertionError(f"bucket_5m_utc not 5-min aligned: {bad.sum()} rows")


def process_one_file(
    path: Path,
    stats: dict[str, Any],
) -> pd.DataFrame | None:
    """Load one daily parquet, validate, clean, aggregate. Returns aggregated DataFrame or None if skipped."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        stats["skip_reason"] = str(e)
        return None

    if not _validate_columns(df):
        stats["skip_reason"] = "missing_required_columns"
        return None

    file_stats = {}
    clean = _validate_and_clean(df, file_stats)
    for k, v in file_stats.items():
        stats[k] = v

    if clean.empty:
        return pd.DataFrame()

    agg = _aggregate_one_day(clean)
    _validate_aggregation(agg)
    return agg


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "files_processed": 0,
        "files_skipped": 0,
        "skip_details": [],
        "raw_rows_read": 0,
        "rows_kept": 0,
        "rows_dropped_by_reason": {},
        "aggregated_rows": 0,
        "buckets_per_date": {},
        "top20_by_contracts_sum": [],
        "top20_by_trade_count": [],
    }

    dates = _date_range(start_date, end_date)
    all_aggs: list[pd.DataFrame] = []

    for d in dates:
        part_dir = input_path / f"date={d}"
        trade_file = part_dir / "trades.parquet"
        if not trade_file.exists():
            logger.info("No file for %s: %s", d, trade_file)
            continue

        out_part = output_path / f"date={d}"
        out_parquet = out_part / "agg_5m.parquet"
        if out_parquet.exists() and not force:
            logger.info("Already processed %s, skipping (use --force to reprocess)", d)
            try:
                existing = pd.read_parquet(out_parquet)
                report["aggregated_rows"] += len(existing)
                report["buckets_per_date"][d] = len(existing)
                all_aggs.append(existing)
            except Exception:
                pass
            report["files_processed"] += 1
            continue

        stats = {}
        agg = process_one_file(trade_file, stats)

        if stats.get("skip_reason"):
            report["files_skipped"] += 1
            report["skip_details"].append({"date": d, "reason": stats["skip_reason"]})
            continue

        report["files_processed"] += 1
        report["raw_rows_read"] += stats.get("raw_rows", 0)
        report["rows_kept"] += stats.get("rows_kept", 0)
        for k, v in stats.items():
            if k.startswith("dropped_"):
                report["rows_dropped_by_reason"][k] = report["rows_dropped_by_reason"].get(k, 0) + v

        if agg is None:
            continue
        if agg.empty:
            report["buckets_per_date"][d] = 0
            continue

        report["aggregated_rows"] += len(agg)
        report["buckets_per_date"][d] = len(agg)
        all_aggs.append(agg)

        out_part.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_parquet, index=False)
        logger.info("Wrote %s -> %s (%d bars)", d, out_parquet, len(agg))

    # Top 20 by contracts_sum and trade_count (from merged agg)
    if all_aggs:
        merged = pd.concat(all_aggs, ignore_index=True)
        if "contracts_sum" in merged.columns:
            top_contracts = (
                merged.groupby("instrument_name")["contracts_sum"]
                .sum()
                .nlargest(20)
            )
            report["top20_by_contracts_sum"] = [{"instrument_name": k, "contracts_sum": float(v)} for k, v in top_contracts.items()]
        if "trade_count" in merged.columns:
            top_trades = (
                merged.groupby("instrument_name")["trade_count"]
                .sum()
                .nlargest(20)
            )
            report["top20_by_trade_count"] = [{"instrument_name": k, "trade_count": int(v)} for k, v in top_trades.items()]

    # QA report
    qa_path = output_path / "qa_report.json"
    with open(qa_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("QA report: %s", qa_path)

    md_path = output_path / "qa_report.md"
    with open(md_path, "w") as f:
        f.write("# 5m trade bars aggregation – QA report\n\n")
        f.write(f"- Files processed: {report['files_processed']}\n")
        f.write(f"- Files skipped: {report['files_skipped']}\n")
        f.write(f"- Raw rows read: {report['raw_rows_read']}\n")
        f.write(f"- Rows kept: {report['rows_kept']}\n")
        f.write(f"- Rows dropped (by reason): {json.dumps(report['rows_dropped_by_reason'], indent=2)}\n")
        f.write(f"- Total aggregated rows: {report['aggregated_rows']}\n\n")
        f.write("## Top 20 instruments by contracts_sum\n\n")
        for e in report["top20_by_contracts_sum"]:
            f.write(f"- {e['instrument_name']}: {e['contracts_sum']}\n")
        f.write("\n## Top 20 instruments by trade_count\n\n")
        for e in report["top20_by_trade_count"]:
            f.write(f"- {e['instrument_name']}: {e['trade_count']}\n")
        f.write("\n## Buckets per date\n\n")
        for d, c in sorted(report["buckets_per_date"].items()):
            f.write(f"- {d}: {c}\n")
    logger.info("QA report (MD): %s", md_path)

    # Optional: single merged parquet
    if all_aggs:
        merged_path = output_path / "agg_5m_merged.parquet"
        pd.concat(all_aggs, ignore_index=True).to_parquet(merged_path, index=False)
        logger.info("Merged parquet: %s", merged_path)

    return report


def load_agg_5m(
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    use_merged: bool = True,
) -> pd.DataFrame:
    """Load aggregated 5m bars from output_dir.

    - If use_merged=True and agg_5m_merged.parquet exists, loads that (ignores date range).
    - Otherwise loads partitioned data: date=YYYY-MM-DD/agg_5m.parquet for each day
      in [start_date, end_date]. start_date/end_date required when use_merged=False or
      when merged file is missing.
    """
    out = Path(output_dir)
    merged_path = out / "agg_5m_merged.parquet"
    if use_merged and merged_path.exists():
        return pd.read_parquet(merged_path)
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date required when merged file is not used or missing")
    dates = _date_range(start_date, end_date)
    parts = [out / f"date={d}" / "agg_5m.parquet" for d in dates]
    existing = [p for p in parts if p.exists()]
    if not existing:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in existing], ignore_index=True)


def main() -> None:
    args = _parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
    )


if __name__ == "__main__":
    main()
