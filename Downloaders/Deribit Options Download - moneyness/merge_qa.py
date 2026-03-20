"""
Merge option candles with index series; add moneyness; write long-format parquet(s).
Partition by month if configured.
"""
import logging
from pathlib import Path

import pandas as pd

from . import config as cfg
from .fetch_candles import run as fetch_run

logger = logging.getLogger(__name__)


def run(force_refresh=False):
    start_ms = int(pd.Timestamp(cfg.START_UTC.replace("Z", "+00:00")).timestamp() * 1000)
    end_ms = int(pd.Timestamp(cfg.END_UTC.replace("Z", "+00:00")).timestamp() * 1000)

    candles = fetch_run(force_refresh=force_refresh)
    if candles.empty:
        logger.warning("No candle data to merge")
        return pd.DataFrame()

    index_df = pd.read_parquet(cfg.INDEX_SERIES_PARQUET)
    index_df = index_df.rename(columns={"open_time": "timestamp_utc"})
    candles = candles.merge(
        index_df[["timestamp_utc", "index_price"]],
        on="timestamp_utc",
        how="left",
    )
    candles["moneyness"] = candles["strike"] / candles["index_price"].replace(0, float("nan"))

    candles = candles[
        (candles["timestamp_utc"] >= start_ms) & (candles["timestamp_utc"] <= end_ms)
    ]
    candles = candles.dropna(subset=["index_price", "moneyness"])

    export_df = candles.copy()
    export_df = export_df.drop(columns=[c for c in ("ts", "partition_month") if c in export_df.columns], errors="ignore")

    if cfg.PARTITION_BY_MONTH:
        candles["ts"] = pd.to_datetime(candles["timestamp_utc"], unit="ms", utc=True)
        candles["partition_month"] = candles["ts"].dt.strftime("%Y-%m")
        for month, grp in candles.groupby("partition_month"):
            out_dir = cfg.ALIGNED_PARQUET_DIR / f"month={month}"
            out_dir.mkdir(parents=True, exist_ok=True)
            grp.drop(columns=["ts", "partition_month"]).to_parquet(out_dir / "data.parquet", index=False)
    export_df.to_parquet(cfg.RAW_LONG_PARQUET, index=False)
    out_path = cfg.RAW_LONG_PARQUET

    logger.info("Merged and saved aligned options: %s rows to %s", len(export_df), out_path)
    return export_df
