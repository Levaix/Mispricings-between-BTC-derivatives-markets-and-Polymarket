"""Fetch 5m index proxy series (BTC-PERPETUAL close) for the window."""
import logging

import pandas as pd

from . import config as cfg
from .deribit_client import RateLimiter, get_tradingview_chart_data
from .discover import utc_to_ms

logger = logging.getLogger(__name__)


def _candles_to_df(result):
    if result.get("status") != "ok":
        return pd.DataFrame()
    ticks = result.get("ticks") or []
    close = result.get("close") or []
    if not ticks:
        return pd.DataFrame()
    rows = []
    for i, t in enumerate(ticks):
        rows.append({"open_time": int(t), "index_price": close[i] if i < len(close) else None})
    return pd.DataFrame(rows)


def run(force_refresh=False):
    start_ms = utc_to_ms(cfg.START_UTC)
    end_ms = utc_to_ms(cfg.END_UTC)

    if cfg.INDEX_SERIES_PARQUET.exists() and not force_refresh:
        df = pd.read_parquet(cfg.INDEX_SERIES_PARQUET)
        logger.info("Loaded index series %s rows from %s", len(df), cfg.INDEX_SERIES_PARQUET)
        return df

    limiter = RateLimiter()
    chunk_ms = getattr(cfg, "CANDLE_CHUNK_DAYS", 1) * 24 * 60 * 60 * 1000
    all_rows = []
    current = start_ms
    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        result = get_tradingview_chart_data(
            cfg.INDEX_PROXY_INSTRUMENT,
            current,
            chunk_end,
            resolution=cfg.RESOLUTION_MINUTES,
            rate_limiter=limiter,
        )
        df_chunk = _candles_to_df(result)
        if not df_chunk.empty:
            all_rows.append(df_chunk)
        current = chunk_end

    if not all_rows:
        logger.warning("No index data returned")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    df = df[df["index_price"].notna() & (df["index_price"] > 0)]
    df.to_parquet(cfg.INDEX_SERIES_PARQUET, index=False)
    logger.info("Saved index series %s rows to %s", len(df), cfg.INDEX_SERIES_PARQUET)
    return df
