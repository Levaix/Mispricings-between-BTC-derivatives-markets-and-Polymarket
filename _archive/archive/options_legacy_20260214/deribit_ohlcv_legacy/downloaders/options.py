"""
BTC options: Deribit API. Instrument universe + 5m OHLC for selected near-ATM options.
Output schema: instrument_name, expiry, strike, call_put, open_time, open, high, low, close, volume, source=deribit.
LEGACY: This module is in archive/deribit_ohlcv_legacy. Use data_pipeline_v2_deribit_quotes for quote-based pipeline.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config
from deribit_client import (
    RateLimiter,
    list_instruments,
    get_tradingview_chart_data,
    get_index_price,
)
from utils.time import get_start_end_ms, ms_to_utc
from utils.storage import add_utc_columns, ensure_dir, is_chunk_done, save_done_chunk

logger = logging.getLogger(__name__)

# 5 min in ms
RESOLUTION_5M_MS = 5 * 60 * 1000


def _expiry_bucket(expiry_ts_ms: int) -> str:
    """Classify expiry as weekly, monthly, or quarterly (Deribit style)."""
    dt = ms_to_utc(expiry_ts_ms)
    day = dt.day
    month = dt.month
    if month in (3, 6, 9, 12) and day >= 25:
        return "quarterly"
    if day >= 25 or (dt.replace(day=1) + pd.Timedelta(days=32)).replace(day=1) - pd.Timedelta(days=1) == dt.date():
        return "monthly"
    return "weekly"


def get_atm_reference(data_root: str) -> float:
    """ATM proxy: Binance futures index first close in window, or Deribit index, or default."""
    base = os.path.join(data_root, "futures", "usdm_perp", config.FUTURES_SYMBOL, "index_klines")
    if os.path.isdir(base):
        for year in ["2025"]:
            ydir = os.path.join(base, f"year={year}")
            if not os.path.isdir(ydir):
                continue
            for month in range(10, 13):
                mdir = os.path.join(ydir, f"month={month:02d}")
                if not os.path.isdir(mdir):
                    continue
                for f in os.listdir(mdir):
                    if not f.endswith(".parquet"):
                        continue
                    try:
                        df = pd.read_parquet(os.path.join(mdir, f))
                        if not df.empty and "close" in df.columns:
                            return float(df["close"].iloc[0])
                    except Exception:
                        pass
    limiter = RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    try:
        idx = get_index_price(config.DERIBIT_INDEX_NAME, rate_limiter=limiter)
        if idx and "index_price" in idx:
            return float(idx["index_price"])
    except Exception as e:
        logger.warning("Deribit index price fallback failed: %s", e)
    return 95000.0


def download_instruments(rate_limiter: Optional[RateLimiter] = None) -> List[Dict[str, Any]]:
    """Fetch BTC option instruments from Deribit: both active and recently expired, then merge."""
    limiter = rate_limiter or RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    active = list_instruments(currency="BTC", kind="option", expired=False, rate_limiter=limiter)
    expired = list_instruments(currency="BTC", kind="option", expired=True, rate_limiter=limiter)
    by_name = {inv.get("instrument_name"): inv for inv in (active + expired) if inv.get("instrument_name")}
    instruments = list(by_name.values())
    ensure_dir(config.OPTIONS_EXCHANGE_INFO_PATH)
    raw_path = os.path.join(config.OPTIONS_EXCHANGE_INFO_PATH, "deribit_instruments.json")
    with open(raw_path, "w") as f:
        json.dump(instruments, f, indent=2)
    logger.info("saved %s Deribit option instruments to %s", len(instruments), raw_path)
    # Parquet table for downstream
    rows = []
    for inv in instruments:
        rows.append({
            "instrument_name": inv.get("instrument_name"),
            "creation_timestamp": inv.get("creation_timestamp"),
            "expiration_timestamp": inv.get("expiration_timestamp"),
            "strike": inv.get("strike"),
            "option_type": inv.get("option_type"),
            "base_currency": inv.get("base_currency"),
            "settlement_currency": inv.get("settlement_currency"),
            "counter_currency": inv.get("counter_currency"),
            "expiry_bucket": _expiry_bucket(inv.get("expiration_timestamp") or 0),
        })
    if rows:
        df = pd.DataFrame(rows)
        table_path = os.path.join(config.OPTIONS_EXCHANGE_INFO_PATH, "deribit_instruments.parquet")
        df.to_parquet(table_path, index=False)
        logger.info("saved instruments table to %s", table_path)
    return instruments


def build_selected_instruments(
    instruments: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
    atm_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Select near-ATM options that were live trading at any time during [start_ms, end_ms].
    """
    if not instruments:
        return []
    df = pd.DataFrame(instruments)
    if "base_currency" in df.columns:
        df = df[df["base_currency"] == "BTC"].copy()
    else:
        df = df.copy()
    if "expiration_timestamp" not in df.columns:
        return []
    df["expiration_timestamp"] = df["expiration_timestamp"].astype("int64")
    if "creation_timestamp" in df.columns:
        df["creation_timestamp"] = pd.to_numeric(df["creation_timestamp"], errors="coerce").fillna(0).astype("int64")
    else:
        df["creation_timestamp"] = 0
    was_live_in_window = (df["creation_timestamp"] <= end_ms) & (df["expiration_timestamp"] >= start_ms)
    df = df[was_live_in_window].copy()
    if df.empty:
        logger.warning("No Deribit options had trading life overlapping window.")
        return []
    if atm_price is None:
        atm_price = get_atm_reference(config.DATA_ROOT)
        logger.info("ATM reference for option selection: %s", atm_price)
    low = atm_price * (1 - config.ATM_PERCENT_RANGE)
    high = atm_price * (1 + config.ATM_PERCENT_RANGE)
    selected = []
    for exp_ts in sorted(df["expiration_timestamp"].unique()):
        sub = df[df["expiration_timestamp"] == exp_ts].copy()
        sub["strike"] = pd.to_numeric(sub["strike"], errors="coerce")
        sub = sub[(sub["strike"] >= low) & (sub["strike"] <= high)]
        calls = sub[sub["option_type"] == "call"].sort_values("strike")
        puts = sub[sub["option_type"] == "put"].sort_values("strike")
        def pick(side_df: pd.DataFrame, limit: int) -> List[Dict]:
            if side_df.empty:
                return []
            side_df = side_df.copy()
            side_df["dist"] = (side_df["strike"] - atm_price).abs()
            side_df = side_df.sort_values("dist").head(limit)
            return side_df.to_dict("records")
        chosen = pick(calls, config.MAX_STRIKES_PER_SIDE) + pick(puts, config.MAX_STRIKES_PER_SIDE)
        selected.extend(chosen)
    return selected


def persist_selected_instruments(selected: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(config.OPTIONS_SELECTED_INSTRUMENTS_PATH))
    with open(config.OPTIONS_SELECTED_INSTRUMENTS_PATH, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info("persisted %s selected instruments to %s", len(selected), config.OPTIONS_SELECTED_INSTRUMENTS_PATH)


def load_selected_instruments() -> List[Dict[str, Any]]:
    if not os.path.isfile(config.OPTIONS_SELECTED_INSTRUMENTS_PATH):
        return []
    with open(config.OPTIONS_SELECTED_INSTRUMENTS_PATH, "r") as f:
        return json.load(f)


def fetch_candles_chunk(
    instrument_name: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> Dict[str, Any]:
    return get_tradingview_chart_data(
        instrument_name,
        start_timestamp=start_ms,
        end_timestamp=end_ms,
        resolution=config.OPTIONS_RESOLUTION_MINUTES,
        rate_limiter=limiter,
    )


def candles_result_to_rows(result: Dict[str, Any], instrument_name: str) -> List[Dict[str, Any]]:
    if result.get("status") != "ok":
        return []
    ticks = result.get("ticks") or []
    if not ticks:
        return []
    open_ = result.get("open") or []
    high = result.get("high") or []
    low = result.get("low") or []
    close = result.get("close") or []
    volume = result.get("volume") or []
    rows = []
    for i, t in enumerate(ticks):
        rows.append({
            "open_time": int(t),
            "open": open_[i] if i < len(open_) else None,
            "high": high[i] if i < len(high) else None,
            "low": low[i] if i < len(low) else None,
            "close": close[i] if i < len(close) else None,
            "volume": volume[i] if i < len(volume) else 0,
        })
    return rows


def download_options_klines_for_instrument(
    instrument_info: Dict[str, Any],
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> int:
    instrument_name = instrument_info.get("instrument_name")
    if not instrument_name:
        return 0
    manifest_key = f"options_klines_{instrument_name}"
    if is_chunk_done(manifest_key, "full"):
        return 0
    chunk_ms = config.OPTIONS_CANDLE_CHUNK_DAYS * 24 * 60 * 60 * 1000
    all_rows = []
    current = start_ms
    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        result = fetch_candles_chunk(instrument_name, current, chunk_end, limiter)
        rows = candles_result_to_rows(result, instrument_name)
        all_rows.extend(rows)
        current = chunk_end
    if not all_rows:
        save_done_chunk(manifest_key, "full")
        return 0
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df = df[df[col].notna() & (df[col] > 0)]
    if df.empty:
        save_done_chunk(manifest_key, "full")
        return 0
    df["instrument_name"] = instrument_name
    df["expiry"] = int(instrument_info.get("expiration_timestamp", 0))
    df["strike"] = float(instrument_info.get("strike", 0))
    df["call_put"] = str(instrument_info.get("option_type", "")).upper()
    df["source"] = "deribit"
    df = add_utc_columns(df)
    safe_name = instrument_name.replace(":", "_")
    base_path = os.path.join(config.OPTIONS_KLINES_PATH, f"instrument={safe_name}")
    df["_year"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.year
    df["_month"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.month
    total = 0
    for (y, m), group in df.groupby(["_year", "_month"]):
        out_dir = os.path.join(base_path, f"year={y}", f"month={m:02d}")
        ensure_dir(out_dir)
        out_file = os.path.join(out_dir, "klines_5m.parquet")
        group = group.drop(columns=["_year", "_month"])
        group.to_parquet(out_file, index=False)
        total += len(group)
    save_done_chunk(manifest_key, "full")
    logger.info("wrote %s options klines %s rows to %s", instrument_name, total, base_path)
    return total


def download_all_options(
    api_key: Optional[str] = None,
    exchange_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], int]:
    start_ms, end_ms = get_start_end_ms()
    limiter = RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    instruments = download_instruments(rate_limiter=limiter)
    selected = load_selected_instruments()
    if not selected:
        selected = build_selected_instruments(instruments, start_ms, end_ms)
        persist_selected_instruments(selected)
    total = 0
    for inv in selected:
        if is_chunk_done(f"options_klines_{inv.get('instrument_name')}", "full"):
            continue
        total += download_options_klines_for_instrument(inv, start_ms, end_ms, limiter)
    return {"optionSymbols": instruments, "deribit_instruments": instruments}, total


def get_options_summary_stats() -> Dict[str, Any]:
    selected = load_selected_instruments()
    if not selected:
        return {"expiries_per_bucket": {}, "instruments_total": 0, "rows_per_month": {}}
    buckets: Dict[str, int] = {}
    for inv in selected:
        b = _expiry_bucket(inv.get("expiration_timestamp") or 0)
        buckets[b] = buckets.get(b, 0) + 1
    rows_per_month: Dict[str, int] = {}
    for inv in selected:
        safe_name = inv.get("instrument_name", "").replace(":", "_")
        base = os.path.join(config.OPTIONS_KLINES_PATH, f"instrument={safe_name}")
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(".parquet"):
                    continue
                try:
                    parts = os.path.normpath(root).split(os.sep)
                    y_idx = next((i for i, p in enumerate(parts) if p.startswith("year=")), None)
                    m_idx = next((i for i, p in enumerate(parts) if p.startswith("month=")), None)
                    if y_idx is not None and m_idx is not None:
                        ym = f"{parts[y_idx].replace('year=', '')}-{parts[m_idx].replace('month=', '')}"
                        df = pd.read_parquet(os.path.join(root, f))
                        rows_per_month[ym] = rows_per_month.get(ym, 0) + len(df)
                except Exception:
                    pass
    return {"expiries_per_bucket": buckets, "instruments_total": len(selected), "rows_per_month": rows_per_month}
