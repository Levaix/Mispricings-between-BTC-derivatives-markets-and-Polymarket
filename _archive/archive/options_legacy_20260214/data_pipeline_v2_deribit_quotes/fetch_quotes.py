"""
Fetch current best bid/ask snapshot for Deribit options (live collection).
Deribit API does not provide historical order book; this module supports building
history by running on a schedule (e.g. every 5 minutes) and appending to raw store.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from deribit_client import RateLimiter, get_order_book, list_instruments

logger = logging.getLogger(__name__)

BAR_MS = 5 * 60 * 1000


def _instrument_meta(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Extract instrument_name, strike, expiry, call_put from Deribit instrument dict or selected_instruments.json."""
    expiry = inv.get("expiration_timestamp") or inv.get("expiry")
    return {
        "instrument_name": inv.get("instrument_name"),
        "strike": float(inv.get("strike", 0)),
        "expiry": int(expiry or 0),
        "call_put": str(inv.get("option_type") or inv.get("call_put") or "").upper()[:1],
    }


def fetch_snapshot_for_instrument(
    instrument_name: str,
    meta: Dict[str, Any],
    limiter: RateLimiter,
) -> Optional[Dict[str, Any]]:
    """
    Fetch current order book for one instrument. Returns one row dict with
    timestamp_utc, instrument_name, strike, expiry, call_put, best_bid_price, best_ask_price,
    best_bid_size, best_ask_size, index_price (and raw mark_iv if present).
    """
    raw = get_order_book(instrument_name, depth=1, rate_limiter=limiter)
    if not raw:
        return None
    ts_ms = raw.get("timestamp")
    if not ts_ms:
        return None
    bid = raw.get("best_bid_price")
    ask = raw.get("best_ask_price")
    bid_size = raw.get("best_bid_amount")
    ask_size = raw.get("best_ask_amount")
    index_price = raw.get("index_price")
    row = {
        "open_time": int(ts_ms),
        "instrument_name": instrument_name,
        "strike": meta.get("strike", 0),
        "expiry": meta.get("expiry", 0),
        "call_put": meta.get("call_put", ""),
        "best_bid_price": float(bid) if bid is not None else None,
        "best_ask_price": float(ask) if ask is not None else None,
        "best_bid_size": float(bid_size) if bid_size is not None else None,
        "best_ask_size": float(ask_size) if ask_size is not None else None,
        "index_price": float(index_price) if index_price is not None else None,
    }
    if raw.get("mark_iv") is not None:
        row["mark_iv_api"] = float(raw["mark_iv"])
    return row


def fetch_snapshots_for_instruments(
    instruments: List[Dict[str, Any]],
    rate_limiter: Optional[RateLimiter] = None,
) -> pd.DataFrame:
    """Fetch current bid/ask snapshot for each instrument; return one DataFrame."""
    limiter = rate_limiter or RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    rows = []
    for inv in instruments:
        name = inv.get("instrument_name")
        if not name:
            continue
        meta = _instrument_meta(inv)
        row = fetch_snapshot_for_instrument(name, meta, limiter)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def append_raw_snapshots(
    df: pd.DataFrame,
    raw_dir: Path,
) -> None:
    """Append snapshot DataFrame to raw store (one parquet per day or single append file)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    df["ts_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["_date"] = df["ts_utc"].dt.date
    for date, grp in df.groupby("_date"):
        out = raw_dir / f"quotes_{date}.parquet"
        existing = pd.read_parquet(out) if out.exists() else pd.DataFrame()
        combined = pd.concat([existing, grp.drop(columns=["_date"])], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time", "instrument_name"], keep="last")
        combined = combined.sort_values(["instrument_name", "open_time"])
        combined.to_parquet(out, index=False)
    logger.info("Appended %s rows to %s", len(df), raw_dir)


def load_selected_instruments_from_config() -> List[Dict[str, Any]]:
    """Load selected instruments from config path (JSON list)."""
    path = getattr(config, "OPTIONS_SELECTED_INSTRUMENTS_PATH", None)
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_instruments_deribit(rate_limiter: Optional[RateLimiter] = None) -> List[Dict[str, Any]]:
    """Load all BTC options from Deribit (active + expired) and filter to selected or near-ATM."""
    limiter = rate_limiter or RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    selected = load_selected_instruments_from_config()
    if selected:
        return selected
    active = list_instruments(currency="BTC", kind="option", expired=False, rate_limiter=limiter)
    expired = list_instruments(currency="BTC", kind="option", expired=True, rate_limiter=limiter)
    by_name = {inv.get("instrument_name"): inv for inv in (active + expired) if inv.get("instrument_name")}
    return list(by_name.values())
