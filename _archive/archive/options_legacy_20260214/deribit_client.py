"""
Deribit API v2 REST client for public market data.
No authentication required. Retry/backoff on 429/5xx, polite pacing between calls.
"""
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Min interval between calls; optional jitter."""

    def __init__(self, min_interval_sec: float = 0.25, jitter_sec: float = 0.05):
        self.min_interval_sec = min_interval_sec
        self.jitter_sec = jitter_sec
        self.last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_call
        target = self.min_interval_sec + random.uniform(0, self.jitter_sec)
        if elapsed < target:
            time.sleep(target - elapsed)
        self.last_call = time.monotonic()


def _request(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Any:
    """
    Call Deribit public REST (GET with query params). Returns JSON-RPC result or raises.
    """
    params = params or {}
    limiter = rate_limiter or RateLimiter(config.DERIBIT_SLEEP_BETWEEN_CALLS_MS / 1000.0)
    url = f"{config.DERIBIT_BASE_URL.rstrip('/')}/public/{method}"
    # Deribit accepts params as query string
    last_error = None
    for attempt in range(config.DERIBIT_MAX_RETRIES):
        limiter.wait()
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=config.DERIBIT_REQUEST_TIMEOUT_SEC,
            )
            logger.debug(
                "deribit %s params=%s status=%s attempt=%s",
                method,
                list(params.keys()),
                resp.status_code,
                attempt + 1,
            )
            if resp.status_code in (429, 418, 503, 502, 500):
                backoff = min(
                    config.DERIBIT_BACKOFF_BASE_SEC * (2**attempt),
                    config.DERIBIT_BACKOFF_MAX_SEC,
                )
                logger.warning(
                    "deribit %s status=%s backoff=%.1fs attempt=%s",
                    method,
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                last_error = resp
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Deribit API error: {data['error']}")
            return data.get("result")
        except requests.exceptions.RequestException as e:
            last_error = e
            backoff = min(
                config.DERIBIT_BACKOFF_BASE_SEC * (2**attempt),
                config.DERIBIT_BACKOFF_MAX_SEC,
            )
            logger.warning("deribit %s request error: %s backoff=%.1fs", method, e, backoff)
            if attempt < config.DERIBIT_MAX_RETRIES - 1:
                time.sleep(backoff)
    if last_error:
        if hasattr(last_error, "response") and last_error.response is not None:
            raise RuntimeError(
                f"Deribit request failed after {config.DERIBIT_MAX_RETRIES} retries: "
                f"{last_error.response.status_code} {last_error.response.text}"
            ) from last_error
        raise RuntimeError(f"Deribit request failed after {config.DERIBIT_MAX_RETRIES} retries: {last_error}") from last_error
    raise RuntimeError(f"Deribit request failed after {config.DERIBIT_MAX_RETRIES} retries")


def list_instruments(
    currency: str = "BTC",
    kind: str = "option",
    expired: bool = True,
    rate_limiter: Optional[RateLimiter] = None,
) -> List[Dict[str, Any]]:
    """
    List instruments. For historical backfill use expired=True to include expired options.
    Returns list of instrument objects (instrument_name, expiration_timestamp, strike, option_type, etc.).
    """
    # Deribit expects query param "true"/"false" (lowercase), not Python True/False
    result = _request(
        "get_instruments",
        params={"currency": currency, "kind": kind, "expired": "true" if expired else "false"},
        rate_limiter=rate_limiter,
    )
    return result if isinstance(result, list) else []


def get_tradingview_chart_data(
    instrument_name: str,
    start_timestamp: int,
    end_timestamp: int,
    resolution: int = 5,
    rate_limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """
    Get OHLC candles (TradingView format). Timestamps in milliseconds.
    resolution: bar size in minutes (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720) or use 1D.
    Returns dict with status, ticks (ms), open, high, low, close, volume (arrays).
    """
    result = _request(
        "get_tradingview_chart_data",
        params={
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": str(resolution),
        },
        rate_limiter=rate_limiter,
    )
    return result if isinstance(result, dict) else {}


def get_index_price(
    index_name: str = "btc_usd",
    rate_limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """Get current index price. Returns dict with index_price, estimated_delivery_price."""
    result = _request(
        "get_index_price",
        params={"index_name": index_name},
        rate_limiter=rate_limiter,
    )
    return result if isinstance(result, dict) else {}


def get_order_book(
    instrument_name: str,
    depth: int = 1,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get order book snapshot: best bid/ask prices and sizes, timestamp, index_price.
    depth: 1–10000; use 1 for best bid/ask only. Returns None on failure.
    """
    try:
        result = _request(
            "get_order_book",
            params={"instrument_name": instrument_name, "depth": depth},
            rate_limiter=rate_limiter,
        )
        return result if isinstance(result, dict) else None
    except Exception as e:
        logger.debug("get_order_book %s: %s", instrument_name, e)
        return None


def get_book_summary_by_instrument(
    instrument_name: str,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Dict[str, Any]]:
    """Get book summary (mark, bid/ask, mark_iv for options). Returns first element of result array or None."""
    try:
        result = _request(
            "get_book_summary_by_instrument",
            params={"instrument_name": instrument_name},
            rate_limiter=rate_limiter,
        )
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else None
    except Exception as e:
        logger.debug("get_book_summary_by_instrument %s: %s", instrument_name, e)
        return None
