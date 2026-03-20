"""
Minimal Deribit API v2 REST client for options v2 pipeline.
Public methods only; rate limit, retry, backoff.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

import requests

from . import config as cfg

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, min_interval_sec: float = None, jitter_sec: float = 0.05):
        self.min_interval_sec = min_interval_sec or (cfg.SLEEP_BETWEEN_CALLS_MS / 1000.0)
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
    params = params or {}
    limiter = rate_limiter or RateLimiter()
    url = f"{cfg.DERIBIT_BASE_URL.rstrip('/')}/public/{method}"
    last_error = None
    for attempt in range(cfg.MAX_RETRIES):
        limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=cfg.REQUEST_TIMEOUT_SEC)
            if resp.status_code in (429, 418, 503, 502, 500):
                backoff = min(
                    cfg.BACKOFF_BASE_SEC * (2**attempt),
                    cfg.BACKOFF_MAX_SEC,
                )
                logger.warning("deribit %s status=%s backoff=%.1fs", method, resp.status_code, backoff)
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
            backoff = min(cfg.BACKOFF_BASE_SEC * (2**attempt), cfg.BACKOFF_MAX_SEC)
            logger.warning("deribit %s request error: %s backoff=%.1fs", method, e, backoff)
            if attempt < cfg.MAX_RETRIES - 1:
                time.sleep(backoff)
    if last_error:
        if hasattr(last_error, "response") and last_error.response is not None:
            raise RuntimeError(
                f"Deribit request failed after {cfg.MAX_RETRIES} retries: "
                f"{last_error.response.status_code} {last_error.response.text}"
            ) from last_error
        raise RuntimeError(f"Deribit request failed after {cfg.MAX_RETRIES} retries: {last_error}") from last_error
    raise RuntimeError(f"Deribit request failed after {cfg.MAX_RETRIES} retries")


def list_instruments(
    currency: str = "BTC",
    kind: str = "option",
    expired: bool = True,
    rate_limiter: Optional[RateLimiter] = None,
) -> List[Dict[str, Any]]:
    """List instruments. Use expired=True for historical options."""
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
    """OHLC candles. Timestamps in ms. resolution = bar minutes (5 for 5m)."""
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
