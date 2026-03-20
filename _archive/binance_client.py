"""
Simple REST client for Binance APIs: GET requests, retry/backoff, rate limiter.
"""
import time
import hashlib
import json
import logging
from typing import Any, Dict, Optional

import requests

import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Max 5 requests/second; sleep 150-250ms between calls by default."""

    def __init__(self, min_interval_sec: float = 0.2):
        self.min_interval_sec = min_interval_sec
        self.last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self.last_call = time.monotonic()


def _params_hash(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Any:
    """
    GET request with retry, exponential backoff on 429/418, and rate limiting.
    Returns JSON-decoded response.
    """
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    params = dict(params or {})
    # Binance requires startTime/endTime as plain int (rejects numpy.int64 etc.)
    for key in ("startTime", "endTime"):
        if key in params and params[key] is not None:
            params[key] = int(params[key])
    limiter = rate_limiter or RateLimiter(config.SLEEP_BETWEEN_CALLS_MS / 1000.0)
    phash = _params_hash(params)

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        limiter.wait()
        try:
            headers = {}
            if api_key:
                headers["X-MBX-APIKEY"] = api_key
            resp = requests.get(
                url,
                params=params,
                headers=headers or None,
                timeout=config.REQUEST_TIMEOUT_SEC,
            )
            logger.info(
                "request %s %s params_hash=%s start=%s end=%s status=%s attempt=%s",
                base_url,
                path,
                phash,
                params.get("startTime"),
                params.get("endTime"),
                resp.status_code,
                attempt + 1,
            )

            if resp.status_code in (429, 418):
                # Exponential backoff: 1s, 2s, 4s, 8s (cap 60s)
                backoff = min(
                    config.BACKOFF_BASE_SEC * (2**attempt),
                    config.BACKOFF_MAX_SEC,
                )
                logger.warning(
                    "rate limited (status=%s), backoff %.1fs attempt %s",
                    resp.status_code,
                    backoff,
                    attempt + 1,
                )
                time.sleep(backoff)
                last_error = resp
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            last_error = e
            backoff = min(
                config.BACKOFF_BASE_SEC * (2**attempt),
                config.BACKOFF_MAX_SEC,
            )
            logger.warning(
                "request error endpoint=%s params_hash=%s attempt=%s error=%s backoff=%.1fs",
                path,
                phash,
                attempt + 1,
                e,
                backoff,
            )
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(backoff)

    if last_error:
        if hasattr(last_error, "response") and last_error.response is not None:
            raise RuntimeError(
                f"Request failed after {config.MAX_RETRIES} retries: {last_error.response.status_code} {last_error.response.text}"
            ) from last_error
        raise RuntimeError(f"Request failed after {config.MAX_RETRIES} retries: {last_error}") from last_error
    raise RuntimeError(f"Request failed after {config.MAX_RETRIES} retries")
