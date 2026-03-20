"""Discover all BTC option instruments from Deribit active in the analysis window."""
import json
import logging
from datetime import datetime, timezone

from . import config as cfg
from .deribit_client import RateLimiter, list_instruments

logger = logging.getLogger(__name__)


def utc_to_ms(s):
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def run(force_refresh=False):
    start_ms = utc_to_ms(cfg.START_UTC)
    end_ms = utc_to_ms(cfg.END_UTC)
    horizon_ms = int(getattr(cfg, "MAX_EXPIRY_HORIZON_DAYS", 180) * 24 * 60 * 60 * 1000)
    max_expiry_ms = end_ms + horizon_ms

    if cfg.INSTRUMENTS_JSON.exists() and not force_refresh:
        with open(cfg.INSTRUMENTS_JSON) as f:
            out = json.load(f)
        logger.info("Loaded %s instruments from %s", len(out), cfg.INSTRUMENTS_JSON)
        return out

    limiter = RateLimiter()
    # IMPORTANT: Need both expired and non-expired instruments. Otherwise (e.g. on 2026-03-01)
    # we only see already-expired series like 1MAR26, which won't cover Jan–Feb well.
    raw_expired = list_instruments(currency="BTC", kind="option", expired=True, rate_limiter=limiter)
    raw_live = list_instruments(currency="BTC", kind="option", expired=False, rate_limiter=limiter)
    raw = (raw_expired or []) + (raw_live or [])

    # Deduplicate by instrument_name
    by_name = {}
    for inv in raw:
        name = inv.get("instrument_name")
        if not name:
            continue
        by_name[name] = inv

    # Keep instruments that overlap the window:
    # - expiration >= window start
    # - creation <= window end (if creation_timestamp provided)
    instruments = []
    for inv in by_name.values():
        exp_ms = int(inv.get("expiration_timestamp") or 0)
        if exp_ms < start_ms:
            continue
        if exp_ms > max_expiry_ms:
            continue
        creation_ms = inv.get("creation_timestamp")
        if creation_ms is not None:
            try:
                creation_ms = int(creation_ms)
            except Exception:
                creation_ms = None
        if creation_ms is not None and creation_ms > end_ms:
            continue
        instruments.append(inv)

    instruments.sort(key=lambda x: (x.get("expiration_timestamp", 0), x.get("strike", 0), x.get("instrument_name", "")))
    with open(cfg.INSTRUMENTS_JSON, "w") as f:
        json.dump(instruments, f, indent=0)
    logger.info("Discovered %s BTC option instruments, saved to %s", len(instruments), cfg.INSTRUMENTS_JSON)
    return instruments
