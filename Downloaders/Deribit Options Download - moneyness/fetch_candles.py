"""
Fetch 5m candles for all instruments in the master universe.
Checkpoint progress; retry on failure; only drop instruments with zero data over full period.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from . import config as cfg
from .deribit_client import RateLimiter, get_tradingview_chart_data
from .discover import utc_to_ms
from .universe import run as universe_run

logger = logging.getLogger(__name__)


def _candles_to_rows(result, instrument_name, instrument_info):
    if result.get("status") != "ok":
        return []
    ticks = result.get("ticks") or []
    o = result.get("open") or []
    h = result.get("high") or []
    l = result.get("low") or []
    c = result.get("close") or []
    v = result.get("volume") or []
    strike = float(instrument_info.get("strike", 0))
    expiry_ms = int(instrument_info.get("expiration_timestamp", 0))
    call_put = str(instrument_info.get("option_type", "")).upper()
    rows = []
    for i, t in enumerate(ticks):
        rows.append({
            "timestamp_utc": int(t),
            "instrument_name": instrument_name,
            "strike": strike,
            "expiry_utc": expiry_ms,
            "call_put": call_put,
            "open": o[i] if i < len(o) else None,
            "high": h[i] if i < len(h) else None,
            "low": l[i] if i < len(l) else None,
            "close": c[i] if i < len(c) else None,
            "volume": v[i] if i < len(v) else 0,
        })
    return rows


def run(force_refresh=False):
    start_ms = utc_to_ms(cfg.START_UTC)
    end_ms = utc_to_ms(cfg.END_UTC)
    master_list, _ = universe_run(force_refresh=force_refresh)
    if not master_list:
        logger.warning("Universe empty")
        return pd.DataFrame()

    # Load instrument details for strike/expiry/call_put
    with open(cfg.INSTRUMENTS_JSON) as f:
        all_instruments = json.load(f)
    by_name = {inv["instrument_name"]: inv for inv in all_instruments if inv.get("instrument_name")}

    checkpoint = {}
    if cfg.CANDLES_CHECKPOINT_JSON.exists() and not force_refresh:
        with open(cfg.CANDLES_CHECKPOINT_JSON) as f:
            checkpoint = json.load(f)

    limiter = RateLimiter()
    # Small chunks so Deribit returns full range (API often returns only last N bars per request)
    chunk_ms = getattr(cfg, "CANDLE_CHUNK_DAYS", 1) * 24 * 60 * 60 * 1000
    all_rows = []
    failed = []
    skipped_zero = []
    _logged_diagnostic = False

    for instrument_name in master_list:
        if checkpoint.get(instrument_name) == "done":
            continue
        inv = by_name.get(instrument_name, {})
        creation_ms = inv.get("creation_timestamp")
        if creation_ms is not None:
            try:
                creation_ms = int(creation_ms)
            except Exception:
                creation_ms = None
        # Don't request periods before the instrument existed
        current = max(start_ms, creation_ms) if creation_ms else start_ms
        if current >= end_ms:
            skipped_zero.append(instrument_name)
            checkpoint[instrument_name] = "done"
            continue
        inst_rows = []
        while current < end_ms:
            chunk_end = min(current + chunk_ms, end_ms)
            try:
                result = get_tradingview_chart_data(
                    instrument_name,
                    current,
                    chunk_end,
                    resolution=cfg.RESOLUTION_MINUTES,
                    rate_limiter=limiter,
                )
                rows = _candles_to_rows(result, instrument_name, inv)
                if not _logged_diagnostic and rows:
                    ticks = result.get("ticks") or []
                    if ticks:
                        from datetime import datetime, timezone
                        t0 = datetime.fromtimestamp(ticks[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                        t1 = datetime.fromtimestamp(ticks[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                        req0 = datetime.fromtimestamp(current / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                        logger.info("Diagnostic: first chunk requested start=%s, got %s bars from %s to %s", req0, len(ticks), t0, t1)
                    _logged_diagnostic = True
                elif not _logged_diagnostic and current == start_ms:
                    status = result.get("status", "?")
                    nticks = len(result.get("ticks") or [])
                    logger.info("Diagnostic: first chunk (requested start=Jan1) got status=%s, ticks=%s", status, nticks)
                    _logged_diagnostic = True
                inst_rows.extend(rows)
            except Exception as e:
                logger.warning("Fetch %s failed: %s", instrument_name, e)
                failed.append({"instrument": instrument_name, "error": str(e)})
                break
            current = chunk_end

        if failed and failed[-1].get("instrument") == instrument_name:
            continue
        if not inst_rows:
            skipped_zero.append(instrument_name)
        else:
            all_rows.extend(inst_rows)
        checkpoint[instrument_name] = "done"
        if len(checkpoint) % 50 == 0:
            with open(cfg.CANDLES_CHECKPOINT_JSON, "w") as f:
                json.dump(checkpoint, f)

    with open(cfg.CANDLES_FAILURES_JSON, "w") as f:
        json.dump({"failed": failed, "skipped_zero_data": skipped_zero}, f, indent=2)

    if failed:
        logger.warning("Failed instruments: %s", len(failed))
    if skipped_zero:
        logger.info("Skipped (zero data) instruments: %s", len(skipped_zero))

    if not all_rows:
        # Don't persist checkpoint when we have no data: next run should re-fetch
        skipped_by_checkpoint = sum(1 for n in master_list if checkpoint.get(n) == "done")
        if skipped_by_checkpoint == len(master_list):
            logger.warning(
                "All %s instruments were skipped (already in checkpoint) but no candle data in memory. "
                "Clearing checkpoint so next run will re-fetch. Run again without --force-refresh.",
                len(master_list),
            )
            if cfg.CANDLES_CHECKPOINT_JSON.exists():
                cfg.CANDLES_CHECKPOINT_JSON.unlink()
        else:
            with open(cfg.CANDLES_CHECKPOINT_JSON, "w") as f:
                json.dump(checkpoint, f)
        return pd.DataFrame()

    with open(cfg.CANDLES_CHECKPOINT_JSON, "w") as f:
        json.dump(checkpoint, f)
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["instrument_name", "timestamp_utc"], keep="last")
    df = df.sort_values(["instrument_name", "timestamp_utc"])
    return df
