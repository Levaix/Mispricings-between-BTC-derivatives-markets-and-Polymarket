"""
Moneyness-based universe selection: per-day index level, in-band strikes, union of instruments.
Saves universe_per_day.json and universe_master.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from . import config as cfg
from .discover import utc_to_ms, run as discover_run
from .index_series import run as index_series_run

logger = logging.getLogger(__name__)


def _day_range(start_ms: int, end_ms: int, cadence_days: int) -> list[tuple[int, int]]:
    """Yield (day_start_ms, day_end_ms) for each day (or cadence) in [start_ms, end_ms]."""
    out = []
    dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
    while dt <= end_dt:
        day_start = int(dt.timestamp() * 1000)
        dt_next = dt + timedelta(days=cadence_days)
        day_end = int(dt_next.timestamp() * 1000) - 1
        day_end = min(day_end, end_ms)
        out.append((day_start, day_end))
        dt = dt_next
    return out


def run(force_refresh: bool = False) -> tuple[list[str], dict]:
    """
    Returns (master_instrument_names, universe_per_day_dict).
    """
    start_ms = utc_to_ms(cfg.START_UTC)
    end_ms = utc_to_ms(cfg.END_UTC)
    horizon_ms = int(getattr(cfg, "MAX_EXPIRY_HORIZON_DAYS", 180) * 24 * 60 * 60 * 1000)
    max_expiry_ms = end_ms + horizon_ms
    max_tte_ms = int(getattr(cfg, "MAX_TTE_DAYS", 90) * 24 * 60 * 60 * 1000)

    if cfg.UNIVERSE_MASTER_JSON.exists() and cfg.UNIVERSE_PER_DAY_JSON.exists() and not force_refresh:
        with open(cfg.UNIVERSE_MASTER_JSON) as f:
            master = json.load(f)
        with open(cfg.UNIVERSE_PER_DAY_JSON) as f:
            per_day = json.load(f)
        logger.info("Loaded universe: %s instruments from cache", len(master))
        return master, per_day

    instruments = discover_run(force_refresh=force_refresh)
    index_df = index_series_run(force_refresh=force_refresh)
    if index_df.empty:
        raise RuntimeError("Index series empty; cannot compute universe")

    # Build daily index level (use median or last of day for that day's band)
    index_df["date"] = pd.to_datetime(index_df["open_time"], unit="ms", utc=True).dt.date
    daily_index = index_df.groupby("date")["index_price"].agg(["median", "last"]).reset_index()
    daily_index["index_price"] = daily_index["median"]  # or "last"

    all_names = set()
    universe_per_day = {}

    day_ranges = _day_range(start_ms, end_ms, cfg.UNIVERSE_SELECTION_CADENCE_DAYS)
    for day_start_ms, day_end_ms in day_ranges:
        dt = datetime.fromtimestamp(day_start_ms / 1000.0, tz=timezone.utc)
        day_date = dt.date().isoformat()
        # Index level for this day
        day_df = daily_index[daily_index["date"].astype(str) == day_date]
        if day_df.empty:
            # Fallback: use closest index from index_df
            idx = index_df[(index_df["open_time"] >= day_start_ms) & (index_df["open_time"] <= day_end_ms)]
            if idx.empty:
                idx = index_df[index_df["open_time"] <= day_end_ms].tail(1)
            if idx.empty:
                idx = index_df.head(1)
            level = float(idx["index_price"].iloc[0]) if not idx.empty else None
        else:
            level = float(day_df["index_price"].iloc[0])
        if level is None or level <= 0:
            universe_per_day[day_date] = []
            continue
        k_min = level * cfg.MONEYNESS_MIN
        k_max = level * cfg.MONEYNESS_MAX
        selected = []
        for inv in instruments:
            # Only consider instruments that are alive during this day/cadence interval
            exp_ms = int(inv.get("expiration_timestamp") or 0)
            if exp_ms and exp_ms < day_start_ms:
                continue
            if exp_ms and exp_ms > max_expiry_ms:
                continue
            # Also require expiry not too far after this day (controls universe size)
            if exp_ms and exp_ms > day_start_ms + max_tte_ms:
                continue
            creation_ms = inv.get("creation_timestamp")
            if creation_ms is not None:
                try:
                    creation_ms = int(creation_ms)
                except Exception:
                    creation_ms = None
            if creation_ms is not None and creation_ms > day_end_ms:
                continue
            strike = inv.get("strike")
            if strike is None:
                continue
            if k_min <= strike <= k_max:
                name = inv.get("instrument_name")
                if name:
                    selected.append(name)
                    all_names.add(name)
        universe_per_day[day_date] = sorted(selected)

    master_list = sorted(all_names)
    with open(cfg.UNIVERSE_MASTER_JSON, "w") as f:
        json.dump(master_list, f, indent=0)
    with open(cfg.UNIVERSE_PER_DAY_JSON, "w") as f:
        json.dump(universe_per_day, f, indent=2)
    logger.info("Universe: %s instruments (union over %s days)", len(master_list), len(universe_per_day))
    return master_list, universe_per_day
