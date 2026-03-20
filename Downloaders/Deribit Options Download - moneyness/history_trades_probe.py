"""Probe Deribit historical trades API (history.deribit.com) for BTC options.

Goal:
- Verify that we can retrieve BTC option trades for Jan 1 – Feb 28, 2026 (UTC),
  including expired contracts, using the bulk historical archive host:
  https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time

This module is **standalone** and does NOT modify the existing pipeline.

Outputs:
- Parquet partitions under data/options_v2_deribit_moneyness/history_trades/date=YYYY-MM-DD/trades.parquet
- Diagnostics printed to stdout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from . import config as cfg
from .discover import utc_to_ms
from .deribit_client import RateLimiter

logger = logging.getLogger(__name__)

HISTORY_BASE_URL = "https://history.deribit.com/api/v2"


@dataclass
class HistoryClient:
    base_url: str = HISTORY_BASE_URL
    rate_limiter: Optional[RateLimiter] = None

    def _request(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call history.deribit.com public method; return JSON-RPC result."""
        url = f"{self.base_url.rstrip('/')}/public/{method}"
        limiter = self.rate_limiter or RateLimiter(
            min_interval_sec=cfg.SLEEP_BETWEEN_CALLS_MS / 1000.0
        )
        last_error: Optional[Exception] = None
        for attempt in range(cfg.MAX_RETRIES):
            limiter.wait()
            try:
                resp = requests.get(url, params=params, timeout=cfg.REQUEST_TIMEOUT_SEC)
                if resp.status_code in (429, 418, 503, 502, 500):
                    backoff = min(
                        cfg.BACKOFF_BASE_SEC * (2**attempt), cfg.BACKOFF_MAX_SEC
                    )
                    logger.warning(
                        "history %s status=%s backoff=%.1fs",
                        method,
                        resp.status_code,
                        backoff,
                    )
                    last_error = RuntimeError(f"{resp.status_code}: {resp.text}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"Deribit history API error: {data['error']}")
                result = data.get("result") or {}
                return result
            except Exception as e:  # requests or JSON
                last_error = e
                backoff = min(
                    cfg.BACKOFF_BASE_SEC * (2**attempt), cfg.BACKOFF_MAX_SEC
                )
                logger.warning(
                    "history %s request error: %s backoff=%.1fs", method, e, backoff
                )
        if last_error is not None:
            raise RuntimeError(
                f"history {method} failed after {cfg.MAX_RETRIES} retries: {last_error}"
            )
        raise RuntimeError(f"history {method} failed without result")

    def get_last_trades_by_currency_and_time(
        self,
        currency: str,
        kind: str,
        start_ts_ms: int,
        end_ts_ms: int,
        count: int = 1000,
        sorting: str = "asc",
    ) -> Dict[str, Any]:
        """Wrapper for public/get_last_trades_by_currency_and_time on history host."""
        params = {
            "currency": currency,
            "kind": kind,
            "start_timestamp": start_ts_ms,
            "end_timestamp": end_ts_ms,
            "count": count,
            "sorting": sorting,
        }
        return self._request("get_last_trades_by_currency_and_time", params)


def _date_range(start_dt: datetime, end_dt: datetime) -> List[datetime]:
    dates: List[datetime] = []
    cur = start_dt
    while cur <= end_dt:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def fetch_trades_for_window() -> Path:
    """Fetch BTC option trades for Jan 1–Feb 28, 2026 via history host; return root dir."""
    start_ms = utc_to_ms("2026-01-01T00:00:00")
    end_ms = utc_to_ms("2026-02-28T23:59:59")
    start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 2, 28, 0, 0, 0, tzinfo=timezone.utc)

    out_root = cfg.DATA_DIR / "history_trades"
    out_root.mkdir(parents=True, exist_ok=True)

    client = HistoryClient()

    for day_dt in _date_range(start_dt, end_dt):
        day_str = day_dt.date().isoformat()
        day_start_ms = int(day_dt.timestamp() * 1000)
        day_end_ms = int((day_dt + timedelta(days=1)).timestamp() * 1000) - 1
        if day_end_ms > end_ms:
            day_end_ms = end_ms

        day_dir = out_root / f"date={day_str}"
        out_path = day_dir / "trades.parquet"
        if out_path.exists():
            logger.info("Skipping %s (already exists)", day_str)
            continue

        logger.info("Fetching trades for %s (%s – %s)", day_str, day_start_ms, day_end_ms)
        all_rows: List[Dict[str, Any]] = []
        cursor_ts = day_start_ms
        while cursor_ts <= day_end_ms:
            result = client.get_last_trades_by_currency_and_time(
                currency="BTC",
                kind="option",
                start_ts_ms=cursor_ts,
                end_ts_ms=day_end_ms,
                count=1000,
                sorting="asc",
            )
            trades = result.get("trades") or []
            if not trades:
                break
            all_rows.extend(trades)
            # Pagination by time: advance cursor to just after last trade timestamp
            last_ts = int(trades[-1]["timestamp"])
            if last_ts >= day_end_ms:
                break
            cursor_ts = last_ts + 1
            if not result.get("has_more", False):
                break

        if not all_rows:
            logger.info("No trades for %s", day_str)
            continue

        df = pd.DataFrame(all_rows)
        day_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s rows for %s to %s", len(df), day_str, out_path)

    return out_root


def analyze_trades(root: Path) -> None:
    """Analyze downloaded trades and print diagnostics."""
    if not root.exists():
        logger.warning("History trades dir %s not found", root)
        return

    files = list(root.rglob("trades.parquet"))
    if not files:
        logger.warning("No trades.parquet files found under %s", root)
        return

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        logger.warning("Trades frame is empty")
        return

    # Total trades
    total_trades = len(df)

    # Unique instruments
    instruments = sorted(df["instrument_name"].unique())
    n_instr = len(instruments)

    # Fetch instrument metadata for expiries via public/get_instrument
    from .deribit_client import _request as public_request  # reuse existing helper

    expiries: Dict[str, int] = {}
    for name in instruments:
        try:
            result = public_request("get_instrument", {"instrument_name": name})
            info = result or {}
            exp_ms = int(info.get("expiration_timestamp") or 0)
            expiries[name] = exp_ms
        except Exception as e:
            logger.warning("get_instrument %s failed: %s", name, e)

    # Expiries covered
    exp_values = [v for v in expiries.values() if v > 0]
    exp_dates = sorted(
        {datetime.fromtimestamp(v / 1000, tz=timezone.utc).date().isoformat() for v in exp_values}
    )

    # Expired contracts in Jan/Feb 2026
    jan1 = datetime(2026, 1, 1, tzinfo=timezone.utc).date()
    feb28 = datetime(2026, 2, 28, tzinfo=timezone.utc).date()
    expired_jan_feb = [
        name
        for name, v in expiries.items()
        if v > 0
        and jan1 <= datetime.fromtimestamp(v / 1000, tz=timezone.utc).date() <= feb28
    ]

    # Diagnostics
    print("=== Deribit history trades probe (BTC options, Jan–Feb 2026) ===")
    print("Total trades retrieved:", total_trades)
    print("Unique instruments:", n_instr)
    print("Expiry dates covered (unique):", exp_dates)
    print("Expired contracts in Jan/Feb 2026:", len(expired_jan_feb))
    print("Sample expired instruments (up to 3):", expired_jan_feb[:3])


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = fetch_trades_for_window()
    analyze_trades(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

