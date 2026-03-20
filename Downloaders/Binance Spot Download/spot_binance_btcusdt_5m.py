"""Fetch Binance BTCUSDT 5-minute spot data for the thesis window and run QA.

Window: 2026-01-01 00:00:00 UTC -> 2026-02-28 23:59:59 UTC

Outputs under `Actual data/Spot Binance/` (relative to repo root):
- spot_5m.parquet
- spot_5m_qa.json
- spot_5m_qa.md

The module is CLI-runnable:

  python -m pipeline.spot_binance_btcusdt_5m

Optional arguments allow overriding symbol, start/end, and output_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BINANCE_BASE_URL = "https://api.binance.com"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
THESIS_START_UTC = "2026-01-01T00:00:00"
THESIS_END_UTC = "2026-02-28T23:59:59"


def _root() -> Path:
    """Repository root (same convention as options config)."""
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Binance BTCUSDT 5m spot data with QA.")
    p.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    p.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    p.add_argument(
        "--start_utc",
        type=str,
        default=THESIS_START_UTC,
        help="Start timestamp (UTC, e.g. 2026-01-01T00:00:00)",
    )
    p.add_argument(
        "--end_utc",
        type=str,
        default=THESIS_END_UTC,
        help="End timestamp (UTC, e.g. 2026-02-28T23:59:59)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_root() / "Actual data" / "Spot Binance"),
        help="Output directory for spot_5m.* and QA reports.",
    )
    return p.parse_args()


def _utc_to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


@dataclass
class SpotFetcher:
    symbol: str
    interval: str = DEFAULT_INTERVAL
    session: requests.Session | None = None

    def _request(self, params: Dict[str, Any]) -> List[Any]:
        url = f"{BINANCE_BASE_URL}/api/v3/klines"
        s = self.session or requests.Session()
        resp = s.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_window(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch klines for [start_ms, end_ms] (open_time in ms)."""
        all_rows: List[List[Any]] = []
        cur = start_ms
        step_ms = 5 * 60 * 1000  # 5 minutes
        limit = 1000
        while cur < end_ms:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": limit,
            }
            data = self._request(params)
            if not data:
                logger.info("No more data returned from Binance at %s", _ms_to_utc(cur))
                break
            all_rows.extend(data)
            last_open = int(data[-1][0])
            next_cur = last_open + step_ms
            if next_cur <= cur:
                # Safety to avoid infinite loop if Binance returns overlapping window.
                next_cur = cur + step_ms
            cur = next_cur
            if len(data) < limit:
                # Reached the end of available data.
                break
        if not all_rows:
            return pd.DataFrame()

        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(all_rows, columns=cols)
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["symbol"] = self.symbol
        return df


def _qa_spot_5m(df: pd.DataFrame, start_ms: int, end_ms: int) -> Dict[str, Any]:
    """Run required QA checks and return a summary dict."""
    qa: Dict[str, Any] = {}
    qa["rows"] = int(len(df))
    if df.empty:
        qa["status"] = "empty"
        return qa

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    qa["start_timestamp_utc"] = df["timestamp_utc"].iloc[0].isoformat()
    qa["end_timestamp_utc"] = df["timestamp_utc"].iloc[-1].isoformat()

    # Expected 5-min grid across the thesis window.
    start_dt = _ms_to_utc(start_ms).replace(second=0, microsecond=0)
    # Last bar should start at 23:55 and end at 23:59:59
    end_bar_dt = _ms_to_utc(end_ms).replace(second=0, microsecond=0)
    # Align end_bar_dt down to nearest 5-minute boundary.
    end_bar_dt = (end_bar_dt - timedelta(minutes=end_bar_dt.minute % 5))

    expected_index = pd.date_range(start=start_dt, end=end_bar_dt, freq="5min", tz="UTC")
    qa["expected_bars"] = int(len(expected_index))

    ts = df["timestamp_utc"]
    unique_ts = ts.drop_duplicates()
    missing = expected_index.difference(unique_ts)
    qa["missing_bars"] = int(len(missing))
    if len(missing) > 0:
        qa["missing_examples"] = (
            [missing[0].isoformat(), missing[-1].isoformat()]
            if len(missing) > 1
            else [missing[0].isoformat()]
        )

    # No duplicate timestamps
    dup_count = int(ts.duplicated().sum())
    qa["duplicate_timestamps"] = dup_count

    # Timestamps UTC & 5-min aligned
    minutes = ts.dt.minute
    seconds = ts.dt.second
    bad_align = ((minutes % 5) != 0) | (seconds != 0)
    qa["bad_alignment_count"] = int(bad_align.sum())

    # Prices > 0, volume >= 0
    price_cols = ["open", "high", "low", "close"]
    for c in price_cols:
        bad = (df[c] <= 0) | df[c].isna()
        qa[f"bad_{c}_count"] = int(bad.sum())
    bad_vol = (df["volume"] < 0) | df["volume"].isna()
    qa["bad_volume_count"] = int(bad_vol.sum())

    # Basic stats
    qa["close_min"] = float(df["close"].min())
    qa["close_max"] = float(df["close"].max())
    qa["volume_min"] = float(df["volume"].min())
    qa["volume_max"] = float(df["volume"].max())

    qa["status"] = "ok"
    return qa


def run_spot_fetch(
    symbol: str,
    interval: str,
    start_utc: str,
    end_utc: str,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    start_ms = _utc_to_ms(start_utc)
    end_ms = _utc_to_ms(end_utc)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetcher = SpotFetcher(symbol=symbol, interval=interval)
    logger.info(
        "Fetching %s %s klines from %s to %s",
        symbol,
        interval,
        _ms_to_utc(start_ms),
        _ms_to_utc(end_ms),
    )
    df = fetcher.fetch_window(start_ms=start_ms, end_ms=end_ms)
    if df.empty:
        logger.warning("No data returned for %s %s", symbol, interval)
        qa = _qa_spot_5m(df, start_ms, end_ms)
        return df, qa

    # Ensure 5-min alignment and trim to [start_ms, end_ms]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    mask_window = (df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)
    df = df.loc[mask_window].copy()

    # Drop duplicates by timestamp_utc, keep last
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")

    # Compute log returns from close
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    qa = _qa_spot_5m(df, start_ms, end_ms)

    # Save parquet
    parquet_path = out_dir / "spot_5m.parquet"
    df_out = df[["timestamp_utc", "open", "high", "low", "close", "volume", "log_return", "symbol"]].copy()
    df_out.to_parquet(parquet_path, index=False)
    logger.info("Wrote spot data: %s (%d rows)", parquet_path, len(df_out))

    # Save QA JSON and MD
    qa_json = out_dir / "spot_5m_qa.json"
    with qa_json.open("w") as f:
        json.dump(qa, f, indent=2)
    logger.info("Wrote QA JSON: %s", qa_json)

    qa_md = out_dir / "spot_5m_qa.md"
    with qa_md.open("w") as f:
        f.write("# Spot BTCUSDT 5m QA report\n\n")
        f.write(f"- Symbol: {symbol}\n")
        f.write(f"- Interval: {interval}\n")
        f.write(f"- Window UTC: {start_utc} to {end_utc}\n")
        f.write(f"- Rows: {qa.get('rows', 0)}\n")
        f.write(f"- Expected 5m bars: {qa.get('expected_bars', 0)}\n")
        f.write(f"- Missing bars: {qa.get('missing_bars', 0)}\n")
        f.write(f"- Duplicate timestamps: {qa.get('duplicate_timestamps', 0)}\n")
        f.write(f"- Bad alignment count: {qa.get('bad_alignment_count', 0)}\n")
        f.write(f"- Bad open count: {qa.get('bad_open_count', 0)}\n")
        f.write(f"- Bad high count: {qa.get('bad_high_count', 0)}\n")
        f.write(f"- Bad low count: {qa.get('bad_low_count', 0)}\n")
        f.write(f"- Bad close count: {qa.get('bad_close_count', 0)}\n")
        f.write(f"- Bad volume count: {qa.get('bad_volume_count', 0)}\n")
        f.write(f"- Close min/max: {qa.get('close_min', 0.0)} / {qa.get('close_max', 0.0)}\n")
        f.write(f"- Volume min/max: {qa.get('volume_min', 0.0)} / {qa.get('volume_max', 0.0)}\n")
    logger.info("Wrote QA MD: %s", qa_md)

    return df_out, qa


def main() -> None:
    args = _parse_args()
    run_spot_fetch(
        symbol=args.symbol,
        interval=args.interval,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

