"""Fetch Binance USD-M perpetual (BTCUSDT) 5m OHLCV, mark price klines, and funding rate.

Thesis window (UTC): 2026-01-01 00:00:00 -> 2026-02-28 23:59:59, 5-minute.

Outputs (DATA only) under `Actual data/Futures Binance/`:
- perp_ohlcv_5m.parquet   (timestamp_utc, open, high, low, close, volume)
- perp_mark_5m.parquet    (timestamp_utc, mark_open, mark_high, mark_low, mark_close)
- funding_rate.parquet   (timestamp_utc, funding_rate; native 8h timestamps)
- qa_futures_binance.json / qa_futures_binance.md

Pipeline code stays here; only outputs go to Actual data/Futures Binance/.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FAPI_BASE = "https://fapi.binance.com"
DEFAULT_SYMBOL = "BTCUSDT"
INTERVAL_5M = "5m"
THESIS_START_UTC = "2026-01-01T00:00:00"
THESIS_END_UTC = "2026-02-28T23:59:59"
STEP_MS_5M = 5 * 60 * 1000
KLINES_LIMIT = 1500
FUNDING_LIMIT = 1000


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Binance Futures BTCUSDT 5m + funding.")
    p.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    p.add_argument("--start_utc", type=str, default=THESIS_START_UTC)
    p.add_argument("--end_utc", type=str, default=THESIS_END_UTC)
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_root() / "Actual data" / "Futures Binance"),
        help="Output directory for parquet and QA.",
    )
    return p.parse_args()


def _utc_to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _fetch_klines(start_ms: int, end_ms: int, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Fetch perp OHLCV klines from fapi/v1/klines."""
    all_rows: List[List[Any]] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL_5M,
            "startTime": cur,
            "endTime": end_ms,
            "limit": KLINES_LIMIT,
        }
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/klines", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_open = int(data[-1][0])
        next_cur = last_open + STEP_MS_5M
        if next_cur <= cur:
            next_cur = cur + STEP_MS_5M
        cur = next_cur
        if len(data) < KLINES_LIMIT:
            break
    if not all_rows:
        return pd.DataFrame()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df


def _fetch_mark_klines(start_ms: int, end_ms: int, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Fetch mark price klines from futures/data/markPriceKlines."""
    all_rows: List[List[Any]] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL_5M,
            "startTime": cur,
            "endTime": end_ms,
            "limit": KLINES_LIMIT,
        }
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/markPriceKlines", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_open = int(data[-1][0])
        next_cur = last_open + STEP_MS_5M
        if next_cur <= cur:
            next_cur = cur + STEP_MS_5M
        cur = next_cur
        if len(data) < KLINES_LIMIT:
            break
    if not all_rows:
        return pd.DataFrame()
    # Response: same shape as klines [ openTime, open, high, low, close, closeTime, ... ]; take first 6
    rows_6 = [r[:6] for r in all_rows]
    df = pd.DataFrame(
        rows_6,
        columns=["open_time", "mark_open", "mark_high", "mark_low", "mark_close", "close_time"],
    )
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in ["mark_open", "mark_high", "mark_low", "mark_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df


def _fetch_funding_rate(start_ms: int, end_ms: int, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Fetch funding rate history (native 8h interval)."""
    all_rows: List[Dict[str, Any]] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cur,
            "endTime": end_ms,
            "limit": FUNDING_LIMIT,
        }
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/fundingRate", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_ft = int(data[-1]["fundingTime"])
        cur = last_ft + 1
        if len(data) < FUNDING_LIMIT:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["symbol"] = df.get("symbol", symbol)
    return df


def _qa_ohlcv_5m(df: pd.DataFrame, start_ms: int, end_ms: int, name: str) -> Dict[str, Any]:
    """QA for 5m OHLCV: no duplicates, UTC, 5-min aligned, coverage, prices>0, volume>=0."""
    qa: Dict[str, Any] = {"dataset": name, "rows": len(df)}
    if df.empty:
        qa["status"] = "empty"
        return qa
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    qa["start_timestamp_utc"] = df["timestamp_utc"].iloc[0].isoformat()
    qa["end_timestamp_utc"] = df["timestamp_utc"].iloc[-1].isoformat()

    start_dt = _ms_to_utc(start_ms).replace(second=0, microsecond=0)
    end_bar_dt = _ms_to_utc(end_ms).replace(second=0, microsecond=0)
    end_bar_dt = end_bar_dt - timedelta(minutes=end_bar_dt.minute % 5)
    expected = pd.date_range(start=start_dt, end=end_bar_dt, freq="5min", tz="UTC")
    qa["expected_bars"] = int(len(expected))
    ts = df["timestamp_utc"]
    unique_ts = ts.drop_duplicates()
    missing = expected.difference(unique_ts)
    qa["missing_bars"] = int(len(missing))
    if len(missing) > 0:
        qa["missing_examples"] = (
            [missing[0].isoformat(), missing[-1].isoformat()]
            if len(missing) > 1
            else [missing[0].isoformat()]
        )

    qa["duplicate_timestamps"] = int(ts.duplicated().sum())
    minutes = ts.dt.minute
    seconds = ts.dt.second
    qa["bad_alignment_count"] = int((((minutes % 5) != 0) | (seconds != 0)).sum())

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            qa[f"bad_{c}_count"] = int(((df[c] <= 0) | df[c].isna()).sum())
    if "volume" in df.columns:
        qa["bad_volume_count"] = int(((df["volume"] < 0) | df["volume"].isna()).sum())
    if "close" in df.columns:
        qa["close_min"] = float(df["close"].min())
        qa["close_max"] = float(df["close"].max())
    qa["status"] = "ok"
    return qa


def _qa_funding(df: pd.DataFrame, start_ms: int, end_ms: int) -> Dict[str, Any]:
    """QA for funding: timestamps monotonic, no obvious gaps; summary."""
    qa: Dict[str, Any] = {"dataset": "funding_rate", "rows": len(df)}
    if df.empty:
        qa["status"] = "empty"
        return qa
    df = df.sort_values("fundingTime").reset_index(drop=True)
    qa["start_timestamp_utc"] = df["timestamp_utc"].iloc[0].isoformat()
    qa["end_timestamp_utc"] = df["timestamp_utc"].iloc[-1].isoformat()
    qa["duplicate_fundingTime"] = int(df["fundingTime"].duplicated().sum())
    # Funding every 8h; check monotonic
    diffs = df["fundingTime"].diff().dropna()
    qa["monotonic"] = bool((diffs >= 0).all())
    # Typical gap 8h = 8*3600*1000 ms
    expected_interval_ms = 8 * 3600 * 1000
    gap_ok = (diffs.abs() - expected_interval_ms).abs() < 60_000  # allow 1 min tolerance
    qa["gaps_count"] = int((~gap_ok & (diffs > 0)).sum()) if len(diffs) else 0
    qa["funding_rate_min"] = float(df["fundingRate"].min())
    qa["funding_rate_max"] = float(df["fundingRate"].max())
    qa["status"] = "ok"
    return qa


def run_futures_fetch(
    symbol: str,
    start_utc: str,
    end_utc: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    start_ms = _utc_to_ms(start_utc)
    end_ms = _utc_to_ms(end_utc)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "symbol": symbol,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "perp_ohlcv_5m": {},
        "perp_mark_5m": {},
        "funding_rate": {},
    }

    # 1) Perp OHLCV 5m
    logger.info("Fetching perp OHLCV 5m from %s to %s", _ms_to_utc(start_ms), _ms_to_utc(end_ms))
    perp = _fetch_klines(start_ms, end_ms, symbol)
    if not perp.empty:
        perp = perp[(perp["open_time"] >= start_ms) & (perp["open_time"] <= end_ms)]
        perp = perp.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    report["perp_ohlcv_5m"] = _qa_ohlcv_5m(perp, start_ms, end_ms, "perp_ohlcv_5m")
    out_cols = ["timestamp_utc", "open", "high", "low", "close", "volume"]
    perp_out = perp[[c for c in out_cols if c in perp.columns]].copy()
    perp_out.to_parquet(out_dir / "perp_ohlcv_5m.parquet", index=False)
    logger.info("Wrote perp_ohlcv_5m.parquet (%d rows)", len(perp_out))

    # 2) Mark price 5m
    logger.info("Fetching mark price 5m")
    mark = _fetch_mark_klines(start_ms, end_ms, symbol)
    if not mark.empty:
        mark = mark[(mark["open_time"] >= start_ms) & (mark["open_time"] <= end_ms)]
        mark = mark.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    report["perp_mark_5m"] = _qa_ohlcv_5m(
        mark.rename(columns={"mark_open": "open", "mark_high": "high", "mark_low": "low", "mark_close": "close"}),
        start_ms, end_ms, "perp_mark_5m",
    ) if not mark.empty else {"dataset": "perp_mark_5m", "rows": 0, "status": "empty"}
    if not mark.empty:
        mark_out = mark[["timestamp_utc", "mark_open", "mark_high", "mark_low", "mark_close"]].copy()
        mark_out.to_parquet(out_dir / "perp_mark_5m.parquet", index=False)
        logger.info("Wrote perp_mark_5m.parquet (%d rows)", len(mark_out))

    # 3) Funding rate (native interval)
    logger.info("Fetching funding rate")
    funding = _fetch_funding_rate(start_ms, end_ms, symbol)
    if not funding.empty:
        funding = funding[(funding["fundingTime"] >= start_ms) & (funding["fundingTime"] <= end_ms)]
        funding = funding.sort_values("fundingTime").drop_duplicates(subset=["fundingTime"], keep="last")
    report["funding_rate"] = _qa_funding(funding, start_ms, end_ms)
    funding_out = funding[["timestamp_utc", "fundingTime", "fundingRate", "symbol"]].copy() if not funding.empty else pd.DataFrame(columns=["timestamp_utc", "fundingTime", "fundingRate", "symbol"])
    funding_out.to_parquet(out_dir / "funding_rate.parquet", index=False)
    logger.info("Wrote funding_rate.parquet (%d rows)", len(funding_out))

    # QA report
    qa_path = out_dir / "qa_futures_binance.json"
    with qa_path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", qa_path)

    md_path = out_dir / "qa_futures_binance.md"
    with md_path.open("w") as f:
        f.write("# Futures Binance QA report\n\n")
        f.write(f"- Symbol: {symbol}\n- Window: {start_utc} to {end_utc}\n\n")
        f.write("## Perp OHLCV 5m\n\n")
        for k, v in report["perp_ohlcv_5m"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Perp Mark 5m\n\n")
        for k, v in report["perp_mark_5m"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Funding rate\n\n")
        for k, v in report["funding_rate"].items():
            f.write(f"- {k}: {v}\n")
    logger.info("Wrote %s", md_path)

    return report


def main() -> None:
    args = _parse_args()
    run_futures_fetch(
        symbol=args.symbol,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
