#!/usr/bin/env python3
"""
Inspect the main datasets (Spot, Futures, Options) and their key variables before analysis.
Prints schema, time range, row count, and sample stats for each.
Run: python scripts/inspect_datasets.py
"""
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config


def _first_parquet(path: str) -> Optional[str]:
    """Return path to first parquet file under path, or None."""
    if not os.path.isdir(path):
        return None
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".parquet"):
                return os.path.join(root, f)
    return None


def _all_parquets(path: str, max_files: int = 3) -> List[str]:
    """Return paths to first max_files parquet files under path."""
    out = []
    if not os.path.isdir(path):
        return out
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".parquet"):
                out.append(os.path.join(root, f))
                if len(out) >= max_files:
                    return out
    return out


def _time_range(df: pd.DataFrame, time_col: str = "open_time") -> str:
    if time_col not in df.columns or df.empty:
        return "n/a"
    mn = df[time_col].min()
    mx = df[time_col].max()
    if hasattr(mn, "item"):
        mn, mx = mn.item(), mx.item()
    try:
        from utils.time import ms_to_utc
        return f"{ms_to_utc(int(mn)).strftime('%Y-%m-%d %H:%M')} – {ms_to_utc(int(mx)).strftime('%Y-%m-%d %H:%M')} UTC"
    except Exception:
        return f"{mn} – {mx}"


def inspect_spot() -> None:
    print("\n" + "=" * 60)
    print("SPOT (Binance BTCUSDT 5m klines)")
    print("=" * 60)
    path = config.SPOT_PATH
    fp = _first_parquet(path)
    if not fp:
        print("  No data found at", path)
        return
    df = pd.read_parquet(fp)
    print("  Path (sample):", fp)
    print("  Shape:", df.shape)
    print("  Time range:", _time_range(df))
    print("\n  Columns (key variables):")
    for c in df.columns:
        dtype = str(df[c].dtype)
        print(f"    - {c}: {dtype}")
    print("\n  Key variables for analysis:")
    print("    - open_time / open_time_utc: bar timestamp (5m grid)")
    print("    - open, high, low, close: OHLC in quote currency (USDT)")
    print("    - volume: base asset volume (BTC)")
    print("    - number_of_trades, taker_buy_*: optional for flow analysis")
    print("    - symbol, source: identifier")
    if not df.empty and "close" in df.columns:
        print("\n  Sample stats (close): min={:.2f} max={:.2f} mean={:.2f}".format(
            df["close"].min(), df["close"].max(), df["close"].mean()))
    print()


def inspect_futures() -> None:
    print("\n" + "=" * 60)
    print("FUTURES (Binance USD-M perpetual / index / funding)")
    print("=" * 60)

    # Perp klines
    path = os.path.join(config.FUTURES_PERP_PATH, "klines")
    fp = _first_parquet(path)
    if fp:
        df = pd.read_parquet(fp)
        print("\n  --- Perp klines (tradable 5m OHLC) ---")
        print("  Path (sample):", fp)
        print("  Shape:", df.shape)
        print("  Time range:", _time_range(df))
        print("  Columns:", list(df.columns))
        print("  Key: open_time, open, high, low, close, volume, symbol, source")
    else:
        print("\n  --- Perp klines: no data at", path)

    # Mark price klines
    path_mark = os.path.join(config.FUTURES_PERP_PATH, "mark_klines")
    fp = _first_parquet(path_mark)
    if fp:
        df = pd.read_parquet(fp)
        print("\n  --- Mark price klines (5m) ---")
        print("  Path (sample):", fp)
        print("  Shape:", df.shape)
        print("  Columns:", list(df.columns))
        print("  Key: open_time, open, high, low, close (mark price)")

    # Index price klines
    path_idx = os.path.join(config.FUTURES_PERP_PATH, "index_klines")
    fp = _first_parquet(path_idx)
    if fp:
        df = pd.read_parquet(fp)
        print("\n  --- Index price klines (5m, reference for options) ---")
        print("  Path (sample):", fp)
        print("  Shape:", df.shape)
        print("  Columns:", list(df.columns))
        print("  Key: open_time, close (index price)")

    # Funding rate
    path_fr = os.path.join(config.FUTURES_PERP_PATH, "funding_rate")
    fp = _first_parquet(path_fr)
    if fp:
        df = pd.read_parquet(fp)
        time_col = "fundingTime" if "fundingTime" in df.columns else "open_time"
        print("\n  --- Funding rate ---")
        print("  Path (sample):", fp)
        print("  Shape:", df.shape)
        if time_col in df.columns:
            mn, mx = df[time_col].min(), df[time_col].max()
            try:
                from utils.time import ms_to_utc
                print("  Time range:", ms_to_utc(int(mn)).strftime("%Y-%m-%d"), "–", ms_to_utc(int(mx)).strftime("%Y-%m-%d"), "UTC")
            except Exception:
                pass
        print("  Columns:", list(df.columns))
        print("  Key: fundingTime (or open_time), fundingRate")
    else:
        print("\n  --- Funding rate: no data at", path_fr)

    print()


def inspect_options() -> None:
    print("\n" + "=" * 60)
    print("OPTIONS (Deribit BTC 5m OHLC) — archived at archive/options_legacy_20260214/")
    print("=" * 60)
    path = config.OPTIONS_KLINES_PATH
    if not os.path.isdir(path):
        print("  Options pipeline archived. Data moved to archive/options_legacy_20260214/data/")
        return
    fp = _first_parquet(path)
    if not fp:
        print("  No data found at", path)
        return
    df = pd.read_parquet(fp)
    print("  Path (sample):", fp)
    print("  Shape:", df.shape)
    print("  Time range:", _time_range(df))
    print("\n  Columns (key variables):")
    for c in df.columns:
        print(f"    - {c}: {df[c].dtype}")
    print("\n  Key variables for analysis:")
    print("    - open_time / open_time_utc: bar timestamp (5m grid)")
    print("    - instrument_name: e.g. BTC-27MAR26-125000-C")
    print("    - expiry: expiration timestamp (ms)")
    print("    - strike: strike price")
    print("    - call_put: CALL or PUT")
    print("    - open, high, low, close: option price (quote)")
    print("    - volume: contracts / base")
    print("    - source: deribit")
    if not df.empty and "close" in df.columns:
        print("\n  Sample stats (close): min={:.4f} max={:.4f} mean={:.4f}".format(
            df["close"].min(), df["close"].max(), df["close"].mean()))
    # Count instruments in sample
    if "instrument_name" in df.columns:
        n_inv = df["instrument_name"].nunique()
        print("  Instruments in this file:", n_inv)
    print()


def main() -> None:
    print("Data root:", config.DATA_ROOT)
    print("Config window:", config.START_UTC, "->", config.END_UTC)
    inspect_spot()
    inspect_futures()
    inspect_options()
    print("=" * 60)
    print("Done. Use these variable names when merging or analysing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
