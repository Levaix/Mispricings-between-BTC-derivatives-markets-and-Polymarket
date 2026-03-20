"""
Align raw quote snapshots to 5-minute UTC grid.
- Floor timestamps to 5m boundary.
- Within-instrument forward-fill only after first observation; no fill before first obs.
- No cross-instrument/cross-strike fill.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BAR_MS = 5 * 60 * 1000


def floor_to_5m(ms: pd.Series) -> pd.Series:
    """Floor millisecond timestamps to 5-minute boundary."""
    return (ms.astype("int64") // BAR_MS) * BAR_MS


def align_raw_to_5m(
    df: pd.DataFrame,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Align raw quotes to 5m grid. Expects columns: open_time, instrument_name, strike, expiry, call_put,
    best_bid_price, best_ask_price, (optional best_bid_size, best_ask_size), index_price.
    Returns one row per (open_time_5m, instrument_name) with forward-filled quote fields;
    rows before first observation for that instrument have NaN (no fill before first obs).
    """
    if df.empty or "instrument_name" not in df.columns or "open_time" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["open_time_5m"] = floor_to_5m(df["open_time"])
    # Within each instrument, take last observation per 5m bin (if multiple)
    agg_cols = [c for c in ["best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size", "index_price"]
                if c in df.columns]
    group_cols = ["instrument_name", "open_time_5m"]
    keep_meta = ["strike", "expiry", "call_put"]
    for c in keep_meta:
        if c in df.columns:
            agg_cols.append(c)
    # Take last in bin (arbitrary but consistent)
    reduced = df.groupby(group_cols, as_index=False).agg({c: "last" for c in agg_cols if c in df.columns})
    reduced = reduced.rename(columns={"open_time_5m": "open_time"})
    # Build full grid if start/end given
    if start_ms is not None and end_ms is not None:
        grid_ms = np.arange((start_ms // BAR_MS) * BAR_MS, end_ms + 1, BAR_MS)
        instruments = reduced["instrument_name"].unique().tolist()
        full = pd.DataFrame(
            [(t, inv) for t in grid_ms for inv in instruments],
            columns=["open_time", "instrument_name"],
        )
        reduced = full.merge(
            reduced,
            on=["open_time", "instrument_name"],
            how="left",
        )
        # Forward-fill within each instrument; then set NaN for rows before first observation
        reduced = reduced.sort_values(["instrument_name", "open_time"])
        quote_cols = [c for c in ["best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size", "index_price", "strike", "expiry", "call_put"]
                     if c in reduced.columns]
        first_obs = reduced.dropna(subset=["best_bid_price"] if "best_bid_price" in reduced.columns else []).groupby("instrument_name")["open_time"].min()
        reduced = reduced.merge(first_obs.rename("first_obs_open_time"), on="instrument_name", how="left")
        for c in quote_cols:
            reduced[c] = reduced.groupby("instrument_name")[c].ffill()
        # No fill before first observation: zero only quote/price columns, not strike/expiry/call_put
        quote_only = [c for c in ["best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size", "index_price"] if c in reduced.columns]
        reduced.loc[reduced["open_time"] < reduced["first_obs_open_time"], quote_only] = np.nan
        reduced = reduced.drop(columns=["first_obs_open_time"], errors="ignore")
    return reduced


def load_raw_quotes(raw_path: Path) -> pd.DataFrame:
    """Load raw quotes from a directory (all parquet) or single parquet file."""
    raw_path = Path(raw_path)
    if raw_path.suffix == ".parquet":
        return pd.read_parquet(raw_path)
    if raw_path.is_dir():
        files = list(raw_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()
