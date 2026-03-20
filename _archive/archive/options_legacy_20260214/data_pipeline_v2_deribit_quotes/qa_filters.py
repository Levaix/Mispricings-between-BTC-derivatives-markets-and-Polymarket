"""
Quote sanity filters: bid>0, ask>0, ask>=bid; optional max spread (configurable).
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def apply_quote_filters(
    df: pd.DataFrame,
    max_spread_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Keep rows with best_bid_price > 0, best_ask_price > 0, best_ask_price >= best_bid_price.
    If max_spread_pct is set, drop rows where (ask - bid) / mid > max_spread_pct / 100.
    """
    if df.empty:
        return df
    out = df.copy()
    bid = out["best_bid_price"].astype(float)
    ask = out["best_ask_price"].astype(float)
    mask = (bid > 0) & (ask > 0) & (ask >= bid)
    n_before = len(out)
    out = out.loc[mask].copy()
    n_after = len(out)
    if n_before > n_after:
        logger.info("Quote filter: dropped %s rows (bid>0, ask>0, ask>=bid)", n_before - n_after)
    if max_spread_pct is not None and not out.empty:
        mid = (out["best_bid_price"].astype(float) + out["best_ask_price"].astype(float)) / 2
        spread = out["best_ask_price"].astype(float) - out["best_bid_price"].astype(float)
        spread_pct = (spread / mid.replace(0, float("nan"))) * 100
        mask2 = spread_pct <= max_spread_pct
        n_drop = (~mask2).sum()
        if n_drop > 0:
            logger.info("Spread filter: dropped %s rows (spread > %.1f%%)", n_drop, max_spread_pct)
        out = out.loc[mask2].copy()
    return out
