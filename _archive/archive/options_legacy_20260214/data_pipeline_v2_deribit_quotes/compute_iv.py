"""
Compute implied volatility from mid quote (bid+ask)/2 using Black-Scholes.
Deribit BTC options: we use Black-Scholes with S = index price (underlying), K = strike,
r = 0 (or config), no dividend. This matches Deribit's mark_iv methodology (spot-style).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ACT/365
SEC_PER_YEAR = 365.25 * 24 * 3600
MS_PER_YEAR = SEC_PER_YEAR * 1000

SIGMA_MIN, SIGMA_MAX = 0.05, 3.0

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
except ImportError:
    norm = None
    brentq = None


def _bs_call(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def _bs_put(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def invert_iv_call(price: float, S: float, K: float, tau: float, sigma_lo: float = 0.001, sigma_hi: float = 5.0) -> float:
    if price <= 0 or S <= 0 or K <= 0 or tau <= 0:
        return np.nan
    no_arb = max(0, S - K)
    if price < no_arb - 1e-6 or price > S + 1e-6:
        return np.nan
    def obj(sig):
        return _bs_call(S, K, tau, sig) - price
    try:
        return float(brentq(obj, sigma_lo, sigma_hi))
    except Exception:
        return np.nan


def invert_iv_put(price: float, S: float, K: float, tau: float, sigma_lo: float = 0.001, sigma_hi: float = 5.0) -> float:
    if price <= 0 or S <= 0 or K <= 0 or tau <= 0:
        return np.nan
    no_arb = max(0, K - S)
    if price < no_arb - 1e-6 or price > K + 1e-6:
        return np.nan
    def obj(sig):
        return _bs_put(S, K, tau, sig) - price
    try:
        return float(brentq(obj, sigma_lo, sigma_hi))
    except Exception:
        return np.nan


def compute_mid_and_iv(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Add mid_price = (best_bid_price + best_ask_price) / 2 and iv_mid (implied vol from mid).
    Expects: open_time, instrument_name, strike, expiry, call_put, best_bid_price, best_ask_price, index_price.
    """
    if norm is None or brentq is None:
        logger.warning("scipy not available; iv_mid will be NaN")
        df = df.copy()
        df["mid_price"] = np.nan
        df["iv_mid"] = np.nan
        return df
    df = df.copy()
    bid = df["best_bid_price"].astype(float)
    ask = df["best_ask_price"].astype(float)
    df["mid_price"] = np.where(
        bid.notna() & ask.notna() & (bid > 0) & (ask > 0),
        (bid + ask) / 2,
        np.nan,
    )
    S = df["index_price"].astype(float)
    K = df["strike"].astype(float)
    expiry_ms = df["expiry"].astype("int64")
    tau = (expiry_ms - df["open_time"].astype("int64")) / MS_PER_YEAR
    call_put = df["call_put"].astype(str).str.upper().str[:1]
    mid = df["mid_price"].values
    iv = np.full(len(df), np.nan)
    for i in range(len(df)):
        if pd.isna(mid[i]) or mid[i] <= 0 or pd.isna(S.iloc[i]) or S.iloc[i] <= 0 or pd.isna(K.iloc[i]) or K.iloc[i] <= 0 or tau.iloc[i] <= 0:
            continue
        cp = call_put.iloc[i] if i < len(call_put) else "C"
        if cp == "C":
            sig = invert_iv_call(mid[i], float(S.iloc[i]), float(K.iloc[i]), float(tau.iloc[i]))
        else:
            sig = invert_iv_put(mid[i], float(S.iloc[i]), float(K.iloc[i]), float(tau.iloc[i]))
        if not np.isnan(sig) and SIGMA_MIN <= sig <= SIGMA_MAX:
            iv[i] = sig
    df["iv_mid"] = iv
    return df
