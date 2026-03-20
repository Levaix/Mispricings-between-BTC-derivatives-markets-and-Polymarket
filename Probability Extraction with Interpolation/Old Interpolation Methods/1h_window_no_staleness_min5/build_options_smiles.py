"""Build options IV smiles using a rolling 1-hour window and evaluate at Polymarket strikes.

For each 5-minute evaluation timestamp t, the smile is estimated from all option rows
with bucket_5m_utc in [t - 60 min, t], then IV is interpolated at Polymarket strikes.
No re-aggregation of source data; outputs go to analysis/ only.

Inputs (read-only)
------------------
- Options aggregates (5m trade bars, per contract):
    Default: "Actual data/Options Deribit/options_5m_agg/agg_5m_merged.parquet"
    (Override via --options_path)

  Required columns:
    instrument_name
    bucket_5m_utc
    open_price, close_price, high_price, low_price, vwap_price
    iv_vwap
    trade_count
    amount_sum
    contracts_sum
    index_price_last
    timestamp_first
    timestamp_last

- Polymarket events (target strikes):
    Default: "Actual data/Polymarket/polymarket_contracts_metadata.parquet"
    (Override via --events_path)

  Expected columns:
    contract_id, expiry_date, strike, (optional) option_type

Outputs (under ./analysis/)
---------------------------
- analysis/options_smile_snapshot_stats.parquet
- analysis/options_smile_event_iv.parquet
- analysis/options_smile_report.json
- analysis/plots/
    - n_unique_strikes_hist.png
    - fit_success_rate_by_day.png
    - example_smiles.png

Notes
-----
- Does NOT compute N(d2) or any probabilities.
- Does NOT write anything under "Actual data/"; only reads from it.
- Uses IV = iv_vwap normalized to decimal (e.g. 55 -> 0.55 when needed).
- Uses shape-preserving PCHIP when SciPy is available; otherwise falls back
  to linear interpolation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - plotting not essential for core logic
    HAS_MPL = False

try:
    from scipy.interpolate import PchipInterpolator

    HAS_PCHIP = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PCHIP = False


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


BUCKET_COL = "bucket_5m_utc"
BAR_MS = 5 * 60 * 1000
ROLLING_WINDOW_MINUTES = 60

# Polymarket daily 17:00 UTC vs Deribit daily 08:00 UTC: for event date Z, use option expiry
# Z if t <= Z 08:00 UTC, else Z+1 (never beyond Z+1).
DAILY_EXPIRY_CUTOFF_HOUR_UTC = 8


def chosen_daily_expiry(event_date_str: str, bucket_ts: pd.Timestamp) -> str:
    """For a Polymarket daily event date Z and timestamp t, return option expiry E in {Z, Z+1}.
    If t <= Z 08:00 UTC return Z; else return Z+1. Never return beyond Z+1."""
    t = bucket_ts
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    cutoff = pd.Timestamp(f"{event_date_str} {DAILY_EXPIRY_CUTOFF_HOUR_UTC:02d}:00:00", tz="UTC")
    if t <= cutoff:
        return event_date_str
    next_day = (pd.Timestamp(event_date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return next_day


def _root() -> Path:
    # This script lives in ./analysis/
    return Path(__file__).resolve().parent.parent


@dataclass
class SmileFit:
    fit_success: bool
    fail_reason: str
    m_min: float
    m_max: float
    n_unique_strikes: int
    total_contracts_sum: float
    interpolator_type: str  # "pchip" or "linear" or ""
    m_nodes: np.ndarray
    iv_nodes: np.ndarray

    def eval(self, m: float) -> float:
        if not self.fit_success:
            return np.nan
        if not (self.m_min <= m <= self.m_max):
            return np.nan
        if self.interpolator_type == "pchip" and HAS_PCHIP:
            interp = PchipInterpolator(self.m_nodes, self.iv_nodes, extrapolate=False)
            return float(interp(m))
        # Linear fallback
        return float(np.interp(m, self.m_nodes, self.iv_nodes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build options IV smiles and evaluate at Polymarket strikes.")
    parser.add_argument(
        "--options_path",
        type=str,
        default=str(_root() / "Actual data" / "Options Deribit" / "options_5m_agg" / "agg_5m_merged.parquet"),
        help="Path to 5m aggregated options parquet (read-only).",
    )
    parser.add_argument(
        "--events_path",
        type=str,
        default=str(_root() / "Actual data" / "Polymarket" / "polymarket_contracts_metadata.parquet"),
        help="Path to Polymarket events metadata (parquet/csv).",
    )
    parser.add_argument(
        "--polymarket_long_path",
        type=str,
        default=str(_root() / "Actual data" / "Polymarket" / "polymarket_probabilities_5m_long.parquet"),
        help="Path to Polymarket 5m long parquet (base table for dataset 2 row count).",
    )
    return parser.parse_args()


def load_options(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Options parquet not found at {path}")
    df = pd.read_parquet(path)
    required = {
        "instrument_name",
        BUCKET_COL,
        "iv_vwap",
        "contracts_sum",
        "amount_sum",
        "index_price_last",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Options file missing required columns: {missing}")
    # Ensure bucket is timezone-aware UTC
    df[BUCKET_COL] = pd.to_datetime(df[BUCKET_COL], utc=True)
    return df


def parse_instrument(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse instrument_name -> expiry_date (YYYY-MM-DD), strike (float), option_type (C/P)."""
    pattern = r"^BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-(C|P)$"
    extracted = df["instrument_name"].str.extract(pattern, expand=True)
    extracted.columns = ["expiry_code", "strike_str", "option_type"]
    df = df.join(extracted)

    parse_fail = df["expiry_code"].isna() | df["strike_str"].isna() | df["option_type"].isna()
    n_fail = int(parse_fail.sum())
    parse_stats = {"parse_fail_rows": n_fail}

    if n_fail:
        sample = df.loc[parse_fail, "instrument_name"].dropna().unique()[:10]
        logger.warning("Instrument parse failed for %d rows. Sample: %s", n_fail, sample)
    df = df.loc[~parse_fail].copy()

    df["strike"] = pd.to_numeric(df["strike_str"], errors="coerce")
    # Convert expiry_code like 27MAR26 -> date
    dt = pd.to_datetime(df["expiry_code"], format="%d%b%y", errors="coerce")
    df["expiry_date"] = dt.dt.strftime("%Y-%m-%d")
    bad_date = df["expiry_date"].isna()
    n_bad_date = int(bad_date.sum())
    if n_bad_date:
        sample = df.loc[bad_date, "instrument_name"].dropna().unique()[:10]
        logger.warning("Expiry date parse failed for %d rows. Sample: %s", n_bad_date, sample)
        df = df.loc[~bad_date].copy()
    parse_stats["bad_expiry_rows"] = n_bad_date

    return df, parse_stats


def normalize_iv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Normalize iv_vwap into decimal units (e.g. 55 -> 0.55)."""
    iv = pd.to_numeric(df["iv_vwap"], errors="coerce")
    pos = iv > 0
    med = float(iv.loc[pos].median()) if pos.any() else float("nan")
    if np.isnan(med):
        logger.warning("Cannot determine IV scale (median is NaN); assuming decimal.")
        df["iv_norm"] = iv
        decision = "unknown_assume_decimal"
        factor = 1.0
    elif med > 3.0:
        # Looks like percent
        df["iv_norm"] = iv / 100.0
        decision = "percent_to_decimal"
        factor = 0.01
    else:
        df["iv_norm"] = iv
        decision = "already_decimal"
        factor = 1.0
    stats = {"iv_median_raw": med, "iv_scale_decision": decision, "iv_scale_factor": factor}
    logger.info("IV normalization decision: %s (median raw=%.4f)", decision, med)
    return df, stats


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Events file not found at {path}. Provide --events_path pointing to a CSV/Parquet "
            "with at least columns: expiry_date, and strike or strike_K."
        )
    if path.suffix.lower() == ".parquet":
        ev = pd.read_parquet(path)
    else:
        ev = pd.read_csv(path)
    if "expiry_date" not in ev.columns:
        raise ValueError(f"Events file {path} must contain column 'expiry_date'.")
    strike_col = "strike_K" if "strike_K" in ev.columns else "strike"
    if strike_col not in ev.columns:
        raise ValueError(f"Events file {path} must contain 'strike_K' or 'strike'.")
    ev = ev.copy()
    ev["expiry_date"] = ev["expiry_date"].astype(str)
    ev["strike_K"] = pd.to_numeric(ev[strike_col], errors="coerce")
    ev = ev.loc[ev["strike_K"].notna()].copy()
    # Option type: Polymarket BTC-above-K is call-like
    if "option_type" not in ev.columns:
        ev["option_type"] = "C"
    else:
        ev["option_type"] = ev["option_type"].fillna("C")
    # Keep only relevant columns
    keep = ["expiry_date", "strike_K", "option_type"]
    if "contract_id" in ev.columns:
        keep.append("contract_id")
    ev = ev[keep].drop_duplicates()
    logger.info("Loaded %d Polymarket events from %s", len(ev), path)
    return ev


def load_polymarket_long(
    long_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    """Load Polymarket 5m long parquet and merge with metadata to get event_date, strike_K per row.
    Returns one row per (timestamp_utc, contract_id) with event_date, strike_K, option_type, prob_yes.
    """
    if not long_path.exists():
        raise FileNotFoundError(f"Polymarket long parquet not found: {long_path}")
    long_df = pd.read_parquet(long_path)
    for col in ["timestamp_utc", "contract_id"]:
        if col not in long_df.columns:
            raise ValueError(f"Polymarket long must have column '{col}'")
    if "prob_yes" not in long_df.columns:
        long_df["prob_yes"] = np.nan

    meta = pd.read_parquet(metadata_path) if metadata_path.suffix.lower() == ".parquet" else pd.read_csv(metadata_path)
    meta = meta.copy()
    if "contract_id" not in meta.columns and "event_id" in meta.columns:
        meta = meta.rename(columns={"event_id": "contract_id"})
    # Normalize expiry/date column to expiry_date
    for col in ["expiry_date", "parsed_date", "date", "expiry"]:
        if col in meta.columns:
            if col != "expiry_date":
                meta = meta.rename(columns={col: "expiry_date"})
            break
    if "expiry_date" not in meta.columns:
        raise ValueError("Metadata must have an expiry/date column (expiry_date, parsed_date, date, or expiry)")
    strike_col = "strike_K" if "strike_K" in meta.columns else "strike"
    if strike_col not in meta.columns:
        raise ValueError(f"Metadata must have strike_K or strike")
    meta["expiry_date"] = meta["expiry_date"].astype(str)
    meta["strike_K"] = pd.to_numeric(meta[strike_col], errors="coerce")
    meta = meta.dropna(subset=["strike_K", "expiry_date"])
    if "option_type" in meta.columns:
        meta = meta[["contract_id", "expiry_date", "strike_K", "option_type"]].drop_duplicates(subset=["contract_id"], keep="first")
    else:
        meta["option_type"] = "C"
        meta = meta[["contract_id", "expiry_date", "strike_K", "option_type"]].drop_duplicates(subset=["contract_id"], keep="first")

    merged = long_df.merge(meta, on="contract_id", how="left")
    # Use expiry_date from merge (may be expiry_date_y if long had expiry_date)
    expiry_col = "expiry_date"
    if expiry_col not in merged.columns and "expiry_date_y" in merged.columns:
        expiry_col = "expiry_date_y"
        if "expiry_date_x" in merged.columns:
            merged = merged.drop(columns=["expiry_date_x"])
        merged = merged.rename(columns={"expiry_date_y": "expiry_date"})
    elif "expiry_date_x" in merged.columns and "expiry_date_y" in merged.columns:
        merged = merged.drop(columns=["expiry_date_x"]).rename(columns={"expiry_date_y": "expiry_date"})
    missing = merged["expiry_date"].isna()
    if missing.any():
        n = int(missing.sum())
        logger.warning("Dropping %d long rows with no metadata (expiry_date/strike_K); contract_id not in metadata.", n)
        merged = merged.loc[~missing].copy()
    logger.info("Loaded Polymarket long: %d rows (after metadata merge)", len(merged))
    return merged


def fit_smile_for_snapshot(
    g: pd.DataFrame,
    snapshot_spot: float,
    min_points: int = 5,
    min_range: float = 0.05,
) -> SmileFit:
    """Build moneyness grid and fit IV(m) for one snapshot."""
    # Aggregate by strike to enforce uniqueness; use weighted IV per strike
    if "contracts_sum" in g.columns and g["contracts_sum"].notna().any():
        w_col = "contracts_sum"
    elif "amount_sum" in g.columns and g["amount_sum"].notna().any():
        w_col = "amount_sum"
    else:
        w_col = None

    def _agg(group: pd.Series) -> float:
        if w_col is None:
            return float(group["iv_norm"].mean())
        w = pd.to_numeric(group[w_col], errors="coerce").fillna(0.0)
        v = group["iv_norm"]
        if w.sum() <= 0:
            return float(v.mean())
        return float((v * w).sum() / w.sum())

    by_strike = (
        g.groupby("strike", group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index(name="iv_strike")
        .dropna(subset=["iv_strike"])
    )
    if by_strike.empty:
        return SmileFit(False, "no_iv_data", np.nan, np.nan, 0, float("nan"), "", np.array([]), np.array([]))

    strike_num = pd.to_numeric(by_strike["strike"], errors="coerce")
    by_strike = by_strike.assign(m=strike_num / snapshot_spot)
    by_strike = by_strike.dropna(subset=["m", "iv_strike"])
    if by_strike.empty:
        return SmileFit(False, "no_moneyness", np.nan, np.nan, 0, float("nan"), "", np.array([]), np.array([]))

    by_strike = by_strike.sort_values("m")
    m_vals = by_strike["m"].to_numpy()
    iv_vals = by_strike["iv_strike"].to_numpy()
    # Enforce uniqueness on m
    m_unique, idx = np.unique(m_vals, return_index=True)
    iv_unique = iv_vals[idx]

    n_unique = int(len(m_unique))
    m_min = float(m_unique.min()) if n_unique else float("nan")
    m_max = float(m_unique.max()) if n_unique else float("nan")
    m_range = m_max - m_min if n_unique else 0.0
    total_contracts_sum = float(
        pd.to_numeric(g.get("contracts_sum", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    )

    if n_unique < min_points:
        return SmileFit(False, "too_few_strikes", m_min, m_max, n_unique, total_contracts_sum, "", m_unique, iv_unique)
    if m_range < min_range:
        return SmileFit(False, "range_too_small", m_min, m_max, n_unique, total_contracts_sum, "", m_unique, iv_unique)
    if not np.isfinite(iv_unique).all() or (iv_unique <= 0).all():
        return SmileFit(False, "invalid_iv", m_min, m_max, n_unique, total_contracts_sum, "", m_unique, iv_unique)

    if HAS_PCHIP and n_unique >= 3:
        interp_type = "pchip"
    else:
        interp_type = "linear"
    return SmileFit(True, "", m_min, m_max, n_unique, total_contracts_sum, interp_type, m_unique, iv_unique)


def build_smiles_and_evaluate(
    options: pd.DataFrame,
    events: pd.DataFrame,
    polymarket_long: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build smiles using a rolling 1-hour window; dataset 2 = one row per Polymarket long row."""
    stats_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    total_windows = 0
    fit_success_count = 0
    fail_reasons = Counter()
    out_of_range_count = 0
    no_snapshot_fit_count = 0
    # Polymarket daily expiry switch: counts for dataset 2 only
    event_rows_same_day_expiry = 0
    event_rows_next_day_expiry = 0
    event_rows_next_day_expiry_no_fit = 0

    # For example plots
    example_smiles: List[Tuple[pd.Timestamp, str, str, np.ndarray, np.ndarray]] = []

    # Map (bucket_ts, expiry_date, option_type) -> (SmileFit, spot_median) for dataset 2 lookup
    fits_map: Dict[Tuple[Any, str, str], Tuple[SmileFit, float]] = {}

    unique_buckets = options[BUCKET_COL].drop_duplicates().sort_values()
    window_delta = pd.Timedelta(minutes=ROLLING_WINDOW_MINUTES)

    # --- Dataset 1 + fits map: unchanged smile/rolling logic; no event evaluation here ---
    for bucket_ts in unique_buckets:
        window_start = bucket_ts - window_delta
        window_mask = (options[BUCKET_COL] >= window_start) & (options[BUCKET_COL] <= bucket_ts)
        window_df = options.loc[window_mask]

        for (expiry_date, opt_type), g in window_df.groupby(["expiry_date", "option_type"], sort=False):
            total_windows += 1
            spot_series = pd.to_numeric(g["index_price_last"], errors="coerce").dropna()
            if spot_series.empty:
                continue
            spot_median = float(spot_series.median())
            if spot_median <= 0:
                continue
            spot_dispersion = float((spot_series.max() - spot_series.min()) / spot_median) if len(spot_series) > 1 else 0.0

            fit = fit_smile_for_snapshot(g, snapshot_spot=spot_median)
            if fit.fit_success:
                fit_success_count += 1
            else:
                fail_reasons[fit.fail_reason] += 1

            stats_rows.append(
                {
                    BUCKET_COL: bucket_ts,
                    "expiry_utc": expiry_date,
                    "option_type": opt_type,
                    "n_rows": int(len(g)),
                    "n_unique_strikes": fit.n_unique_strikes,
                    "total_contracts_sum": fit.total_contracts_sum,
                    "spot_snapshot": spot_median,
                    "spot_dispersion": spot_dispersion,
                    "moneyness_min": fit.m_min,
                    "moneyness_max": fit.m_max,
                    "fit_success": fit.fit_success,
                    "fit_fail_reason": fit.fail_reason,
                }
            )
            fits_map[(bucket_ts, expiry_date, opt_type)] = (fit, spot_median)

            if fit.fit_success and len(example_smiles) < 3:
                example_smiles.append((bucket_ts, expiry_date, opt_type, fit.m_nodes.copy(), fit.iv_nodes.copy()))

    # --- Dataset 2: one row per Polymarket long row (base table = polymarket_probabilities_5m_long) ---
    row_count_polymarket_long = int(len(polymarket_long)) if polymarket_long is not None and not polymarket_long.empty else 0
    if polymarket_long is not None and not polymarket_long.empty:
        long_df = polymarket_long.copy()
        long_df["timestamp_utc"] = pd.to_datetime(long_df["timestamp_utc"], utc=True)
        # Align to 5m bucket so we can look up fits_map (keys are from options bucket_5m_utc)
        long_df[BUCKET_COL] = long_df["timestamp_utc"].dt.floor("5min")
        row_count_polymarket_long = len(long_df)

        for idx, row_long in long_df.iterrows():
            bucket_ts = row_long[BUCKET_COL]
            event_date_z = str(row_long["expiry_date"])
            strike_k = float(row_long["strike_K"])
            opt_type = row_long.get("option_type", "C")
            if pd.isna(opt_type):
                opt_type = "C"
            prob_yes = row_long.get("prob_yes", np.nan)
            chosen_e = chosen_daily_expiry(event_date_z, bucket_ts)
            fit, spot_median = fits_map.get((bucket_ts, chosen_e, opt_type), (None, None))
            same_day = chosen_e == event_date_z
            next_day = not same_day

            if fit is None:
                spot_median = np.nan
                no_snapshot_fit_count += 1
                if next_day:
                    event_rows_next_day_expiry_no_fit += 1
                m_event = np.nan
                iv_hat = np.nan
                out_of_range = True
                n_unique_strikes = 0
                moneyness_min = np.nan
                moneyness_max = np.nan
                total_contracts_sum = 0.0
                fit_success = False
                no_snapshot_fit = True
            else:
                no_snapshot_fit = False
                spot_median = spot_median if spot_median is not None else np.nan
                m_event = strike_k / spot_median if np.isfinite(spot_median) and spot_median > 0 else np.nan
                n_unique_strikes = fit.n_unique_strikes
                moneyness_min = fit.m_min
                moneyness_max = fit.m_max
                total_contracts_sum = fit.total_contracts_sum
                fit_success = fit.fit_success
                if not fit.fit_success:
                    iv_hat = np.nan
                    out_of_range = True
                    no_snapshot_fit_count += 1
                    no_snapshot_fit = True
                    if next_day:
                        event_rows_next_day_expiry_no_fit += 1
                else:
                    if fit.m_min <= m_event <= fit.m_max:
                        iv_hat = fit.eval(m_event)
                        if not np.isfinite(iv_hat):
                            out_of_range = True
                            out_of_range_count += 1
                        else:
                            out_of_range = False
                    else:
                        iv_hat = np.nan
                        out_of_range = True
                        out_of_range_count += 1

            if same_day:
                event_rows_same_day_expiry += 1
            else:
                event_rows_next_day_expiry += 1

            event_rows.append({
                BUCKET_COL: bucket_ts,
                "event_date": event_date_z,
                "strike_K": strike_k,
                "prob_yes": prob_yes,
                "contract_id": row_long.get("contract_id"),
                "chosen_option_expiry": chosen_e,
                "expiry_utc": chosen_e,
                "option_type": opt_type,
                "spot_snapshot": spot_median if np.isfinite(spot_median) else np.nan,
                "moneyness_event": m_event,
                "iv_hat": iv_hat,
                "fit_success": fit_success,
                "out_of_range": out_of_range,
                "no_snapshot_fit": no_snapshot_fit,
                "n_unique_strikes": n_unique_strikes,
                "moneyness_min": moneyness_min,
                "moneyness_max": moneyness_max,
                "total_contracts_sum": total_contracts_sum,
            })

    snapshot_stats = pd.DataFrame(stats_rows)
    event_iv = pd.DataFrame(event_rows)

    # Sanity check: snapshot_stats must be one row per (bucket_5m_utc, expiry_utc, option_type)
    snapshot_sanity: Dict[str, Any] = {}
    validation_one_row_per_timestamp = False
    validation_warning: Optional[str] = None
    if not snapshot_stats.empty:
        n_unique_ts = int(snapshot_stats[BUCKET_COL].nunique())
        n_snapshot_rows = int(len(snapshot_stats))
        rows_per_ts = snapshot_stats.groupby(BUCKET_COL, sort=False).size()
        avg_rows_per_ts = float(rows_per_ts.mean()) if len(rows_per_ts) else 0.0
        snapshot_sanity = {
            "n_unique_timestamps": n_unique_ts,
            "n_snapshot_rows": n_snapshot_rows,
            "avg_rows_per_timestamp": round(avg_rows_per_ts, 2),
        }
        # Validation: must have multiple rows per timestamp for at least some timestamps
        if n_snapshot_rows == n_unique_ts and n_snapshot_rows > 0:
            validation_one_row_per_timestamp = True
            validation_warning = (
                "Snapshot stats has exactly one row per timestamp; expected multiple rows per "
                "timestamp (one per expiry/option_type). Check grouping logic."
            )
            logger.warning(validation_warning)

    # Dataset 2: one row per Polymarket long row; key (bucket_5m_utc, event_date, strike_K) must be unique
    event_key = [BUCKET_COL, "event_date", "strike_K"]
    row_count_event_iv_output = int(len(event_iv))
    duplicate_event_keys = 0
    if not event_iv.empty:
        dup_mask = event_iv.duplicated(subset=event_key, keep="first")
        duplicate_event_keys = int(dup_mask.sum())
        if duplicate_event_keys > 0:
            logger.warning(
                "Dataset 2 has %d duplicate (bucket, event_date, strike_K) keys; dropping to keep first.",
                duplicate_event_keys,
            )
            event_iv = event_iv.drop_duplicates(subset=event_key, keep="first")
            row_count_event_iv_output = int(len(event_iv))
    row_count_match_ok = (row_count_event_iv_output == row_count_polymarket_long)
    if row_count_polymarket_long and row_count_event_iv_output != row_count_polymarket_long:
        logger.warning(
            "Dataset 2 row count mismatch: Polymarket long=%d, event_iv=%d",
            row_count_polymarket_long, row_count_event_iv_output,
        )
    usable_iv_count = int((event_iv["iv_hat"].notna() & np.isfinite(event_iv["iv_hat"])).sum()) if not event_iv.empty else 0

    # Invalid expiry assignments: t <= Z 08:00 => expiry must be Z; t > Z 08:00 => expiry must be Z+1
    invalid_expiry_assignments = 0
    if not event_iv.empty and "event_date" in event_iv.columns and "expiry_utc" in event_iv.columns:
        bucket_dt = pd.to_datetime(event_iv[BUCKET_COL], utc=True)
        ed = pd.to_datetime(event_iv["event_date"])
        ex = pd.to_datetime(event_iv["expiry_utc"])
        on_event_date = (bucket_dt.dt.date == ed.dt.date).values
        hour = bucket_dt.dt.hour.values
        minute = bucket_dt.dt.minute.values
        at_or_before_08 = (hour < DAILY_EXPIRY_CUTOFF_HOUR_UTC) | (
            (hour == DAILY_EXPIRY_CUTOFF_HOUR_UTC) & (minute == 0)
        )
        same_day_required = on_event_date & at_or_before_08
        next_day_required = on_event_date & ~at_or_before_08
        used_same = (event_iv["expiry_utc"] == event_iv["event_date"]).values
        next_day_date = (ed + pd.Timedelta(days=1)).dt.date
        used_next = (ex.dt.date.values == next_day_date.values)
        invalid_expiry_assignments = int(
            np.sum((same_day_required & ~used_same) | (next_day_required & ~used_next))
        )
        if invalid_expiry_assignments > 0:
            logger.warning("Dataset 2: %d rows have invalid expiry assignment (rule: Z 08:00 switch).", invalid_expiry_assignments)

    # Polymarket daily expiry switch: report and validation (dataset 2 only)
    # Rule: for event date Z, use option expiry Z if t <= Z 08:00 UTC else Z+1 (never beyond Z+1).
    expiry_switch_report: Dict[str, Any] = {
        "polymarket_daily_expiry_switch_rule": True,
        "description": "For Polymarket event date Z: option expiry = Z if t <= Z 08:00 UTC, else Z+1.",
        "event_rows_same_day_expiry": event_rows_same_day_expiry,
        "event_rows_next_day_expiry": event_rows_next_day_expiry,
        "event_rows_next_day_expiry_no_fit": event_rows_next_day_expiry_no_fit,
    }
    validation_expiry_warning: Optional[str] = None
    if not event_iv.empty and "event_date" in event_iv.columns and "expiry_utc" in event_iv.columns:
        ed = pd.to_datetime(event_iv["event_date"])
        ex = pd.to_datetime(event_iv["expiry_utc"])
        max_allowed = ed + pd.Timedelta(days=1)
        violation = (ex > max_allowed).any()
        if violation:
            validation_expiry_warning = (
                "Some event_iv rows have option expiry_utc later than event_date + 1 day; "
                "daily rule allows only Z or Z+1."
            )
            logger.warning(validation_expiry_warning)
        # Sanity: on event_date, t <= 08:00 -> same-day expiry; t > 08:00 -> next-day expiry
        bucket_dt = pd.to_datetime(event_iv[BUCKET_COL], utc=True)
        bucket_date = bucket_dt.dt.date
        event_date_only = pd.to_datetime(event_iv["event_date"]).dt.date
        on_event_date = (bucket_date == event_date_only).values
        hour = bucket_dt.dt.hour.values
        minute = bucket_dt.dt.minute.values
        at_or_before_08 = (hour < DAILY_EXPIRY_CUTOFF_HOUR_UTC) | (
            (hour == DAILY_EXPIRY_CUTOFF_HOUR_UTC) & (minute == 0)
        )
        same_day_expected = on_event_date & at_or_before_08
        next_day_expected = on_event_date & ~at_or_before_08
        used_same = (event_iv["expiry_utc"] == event_iv["event_date"]).values
        ex_dates = pd.to_datetime(event_iv["expiry_utc"])
        next_day_dates = ed + pd.Timedelta(days=1)
        used_next = (ex_dates.values == next_day_dates.values)  # compare numpy datetimes
        if np.any(same_day_expected & ~used_same) or np.any(next_day_expected & ~used_next):
            logger.warning(
                "Expiry rule sanity: some rows on event date have unexpected same-day vs next-day choice."
            )

    n_ev = int(len(event_iv))
    pct_usable = round(100.0 * usable_iv_count / n_ev, 2) if n_ev else 0.0
    pct_out_of_range = round(100.0 * out_of_range_count / n_ev, 2) if n_ev else 0.0
    pct_no_snapshot_fit = round(100.0 * no_snapshot_fit_count / n_ev, 2) if n_ev else 0.0

    summary: Dict[str, Any] = {
        "total_windows": total_windows,
        "fit_success_count": fit_success_count,
        "fit_fail_counts": dict(fail_reasons),
        "n_event_rows": n_ev,
        "row_count_polymarket_long": row_count_polymarket_long,
        "row_count_event_iv_output": row_count_event_iv_output,
        "row_count_match_ok": row_count_match_ok,
        "duplicate_event_keys": duplicate_event_keys,
        "invalid_expiry_assignments": invalid_expiry_assignments,
        "out_of_range_count": int(out_of_range_count),
        "no_snapshot_fit_count": int(no_snapshot_fit_count),
        "usable_iv_count": usable_iv_count,
        "pct_rows_usable_iv": pct_usable,
        "pct_rows_out_of_range": pct_out_of_range,
        "pct_rows_no_snapshot_fit": pct_no_snapshot_fit,
        "same_day_expiry_count": event_rows_same_day_expiry,
        "next_day_expiry_count": event_rows_next_day_expiry,
        "snapshot_stats_sanity": snapshot_sanity,
        "validation_one_row_per_timestamp": validation_one_row_per_timestamp,
        "validation_warning": validation_warning,
        "expiry_switch": expiry_switch_report,
        "validation_expiry_warning": validation_expiry_warning,
    }

    return snapshot_stats, event_iv, {"summary": summary, "example_smiles": example_smiles}


def make_plots(snapshot_stats: pd.DataFrame, aux: Dict[str, Any], out_dir: Path) -> None:
    if not HAS_MPL or snapshot_stats.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram: n_unique_strikes per window
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(snapshot_stats["n_unique_strikes"].dropna(), bins=30, color="steelblue", alpha=0.8)
        ax.set_xlabel("n_unique_strikes per window")
        ax.set_ylabel("count")
        ax.set_title("Distribution of unique strikes per window")
        fig.tight_layout()
        fig.savefig(plots_dir / "n_unique_strikes_hist.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create n_unique_strikes_hist plot: %s", e)

    # 2) Time series: % windows with successful fit per day
    try:
        df = snapshot_stats.copy()
        df["date"] = pd.to_datetime(df[BUCKET_COL]).dt.floor("D")
        daily = df.groupby("date")["fit_success"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(daily["date"], daily["fit_success"] * 100.0, marker="o")
        ax.set_ylabel("% windows with successful fit")
        ax.set_xlabel("date")
        ax.set_title("Smile fit success rate by day")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / "fit_success_rate_by_day.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create fit_success_rate_by_day plot: %s", e)

    # 3) Example smile plots
    try:
        example_smiles = aux.get("example_smiles") or []
        if not example_smiles:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for (bucket_ts, expiry_date, opt_type, m_nodes, iv_nodes) in example_smiles:
            m_grid = np.linspace(m_nodes.min(), m_nodes.max(), 200)
            if HAS_PCHIP and len(m_nodes) >= 3:
                interp = PchipInterpolator(m_nodes, iv_nodes, extrapolate=False)
                iv_grid = interp(m_grid)
            else:
                iv_grid = np.interp(m_grid, m_nodes, iv_nodes)
            label = f"{expiry_date} {opt_type} {bucket_ts}"
            ax.plot(m_grid, iv_grid, label=label)
            ax.scatter(m_nodes, iv_nodes, s=10, alpha=0.7)
        ax.set_xlabel("moneyness (strike / spot)")
        ax.set_ylabel("IV (decimal)")
        ax.set_title("Example per-window smiles")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "example_smiles.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create example_smiles plot: %s", e)


def _clean_previous_outputs(analysis_dir: Path) -> None:
    """Delete previous pipeline outputs and recreate analysis/plots/ empty."""
    for name in [
        "options_smile_snapshot_stats.parquet",
        "options_smile_event_iv.parquet",
        "options_smile_report.json",
    ]:
        p = analysis_dir / name
        if p.exists():
            p.unlink()
            logger.info("Deleted %s", p)
    plots_dir = analysis_dir / "plots"
    if plots_dir.exists():
        for f in plots_dir.iterdir():
            f.unlink()
        logger.info("Cleaned %s", plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    root = _root()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _clean_previous_outputs(analysis_dir)

    options_path = Path(args.options_path)
    events_path = Path(args.events_path)
    polymarket_long_path = Path(args.polymarket_long_path)

    logger.info("Loading options from %s", options_path)
    options_df = load_options(options_path)
    logger.info("Options rows: %d", len(options_df))

    logger.info("Parsing instrument_name")
    options_df, parse_stats = parse_instrument(options_df)

    logger.info("Normalizing IV")
    options_df, iv_stats = normalize_iv(options_df)

    logger.info("Loading Polymarket events from %s", events_path)
    events_df = load_events(events_path)

    logger.info("Loading Polymarket 5m long from %s (base table for dataset 2)", polymarket_long_path)
    polymarket_long = load_polymarket_long(polymarket_long_path, events_path)

    logger.info("Building smiles and evaluating at event strikes (dataset 2 = one row per long row)")
    snapshot_stats, event_iv, aux = build_smiles_and_evaluate(
        options_df, events_df, polymarket_long=polymarket_long
    )

    # Write parquet outputs
    snapshot_path = analysis_dir / "options_smile_snapshot_stats.parquet"
    event_iv_path = analysis_dir / "options_smile_event_iv.parquet"

    if not snapshot_stats.empty:
        snapshot_stats.to_parquet(snapshot_path, index=False)
        logger.info("Wrote %s (%d rows)", snapshot_path, len(snapshot_stats))
    else:
        logger.warning("No snapshot stats to write.")

    if not event_iv.empty:
        event_iv.to_parquet(event_iv_path, index=False)
        logger.info("Wrote %s (%d rows)", event_iv_path, len(event_iv))
    else:
        logger.warning("No event IV rows to write.")

    # Build report JSON
    summary = aux.get("summary", {})
    if not summary.get("row_count_match_ok"):
        summary["sanity_check_warning"] = (
            "Dataset 2 row count does not match Polymarket long; every long row must appear exactly once."
        )
    report = {
        "options_path": str(options_path),
        "events_path": str(events_path),
        "polymarket_long_path": str(polymarket_long_path),
        "parse_stats": parse_stats,
        "iv_stats": iv_stats,
        "summary": summary,
    }
    report_path = analysis_dir / "options_smile_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # Plots
    make_plots(snapshot_stats, aux, analysis_dir)


if __name__ == "__main__":
    main()

