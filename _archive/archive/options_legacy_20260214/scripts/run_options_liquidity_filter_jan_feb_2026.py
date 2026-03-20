#!/usr/bin/env python3
"""
Liquid options subset filter + sanity report for Jan–Feb 2026 options panel.
Additive: does not change existing download or Step 1 code.
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.time import get_start_end_ms, window_output_suffix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths (same window folder as main pipeline)
DATA_ROOT = Path(config.DATA_ROOT)
OPTIONS_DATA_DIR = DATA_ROOT / "options_jan_feb_2026"
SELECTED_INSTRUMENTS_PATH = OPTIONS_DATA_DIR / "selected_instruments.json"
DERIBIT_INSTRUMENTS_JSON = OPTIONS_DATA_DIR / "deribit_instruments.json"
_start_ms, _end_ms = get_start_end_ms()
OUTPUT_DIR = ROOT / "outputs" / window_output_suffix(_start_ms, _end_ms)
PANEL_PATH = OUTPUT_DIR / "options_panel_long_5m.parquet"
COVERAGE_PATH = OUTPUT_DIR / "options_coverage_summary.parquet"

# Filter thresholds
MIN_NONZERO_BARS = 200
MIN_TOTAL_VOLUME = 500.0

# Usable expiry: at least 3 calls, 3 puts, 6 strikes
MIN_CALLS_USABLE = 3
MIN_PUTS_USABLE = 3
MIN_STRIKES_USABLE = 6


def _ms_to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_panel() -> pd.DataFrame:
    """Load options long panel; require instrument_name, open_time, volume."""
    if not PANEL_PATH.exists():
        logger.error("Panel not found: %s", PANEL_PATH)
        return pd.DataFrame()
    df = pd.read_parquet(PANEL_PATH)
    if "instrument_name" not in df.columns or "open_time" not in df.columns or "volume" not in df.columns:
        logger.error("Panel missing required columns (instrument_name, open_time, volume)")
        return pd.DataFrame()
    return df


def load_instrument_metadata() -> Optional[pd.DataFrame]:
    """Load instrument metadata (strike, expiry, call_put) from selected_instruments or deribit_instruments."""
    # Prefer selected_instruments.json (one row per selected instrument)
    for path in (SELECTED_INSTRUMENTS_PATH, DERIBIT_INSTRUMENTS_JSON):
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue
        if not isinstance(data, list) or not data:
            continue
        rows = []
        for inv in data:
            name = inv.get("instrument_name")
            if not name:
                continue
            exp = inv.get("expiration_timestamp") or inv.get("expiry")
            strike = inv.get("strike")
            opt_type = inv.get("option_type") or inv.get("call_put")
            rows.append({
                "instrument_name": name,
                "expiry": int(exp) if exp is not None else None,
                "strike": float(strike) if strike is not None else None,
                "call_put": str(opt_type).upper() if opt_type else "",
            })
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["instrument_name"], keep="first")
    return None


def ensure_metadata(panel: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Ensure panel has strike, expiry, call_put; merge from meta if missing."""
    needed = ["strike", "expiry", "call_put"]
    if meta is None or meta.empty:
        return panel
    merge_cols = ["instrument_name"] + [c for c in needed if c in meta.columns]
    panel = panel.merge(meta[merge_cols], on="instrument_name", how="left", suffixes=("", "_meta"))
    for c in needed:
        if f"{c}_meta" in panel.columns:
            if c in panel.columns:
                panel[c] = panel[c].fillna(panel[f"{c}_meta"])
            else:
                panel[c] = panel[f"{c}_meta"]
            panel = panel.drop(columns=[f"{c}_meta"])
    return panel


def compute_per_instrument_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Group by instrument_name; compute n_bars, n_nonzero_bars, trade_bar_share, total_volume, etc."""
    grp = panel.groupby("instrument_name").agg(
        n_bars=("open_time", "count"),
        n_nonzero_bars=("volume", lambda s: (s.fillna(0) > 0).sum()),
        total_volume=("volume", "sum"),
        start_utc_ms=("open_time", "min"),
        end_utc_ms=("open_time", "max"),
    ).reset_index()
    grp["trade_bar_share"] = grp["n_nonzero_bars"] / grp["n_bars"].replace(0, float("nan"))
    # median_volume_nonzero
    def med_nonzero(s):
        pos = s[s.fillna(0) > 0]
        return pos.median() if len(pos) else float("nan")
    grp["median_volume_nonzero"] = panel.groupby("instrument_name")["volume"].apply(med_nonzero).values
    grp["start_utc"] = grp["start_utc_ms"].map(lambda x: _ms_to_utc_iso(int(x)) if pd.notna(x) else "")
    grp["end_utc"] = grp["end_utc_ms"].map(lambda x: _ms_to_utc_iso(int(x)) if pd.notna(x) else "")
    grp = grp.drop(columns=["start_utc_ms", "end_utc_ms"])

    # Attach expiry, strike, call_put from panel (first per instrument)
    for col in ["expiry", "strike", "call_put"]:
        if col in panel.columns:
            first = panel.groupby("instrument_name")[col].first().reset_index()
            first = first.rename(columns={col: col})
            grp = grp.merge(first, on="instrument_name", how="left")
    grp["pass_liquid_filter"] = (grp["n_nonzero_bars"] >= MIN_NONZERO_BARS) & (grp["total_volume"] >= MIN_TOTAL_VOLUME)
    if "expiry" in grp.columns:
        grp["expiry_utc"] = grp["expiry"].apply(lambda x: _ms_to_utc_iso(int(x)) if pd.notna(x) and x else "")
    return grp


def apply_liquid_filter(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep instruments with n_nonzero_bars >= 200 and total_volume >= 500."""
    return metrics[metrics["pass_liquid_filter"]].copy()


def expiry_distribution(metrics: pd.DataFrame) -> Dict[str, int]:
    """Counts by expiry year-month."""
    if "expiry" not in metrics.columns or metrics["expiry"].isna().all():
        return {}
    hist = defaultdict(int)
    for exp in metrics["expiry"].dropna().astype("int64"):
        dt = datetime.fromtimestamp(exp / 1000.0, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        hist[key] += 1
    return dict(sorted(hist.items()))


def coverage_per_expiry(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per expiry: n_calls, n_puts, strike_count, strike_min/max/median, usable."""
    if "expiry" not in metrics.columns or metrics["expiry"].isna().all():
        return pd.DataFrame()
    rows = []
    for exp, grp in metrics.groupby("expiry"):
        exp_ts = int(exp) if pd.notna(exp) else None
        if exp_ts is None:
            continue
        n_calls = (grp["call_put"] == "CALL").sum() if "call_put" in grp.columns else 0
        n_puts = (grp["call_put"] == "PUT").sum() if "call_put" in grp.columns else 0
        strikes = grp["strike"].dropna()
        strike_count = strikes.nunique()
        strike_min = strikes.min() if len(strikes) else None
        strike_max = strikes.max() if len(strikes) else None
        strike_median = strikes.median() if len(strikes) else None
        usable = (n_calls >= MIN_CALLS_USABLE and n_puts >= MIN_PUTS_USABLE and strike_count >= MIN_STRIKES_USABLE)
        rows.append({
            "expiry": exp_ts,
            "expiry_utc": _ms_to_utc_iso(exp_ts),
            "n_calls": n_calls,
            "n_puts": n_puts,
            "strike_count": strike_count,
            "strike_min": strike_min,
            "strike_max": strike_max,
            "strike_median": strike_median,
            "n_instruments": len(grp),
            "usable": usable,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Liquid options subset filter for Jan–Feb 2026")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load panel
    logger.info("Loading options panel: %s", PANEL_PATH)
    panel = load_panel()
    if panel.empty:
        return
    meta = load_instrument_metadata()
    panel = ensure_metadata(panel, meta)

    # 2) Per-instrument metrics
    metrics_all = compute_per_instrument_metrics(panel)
    liquid = apply_liquid_filter(metrics_all)
    liquid_names = set(liquid["instrument_name"].tolist())

    # 3) Filtered long panel
    panel_liquid = panel[panel["instrument_name"].isin(liquid_names)].copy()

    # 4) Sanity: expiry distribution and coverage per expiry
    exp_hist = expiry_distribution(liquid)
    coverage = coverage_per_expiry(liquid)
    usable_expiries = coverage[coverage["usable"]] if not coverage.empty else pd.DataFrame()

    # 5) Optional: moneyness using median strike as ATM proxy (no Binance index for Jan–Feb here)
    if "strike" in liquid.columns and liquid["strike"].notna().any():
        atm_proxy = float(liquid["strike"].median())
        liquid["moneyness"] = liquid["strike"] / atm_proxy
        if not coverage.empty:
            coverage["moneyness_min"] = None
            coverage["moneyness_median"] = None
            coverage["moneyness_max"] = None
            for idx, row in coverage.iterrows():
                exp = row["expiry"]
                sub = liquid[liquid["expiry"] == exp]["moneyness"].dropna()
                if len(sub):
                    coverage.at[idx, "moneyness_min"] = sub.min()
                    coverage.at[idx, "moneyness_median"] = sub.median()
                    coverage.at[idx, "moneyness_max"] = sub.max()

    # 6) Save outputs
    metrics_all.to_parquet(OUTPUT_DIR / "liquid_instruments_summary.parquet", index=False)
    if not liquid.empty:
        liquid_list = liquid[["instrument_name", "expiry", "strike", "call_put"]].drop_duplicates()
        liquid_list.to_csv(OUTPUT_DIR / "liquid_instruments_list.csv", index=False)
    panel_liquid.to_parquet(OUTPUT_DIR / "options_panel_long_5m_liquid.parquet", index=False)

    # 7) Console summary (copy/paste friendly)
    print("\n" + "=" * 70)
    print("LIQUID OPTIONS SUBSET — JAN–FEB 2026")
    print("=" * 70)
    print(f"Filters used: n_nonzero_bars >= {MIN_NONZERO_BARS}, total_volume >= {MIN_TOTAL_VOLUME}")
    print(f"Instruments: before -> after  ({len(metrics_all)} -> {len(liquid)})")
    print(f"Panel rows:   before -> after  ({len(panel)} -> {len(panel_liquid)})")
    if not metrics_all.empty and "trade_bar_share" in metrics_all.columns:
        tbs = metrics_all["trade_bar_share"].dropna()
        if len(tbs):
            print(f"trade_bar_share (before filter): median={tbs.median():.2%}, p25={tbs.quantile(0.25):.2%}, p75={tbs.quantile(0.75):.2%}")
    if not liquid.empty and "trade_bar_share" in liquid.columns:
        tbs = liquid["trade_bar_share"].dropna()
        if len(tbs):
            print(f"trade_bar_share (after filter):  median={tbs.median():.2%}, p25={tbs.quantile(0.25):.2%}, p75={tbs.quantile(0.75):.2%}")

    print("\nExpiry year-month counts (after filter):")
    for k, v in sorted(exp_hist.items()):
        print(f"  {k}: {v}")

    if not coverage.empty:
        print("\nDistinct expiry dates (top 15):")
        for _, r in coverage.head(15).iterrows():
            print(f"  {r.get('expiry_utc', r.get('expiry'))}")

    print("\nUsable expiries (n_calls>=3, n_puts>=3, strike_count>=6):")
    if usable_expiries.empty:
        print("  (none)")
    else:
        for _, r in usable_expiries.iterrows():
            print(f"  {r.get('expiry_utc')}  n_calls={r['n_calls']}  n_puts={r['n_puts']}  strike_count={r['strike_count']}")

    print("\nTop 10 instruments by total_volume (after filter):")
    if liquid.empty:
        print("  (none)")
    else:
        top_vol = liquid.nlargest(10, "total_volume")[["instrument_name", "total_volume", "n_nonzero_bars", "n_bars"]]
        print(top_vol.to_string(index=False))

    print("\nTop 10 instruments by n_nonzero_bars (after filter):")
    if liquid.empty:
        print("  (none)")
    else:
        top_nz = liquid.nlargest(10, "n_nonzero_bars")[["instrument_name", "n_nonzero_bars", "total_volume", "n_bars"]]
        print(top_nz.to_string(index=False))

    print("\n" + "=" * 70)
    print("Saved:")
    print("  ", OUTPUT_DIR / "liquid_instruments_summary.parquet")
    print("  ", OUTPUT_DIR / "liquid_instruments_list.csv")
    print("  ", OUTPUT_DIR / "options_panel_long_5m_liquid.parquet")
    print("=" * 70)


if __name__ == "__main__":
    main()
