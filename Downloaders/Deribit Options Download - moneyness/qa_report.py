"""
QA report: coverage by day, instrument stats, timestamp quality, value checks.
Output: reports/options_v2_qa/qa_report.md and qa_report.json
"""
import json
import logging
from pathlib import Path

import pandas as pd

from . import config as cfg


def _instruments_summary(df):
    out = {"pulled": int(df["instrument_name"].nunique())}
    if cfg.CANDLES_FAILURES_JSON.exists():
        try:
            with open(cfg.CANDLES_FAILURES_JSON) as f:
                data = json.load(f)
            failed_n = len(data.get("failed", []))
            skipped_n = len(data.get("skipped_zero_data", []))
            out["attempted"] = out["pulled"] + failed_n + skipped_n
            out["failed_count"] = failed_n
            out["skipped_zero_count"] = skipped_n
        except Exception:
            pass
    return out

logger = logging.getLogger(__name__)


def run(df=None):
    if df is None:
        if not cfg.RAW_LONG_PARQUET.exists():
            logger.warning("No aligned parquet found at %s", cfg.RAW_LONG_PARQUET)
            return {}
        df = pd.read_parquet(cfg.RAW_LONG_PARQUET)

    if df.empty:
        report = {"n_rows": 0, "n_instruments": 0, "message": "No data"}
        _write_report(report, df)
        return report

    df["ts"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
    df["date"] = df["ts"].dt.date.astype(str)

    # Coverage by day
    bars_per_day = df.groupby("date").agg(
        n_bars=("timestamp_utc", "nunique"),
        n_instruments=("instrument_name", "nunique"),
        total_volume=("volume", "sum"),
    ).reset_index()
    missing_dates = _missing_dates_in_range_from_bars(
        bars_per_day, df["date"].min(), df["date"].max()
    )
    days_with_data = bars_per_day["date"].nunique()
    days_expected = (pd.Timestamp(cfg.END_UTC) - pd.Timestamp(cfg.START_UTC)).days + 1

    # Duplicates
    dup = df.duplicated(subset=["instrument_name", "timestamp_utc"]).sum()

    # Value checks
    bad_price = ((df[["open", "high", "low", "close"]] < 0).any(axis=1)).sum()
    bad_vol = (df["volume"] < 0).sum()
    moneyness_ok = (df["moneyness"] >= cfg.MONEYNESS_MIN) & (df["moneyness"] <= cfg.MONEYNESS_MAX)
    moneyness_outliers = (~moneyness_ok).sum()

    # 5m grid alignment (timestamp % 300000 should be 0 for 5m)
    grid_ok = (df["timestamp_utc"] % (5 * 60 * 1000) == 0).mean() * 100

    # Distribution summaries
    moneyness_q = df["moneyness"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
    strike_q = df["strike"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
    call_put = df["call_put"].value_counts().to_dict()
    expiry_dates = df["expiry_utc"].apply(lambda x: pd.Timestamp(x, unit="ms").date().isoformat()).value_counts().head(20).to_dict()

    report = {
        "n_rows": int(len(df)),
        "n_instruments": int(df["instrument_name"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "day_coverage": {
            "days_with_data": int(days_with_data),
            "days_expected": int(days_expected),
            "missing_dates_count": len(missing_dates),
            "missing_dates_sample": list(missing_dates)[:10],
        },
        "instruments": _instruments_summary(df),
        "timestamp_quality": {
            "duplicate_key_count": int(dup),
            "pct_on_5m_grid": round(float(grid_ok), 2),
        },
        "value_checks": {
            "bad_price_count": int(bad_price),
            "bad_volume_count": int(bad_vol),
            "moneyness_outliers_count": int(moneyness_outliers),
        },
        "distribution": {
            "moneyness_quantiles": {str(k): round(v, 4) for k, v in moneyness_q.items()},
            "strike_quantiles": {str(k): round(v, 2) for k, v in strike_q.items()},
            "call_put": call_put,
            "top_expiries": expiry_dates,
        },
    }
    _write_report(report, df, bars_per_day, missing_dates)
    return report




def _missing_dates_in_range_from_bars(bars_per_day, start_str, end_str):
    if not start_str or not end_str:
        return []
    dr = pd.date_range(start=start_str, end=end_str, freq="D")
    have = set(bars_per_day["date"].astype(str))
    missing = [d.date().isoformat() for d in dr if d.date().isoformat() not in have]
    return missing


def _write_report(report, df, bars_per_day=None, missing_dates=None):
    if missing_dates is None:
        missing_dates = []
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.REPORTS_DIR / "qa_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md = [
        "# Options v2 QA Report",
        "",
        "## Summary",
        f"- **Rows:** {report.get('n_rows', 0):,}",
        f"- **Instruments:** {report.get('n_instruments', 0)}",
        f"- **Date range:** {report.get('date_min', '')} to {report.get('date_max', '')}",
        "",
        "## Day coverage",
        f"- Days with data: {report.get('day_coverage', {}).get('days_with_data', 0)}",
        f"- Missing dates (sample): {report.get('day_coverage', {}).get('missing_dates_sample', [])[:5]}",
        "",
        "## Timestamp quality",
        f"- Duplicate (instrument, timestamp): {report.get('timestamp_quality', {}).get('duplicate_key_count', 0)}",
        f"- % on 5m grid: {report.get('timestamp_quality', {}).get('pct_on_5m_grid', 0)}%",
        "",
        "## Value checks",
        f"- Bad price count: {report.get('value_checks', {}).get('bad_price_count', 0)}",
        f"- Bad volume count: {report.get('value_checks', {}).get('bad_volume_count', 0)}",
        f"- Moneyness outliers: {report.get('value_checks', {}).get('moneyness_outliers_count', 0)}",
        "",
        "## Distribution",
        "- Moneyness quantiles: " + str(report.get("distribution", {}).get("moneyness_quantiles", {})),
        "- Call/Put: " + str(report.get("distribution", {}).get("call_put", {})),
        "",
    ]
    (cfg.REPORTS_DIR / "qa_report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("QA report written to %s", cfg.REPORTS_DIR)
