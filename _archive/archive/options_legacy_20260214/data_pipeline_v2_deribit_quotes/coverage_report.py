"""
Coverage metrics per instrument: 5m bins filled, first/last observation, % forward-filled.
Output: JSON and optional markdown report.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

BAR_MS = 5 * 60 * 1000


def build_coverage_report(
    aligned: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """
    aligned must have open_time, instrument_name, and at least one quote column (e.g. mid_price or best_bid_price).
    Returns dict with window_start_ms, window_end_ms, n_expected_bars, per_instrument list.
    """
    if aligned.empty or "instrument_name" not in aligned.columns:
        return {"window_start_ms": start_ms, "window_end_ms": end_ms, "per_instrument": []}
    n_expected = (end_ms - start_ms) // BAR_MS + 1
    value_col = "mid_price" if "mid_price" in aligned.columns else "best_bid_price"
    if value_col not in aligned.columns:
        value_col = [c for c in ["best_bid_price", "best_ask_price", "iv_mid"] if c in aligned.columns]
        value_col = value_col[0] if value_col else None
    if value_col is None:
        return {"window_start_ms": start_ms, "window_end_ms": end_ms, "per_instrument": []}
    rows: List[Dict[str, Any]] = []
    for inv, grp in aligned.groupby("instrument_name"):
        grp = grp.sort_values("open_time")
        filled = grp[value_col].notna()
        n_filled = filled.sum()
        first_obs = grp.loc[filled, "open_time"].min() if filled.any() else None
        last_obs = grp.loc[filled, "open_time"].max() if filled.any() else None
        pct_filled = 100.0 * n_filled / n_expected if n_expected > 0 else 0.0
        rows.append({
            "instrument_name": inv,
            "n_bars_filled": int(n_filled),
            "n_expected_bars": n_expected,
            "pct_bars_filled": round(pct_filled, 2),
            "first_obs_open_time_ms": int(first_obs) if first_obs is not None else None,
            "last_obs_open_time_ms": int(last_obs) if last_obs is not None else None,
        })
    return {
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "n_expected_bars": n_expected,
        "n_instruments": len(rows),
        "per_instrument": rows,
    }


def write_coverage_report(
    report: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Write report to out_dir/deribit_quotes_coverage.json and deribit_quotes_coverage.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "deribit_quotes_coverage.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", json_path)
    md_path = out_dir / "deribit_quotes_coverage.md"
    lines = [
        "# Deribit quotes coverage",
        "",
        f"- Window: {report.get('window_start_ms')} – {report.get('window_end_ms')} ms",
        f"- Expected 5m bars: {report.get('n_expected_bars')}",
        f"- Instruments: {report.get('n_instruments')}",
        "",
        "## Per-instrument",
        "",
        "| instrument_name | n_bars_filled | pct_bars_filled | first_obs_ms | last_obs_ms |",
        "|------------------|---------------|-----------------|--------------|-------------|",
    ]
    for r in report.get("per_instrument", []):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                r.get("instrument_name", ""),
                r.get("n_bars_filled", ""),
                r.get("pct_bars_filled", ""),
                r.get("first_obs_open_time_ms") or "",
                r.get("last_obs_open_time_ms") or "",
            )
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", md_path)
