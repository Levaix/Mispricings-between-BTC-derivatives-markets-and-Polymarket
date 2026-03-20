"""
Step 5: Build master 5-minute panel dataset (Polymarket aligned).

Reads data/polymarket/probabilities_5m.parquet and markets_selected.csv,
builds a full UTC 5m time grid for the window, and outputs long (and optional wide)
panel under outputs/window=.../panel/. No dependency on question text; uses
market_id / event_id / expiry / strike from metadata only.

Usage:
  python scripts/step5_build_master_panel.py --window_dir "outputs/window=2026-01-01_2026-02-15"
  python scripts/step5_build_master_panel.py --window_dir "outputs/window=2026-01-01_2026-02-15" --overwrite --allow_missing_markets

Exit: 0 on success; raises or exits 1 on failure.

Error handling:
- Missing probability or metadata file: raises FileNotFoundError with expected paths.
- Timestamps not on 5m grid: rounded down to 5m boundary with a warning.
- window_dir dates vs metadata expiry dates differ: warn and use window_dir as authoritative.
- Event in probabilities but not in metadata: warn and drop those rows.
- Event in metadata but no probability rows: warn and drop; fail if n_markets < expected unless --allow_missing_markets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Default window (from window_dir name convention: window=START_END)
WINDOW_START = "2026-01-01 00:00:00"
WINDOW_END = "2026-02-15 23:55:00"
GRID_FREQ_MS = 5 * 60 * 1000  # 5 minutes in ms

EXPECTED_N_MARKETS = 506
EXPECTED_N_DATES = 46
MIN_POINTS_PER_MARKET = 50
MAX_MISSING_PCT_DEFAULT = 90.0  # fail if any market has > this % missing (catastrophic); use e.g. 2 for strict
PROB_EPS = 1e-9


def _parse_window_from_dir(window_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse start/end from folder name window=YYYY-MM-DD_YYYY-MM-DD; fallback to defaults."""
    name = window_dir.name
    if name.startswith("window="):
        part = name.replace("window=", "")
        if "_" in part:
            a, b = part.split("_", 1)
            try:
                start = pd.Timestamp(a + " 00:00:00", tz="UTC")
                end = pd.Timestamp(b + " 23:55:00", tz="UTC")
                return start, end
            except Exception:
                pass
    start = pd.Timestamp(WINDOW_START, tz="UTC")
    end = pd.Timestamp(WINDOW_END, tz="UTC")
    return start, end


def _build_5m_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Build canonical 5m UTC grid with ts_utc and open_time_ms."""
    ts = pd.date_range(start=start, end=end, freq="5min", tz="UTC")
    grid = pd.DataFrame({
        "ts_utc": ts,
        "open_time_ms": ts.astype("int64") // 10**6,
    })
    return grid


def _ensure_5m_grid(times_ms: pd.Series) -> tuple[pd.Series, bool]:
    """Round ms timestamps to 5m boundary (floor). Returns (rounded_ms, any_off_grid)."""
    rounded = (times_ms // GRID_FREQ_MS) * GRID_FREQ_MS
    off_grid = (times_ms != rounded).any()
    return rounded, off_grid


def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Load markets_selected.csv; normalize to market_id, event_id, expiry_date, expiry_ts, strike (no question text)."""
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Markets metadata not found: {meta_path}\n"
            f"Expected: data/polymarket/markets_selected.csv"
        )
    df = pd.read_csv(meta_path)
    required = {"event_id", "market_id", "parsed_date", "strike"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Markets metadata missing columns: {missing}. Path: {meta_path}")
    # expiry_ts: prefer expiry_utc, else expiry_ms, else parsed_date 17:00
    if "expiry_utc" in df.columns:
        df["expiry_ts"] = pd.to_datetime(df["expiry_utc"], utc=True, errors="coerce")
    elif "expiry_ms" in df.columns:
        df["expiry_ts"] = pd.to_datetime(df["expiry_ms"], unit="ms", utc=True, errors="coerce")
    else:
        df["expiry_ts"] = pd.to_datetime(df["parsed_date"].astype(str) + " 17:00:00", utc=True, errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_ts"], utc=True).dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    # Keep market_id consistent (int if possible)
    df["market_id"] = pd.to_numeric(df["market_id"], errors="coerce").fillna(df["market_id"].astype(str))
    return df[["event_id", "market_id", "expiry_date", "expiry_ts", "strike"]].dropna(subset=["event_id", "market_id"])


def load_probabilities(probs_path: Path) -> pd.DataFrame:
    """Load probabilities_5m.parquet; normalize to event_id, open_time_ms, prob_yes; round to 5m."""
    if not probs_path.exists():
        raise FileNotFoundError(
            f"Probability history not found: {probs_path}\n"
            f"Expected: data/polymarket/probabilities_5m.parquet"
        )
    df = pd.read_parquet(probs_path)
    for col in ["event_id", "probability_pm"]:
        if col not in df.columns:
            raise ValueError(f"Probability file missing column: {col}. Path: {probs_path}")
    time_col = "open_time" if "open_time" in df.columns else None
    if not time_col or df[time_col].dtype.kind not in "iu":
        raise ValueError(f"Probability file must have integer open_time (ms). Path: {probs_path}")
    open_ms = pd.Series(df[time_col].values, dtype="int64")
    open_ms_5m, off_grid = _ensure_5m_grid(open_ms)
    if off_grid:
        print("  WARNING: Some timestamps were not on 5m grid; rounded down to 5m boundary.")
    df = df.assign(
        open_time_ms=open_ms_5m.values,
        prob_yes=pd.to_numeric(df["probability_pm"], errors="coerce").clip(0.0, 1.0),
    )
    return df[["event_id", "open_time_ms", "prob_yes"]].dropna(subset=["event_id", "open_time_ms", "prob_yes"])


def build_long_panel(
    grid: pd.DataFrame,
    meta: pd.DataFrame,
    probs: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build long panel: one row per (ts_utc, market_id) with ts_utc, open_time_ms, market_id, expiry_date, expiry_ts, strike, prob_yes."""
    # Join probs with meta on event_id
    probs = probs.merge(meta[["event_id", "market_id", "expiry_date", "expiry_ts", "strike"]], on="event_id", how="inner")
    # Map open_time_ms -> ts_utc
    probs["ts_utc"] = pd.to_datetime(probs["open_time_ms"], unit="ms", utc=True)
    # Restrict to window
    probs = probs[(probs["ts_utc"] >= window_start) & (probs["ts_utc"] <= window_end)]
    # Full grid x markets
    markets = meta[["market_id", "expiry_date", "expiry_ts", "strike"]].drop_duplicates("market_id")
    full = grid.merge(markets, how="cross")
    # Merge in prob_yes (left join); grid already has open_time_ms so drop from agg to avoid duplicate
    agg = probs.groupby(["ts_utc", "market_id"], as_index=False)["prob_yes"].first()
    full = full.merge(agg, on=["ts_utc", "market_id"], how="left")
    full["open_time_ms"] = full["open_time_ms"].astype("int64")
    # Column order
    return full[["ts_utc", "open_time_ms", "market_id", "expiry_date", "expiry_ts", "strike", "prob_yes"]]


def build_wide_panel_from_probs(
    grid: pd.DataFrame,
    meta: pd.DataFrame,
    probs: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Wide panel: one row per ts_utc, columns open_time_ms + one per event_id (prob_<event_id>)."""
    probs = probs.merge(meta[["event_id", "market_id"]], on="event_id", how="inner")
    probs["ts_utc"] = pd.to_datetime(probs["open_time_ms"], unit="ms", utc=True)
    probs = probs[(probs["ts_utc"] >= window_start) & (probs["ts_utc"] <= window_end)]
    # Dedupe (ts_utc, event_id) keep first
    probs = probs.groupby(["ts_utc", "event_id"], as_index=False).agg({"prob_yes": "first", "open_time_ms": "first"})
    wide = probs.pivot(index="ts_utc", columns="event_id", values="prob_yes")
    wide.columns = [f"prob_{c}" for c in wide.columns]
    wide = wide.reset_index()
    wide = grid[["ts_utc", "open_time_ms"]].merge(wide, on="ts_utc", how="left")
    return wide


def run_qa(
    long_df: pd.DataFrame,
    meta: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    expected_n_markets: int,
    expected_n_dates: int,
    min_points: int,
    max_missing_pct: float,
    allow_missing_markets: bool,
) -> tuple[bool, dict]:
    """Run QA checks; return (all_pass, summary_dict)."""
    summary = {}
    all_pass = True

    # Expected markets
    n_markets = long_df["market_id"].nunique()
    summary["n_markets"] = int(n_markets)
    if n_markets != expected_n_markets and not allow_missing_markets:
        print(f"  [QA] FAIL: n_markets={n_markets} (expected {expected_n_markets})")
        all_pass = False
    else:
        print(f"  [QA] PASS: n_markets={n_markets} (expected {expected_n_markets})")

    # Expected expiry dates
    n_dates = long_df["expiry_date"].nunique()
    summary["n_expiry_dates"] = int(n_dates)
    if n_dates != expected_n_dates:
        print(f"  [QA] FAIL: n_expiry_dates={n_dates} (expected {expected_n_dates})")
        all_pass = False
    else:
        print(f"  [QA] PASS: n_expiry_dates={n_dates} (expected {expected_n_dates})")

    # Min points per market (non-null prob_yes)
    points = long_df.groupby("market_id")["prob_yes"].apply(lambda x: x.notna().sum())
    below = points[points < min_points]
    summary["min_points_per_market"] = int(points.min())
    summary["markets_below_min_points"] = int(len(below))
    if len(below) > 0:
        print(f"  [QA] FAIL: {len(below)} markets with < {min_points} points (min={points.min()})")
        all_pass = False
    else:
        print(f"  [QA] PASS: each market has >= {min_points} points (min={points.min()})")

    # Missing % per market: expected = grid points from window_start up to market's expiry_ts (inclusive)
    grid_ts = long_df[["ts_utc"]].drop_duplicates().sort_values("ts_utc")
    def expected_points(market_id):
        exp = meta[meta["market_id"] == market_id]["expiry_ts"].iloc[0]
        return (grid_ts["ts_utc"] <= exp).sum()
    expected_per_market = long_df["market_id"].unique()
    expected_per_market = pd.Series({m: expected_points(m) for m in expected_per_market})
    actual = long_df.groupby("market_id")["prob_yes"].apply(lambda x: x.notna().sum())
    actual = actual.reindex(expected_per_market.index).fillna(0)
    missing_pct = (1 - actual / expected_per_market) * 100
    summary["pct_missing_per_market_max"] = float(missing_pct.max())
    summary["pct_missing_per_market_median"] = float(missing_pct.median())
    if (missing_pct > max_missing_pct).any():
        n_bad = (missing_pct > max_missing_pct).sum()
        print(f"  [QA] FAIL: {n_bad} markets with > {max_missing_pct}% missing (max={missing_pct.max():.2f}%)")
        all_pass = False
    else:
        print(f"  [QA] PASS: no market with > {max_missing_pct}% missing (max={missing_pct.max():.2f}%, median={missing_pct.median():.2f}%)")

    # ts_utc within window
    ts_min = long_df["ts_utc"].min()
    ts_max = long_df["ts_utc"].max()
    summary["ts_utc_min"] = str(ts_min)
    summary["ts_utc_max"] = str(ts_max)
    if ts_min < window_start or ts_max > window_end:
        print(f"  [QA] FAIL: ts_utc range [{ts_min}, {ts_max}] outside window [{window_start}, {window_end}]")
        all_pass = False
    else:
        print(f"  [QA] PASS: ts_utc in window [{window_start}, {window_end}]")

    # prob_yes in [0,1]
    probs = long_df["prob_yes"].dropna()
    if len(probs) > 0:
        in_range = ((probs >= -PROB_EPS) & (probs <= 1 + PROB_EPS)).all()
        summary["prob_yes_min"] = float(probs.min())
        summary["prob_yes_max"] = float(probs.max())
        if not in_range:
            print(f"  [QA] FAIL: prob_yes outside [0,1] (min={probs.min()}, max={probs.max()})")
            all_pass = False
        else:
            print(f"  [QA] PASS: prob_yes in [0,1]")
    else:
        print("  [QA] WARN: no non-null prob_yes to check")

    # No duplicates (ts_utc, market_id)
    dupes = long_df.duplicated(subset=["ts_utc", "market_id"]).sum()
    summary["duplicate_rows"] = int(dupes)
    if dupes > 0:
        print(f"  [QA] FAIL: {dupes} duplicate (ts_utc, market_id) rows")
        all_pass = False
    else:
        print(f"  [QA] PASS: no duplicate (ts_utc, market_id)")

    summary["qa_pass"] = all_pass
    return all_pass, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: Build master 5m panel (Polymarket aligned)")
    parser.add_argument("--window_dir", type=str, required=True, help="e.g. outputs/window=2026-01-01_2026-02-15")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing panel files")
    parser.add_argument("--out_format", type=str, default="parquet", choices=["parquet"], help="Output format (default parquet)")
    parser.add_argument("--expected_n_markets", type=int, default=EXPECTED_N_MARKETS)
    parser.add_argument("--expected_n_dates", type=int, default=EXPECTED_N_DATES)
    parser.add_argument("--min_points", type=int, default=MIN_POINTS_PER_MARKET)
    parser.add_argument("--max_missing_pct", type=float, default=MAX_MISSING_PCT_DEFAULT)
    parser.add_argument("--allow_missing_markets", action="store_true", help="Do not fail if n_markets < expected")
    args = parser.parse_args()

    window_dir = Path(args.window_dir)
    if not window_dir.is_absolute():
        window_dir = (ROOT / args.window_dir).resolve()
    if not window_dir.exists():
        print(f"ERROR: window_dir does not exist: {window_dir}")
        return 1

    data_pm = ROOT / "data" / "polymarket"
    probs_path = data_pm / "probabilities_5m.parquet"
    meta_path = data_pm / "markets_selected.csv"

    print("Step 5: Build master 5m panel")
    print("  Window dir:", window_dir)

    window_start, window_end = _parse_window_from_dir(window_dir)
    print("  Window:", window_start, "->", window_end)

    meta = load_metadata(meta_path)
    print("  Loaded metadata:", len(meta), "markets (by event_id)")

    probs_raw = load_probabilities(probs_path)
    print("  Loaded probabilities:", len(probs_raw), "rows")

    # Markets in probs but not in metadata -> warn and drop
    in_probs = set(probs_raw["event_id"].unique())
    in_meta = set(meta["event_id"].unique())
    only_probs = in_probs - in_meta
    only_meta = in_meta - in_probs
    if only_probs:
        print(f"  WARNING: {len(only_probs)} event_ids in probabilities not in metadata; dropping those rows.")
        probs_raw = probs_raw[probs_raw["event_id"].isin(in_meta)]
    if only_meta:
        print(f"  WARNING: {len(only_meta)} event_ids in metadata have no probability rows.")
    meta = meta[meta["event_id"].isin(probs_raw["event_id"].unique())]
    n_after = meta["market_id"].nunique()
    if n_after < args.expected_n_markets and not args.allow_missing_markets:
        print(f"  ERROR: After dropping missing, n_markets={n_after} < expected {args.expected_n_markets}. Use --allow_missing_markets to proceed.")
        return 1

    # Window vs metadata dates
    meta_dates = set(meta["expiry_date"].dropna().astype(str))
    window_dates = set(pd.date_range(window_start.date(), window_end.date(), freq="D").strftime("%Y-%m-%d"))
    if meta_dates and window_dates and (meta_dates - window_dates or window_dates - meta_dates):
        print("  WARNING: window_dir dates and metadata expiry dates differ; using window_dir as authoritative.")

    grid = _build_5m_grid(window_start, window_end)
    print("  Grid size:", len(grid), "timestamps")

    long_df = build_long_panel(grid, meta, probs_raw.copy(), window_start, window_end)
    print("  Long panel:", len(long_df), "rows")

    panel_dir = window_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    long_path = panel_dir / "master_panel_long.parquet"
    if long_path.exists() and not args.overwrite:
        print("  Skip writing (exists and no --overwrite):", long_path)
    else:
        long_df.to_parquet(long_path, index=False)
        print("  Wrote:", long_path)

    wide_df = build_wide_panel_from_probs(grid, meta, probs_raw.copy(), window_start, window_end)
    wide_path = panel_dir / "master_panel_wide.parquet"
    if wide_path.exists() and not args.overwrite:
        print("  Skip writing (exists and no --overwrite):", wide_path)
    else:
        wide_df.to_parquet(wide_path, index=False)
        print("  Wrote:", wide_path)

    all_pass, summary = run_qa(
        long_df, meta, window_start, window_end,
        args.expected_n_markets, args.expected_n_dates, args.min_points, args.max_missing_pct,
        args.allow_missing_markets,
    )
    summary["n_rows_long"] = int(len(long_df))
    summary["n_rows_wide"] = int(len(wide_df))
    summary["n_ts_utc"] = int(grid.shape[0])

    with open(panel_dir / "step5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# Step 5 report",
        "",
        f"- Long panel: {long_path.name} ({summary['n_rows_long']:,} rows)",
        f"- Wide panel: {wide_path.name} ({summary['n_rows_wide']:,} rows)",
        f"- QA pass: {summary['qa_pass']}",
        "",
        "## Schema (long)",
        "- ts_utc: datetime64[ns, UTC]",
        "- open_time_ms: int64 (ms since epoch)",
        "- market_id: int or str",
        "- expiry_date: date",
        "- expiry_ts: datetime64[ns, UTC]",
        "- strike: int",
        "- prob_yes: float64 [0,1]",
    ]
    with open(panel_dir / "step5_report.md", "w") as f:
        f.write("\n".join(report_lines))

    print("")
    print("PASS" if all_pass else "FAIL")
    print("  Summary:", panel_dir / "step5_summary.json")
    print("  Report:", panel_dir / "step5_report.md")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
