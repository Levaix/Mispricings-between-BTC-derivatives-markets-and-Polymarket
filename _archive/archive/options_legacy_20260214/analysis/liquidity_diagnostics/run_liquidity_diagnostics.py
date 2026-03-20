#!/usr/bin/env python3
"""
Liquidity / volume stability diagnostics for Deribit BTC options.
Answers: Do liquid strikes move with spot? Does volume concentrate around ATM and shift Jan → Feb?

Outputs: analysis/liquidity_diagnostics/tables/ (CSV/parquet), plots/ (PNG), REPORT.md

To run on a full (non-liquid) options panel, pass --options-path to the full panel parquet
and --output-suffix full so outputs go to separate files (e.g. heatmap_volume_date_strike_full.png).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = SCRIPT_DIR
TABLES_DIR = OUT_DIR / "tables"
PLOTS_DIR = OUT_DIR / "plots"

# Default window (Jan–Feb 2026)
DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-02-15"
TOP_K = 15
MONEYNESS_BINS = np.round(np.arange(0.80, 1.21, 0.02), 2)  # 0.80, 0.82, ..., 1.20
ATM_BAND = (0.95, 1.05)
NEAR_ATM_BAND = (0.90, 1.10)


def _moneyness_bin_labels() -> list[str]:
    return [f"{MONEYNESS_BINS[i]:.2f}-{MONEYNESS_BINS[i+1]:.2f}" for i in range(len(MONEYNESS_BINS) - 1)]


def load_options_data(path: Path | None, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """Load options panel from CSV or parquet; normalize columns; filter to window."""
    if path is None or not path.exists():
        # Try defaults
        candidates = [
            ROOT / "outputs" / "window=2026-01-01_2026-02-15" / "options_panel_long_5m_liquid_aligned.parquet",
            ROOT / "outputs" / "window=2026-01-01_2026-02-15" / "options_panel_long_5m.parquet",
            ROOT / "outputs" / "options_panel_long_5m.csv",
        ]
        for p in candidates:
            if p.exists():
                path = p
                break
    if path is None:
        raise FileNotFoundError("No options panel found. Tried window parquet and outputs/options_panel_long_5m.csv")
    logger.info("Loading options from %s", path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    # Normalize time
    if "open_time" in df.columns and df["open_time"].dtype in (np.int64, np.float64):
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    elif "open_time_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["open_time_utc"], utc=True)
    elif "timestamp_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    else:
        raise ValueError("Need open_time or open_time_utc or timestamp_utc")
    df["date"] = df["ts"].dt.date
    df["week"] = df["ts"].dt.strftime("%Y-W%W")
    # index_close
    if "index_close" not in df.columns:
        master_path = ROOT / "outputs" / "window=2026-01-01_2026-02-15" / "master_panel_5m.parquet"
        if master_path.exists():
            master = pd.read_parquet(master_path)
            if "open_time" in master.columns and "index_close" in master.columns:
                master = master[["open_time", "index_close"]].drop_duplicates()
                df = df.merge(master, on="open_time", how="left")
        if "index_close" not in df.columns:
            df["index_close"] = np.nan
    # moneyness
    if "moneyness_proxy" in df.columns:
        df["moneyness"] = df["moneyness_proxy"]
    elif "index_close" in df.columns and "strike" in df.columns:
        df["moneyness"] = df["strike"] / df["index_close"].replace(0, np.nan)
    else:
        df["moneyness"] = np.nan
    # expiry_utc if missing
    if "expiry_utc" not in df.columns and "expiry" in df.columns:
        df["expiry_utc"] = pd.to_datetime(df["expiry"], unit="ms", utc=True)
    # call_put normalize
    if "call_put" not in df.columns and "option_type" in df.columns:
        df["call_put"] = df["option_type"].astype(str).str.upper().str[:1]
    # volume
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    # Filter window
    df = df[(df["ts"] >= window_start) & (df["ts"] <= window_end)].copy()
    logger.info("Loaded %s rows, date range %s – %s", len(df), df["date"].min(), df["date"].max())
    return df


def check_open_interest(df: pd.DataFrame) -> bool:
    oi_cols = [c for c in df.columns if "open_interest" in c.lower() or c.lower() == "oi"]
    if oi_cols:
        logger.info("Open interest column(s) found: %s", oi_cols)
        return True
    logger.info("OI not found in dataset; proceeding with volume only.")
    return False


# --- 1) Top liquid contracts over time ---
def top_liquid_contracts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vol = df.groupby(["date", "instrument_name"])["volume"].sum().reset_index()
    daily_total = vol.groupby("date")["volume"].sum().rename("day_total")
    vol = vol.merge(daily_total, on="date", how="left")
    vol["share"] = np.where(vol["day_total"] > 0, vol["volume"] / vol["day_total"], 0)
    top_daily = vol.sort_values(["date", "volume"], ascending=[True, False]).groupby("date").head(TOP_K)
    # Weekly
    vol_week = df.groupby(["week", "instrument_name"])["volume"].sum().reset_index()
    week_total = vol_week.groupby("week")["volume"].sum().rename("week_total")
    vol_week = vol_week.merge(week_total, on="week", how="left")
    vol_week["share"] = np.where(vol_week["week_total"] > 0, vol_week["volume"] / vol_week["week_total"], 0)
    top_weekly = vol_week.sort_values(["week", "volume"], ascending=[True, False]).groupby("week").head(TOP_K)
    return top_daily, top_weekly


# --- 2) Strike-level liquidity ---
def strike_level_volume(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["date", "strike", "expiry_utc", "call_put"])["volume"].sum().reset_index()


def pivot_volume_heatmap(vol: pd.DataFrame, index_col: str, col_col: str, value_col: str = "volume") -> pd.DataFrame:
    return vol.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="sum", fill_value=0)


# --- 3) Moneyness-based liquidity ---
def add_moneyness_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["moneyness_bin"] = pd.cut(df["moneyness"], bins=MONEYNESS_BINS, include_lowest=True, labels=_moneyness_bin_labels())
    return df


def moneyness_volume_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moneyness_bin(df)
    df = df[df["moneyness_bin"].notna()]
    return df.groupby(["date", "moneyness_bin"], observed=True)["volume"].sum().reset_index()


def atm_band_share_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["moneyness"].notna()]
    daily_total = df.groupby("date")["volume"].sum().rename("day_total")
    df["in_atm"] = (df["moneyness"] >= ATM_BAND[0]) & (df["moneyness"] <= ATM_BAND[1])
    df["in_near_atm"] = (df["moneyness"] >= NEAR_ATM_BAND[0]) & (df["moneyness"] <= NEAR_ATM_BAND[1])
    atm_vol = df[df["in_atm"]].groupby("date")["volume"].sum().rename("atm_volume")
    near_vol = df[df["in_near_atm"]].groupby("date")["volume"].sum().rename("near_atm_volume")
    out = daily_total.to_frame().join(atm_vol).join(near_vol).fillna(0)
    out["atm_share"] = np.where(out["day_total"] > 0, out["atm_volume"] / out["day_total"], 0)
    out["near_atm_share"] = np.where(out["day_total"] > 0, out["near_atm_volume"] / out["day_total"], 0)
    return out.reset_index()


# --- 4) Stability summary ---
def concentration_top_strikes(df: pd.DataFrame) -> pd.DataFrame:
    by_strike_date = df.groupby(["date", "strike"])["volume"].sum().reset_index()
    daily_total = by_strike_date.groupby("date")["volume"].sum().rename("day_total")
    by_strike_date = by_strike_date.merge(daily_total, on="date")
    by_strike_date = by_strike_date.sort_values(["date", "volume"], ascending=[True, False])
    top5 = by_strike_date.groupby("date").head(5).groupby("date")["volume"].sum().rename("top5_vol")
    top10 = by_strike_date.groupby("date").head(10).groupby("date")["volume"].sum().rename("top10_vol")
    out = daily_total.to_frame().join(top5).join(top10).fillna(0)
    out["pct_top5"] = np.where(out["day_total"] > 0, 100 * out["top5_vol"] / out["day_total"], 0)
    out["pct_top10"] = np.where(out["day_total"] > 0, 100 * out["top10_vol"] / out["day_total"], 0)
    return out.reset_index()


def moneyness_distribution_l1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day distribution of volume across moneyness bins; L1 vs previous day."""
    df = add_moneyness_bin(df)
    df = df[df["moneyness_bin"].notna()]
    daily_bin = df.groupby(["date", "moneyness_bin"], observed=True)["volume"].sum().unstack(fill_value=0)
    daily_bin = daily_bin.div(daily_bin.sum(axis=1), axis=0)
    daily_bin = daily_bin.fillna(0)
    l1_list = []
    prev = None
    for date, row in daily_bin.iterrows():
        if prev is not None:
            l1 = (row - prev).abs().sum() / 2  # L1 distance
            l1_list.append({"date": date, "l1_prev": l1})
        prev = row.values
    return pd.DataFrame(l1_list) if l1_list else pd.DataFrame(columns=["date", "l1_prev"])


def run_diagnostics(options_path: Path | None, start: str, end: str, output_suffix: str | None = None) -> None:
    suf = f"_{output_suffix}" if output_suffix else ""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    window_start = pd.Timestamp(start, tz="UTC")
    window_end = pd.Timestamp(end + " 23:59:59", tz="UTC")
    df = load_options_data(options_path, window_start, window_end)
    if df.empty:
        logger.error("No data in window; aborting.")
        return
    has_oi = check_open_interest(df)
    # --- 1) Top liquid ---
    top_daily, top_weekly = top_liquid_contracts(df)
    top_daily.to_csv(TABLES_DIR / f"top_liquid_contracts_daily{suf}.csv", index=False)
    top_weekly.to_csv(TABLES_DIR / f"top_liquid_contracts_weekly{suf}.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (date, grp) in enumerate(top_daily.groupby("date")):
            tot = grp["volume"].sum()
            if tot == 0:
                continue
            top3 = grp.head(3)
            ax.bar(i, tot, color="steelblue", alpha=0.8)
            ax.text(i, tot, f"{date}\n" + "\n".join(top3["instrument_name"].str[:20]), fontsize=6, ha="center", va="bottom")
        ax.set_xticks(range(len(top_daily["date"].unique())))
        ax.set_xticklabels(sorted(top_daily["date"].unique()), rotation=45, ha="right")
        ax.set_ylabel("Volume (daily)")
        ax.set_title("Daily total volume (top-3 instruments labeled)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"top_liquid_daily{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("Plot top_liquid_daily failed: %s", e)
    # --- 2) Strike-level ---
    strike_vol = strike_level_volume(df)
    strike_vol.to_csv(TABLES_DIR / f"strike_expiry_volume_daily{suf}.csv", index=False)
    heat_daily_strike = pivot_volume_heatmap(
        strike_vol.groupby(["date", "strike"])["volume"].sum().reset_index(),
        "date", "strike",
    )
    heat_daily_strike.to_csv(TABLES_DIR / f"heatmap_volume_by_date_strike{suf}.csv")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(heat_daily_strike.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(0, heat_daily_strike.shape[1], max(1, heat_daily_strike.shape[1] // 15)))
        ax.set_xticklabels([str(heat_daily_strike.columns[i]) for i in range(0, heat_daily_strike.shape[1], max(1, heat_daily_strike.shape[1] // 15))])
        ax.set_yticks(range(heat_daily_strike.shape[0]))
        ax.set_yticklabels(heat_daily_strike.index.astype(str), fontsize=7)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Date")
        ax.set_title("Volume by date and strike (heatmap)")
        plt.colorbar(im, ax=ax, label="Volume")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"heatmap_volume_date_strike{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
        # By expiry: one plot per expiry (small multiples)
        expiries = strike_vol["expiry_utc"].dropna().unique()[:6]
        n_exp = len(expiries)
        if n_exp:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            for idx, exp in enumerate(expiries):
                sub = strike_vol[strike_vol["expiry_utc"].astype(str) == str(exp)]
                if sub.empty:
                    continue
                piv = pivot_volume_heatmap(sub, "date", "strike")
                ax = axes[idx]
                im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
                ax.set_title(str(exp)[:10])
                ax.set_xlabel("Strike")
            for j in range(n_exp, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Volume by date and strike (by expiry)")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"heatmap_volume_by_expiry{suf}.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logger.warning("Strike heatmap plot failed: %s", e)
    # --- 3) Moneyness ---
    money_daily = moneyness_volume_daily(df)
    money_daily.to_csv(TABLES_DIR / f"moneyness_volume_daily{suf}.csv", index=False)
    atm_series = atm_band_share_series(df)
    atm_series.to_csv(TABLES_DIR / f"atm_near_atm_share_daily{suf}.csv", index=False)
    heat_money = pivot_volume_heatmap(money_daily, "date", "moneyness_bin")
    heat_money.to_csv(TABLES_DIR / f"heatmap_volume_date_moneyness{suf}.csv")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(heat_money.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(heat_money.shape[0]))
        ax.set_yticklabels(heat_money.index.astype(str), fontsize=7)
        ax.set_xticks(range(heat_money.shape[1]))
        ax.set_xticklabels(heat_money.columns, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Moneyness bin")
        ax.set_ylabel("Date")
        ax.set_title("Volume by date and moneyness (strike/index)")
        plt.colorbar(im, ax=ax, label="Volume")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"heatmap_volume_date_moneyness{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(atm_series["date"].astype(str), atm_series["atm_share"], label=f"ATM {ATM_BAND[0]}-{ATM_BAND[1]}", marker="o", markersize=4)
        ax.plot(atm_series["date"].astype(str), atm_series["near_atm_share"], label=f"Near-ATM {NEAR_ATM_BAND[0]}-{NEAR_ATM_BAND[1]}", marker="s", markersize=4)
        ax.set_ylabel("Share of daily volume")
        ax.set_xlabel("Date")
        ax.set_title("Volume share in ATM and near-ATM bands over time")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"atm_band_share_timeseries{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("Moneyness plot failed: %s", e)
    # --- 4) Stability summary ---
    conc = concentration_top_strikes(df)
    conc.to_csv(TABLES_DIR / f"concentration_top5_top10_daily{suf}.csv", index=False)
    l1_df = moneyness_distribution_l1(df)
    l1_df.to_csv(TABLES_DIR / f"moneyness_distribution_l1_prev_day{suf}.csv", index=False)
    summary = conc.merge(atm_series[["date", "atm_share", "near_atm_share"]], on="date", how="left")
    summary = summary.merge(l1_df, on="date", how="left")
    summary["atm_share_pct"] = summary["atm_share"] * 100
    summary["near_atm_share_pct"] = summary["near_atm_share"] * 100
    summary.to_csv(TABLES_DIR / f"stability_summary_daily{suf}.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes[0, 0]
        ax.bar(range(len(conc)), conc["pct_top5"], label="Top 5 strikes", alpha=0.8)
        ax.bar(range(len(conc)), conc["pct_top10"] - conc["pct_top5"], bottom=conc["pct_top5"], label="Strikes 6–10", alpha=0.8)
        ax.set_xticks(range(len(conc)))
        ax.set_xticklabels(conc["date"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("% of daily volume")
        ax.set_title("Concentration: top 5 vs top 10 strikes")
        ax.legend()
        ax = axes[0, 1]
        ax.plot(range(len(summary)), summary["atm_share_pct"], marker="o", label="ATM 0.95–1.05")
        ax.plot(range(len(summary)), summary["near_atm_share_pct"], marker="s", label="Near-ATM 0.90–1.10")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["date"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("% volume")
        ax.set_title("Volume share in ATM bands")
        ax.legend()
        ax = axes[1, 0]
        if not l1_df.empty:
            ax.plot(range(len(l1_df)), l1_df["l1_prev"], marker="o", color="green")
            ax.set_xticks(range(len(l1_df)))
            ax.set_xticklabels(l1_df["date"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("L1 distance")
        ax.set_title("Moneyness distribution shift vs previous day")
        ax = axes[1, 1]
        ax.axis("off")
        ax.text(0.1, 0.9, "Summary metrics (see stability_summary_daily.csv)", fontsize=12)
        ax.text(0.1, 0.7, f"Date range: {df['date'].min()} – {df['date'].max()}", fontsize=10)
        ax.text(0.1, 0.5, f"OI in data: {has_oi}", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"stability_metrics{suf}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("Stability plot failed: %s", e)
    # --- Report ---
    write_report(df, summary, has_oi, output_suffix)
    logger.info("Done. Tables: %s  Plots: %s  Report: %s", TABLES_DIR, PLOTS_DIR, OUT_DIR / f"REPORT{suf}.md")


def write_report(df: pd.DataFrame, summary: pd.DataFrame, has_oi: bool, output_suffix: str | None = None) -> None:
    suf = f"_{output_suffix}" if output_suffix else ""
    md = [
        "# Liquidity diagnostics report",
        "",
        "## Data",
        f"- Window: {df['date'].min()} – {df['date'].max()}",
        f"- Rows: {len(df):,}",
        f"- Unique instruments: {df['instrument_name'].nunique()}",
        f"- Open interest in dataset: **{'Yes' if has_oi else 'No (volume only)'}**",
        "",
        "## Do liquid strikes move with spot?",
        "",
        "1. **Top liquid contracts over time**",
        f"   - `tables/top_liquid_contracts_daily{suf}.csv` — Top {TOP_K} contracts by volume per day (with share of day total).",
        f"   - `tables/top_liquid_contracts_weekly{suf}.csv` — Same by week.",
        f"   - `plots/top_liquid_daily{suf}.png` — Daily total volume with top-3 instruments labeled.",
        "   - If the same few instruments stay at the top across Jan and Feb, liquidity is stable; if they change, liquid strikes are moving.",
        "",
        "2. **Strike-level liquidity**",
        f"   - `tables/strike_expiry_volume_daily{suf}.csv` — Volume by date, strike, expiry, call_put.",
        f"   - `tables/heatmap_volume_by_date_strike{suf}.csv` — Pivot: rows=date, columns=strike, cell=volume.",
        f"   - `plots/heatmap_volume_date_strike{suf}.png` — Heatmap of volume by date vs strike.",
        f"   - `plots/heatmap_volume_by_expiry{suf}.png` — Small multiples per expiry.",
        "   - Heat moving along the strike axis over time suggests liquidity following spot.",
        "",
        "3. **Moneyness-based liquidity (main diagnostic)**",
        f"   - `tables/moneyness_volume_daily{suf}.csv` — Volume per day per moneyness bin (strike/index).",
        f"   - `tables/atm_near_atm_share_daily{suf}.csv` — Daily share of volume in ATM (0.95–1.05) and near-ATM (0.90–1.10).",
        f"   - `plots/heatmap_volume_date_moneyness{suf}.png` — Heatmap: date vs moneyness bin.",
        f"   - `plots/atm_band_share_timeseries{suf}.png` — Time series of % volume in ATM and near-ATM bands.",
        "   - If volume stays concentrated in the same moneyness band (e.g. 0.98–1.02) as time passes, liquidity tracks spot. If the band shifts or disperses, regime change.",
        "",
        "4. **Stability summary metrics**",
        f"   - `tables/stability_summary_daily{suf}.csv` — Per day: pct_top5, pct_top10, atm_share_pct, near_atm_share_pct, l1_prev (moneyness dist shift vs previous day).",
        f"   - `tables/concentration_top5_top10_daily{suf}.csv` — Raw concentration.",
        f"   - `tables/moneyness_distribution_l1_prev_day{suf}.csv` — L1 distance of volume-by-moneyness vs previous day.",
        f"   - `plots/stability_metrics{suf}.png` — Four panels: concentration, ATM share, L1 shift, summary text.",
        "",
        "## Quick takeaway",
        "",
    ]
    if not summary.empty and "atm_share_pct" in summary.columns:
        mean_atm = summary["atm_share_pct"].mean()
        mean_near = summary["near_atm_share_pct"].mean()
        md.append(f"- Mean % volume in ATM band (0.95–1.05): **{mean_atm:.1f}%**.")
        md.append(f"- Mean % volume in near-ATM band (0.90–1.10): **{mean_near:.1f}%**.")
    if not summary.empty and "pct_top5" in summary.columns:
        md.append(f"- Mean % volume in top 5 strikes: **{summary['pct_top5'].mean():.1f}%**.")
    md.append("")
    md.append("Open the heatmaps and ATM band time series to see whether liquid moneyness is stable or shifts between Jan and Feb.")
    (OUT_DIR / f"REPORT{suf}.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Liquidity diagnostics for Deribit options")
    parser.add_argument("--options-path", type=Path, default=None, help="Path to options panel CSV/parquet")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--output-suffix", type=str, default=None, help="Suffix for output files (e.g. 'full' -> heatmap_volume_date_strike_full.png)")
    args = parser.parse_args()
    run_diagnostics(args.options_path, args.start, args.end, args.output_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
