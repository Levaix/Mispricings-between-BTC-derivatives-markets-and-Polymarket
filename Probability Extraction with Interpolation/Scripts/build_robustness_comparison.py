"""
Build robustness comparison across methodology variants 1-4 from existing JSON outputs.
Reads options_smile_report.json and options_iv_timestamp_coverage.json from each folder.
Writes: robustness_method_comparison.csv, robustness_method_comparison.md, robustness_plots/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

ANALYSIS_DIR = Path(__file__).resolve().parent

METHODS = [
    ("1h_window_no_staleness_min5", "Method 1: 1h / no staleness / min5"),
    ("3h_window_staleness90_min5", "Method 2: 3h / 90m stale / min5"),
    ("3h_window_staleness90_min3_gap10k", "Method 3: 3h / 90m stale / min3 / gap10k"),
    ("3h_window_staleness120_min3_gap10k", "Method 4: 3h / 120m stale / min3 / gap10k"),
]

METHOD_DESCRIPTIONS = {
    "Method 1: 1h / no staleness / min5": {
        "lookback_window": "1h",
        "staleness_rule": "none",
        "min_strikes": 5,
        "strike_gap_rule": "none",
        "latest_trade_per_strike": "no",
    },
    "Method 2: 3h / 90m stale / min5": {
        "lookback_window": "3h",
        "staleness_rule": "90 min",
        "min_strikes": 5,
        "strike_gap_rule": "none",
        "latest_trade_per_strike": "no",
    },
    "Method 3: 3h / 90m stale / min3 / gap10k": {
        "lookback_window": "3h",
        "staleness_rule": "90 min",
        "min_strikes": 3,
        "strike_gap_rule": "<=10k",
        "latest_trade_per_strike": "yes",
    },
    "Method 4: 3h / 120m stale / min3 / gap10k": {
        "lookback_window": "3h",
        "staleness_rule": "120 min",
        "min_strikes": 3,
        "strike_gap_rule": "<=10k",
        "latest_trade_per_strike": "yes",
    },
}

MAIN_TRADEOFF = {
    "Method 1: 1h / no staleness / min5": "Most conservative; lowest coverage, many rows without fit.",
    "Method 2: 3h / 90m stale / min5": "Better coverage with 3h + staleness; more out-of-range.",
    "Method 3: 3h / 90m stale / min3 / gap10k": "Latest-trade + gap cap; high coverage, some gap fails.",
    "Method 4: 3h / 120m stale / min3 / gap10k": "Relaxed staleness; highest coverage, highest out-of-range.",
}


def _get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data


def load_method_data(folder: str) -> tuple[Optional[Dict], Optional[Dict]]:
    base = ANALYSIS_DIR / folder
    report_path = base / "options_smile_report.json"
    coverage_path = base / "options_iv_timestamp_coverage.json"
    report = None
    coverage = None
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    if coverage_path.exists():
        with open(coverage_path, encoding="utf-8") as f:
            coverage = json.load(f)
    return report, coverage


def build_row(folder: str, label: str, report: Optional[Dict], coverage: Optional[Dict]) -> Dict[str, Any]:
    s = _get(report or {}, "summary") or {}
    ff = _get(report or {}, "summary", "fit_fail_counts") or {}
    td = _get(coverage or {}, "timestamp_diagnostics") or {}
    ivd = _get(coverage or {}, "iv_density_per_timestamp") or {}
    stal = _get(report or {}, "summary", "staleness_diagnostics") or {}
    gapd = _get(report or {}, "summary", "strike_gap_diagnostics") or {}

    total_windows = _get(report or {}, "summary", "total_windows")
    fit_success = _get(report or {}, "summary", "fit_success_count")
    fit_success_rate = (fit_success / total_windows * 100) if total_windows else None
    usable_iv_count = _get(report or {}, "summary", "usable_iv_count")
    usable_iv_rate_among_fits = (usable_iv_count / fit_success * 100) if (fit_success and usable_iv_count is not None) else None

    desc = METHOD_DESCRIPTIONS.get(label, {})
    row = {
        "method_label": label,
        "folder": folder,
        "lookback_window": desc.get("lookback_window", ""),
        "staleness_rule": desc.get("staleness_rule", ""),
        "min_strikes": desc.get("min_strikes", ""),
        "strike_gap_rule": desc.get("strike_gap_rule", ""),
        "latest_trade_per_strike": desc.get("latest_trade_per_strike", ""),
        "usable_iv_rows": _get(coverage or {}, "usable_iv_rows"),
        "pct_rows_usable_iv": _get(coverage or {}, "pct_rows_with_iv") or s.get("pct_rows_usable_iv"),
        "timestamps_with_iv": _get(coverage or {}, "timestamp_diagnostics", "timestamps_with_iv"),
        "pct_timestamps_with_iv": _get(coverage or {}, "timestamp_diagnostics", "pct_timestamps_with_iv"),
        "avg_usable_iv_per_timestamp": _get(coverage or {}, "iv_density_per_timestamp", "avg_usable_iv_per_timestamp"),
        "median_usable_iv_per_timestamp": _get(coverage or {}, "iv_density_per_timestamp", "median_usable_iv_per_timestamp"),
        "total_windows": total_windows,
        "fit_success_count": fit_success,
        "fit_success_rate_pct": round(fit_success_rate, 4) if fit_success_rate is not None else None,
        "too_few_strikes": ff.get("too_few_strikes"),
        "range_too_small": ff.get("range_too_small"),
        "no_iv_data": ff.get("no_iv_data"),
        "out_of_range_count": s.get("out_of_range_count"),
        "pct_rows_out_of_range": s.get("pct_rows_out_of_range"),
        "no_snapshot_fit_count": s.get("no_snapshot_fit_count"),
        "pct_rows_no_snapshot_fit": s.get("pct_rows_no_snapshot_fit"),
        "total_retained_strikes_after_staleness_filter": stal.get("total_retained_strikes_after_staleness_filter")
            or stal.get("total_retained_strikes_after_staleness"),
        "total_discarded_stale_strikes": stal.get("total_discarded_stale_strikes"),
        "avg_retained_strikes_per_snapshot": stal.get("avg_retained_strikes_per_snapshot"),
        "avg_discarded_stale_strikes_per_snapshot": stal.get("avg_discarded_stale_strikes_per_snapshot"),
        "in_range_gap_rule_fail_count": gapd.get("in_range_gap_rule_fail_count"),
        "avg_local_gap_successful_interpolation": gapd.get("avg_local_gap_successful_interpolation"),
        "max_local_gap_successful_interpolation": gapd.get("max_local_gap_successful_interpolation"),
        "usable_iv_rate_among_fits_pct": round(usable_iv_rate_among_fits, 4) if usable_iv_rate_among_fits is not None else None,
        "main_tradeoff": MAIN_TRADEOFF.get(label, ""),
    }
    return row


def _na(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float):
        return str(round(v, 4)) if v == int(v) else str(round(v, 4))
    return str(v)


def write_csv(rows: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def write_md(rows: List[Dict], out_path: Path) -> None:
    lines = [
        "# Robustness comparison: options smile methodology variants 1–4",
        "",
        "Comparison built from existing `options_smile_report.json` and `options_iv_timestamp_coverage.json` in each methodology folder. No pipelines were rerun.",
        "",
        "## Summary table",
        "",
        "| Method | Coverage (usable IV %) | Fit success % | Out-of-range % | Avg usable event rows per successful smile | Main tradeoff |",
        "|--------|-------------------------|---------------|----------------|-------------------------------------------|----------------|",
    ]
    for r in rows:
        cov = _na(r["pct_rows_usable_iv"])
        fit = _na(r["fit_success_rate_pct"])
        oor = _na(r["pct_rows_out_of_range"])
        uiaf = _na(r["usable_iv_rate_among_fits_pct"])
        tradeoff = (r.get("main_tradeoff") or "").replace("|", "\\|")
        lines.append(f"| {r['method_label']} | {cov} | {fit} | {oor} | {uiaf} | {tradeoff} |")
    lines.extend([
        "",
        "*Avg usable event rows per successful smile* = usable_iv_count / fit_success_count. Because multiple Polymarket strikes map to the same option smile, several event rows can be produced from one successful snapshot fit.",
        "",
        "## Method descriptions",
        "",
        "| Method | Lookback | Staleness | Min strikes | Strike gap | Latest trade per strike |",
        "|--------|----------|-----------|-------------|------------|--------------------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {r['lookback_window']} | {r['staleness_rule'] or '—'} | "
            f"{r['min_strikes']} | {r['strike_gap_rule'] or '—'} | {r['latest_trade_per_strike']} |"
        )
    lines.extend([
        "",
        "## Core outcome metrics",
        "",
        "| Method | usable_iv_rows | pct_rows_usable_iv | timestamps_with_iv | pct_timestamps_with_iv | avg_usable_iv_per_ts | median_usable_iv_per_ts |",
        "|--------|----------------|--------------------|--------------------|-------------------------|------------------------|--------------------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {_na(r['usable_iv_rows'])} | {_na(r['pct_rows_usable_iv'])} | "
            f"{_na(r['timestamps_with_iv'])} | {_na(r['pct_timestamps_with_iv'])} | "
            f"{_na(r['avg_usable_iv_per_timestamp'])} | {_na(r['median_usable_iv_per_timestamp'])} |"
        )
    lines.extend([
        "",
        "## Smile construction metrics",
        "",
        "| Method | total_windows | fit_success_count | fit_success_rate_pct | too_few_strikes | range_too_small | no_iv_data |",
        "|--------|----------------|--------------------|-----------------------|-----------------|----------------|------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {_na(r['total_windows'])} | {_na(r['fit_success_count'])} | "
            f"{_na(r['fit_success_rate_pct'])} | {_na(r['too_few_strikes'])} | {_na(r['range_too_small'])} | {_na(r['no_iv_data'])} |"
        )
    lines.extend([
        "",
        "## Failure / tradeoff metrics",
        "",
        "| Method | out_of_range_count | pct_rows_out_of_range | no_snapshot_fit_count | pct_rows_no_snapshot_fit |",
        "|--------|--------------------|------------------------|------------------------|---------------------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {_na(r['out_of_range_count'])} | {_na(r['pct_rows_out_of_range'])} | "
            f"{_na(r['no_snapshot_fit_count'])} | {_na(r['pct_rows_no_snapshot_fit'])} |"
        )
    lines.extend([
        "",
        "## Staleness / filtering (methods 2–4)",
        "",
        "| Method | total_retained_after_staleness | total_discarded_stale | avg_retained_per_snapshot | avg_discarded_per_snapshot |",
        "|--------|-------------------------------|------------------------|----------------------------|----------------------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {_na(r['total_retained_strikes_after_staleness_filter'])} | "
            f"{_na(r['total_discarded_stale_strikes'])} | {_na(r['avg_retained_strikes_per_snapshot'])} | "
            f"{_na(r['avg_discarded_stale_strikes_per_snapshot'])} |"
        )
    lines.extend([
        "",
        "## Strike gap diagnostics (methods 3–4)",
        "",
        "| Method | in_range_gap_rule_fail_count | avg_local_gap_success | max_local_gap_success |",
        "|--------|------------------------------|------------------------|----------------------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['method_label']} | {_na(r['in_range_gap_rule_fail_count'])} | "
            f"{_na(r['avg_local_gap_successful_interpolation'])} | {_na(r['max_local_gap_successful_interpolation'])} |"
        )
    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "**Progression from Method 1 to Method 4**",
        "",
        "- **Method 1 (1h / no staleness / min5)** is the baseline: shortest lookback, no staleness filter, strict minimum of 5 strikes. It yields the lowest row-level IV coverage (about 16.5% of Polymarket rows with usable IV) but the highest share of rows without any snapshot fit (about 64%), reflecting many timestamps or expiry/type combinations with insufficient option data.",
        "",
        "- **Method 2 (3h / 90m stale / min5)** lengthens the lookback to 3 hours and adds a 90-minute staleness filter, keeping min 5 strikes. More snapshots succeed (higher fit success rate), and both row-level and timestamp-level coverage improve. The share of rows with no snapshot fit drops; the share of rows out of range rises slightly as more interpolations are attempted. Staleness diagnostics show a substantial share of candidate strikes discarded, so the filter is active.",
        "",
        "- **Method 3 (3h / 90m stale / min3 / gap10k)** relaxes the strike requirement to 3 and adds the latest-trade-per-strike rule plus a local strike gap cap of 10k. Fit success rate and usable IV coverage increase further; fewer rows have no snapshot fit. The gap rule causes a small number of in-range gap failures. Tradeoff: higher coverage but more interpolations that may span wider strike distances (capped at 10k).",
        "",
        "- **Method 4 (3h / 120m stale / min3 / gap10k)** keeps the same rules as Method 3 but relaxes staleness to 120 minutes. More strikes are retained after the staleness filter (higher avg retained per snapshot, lower discarded). This yields the highest row-level and timestamp-level coverage and the highest fit success rate, with the lowest no-snapshot-fit share. Out-of-range share is highest, as more marginal interpolations are attempted.",
        "",
        "**Main tradeoffs**",
        "",
        "- **Coverage vs conservatism:** Moving from Method 1 to 4, usable IV coverage (both by rows and by timestamps) increases monotonically. The cost is more out-of-range outcomes (target strike outside or gap-fail) and, in Methods 3–4, use of thinner strike sets (min 3) and a hard gap cap rather than excluding wide gaps entirely.",
        "",
        "- **Staleness:** Tightening staleness (90 min) discards more strikes and reduces coverage; relaxing to 120 min (Method 4) retains more strikes and increases coverage, at the expense of using slightly older quotes.",
        "",
        "**Best balance (diagnostics only)**",
        "",
        "The diagnostics do not define a single “best” method; they summarize tradeoffs. Method 1 is most conservative (least coverage, strictest rules). Method 4 achieves the highest coverage and lowest no-snapshot-fit rate but the highest out-of-range share. Methods 2 and 3 sit in between. For a thesis, the choice depends on whether the analysis prioritizes coverage (favoring 3 or 4) or methodological conservatism (favoring 1 or 2).",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def make_plots(rows: List[Dict], plots_dir: Path) -> None:
    if not HAS_MPL or not rows:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["method_label"] for r in rows]
    short = ["M1", "M2", "M3", "M4"]

    # A) Bar chart: pct_rows_usable_iv, pct_timestamps_with_iv
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    w = 0.35
    pct_rows = [float(r["pct_rows_usable_iv"] or 0) for r in rows]
    pct_ts = [float(r["pct_timestamps_with_iv"] or 0) for r in rows]
    ax.bar([i - w/2 for i in x], pct_rows, w, label="% rows with usable IV", color="steelblue", alpha=0.9)
    ax.bar([i + w/2 for i in x], pct_ts, w, label="% timestamps with IV", color="darkorange", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylabel("%")
    ax.set_title("Coverage: row-level vs timestamp-level usable IV")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(plots_dir / "coverage_pct_rows_and_timestamps.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots_dir / 'coverage_pct_rows_and_timestamps.png'}")

    # B) Tradeoff: pct_rows_usable_iv vs pct_rows_out_of_range
    fig, ax = plt.subplots(figsize=(6, 5))
    usable = [float(r["pct_rows_usable_iv"] or 0) for r in rows]
    oor = [float(r["pct_rows_out_of_range"] or 0) for r in rows]
    colors = ["#2e7d32", "#1976d2", "#7b1fa2", "#c62828"]
    for i, (u, o) in enumerate(zip(usable, oor)):
        ax.scatter(o, u, s=120, c=colors[i], label=short[i], zorder=3)
        ax.annotate(short[i], (o, u), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("% rows out of range (or gap fail)")
    ax.set_ylabel("% rows with usable IV")
    ax.set_title("Tradeoff: usable IV coverage vs out-of-range share")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(oor) * 1.15 if oor else 35)
    ax.set_ylim(0, max(usable) * 1.1 if usable else 30)
    fig.tight_layout()
    fig.savefig(plots_dir / "tradeoff_usable_iv_vs_out_of_range.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots_dir / 'tradeoff_usable_iv_vs_out_of_range.png'}")

    # C) Fit success rate and no_snapshot_fit rate
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    fit_rate = [float(r["fit_success_rate_pct"] or 0) for r in rows]
    no_fit = [float(r["pct_rows_no_snapshot_fit"] or 0) for r in rows]
    ax1.bar([i - 0.2 for i in x], fit_rate, 0.4, label="Fit success rate (%)", color="steelblue", alpha=0.9)
    ax1.set_ylabel("Fit success rate (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], no_fit, 0.4, label="% rows no snapshot fit", color="coral", alpha=0.9)
    ax2.set_ylabel("% rows with no snapshot fit", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short)
    ax1.set_title("Smile construction: fit success rate vs no-snapshot-fit share")
    fig.tight_layout()
    fig.savefig(plots_dir / "fit_success_and_no_snapshot_fit.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots_dir / 'fit_success_and_no_snapshot_fit.png'}")


def main() -> None:
    rows = []
    for folder, label in METHODS:
        report, coverage = load_method_data(folder)
        row = build_row(folder, label, report, coverage)
        rows.append(row)

    write_csv(rows, ANALYSIS_DIR / "robustness_method_comparison.csv")
    write_md(rows, ANALYSIS_DIR / "robustness_method_comparison.md")
    make_plots(rows, ANALYSIS_DIR / "robustness_plots")
    print("Done.")


if __name__ == "__main__":
    main()
