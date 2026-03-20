"""EDA module for BTC options (Deribit, 5-minute aggregates).

This script:
  - Loads aggregated 5-minute options data from Actual data (read-only)
  - Constructs basic moneyness and time-to-maturity variables
  - Explores implied volatility, moneyness, strikes, maturities and volumes
  - Produces summary tables and publication-quality figures

All outputs are written ONLY under:
  EDA/Options/

IMPORTANT: The script NEVER writes to `Actual data/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CODE_ROOT = Path(__file__).resolve().parents[3]

# Correct path to the aggregated 5-minute options dataset
OPTIONS_PATH = (
    CODE_ROOT
    / "Actual data"
    / "Options Deribit"
    / "options_5m_agg"
    / "agg_5m_merged.parquet"
)

OUTPUT_ROOT = CODE_ROOT / "EDA" / "Options"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_options_data(path: Path) -> pd.DataFrame:
    """Load Deribit options 5-minute aggregates and standardize columns.

    The raw file does not contain explicit expiry/strike/type columns,
    so these are parsed from the Deribit `instrument_name`, which typically
    has the form `BTC-30DEC22-20000-C`.
    """
    logger.info("Loading options dataset from %s", path)
    df = pd.read_parquet(path)
    cols_lower = {c.lower(): c for c in df.columns}

    # Detect timestamp column (use bucket_5m_utc or similar)
    ts_col = None
    for key in ["bucket_5m_utc", "timestamp", "timestamp_utc", "time", "datetime"]:
        if key in cols_lower:
            ts_col = cols_lower[key]
            break
    if ts_col is None:
        raise KeyError(
            "Could not detect timestamp column in options dataset; "
            f"columns available: {list(df.columns)}"
        )

    # Detect implied volatility column (iv_vwap in this aggregated file)
    iv_col = None
    for key in ["iv_vwap", "iv", "implied_vol", "implied_volatility"]:
        if key in cols_lower:
            iv_col = cols_lower[key]
            break
    if iv_col is None:
        raise KeyError("Could not detect implied volatility column in options dataset.")

    # Detect instrument_name for strike/expiry/type parsing
    inst_col = None
    for key in ["instrument_name", "instrument", "symbol"]:
        if key in cols_lower:
            inst_col = cols_lower[key]
            break
    if inst_col is None:
        raise KeyError("Could not detect instrument_name column in options dataset.")

    # Optional volume / index columns
    contracts_col = None
    for key in ["contracts_sum", "contracts", "volume_contracts"]:
        if key in cols_lower:
            contracts_col = cols_lower[key]
            break

    amount_col = None
    for key in ["amount_sum", "notional", "volume_usd"]:
        if key in cols_lower:
            amount_col = cols_lower[key]
            break

    index_price_col = None
    for key in ["index_price_last", "index_price", "spot", "underlying_price"]:
        if key in cols_lower:
            index_price_col = cols_lower[key]
            break
    if index_price_col is None:
        raise KeyError("Could not detect index_price column in options dataset.")

    df = df.copy()
    rename_map = {
        ts_col: "timestamp",
        inst_col: "instrument_name",
        iv_col: "iv",
        index_price_col: "index_price",
    }
    if contracts_col:
        rename_map[contracts_col] = "contracts_sum"
    if amount_col:
        rename_map[amount_col] = "amount_sum"
    df.rename(columns=rename_map, inplace=True)

    # Parse timestamps as UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Failed to parse some option timestamps as UTC.")

    # Parse expiry, strike and option_type from instrument_name
    def _parse_instrument(name: str) -> Dict[str, Optional[object]]:
        # Expected pattern: UNDERLYING-DAYMONYY-STRIKE-TYPE, e.g. BTC-30DEC22-20000-C
        try:
            parts = name.split("-")
            if len(parts) < 4:
                return {"expiry": None, "strike": None, "option_type": None}
            expiry_str = parts[1]
            strike_str = parts[2]
            cp = parts[3].upper()

            # Try Deribit-style expiry formats
            for fmt in ("%d%b%y", "%d%b%Y"):
                try:
                    exp = pd.to_datetime(expiry_str, format=fmt, utc=True)
                    break
                except Exception:
                    exp = None
            strike = float(strike_str)
            opt_type = "call" if cp.startswith("C") else "put" if cp.startswith("P") else cp
            return {"expiry": exp, "strike": strike, "option_type": opt_type}
        except Exception:
            return {"expiry": None, "strike": None, "option_type": None}

    parsed = df["instrument_name"].astype(str).map(_parse_instrument)
    parsed_df = pd.DataFrame(list(parsed))
    df = pd.concat([df, parsed_df], axis=1)

    if df["expiry"].isna().all():
        raise ValueError("Failed to parse any expiries from instrument_name.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add tau, moneyness, and IV-clean columns.

    - tau_days, tau_years: time to expiry; important for term structure.
    - moneyness: strike/index; standard way to scale strikes across levels.
    - iv_clean: implied volatility restricted to a plausible range.
    """
    df = df.copy()
    # Time to expiry in days/years
    delta = (df["expiry"] - df["timestamp"]).dt.total_seconds() / 86400.0
    df["tau_days"] = delta
    df["tau_years"] = df["tau_days"] / 365.0

    # Moneyness: K / S (strike divided by underlying index level)
    df["moneyness"] = df["strike"] / df["index_price"]

    # Clean IV: drop obvious outliers / bad fits
    df["iv_clean"] = df["iv"].where((df["iv"] > 0.0) & (df["iv"] <= 3.0))
    return df


def summary_table(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compute summary stats for key numeric variables and write CSV+MD."""
    variables = [
        "iv",
        "moneyness",
        "tau_days",
        "strike",
        "index_price",
        "contracts_sum",
        "amount_sum",
    ]

    def _describe(series: pd.Series) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p5": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }

    rows: List[Dict[str, float]] = []
    for var in variables:
        if var not in df.columns:
            continue
        stats_dict = _describe(df[var])
        stats_dict["variable"] = var
        rows.append(stats_dict)

    out_df = pd.DataFrame(rows)[
        [
            "variable",
            "count",
            "mean",
            "std",
            "min",
            "p5",
            "p25",
            "median",
            "p75",
            "p95",
            "max",
        ]
    ]
    out_df.to_csv(out_path, index=False)

    # Markdown version
    md_path = out_path.with_suffix(".md")
    cols = list(out_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in out_df.iterrows():
        vals = [str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved options summary stats to %s and %s", out_path, md_path)
    return out_df


def option_type_tables(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Frequency table for option types (calls vs puts)."""
    counts = df["option_type"].value_counts(dropna=False).rename_axis("option_type")
    out_df = counts.reset_index(name="count")
    out_df.to_csv(out_path, index=False)

    # Markdown version
    md_path = out_path.with_suffix(".md")
    lines = []
    lines.append("| option_type | count |")
    lines.append("| --- | --- |")
    for _, r in out_df.iterrows():
        lines.append(f"| {r['option_type']} | {r['count']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved option type counts to %s and %s", out_path, md_path)
    return out_df


def format_time_axis(ax, timestamps: pd.Series) -> None:
    """Format time axis with a small number of date ticks."""
    locator = mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25])
    formatter = mdates.DateFormatter("%d-%b")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_iv_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["iv_clean"].dropna()
    plt.hist(vals, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of Implied Volatility (cleaned)")
    plt.xlabel("Implied volatility")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "iv_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved IV distribution to %s", out_path)


def plot_moneyness_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["moneyness"].dropna()
    plt.hist(vals, bins=100, density=True, alpha=0.7)
    plt.title("Distribution of Moneyness (K / S)")
    plt.xlabel("Moneyness (strike / index price)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "moneyness_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved moneyness distribution to %s", out_path)


def plot_iv_vs_moneyness(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sub = df.dropna(subset=["iv_clean", "moneyness"])
    plt.scatter(sub["moneyness"], sub["iv_clean"], s=5, alpha=0.3)
    plt.title("Implied Volatility vs Moneyness")
    plt.xlabel("Moneyness (strike / index price)")
    plt.ylabel("Implied volatility (clean)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "iv_vs_moneyness.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved IV vs moneyness scatter to %s", out_path)


def plot_iv_vs_maturity(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sub = df.dropna(subset=["iv_clean", "tau_days"])
    plt.scatter(sub["tau_days"], sub["iv_clean"], s=5, alpha=0.3)
    plt.title("Implied Volatility vs Time to Maturity")
    plt.xlabel("Time to maturity (days)")
    plt.ylabel("Implied volatility (clean)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "iv_vs_maturity.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved IV vs maturity scatter to %s", out_path)


def plot_strike_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    vals = df["strike"].dropna()
    plt.hist(vals, bins=100, density=False, alpha=0.7)
    plt.title("Strike Distribution")
    plt.xlabel("Strike")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "strike_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved strike distribution to %s", out_path)


def plot_index_price_timeseries(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(df["timestamp"], df["index_price"], linewidth=0.8)
    plt.title("Index Price (Underlying BTC Spot) Over Time")
    ax = plt.gca()
    format_time_axis(ax, df["timestamp"])
    plt.xlabel("Time")
    plt.ylabel("Index price (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "index_price_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved index price timeseries to %s", out_path)


def plot_example_vol_smile(df: pd.DataFrame) -> None:
    """Plot an example volatility smile (IV vs strike).

    In earlier versions, this figure was based on a single timestamp and could
    degenerate to a single point if very few strikes passed the IV cleaning
    filters. To guarantee a meaningful smile for interpretation, we now
    construct a representative cross-section by taking the median cleaned IV
    for each strike across the full sample.
    """
    sub = df.dropna(subset=["iv_clean", "strike"]).copy()
    if sub.empty:
        logger.warning("No rows with non-missing iv_clean and strike for smile example.")
        return

    # Aggregate to a representative smile: median iv_clean per strike.
    smile = (
        sub.groupby("strike", as_index=False)["iv_clean"]
        .median()
        .rename(columns={"iv_clean": "iv_median"})
        .sort_values("strike")
    )
    if smile.shape[0] < 3:
        logger.warning(
            "Fewer than 3 strikes available for example smile after aggregation."
        )
        return

    plt.figure(figsize=(6, 4))
    plt.plot(smile["strike"], smile["iv_median"], marker="o", linestyle="-", linewidth=1)
    plt.title("Example Volatility Smile (median IV by strike)")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility (clean)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / "example_volatility_smile.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved example volatility smile to %s", out_path)


def plot_volume_distributions(df: pd.DataFrame) -> None:
    # Contracts traded
    if "contracts_sum" in df.columns:
        plt.figure(figsize=(6, 4))
        vals = df["contracts_sum"].dropna()
        plt.hist(vals, bins=100, density=False, alpha=0.7)
        plt.title("Contracts Traded Distribution")
        plt.xlabel("contracts_sum")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = FIGURES_DIR / "contracts_distribution.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Saved contracts distribution to %s", out_path)

    # Notional (amount_sum)
    if "amount_sum" in df.columns:
        plt.figure(figsize=(6, 4))
        vals = df["amount_sum"].dropna()
        plt.hist(vals, bins=100, density=False, alpha=0.7)
        plt.title("Notional Trade Size Distribution")
        plt.xlabel("amount_sum")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = FIGURES_DIR / "notional_distribution.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Saved notional distribution to %s", out_path)


# ---------------------------------------------------------------------------
# Diagnostics and markdown summary
# ---------------------------------------------------------------------------

def write_diagnostics(df: pd.DataFrame) -> str:
    out_path = DIAGNOSTICS_DIR / "options_eda_diagnostics.txt"

    iv_min, iv_max = float(df["iv"].min()), float(df["iv"].max())
    mny_min, mny_max = float(df["moneyness"].min()), float(df["moneyness"].max())

    counts = df["option_type"].value_counts(dropna=False)
    n_calls = int(counts.get("call", 0) + counts.get("C", 0))
    n_puts = int(counts.get("put", 0) + counts.get("P", 0))

    lines: List[str] = []
    lines.append("Options EDA – Diagnostics")
    lines.append("==========================")
    lines.append(f"Dataset path: {OPTIONS_PATH}")
    lines.append(f"Row count: {len(df)}")
    lines.append(
        f"Timestamp range: {df['timestamp'].min().isoformat()} to {df['timestamp'].max().isoformat()}"
    )
    lines.append(
        f"Expiry range:    {df['expiry'].min().isoformat()} to {df['expiry'].max().isoformat()}"
    )
    lines.append(f"IV min/max:      {iv_min:.6f} / {iv_max:.6f}")
    lines.append(f"Moneyness min/max: {mny_min:.6f} / {mny_max:.6f}")
    lines.append("")
    lines.append(f"Number of calls: {n_calls}")
    lines.append(f"Number of puts:  {n_puts}")
    lines.append("")
    lines.append("Script status: SUCCESS")
    lines.append("")
    lines.append("Tables created:")
    lines.append(f"  - {TABLES_DIR / 'options_summary_stats.csv'}")
    lines.append(f"  - {TABLES_DIR / 'options_summary_stats.md'}")
    lines.append(f"  - {TABLES_DIR / 'options_type_counts.csv'}")
    lines.append(f"  - {TABLES_DIR / 'options_type_counts.md'}")
    lines.append(f"  - {TABLES_DIR / 'options_eda_summary.md'}")
    lines.append("")
    lines.append("Figures created:")
    lines.append(f"  - {FIGURES_DIR / 'iv_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'moneyness_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'iv_vs_moneyness.png'}")
    lines.append(f"  - {FIGURES_DIR / 'iv_vs_maturity.png'}")
    lines.append(f"  - {FIGURES_DIR / 'strike_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'index_price_timeseries.png'}")
    lines.append(f"  - {FIGURES_DIR / 'example_volatility_smile.png'}")
    lines.append(f"  - {FIGURES_DIR / 'contracts_distribution.png'}")
    lines.append(f"  - {FIGURES_DIR / 'notional_distribution.png'}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote options diagnostics to %s", out_path)
    return text


def write_markdown_summary(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    type_counts_df: pd.DataFrame,
) -> None:
    """High-level markdown overview for thesis."""
    out_path = TABLES_DIR / "options_eda_summary.md"

    n_obs = len(df)
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    n_strikes = df["strike"].nunique()
    n_expiries = df["expiry"].nunique()

    counts = type_counts_df.set_index("option_type")["count"].to_dict()
    total_type = sum(counts.values())
    share_calls = (counts.get("call", 0) + counts.get("C", 0)) / total_type if total_type else 0.0
    share_puts = (counts.get("put", 0) + counts.get("P", 0)) / total_type if total_type else 0.0

    iv_stats = summary_df[summary_df["variable"] == "iv"].iloc[0].to_dict()

    lines: List[str] = []
    lines.append("# Options EDA – Summary")
    lines.append("")
    lines.append("This module summarizes the Deribit BTC options dataset used for")
    lines.append("constructing the volatility smile and surface in the thesis.")
    lines.append("")
    lines.append("## Dataset description")
    lines.append("")
    lines.append(f"- **Source**: `Actual data/Options Deribit/agg_5m_merged.parquet`")
    lines.append(f"- **Number of observations**: {n_obs}")
    lines.append(
        f"- **Sample period**: {ts_min.isoformat()} to {ts_max.isoformat()}"
    )
    lines.append(f"- **Number of unique strikes**: {n_strikes}")
    lines.append(f"- **Number of unique expiries**: {n_expiries}")
    lines.append("")
    lines.append("## Option type mix")
    lines.append("")
    lines.append(
        f"- **Share of calls**: {share_calls:.2%} (counting labels 'call'/'C')"
    )
    lines.append(
        f"- **Share of puts**: {share_puts:.2%} (counting labels 'put'/'P')"
    )
    lines.append("")
    lines.append("## Implied volatility (IV)")
    lines.append("")
    lines.append(
        f"- **IV mean**: {iv_stats['mean']:.4f}, **std**: {iv_stats['std']:.4f}"
    )
    lines.append(
        f"- **IV 5th–95th pct**: {iv_stats['p5']:.4f} – {iv_stats['p95']:.4f}"
    )
    lines.append(
        f"- **IV range**: {iv_stats['min']:.4f} – {iv_stats['max']:.4f} (raw, before `iv_clean` filter)"
    )
    lines.append("")
    lines.append("Implied volatility reflects the market's view of future uncertainty")
    lines.append("embedded in option prices. It is the central object for volatility")
    lines.append("smile and surface analysis.")
    lines.append("")
    lines.append("## Moneyness definition")
    lines.append("")
    lines.append(
        r"- **Moneyness** is defined as \( K / S \), where \( K \) is the strike"
    )
    lines.append(r"  and \( S \) is the underlying index price. Values near 1")
    lines.append("  correspond to at-the-money options; values above/below 1")
    lines.append("  correspond to out-of-the-money calls/puts.")
    lines.append("")
    lines.append("## Key figures produced")
    lines.append("")
    lines.append("- IV and moneyness distributions")
    lines.append("- IV vs moneyness scatter and IV vs maturity scatter")
    lines.append("- Strike and notional/contract volume distributions")
    lines.append("- Index price time series")
    lines.append("- An example **volatility smile** (IV vs strike at a fixed time)")
    lines.append("")
    lines.append(
        "The volatility smile figure highlights how IV varies across strikes "
        "for a given timestamp, which is crucial for understanding departures "
        "from the Black–Scholes flat-volatility assumption."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote options EDA markdown summary to %s", out_path)


def main() -> None:
    ensure_output_dirs()

    df = load_options_data(OPTIONS_PATH)
    df = add_derived_variables(df)

    summary_df = summary_table(df, TABLES_DIR / "options_summary_stats.csv")
    type_counts_df = option_type_tables(df, TABLES_DIR / "options_type_counts.csv")

    # Figures
    plot_iv_distribution(df)
    plot_moneyness_distribution(df)
    plot_iv_vs_moneyness(df)
    plot_iv_vs_maturity(df)
    plot_strike_distribution(df)
    plot_index_price_timeseries(df)
    plot_example_vol_smile(df)
    plot_volume_distributions(df)

    # Text outputs
    write_markdown_summary(df, summary_df, type_counts_df)
    diag_text = write_diagnostics(df)

    # Print diagnostics to console
    print()
    print(diag_text)


if __name__ == "__main__":
    main()

