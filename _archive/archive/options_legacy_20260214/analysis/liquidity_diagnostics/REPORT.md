# Liquidity diagnostics report

## Data
- Window: 2026-01-01 – 2026-02-15
- Rows: 227,579
- Unique instruments: 37
- Open interest in dataset: **No (volume only)**

## Do liquid strikes move with spot?

1. **Top liquid contracts over time**
   - `tables/top_liquid_contracts_daily.csv` — Top 15 contracts by volume per day (with share of day total).
   - `tables/top_liquid_contracts_weekly.csv` — Same by week.
   - `plots/top_liquid_daily.png` — Daily total volume with top-3 instruments labeled.
   - If the same few instruments stay at the top across Jan and Feb, liquidity is stable; if they change, liquid strikes are moving.

2. **Strike-level liquidity**
   - `tables/strike_expiry_volume_daily.csv` — Volume by date, strike, expiry, call_put.
   - `tables/heatmap_volume_by_date_strike.csv` — Pivot: rows=date, columns=strike, cell=volume.
   - `plots/heatmap_volume_date_strike.png` — Heatmap of volume by date vs strike.
   - `plots/heatmap_volume_by_expiry.png` — Small multiples per expiry.
   - Heat moving along the strike axis over time suggests liquidity following spot.

3. **Moneyness-based liquidity (main diagnostic)**
   - `tables/moneyness_volume_daily.csv` — Volume per day per moneyness bin (strike/index).
   - `tables/atm_near_atm_share_daily.csv` — Daily share of volume in ATM (0.95–1.05) and near-ATM (0.90–1.10).
   - `plots/heatmap_volume_date_moneyness.png` — Heatmap: date vs moneyness bin.
   - `plots/atm_band_share_timeseries.png` — Time series of % volume in ATM and near-ATM bands.
   - If volume stays concentrated in the same moneyness band (e.g. 0.98–1.02) as time passes, liquidity tracks spot. If the band shifts or disperses, regime change.

4. **Stability summary metrics**
   - `tables/stability_summary_daily.csv` — Per day: pct_top5, pct_top10, atm_share_pct, near_atm_share_pct, l1_prev (moneyness dist shift vs previous day).
   - `tables/concentration_top5_top10_daily.csv` — Raw concentration.
   - `tables/moneyness_distribution_l1_prev_day.csv` — L1 distance of volume-by-moneyness vs previous day.
   - `plots/stability_metrics.png` — Four panels: concentration, ATM share, L1 shift, summary text.

## Quick takeaway

- Mean % volume in ATM band (0.95–1.05): **17.7%**.
- Mean % volume in near-ATM band (0.90–1.10): **28.0%**.
- Mean % volume in top 5 strikes: **97.1%**.

Open the heatmaps and ATM band time series to see whether liquid moneyness is stable or shifts between Jan and Feb.

---

## Full-panel diagnostics (all options, not liquid-only)

The diagnostics above use the **liquid-aligned** options panel (37 instruments). There is currently **no full (all-options) panel for Jan–Feb 2026** in the repo: both `options_panel_long_5m.parquet` and `options_panel_long_5m_liquid_aligned.parquet` in the window folder are the same liquid subset.

To produce heatmaps and ATM band time series on the **full** options set:

1. **Generate the full panel**  
   Run the options pipeline for Jan–Feb 2026 so it writes `options_panel_long_5m.parquet` with all instruments (e.g. from `run_options_jan_feb_2026.py` or the pipeline step that builds the panel from klines). **Before** running the liquidity filter or Step 1 (which overwrites that file with the liquid subset), copy the full panel to a dedicated path, e.g.  
   `outputs/window=2026-01-01_2026-02-15/options_panel_long_5m_full.parquet`.

2. **Run diagnostics with a suffix**  
   From the repo root:
   ```bash
   python analysis/liquidity_diagnostics/run_liquidity_diagnostics.py --start 2026-01-01 --end 2026-02-15 --options-path "outputs/window=2026-01-01_2026-02-15/options_panel_long_5m_full.parquet" --output-suffix full
   ```
   Outputs will be written with a `_full` suffix (e.g. `heatmap_volume_date_strike_full.png`, `atm_band_share_timeseries_full.png`, `REPORT_full.md`) so they do not overwrite the liquid-based diagnostics.