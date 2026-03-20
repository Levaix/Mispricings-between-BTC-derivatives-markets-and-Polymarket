# Options pipeline legacy (archived 2026-02-14)

All options-related pipeline components, data, and outputs moved here so the main repo can rebuild options sourcing from scratch.

## What was archived

| Location | Contents |
|----------|----------|
| `data/` | `options_jan_feb_2026/` (klines, instruments, selected_instruments.json), `OPTIONS_DATA_OVERVIEW.md` |
| `downloaders/` | `options.py` (real implementation; repo has stub) |
| `deribit_client.py` | Deribit API client (rate limit, order book, instruments) |
| `data_pipeline_v2_deribit_quotes/` | Quote-based pipeline (fetch, align, IV, QA, coverage) |
| `deribit_ohlcv_legacy/` | Legacy OHLCV downloader (from previous archive) |
| `scripts/` | `run_options_jan_feb_2026.py`, `run_options_liquidity_filter_jan_feb_2026.py`, `options_viability_audit.py`, `run_deribit_quotes_v2.py`, `verify_options_oct_dec_2025.py`, `run_step3_derivatives_probability.py`, `clean_step3_events.py`, `run_step3_full_sanity_audit.py` |
| `outputs/` | `options_panel_long_5m.*`, `options_coverage_summary.*`, `options_viability*`, `plots/options_missing_pct_hist.png` |
| `outputs_jan_feb_2026/` | Full options outputs (panels, coverage, viability) |
| `outputs_window/` | Window options parquets, step3 (IV, derivatives probability), eda options tables/figures |
| `analysis/` | `liquidity_diagnostics/` (heatmaps, ATM band, stability metrics) |

## Old approach

- **Wide strike range (Jan–Feb 2026):** Selected instruments across many expiries and strikes caused **low volume and stale bars** for most contracts.
- **Trade-based OHLCV only:** Deribit 5m candles from trades; no historical bid/ask. Many bars had zero volume.
- **Liquidity filter:** Scripts filtered to a liquid subset (~37 instruments) for the main analysis; full panel was rarely retained.
- **Step 3:** IV extraction from near-K options and derivatives-implied probability P^OPT.

## How to restore / run

1. **Copy back to repo root** (or adjust `sys.path` and paths in scripts):
   - `data/options_jan_feb_2026/` → `data/options_jan_feb_2026/`
   - `downloaders/options.py` → `downloaders/options.py` (overwrite stub)
   - `deribit_client.py` → repo root
   - `data_pipeline_v2_deribit_quotes/` → repo root

2. **Update `config.py`** to point `OPTIONS_KLINES_PATH`, `OPTIONS_SELECTED_INSTRUMENTS_PATH`, etc. back to the data paths.

3. **Entrypoints:**
   - Build full panel: `python scripts/run_options_jan_feb_2026.py`
   - Build liquid subset: `python scripts/run_options_liquidity_filter_jan_feb_2026.py`
   - Deribit v2 quotes: `python scripts/run_deribit_quotes_v2.py`
   - Step 3 (IV / derivatives probability): `python scripts/run_step3_derivatives_probability.py --window_dir outputs/window=2026-01-01_2026-02-15`
   - Liquidity diagnostics: `python analysis/liquidity_diagnostics/run_liquidity_diagnostics.py --start 2026-01-01 --end 2026-02-15`

4. **Dependencies:** Same as main repo (`pandas`, `numpy`, `scipy` for Step 3, etc.).
