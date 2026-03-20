# Thesis Data Pipeline (Jan–Feb 2026)

The **main analysis window** is **2026-01-01 00:00 UTC** → **2026-02-15 23:55 UTC** (5-minute bars, BTC only).

## Data produced

- **Spot:** BTCUSDT 5m OHLCV klines (Binance)
- **USD-M Futures:** Perp 5m klines, mark price, index price, funding rate (Binance). Open interest and continuous quarterly klines are **off by default** (OI not available historically; use `--with-oi` / `--with-continuous` in backfill if needed).
- **Options:** **v2 (moneyness-based)** pipeline rebuilds BTC options from Deribit for Jan–Feb 2026: moneyness-band universe, 5m candles, index proxy, QA report and plots. See **`pipeline/options_v2_deribit_moneyness/`** and run with `python scripts/run_options_v2_moneyness.py`. Legacy components are archived at **`archive/options_legacy_20260214/`**.

Output: **Parquet** under `./data/`; pipeline outputs go to **windowed subfolders** `./outputs/window=YYYY-MM-DD_YYYY-MM-DD/` (e.g. `outputs/window=2026-01-01_2026-02-15/`).

## Setup

```bash
cd "c:\Users\shf\OneDrive - SHS Gesellschaft für Beteiligungsmanagement mbH\Dokumente\Data Thesis\Code"
pip install -r requirements.txt
```

Optional: set `BINANCE_API_KEY` if required for higher rate limits (not needed for public market data).

## Run

**Main pipeline (recommended):** Backfill Spot + Futures for the config window, then Step 1 QA + alignment (spot, futures, funding; no options until rebuilt).

```bash
python scripts/run_pipeline.py [--overwrite]
python scripts/run_pipeline.py --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --overwrite
python scripts/run_pipeline.py --no-backfill   # skip download, run Step 1 only
```

**Backfill only** (Spot + Futures; OI/continuous/options skipped by default):

```bash
python scripts/backfill_binance_oct_dec_2025.py
python scripts/backfill_binance_oct_dec_2025.py --with-oi --with-continuous   # optional; --with-options raises (pipeline archived)
```

Runs from the `Code` folder. Writes to `./data/`; pipeline outputs to `./outputs/window=2026-01-01_2026-02-15/`.

## Verify downloaded data

After the backfill (or if you see `startTime is invalid` or empty Futures/Options), check completeness:

```bash
python scripts/verify_downloaded_data.py
```

This compares expected chunks (from the config date range) with `data/_manifest/done_chunks.json`, counts Parquet rows for Spot, Futures, and Options, and reports missing chunks and total rows so you can see what was actually downloaded.

**Inspect datasets before analysis** (schema, key variables, time range, sample stats):

```bash
python scripts/inspect_datasets.py
```

A static reference of key variables and merge keys is in `data/DATA_SCHEMA.md`.  
A **detailed data overview** for analysis planning (variables, timestamps, merge strategy, limitations, suggested use) is in `data/DATA_OVERVIEW.md`.

**Options v2 (moneyness-based):** `python scripts/run_options_v2_moneyness.py [--force-refresh]` — builds raw aligned options dataset and QA in `data/options_v2_deribit_moneyness/` and `reports/options_v2_qa/`.

**Legacy options scripts:** moved to `archive/options_legacy_20260214/scripts/`. Restore from the archive to run.

## Step 1: Data QA + alignment (pipeline)

Build the **master 5m panel** (one row per 5-minute bar with spot, perp, index, mark, and funding aligned by time), plus options long table, QA report, and sanity plots.

**What the master panel is:** One row per 5m bar. Spot close/volume, perp close/volume, index close, mark close, and funding rate are joined so you see spot, futures, and funding side by side. Options stay in a separate long-format file (one row per instrument per bar).

**Data needed:** Step 1 loads from `data/spot/...`, `data/futures/usdm_perp/...` for months overlapping the config window. Options pipeline is archived; Step 1 runs without options (options outputs empty/skipped).

```bash
python scripts/run_step1_qa_alignment.py [--overwrite]
python scripts/run_step1_qa_alignment.py --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --overwrite
```

Outputs are written under **`./outputs/window=YYYY-MM-DD_YYYY-MM-DD/`** (e.g. `outputs/window=2026-01-01_2026-02-15/`):

- `master_panel_5m.parquet` — 5m grid with spot_*, perp_*, index_close, mark_close, funding_rate_5m, fundingTime_last
- `qa_report.parquet` / `qa_report.csv` — QA summary (n_rows, time range, duplicates, missing share)
- `plots/` — spot–index spread, perp–index basis, funding rate over time
- Options outputs (`options_panel_long_5m.parquet`, etc.) are written only when options data is provided (legacy archived).

## Layout

- `config.py` — API URLs, **default window Jan–Feb 2026**, paths, rate limits (Binance + Deribit)
- `utils/time.py` — ms ↔ UTC, chunk ranges, `parse_window_start_end`, `window_output_suffix`
- `downloaders/spot.py` — Spot klines (Binance)
- `downloaders/futures.py` — Futures (perp, mark, index, funding; OI/continuous off by default)
- `downloaders/options.py` — Stub (options pipeline archived at `archive/options_legacy_20260214/`)
- `scripts/backfill_binance_oct_dec_2025.py` — Backfill Spot + Futures (optional OI/continuous/options via flags)
- `scripts/run_pipeline.py` — **Main runner:** backfill + Step 1; windowed outputs (options pipeline archived)
- `pipeline/step1_qa_alignment.py` — Step 1: load, standardize, master grid, funding/options alignment, QA report, plots
- `scripts/run_step1_qa_alignment.py` — Run Step 1 only (optional `--start`/`--end`/`--options-liquid-path`; writes to `./outputs/window=.../`)

## Rate limits

- ~5 requests/second, 200 ms between calls
- On HTTP 429/418: exponential backoff (1s, 2s, 4s, … max 60s), up to 8 retries
