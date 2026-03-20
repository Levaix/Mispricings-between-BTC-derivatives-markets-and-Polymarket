# Legacy: Deribit options OHLCV pipeline

This folder contains the **previous** Deribit options pipeline based on **trade-based 5m OHLCV candles** (`get_tradingview_chart_data`). It has been superseded by the **quote-based pipeline** (best bid/ask → 5m aligned → IV from mid) in `data_pipeline_v2_deribit_quotes/` and `scripts/run_deribit_quotes_v2.py`.

## Contents

- **downloaders/options.py** — Options downloader using Deribit `get_tradingview_chart_data` (OHLCV per instrument).
- **scripts/run_options_jan_feb_2026.py** — Jan–Feb 2026 options backfill: download candles, build long panel, viability audit.
- **scripts/run_options_liquidity_filter_jan_feb_2026.py** — Liquidity filter: produce liquid subset panel from the long panel.

## How to run (from repo root)

Ensure you are in the **Code** directory (repo root). Then:

```bash
# Use legacy folder first in path so downloaders.options is loaded from here
set PYTHONPATH=archive/deribit_ohlcv_legacy;%PYTHONPATH%
python archive/deribit_ohlcv_legacy/scripts/run_options_jan_feb_2026.py [--overwrite]
python archive/deribit_ohlcv_legacy/scripts/run_options_liquidity_filter_jan_feb_2026.py [--overwrite]
```

On Linux/macOS:

```bash
export PYTHONPATH=archive/deribit_ohlcv_legacy:$PYTHONPATH
python archive/deribit_ohlcv_legacy/scripts/run_options_jan_feb_2026.py --overwrite
python archive/deribit_ohlcv_legacy/scripts/run_options_liquidity_filter_jan_feb_2026.py --overwrite
```

Config (e.g. `config.OPTIONS_KLINES_PATH`, `config.OPTIONS_SELECTED_INSTRUMENTS_PATH`) is read from the repo root `config.py`. Data and outputs use the same paths as before (e.g. `data/options_jan_feb_2026/`, `outputs/window=.../`).

## Why legacy

- OHLCV candles are **trade-based** and very **sparse** (many zero-volume bars).
- The **quote-based pipeline** uses best bid/ask (when available) and computes IV from mid, giving denser and more stable series for analysis.
