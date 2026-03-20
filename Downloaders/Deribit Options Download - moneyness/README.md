# Options v2: Moneyness-based Deribit BTC options (Jan–Feb 2026)

Rebuild BTC options dataset from scratch via Deribit API for **2026-01-01 00:00 UTC → 2026-02-28 23:59 UTC**, 5-minute resolution, using **moneyness-based universe selection** (not fixed strikes).

## Goal

- Dataset suitable for later: implied vol surfaces σ(K,T,t), risk-neutral probabilities N(d2), comparison to Polymarket.
- **Universe:** Options in moneyness band (e.g. 0.70–1.30 × index) per day; union across days. Broad inclusion; no aggressive volume filter (only drop instruments with zero data over the full period).

## What is fetched

- **Instruments:** All BTC options (Deribit `get_instruments`, expired=true).
- **Index proxy:** BTC-PERPETUAL 5m candles (close used as index for moneyness).
- **Option candles:** 5m OHLCV per instrument in the union universe (`get_tradingview_chart_data`).

## Outputs

| Output | Location |
|--------|----------|
| Raw aligned long-format parquet | `data/options_v2_deribit_moneyness/options_long_5m.parquet` |
| Partitioned by month | `data/options_v2_deribit_moneyness/aligned/month=YYYY-MM/data.parquet` |
| QA report (Markdown + JSON) | `reports/options_v2_qa/qa_report.md`, `qa_report.json` |
| Inspection plots | `reports/options_v2_qa/plot_*.png` |
| Intermediate | `data/options_v2_deribit_moneyness/instruments_btc_options.json`, `index_5m.parquet`, `universe_*.json`, `candles_checkpoint.json`, `candles_failures.json` |

## Schema (long parquet)

- `timestamp_utc`, `instrument_name`, `strike`, `expiry_utc`, `call_put`
- `open`, `high`, `low`, `close`, `volume`
- `index_price`, `moneyness` (= strike / index_price)

## Run

From repo root:

```bash
python scripts/run_options_v2_moneyness.py
python scripts/run_options_v2_moneyness.py --force-refresh
```

Or:

```bash
python -m pipeline.options_v2_deribit_moneyness.run [--force-refresh]
```

## Modules

- `config.py` — Window, moneyness band, paths, Deribit settings.
- `deribit_client.py` — Rate-limited Deribit REST (list_instruments, get_tradingview_chart_data).
- `discover.py` — Fetch all BTC options; filter by expiry in window.
- `index_series.py` — Fetch BTC-PERPETUAL 5m → index proxy series.
- `universe.py` — Per-day moneyness band → union instrument list.
- `fetch_candles.py` — Pull candles with checkpoint and retry; record failures.
- `merge_qa.py` — Merge candles + index, add moneyness, write parquet(s).
- `qa_report.py` — Coverage, duplicates, value checks, distributions → md + json.
- `plots.py` — Moneyness vs date, daily instrument count, strike coverage vs date.

## Sanity checks (QA report)

- Coverage by day; missing dates.
- Instrument attempted vs pulled; failures and zero-data skipped.
- Timestamp UTC and 5m grid; duplicates (instrument, timestamp).
- Prices ≥ 0, volume ≥ 0; moneyness in band (outliers flagged).
- Distributions: moneyness, strike, expiries, call/put.

No surface building or IV inversion in this milestone—only the raw aligned dataset and inspection artifacts.
