# Options data: how it was compiled and configured

This document describes how the **options** dataset was built: source, instrument selection, download, storage layout, schema, and how Step 1 uses it to produce the options long panel and coverage summary. Use it to verify configuration and reproducibility.

---

## 1. Source and role

- **Exchange:** **Deribit** (REST API v2, public endpoints). Options are **not** from Binance (Binance options were replaced with Deribit for this thesis).
- **Underlying:** BTC (currency `BTC` on Deribit).
- **Purpose:** 5-minute OHLC (and volume) for a **selected** set of near-ATM BTC options, so that option prices and liquidity can be aligned with the master 5m panel (spot, futures, funding) and used for analysis (e.g. IV, mispricing, implied probabilities).
- **Time window:** Same as the rest of the pipeline: **2025-10-01 00:00:00 UTC** to **2025-12-31 23:55:00 UTC** (from `config.START_DT` / `config.END_DT`).

---

## 2. Configuration (config.py)

Relevant options and Deribit settings:

| Config | Value | Meaning |
|--------|--------|--------|
| `OPTIONS_KLINES_PATH` | `data/options/klines` | Root under which instrument-partitioned parquet is written. |
| `OPTIONS_SELECTED_INSTRUMENTS_PATH` | `data/options/selected_instruments.json` | Persisted list of selected instruments (reproducible runs). |
| `OPTIONS_EXCHANGE_INFO_PATH` | `data/options` | Directory for raw instrument list and tables. |
| `ATM_PERCENT_RANGE` | `0.10` | Strikes within ±10% of ATM are “near-ATM”. |
| `MAX_STRIKES_PER_SIDE` | `10` | At most 10 calls and 10 puts per expiry (nearest to ATM). |
| `OPTIONS_RESOLUTION_MINUTES` | `5` | Candle resolution in minutes (5m bars). |
| `OPTIONS_CANDLE_CHUNK_DAYS` | `7` | Candle requests are chunked by 7 days to respect API limits. |
| `DERIBIT_BASE_URL` | `https://www.deribit.com/api/v2` | Deribit API base. |
| `DERIBIT_INDEX_NAME` | `btc_usd` | Index used for fallback ATM when no local index klines exist. |
| `DERIBIT_SLEEP_BETWEEN_CALLS_MS` | `250` | Minimum delay between API calls (rate limiting). |

Other Deribit settings (`DERIBIT_MAX_RETRIES`, `DERIBIT_BACKOFF_*`, `DERIBIT_REQUEST_TIMEOUT_SEC`) control retries and timeouts.

---

## 3. Instrument universe and selection

### 3.1 Fetching the universe

- **API:** `get_instruments` (Deribit public API) with `currency=BTC`, `kind=option`.
- **Two calls:** One with `expired=false` (active options), one with `expired=true` (expired options). Results are merged by `instrument_name` so we have both currently listed and recently expired instruments. This is necessary because options that were trading in Oct–Dec 2025 may have expired by the time you run the backfill; Deribit only returns a subset of expired options when `expired=true`.
- **Output:** Full list is written to `data/options/deribit_instruments.json` (and optionally a parquet table under `data/options/deribit_instruments.parquet`) for reference. Each instrument has fields such as `instrument_name`, `creation_timestamp`, `expiration_timestamp`, `strike`, `option_type` (call/put), `base_currency`, etc.

### 3.2 Selection criteria (who gets 5m klines)

Selection is implemented in `build_selected_instruments()` in `downloaders/options.py`. Only instruments that satisfy **all** of the following are kept:

1. **Base currency:** `base_currency == "BTC"`.
2. **Trading life overlaps the window:**  
   - `creation_timestamp <= end_ms` (listed by the end of the window),  
   - `expiration_timestamp >= start_ms` (not expired before the window start).  
   So we keep options that were **live at any time** between 2025-10-01 and 2025-12-31, not only those that *expire* in that range.
3. **Near-ATM strikes:**  
   - ATM reference: first available close from Binance futures **index** klines in the window; if none, Deribit `get_index_price(DERIBIT_INDEX_NAME)`; if that fails, fallback `95000.0`.  
   - Strike must be in `[ATM × (1 - ATM_PERCENT_RANGE), ATM × (1 + ATM_PERCENT_RANGE)]`, i.e. ±10% of ATM.
4. **Cap per expiry:** For each distinct `expiration_timestamp`, we keep at most **10 calls** and **10 puts**, chosen by **distance to ATM** (nearest first).

So the **dataset** is: all BTC options on Deribit that were trading in the window, restricted to near-ATM and to a bounded number of strikes per expiry.

### 3.3 Persistence

- The selected list is written to **`data/options/selected_instruments.json`** so that re-runs of the backfill use the **same** set of instruments and do not re-select (e.g. with a different ATM).  
- If `selected_instruments.json` already exists, the download step **loads** it and skips re-running selection. To change the universe you would delete this file and re-run the options download so that `build_selected_instruments()` runs again with current config and ATM logic.

---

## 4. Downloading 5m candles

### 4.1 API

- **Endpoint:** Deribit `get_tradingview_chart_data`.
- **Parameters:** `instrument_name`, `start_timestamp`, `end_timestamp` (milliseconds), `resolution` (minutes; we use `5`).
- **Response:** Arrays `ticks` (bar start times in ms), `open`, `high`, `low`, `close`, `volume` (one value per bar). Volume is as reported by Deribit (contracts or base; see exchange docs).

### 4.2 Chunking

- The time window is split into chunks of **`OPTIONS_CANDLE_CHUNK_DAYS`** (7 days) to avoid overly large responses and rate limits.
- For each selected instrument, we request chunks from `start_ms` to `end_ms`, concatenate the parsed bars, then deduplicate by `open_time` (keep last) and sort.

### 4.3 Row-level checks

- Rows with null or non-positive OHLC are dropped (invalid candles).
- Each bar is stored with: `open_time` (int64 ms), `open`, `high`, `low`, `close`, `volume`, plus instrument metadata added below.

### 4.4 Schema written to parquet

Per-instrument parquet files contain:

| Column | Type | Description |
|--------|------|-------------|
| `instrument_name` | str | Deribit id, e.g. `BTC-27MAR26-125000-C`. |
| `expiry` | int64 | Expiration timestamp (ms). |
| `strike` | float | Strike price. |
| `call_put` | str | `CALL` or `PUT`. |
| `open_time` | int64 | Bar start (ms UTC). |
| `open` | float | Open option price. |
| `high` | float | High option price. |
| `low` | float | Low option price. |
| `close` | float | Close option price. |
| `volume` | float | Volume (as from API). |
| `source` | str | Always `deribit`. |

UTC ISO strings (e.g. `open_time_utc`) may be added by the storage helper. No implied volatility or bid/ask is stored; IV can be computed later from price, strike, expiry, and underlying.

---

## 5. Storage layout

Partitioning is by **instrument** and then by **year/month** of the bar’s `open_time`:

```
data/options/klines/
  instrument=BTC-27MAR26-125000-C/   (colon in name replaced by underscore in path)
    year=2025/
      month=10/
        klines_5m.parquet
      month=11/
        klines_5m.parquet
      ...
  instrument=BTC-3APR26-120000-P/
    year=2025/
      month=10/
        klines_5m.parquet
      ...
```

- Instrument names are made filesystem-safe by replacing `:` with `_`.
- One parquet file per (instrument, year, month); filename `klines_5m.parquet`.  
- Checkpointing: the downloader can mark an instrument as done (e.g. in `data/_manifest/`) so that re-runs skip already-downloaded instruments.

---

## 6. Step 1: how options are used in the pipeline

Step 1 does **not** re-download options; it **reads** the parquet written by the options downloader and then aligns and enriches it.

### 6.1 Loading

- **Paths:** For each `(year, month)` in the same month list used for spot/futures (window months plus one month before/after), Step 1 looks under `data/options/klines/instrument=*/year=Y/month=M/*.parquet`.
- All such parquet files are read and concatenated into one **options** DataFrame.
- `open_time` is coerced to int64; rows outside `[start_ms, end_ms]` are dropped.
- Deduplication: key `(instrument_name, open_time)`; keep last row per key. Duplicate count is recorded for the QA report.

### 6.2 Attaching index and derived columns

- **Index:** The master 5m panel has `index_close` (Binance futures index price per bar). Step 1 joins this to the options table on `open_time`, so each option bar gets the **underlying index close** for that 5m bar. This is the correct reference for moneyness and, later, IV.
- **Derived columns:**  
  - **`expiry_bucket`:** Classification of expiry into `weekly`, `monthly`, or `quarterly` (by expiry date; same logic as in the downloader).  
  - **`moneyness_proxy`:** `strike / index_close` (NaN when `index_close` is 0).  

No implied volatility is computed in Step 1; the dataset is “raw” option OHLC + index + identifiers.

### 6.3 Options coverage summary

- **Per instrument:** For each `instrument_name`, Step 1 computes:
  - **n_bars:** Number of distinct `open_time` in the window for that instrument.
  - **expected_bars:** Number of 5m bars from that instrument’s first to last `open_time` in the data (continuous 5m grid length).
  - **missing_pct:** `(1 - n_bars / expected_bars) * 100` (share of “missing” bars in that range).
  - **zero_volume_share:** Fraction of that instrument’s rows with `volume == 0`.
  - **volume_min, volume_max, volume_median.**
  - **is_sparse:** `True` if `missing_pct > 30` or (when configured) `median_volume == 0` (tuneable constants in Step 1).
  - **start_time_utc, end_time_utc:** First and last bar time for that instrument.

This is written to **`outputs/options_coverage_summary.parquet`** (and can be exported to CSV) so you can inspect coverage and sparsity per instrument.

### 6.4 Outputs

- **`outputs/options_panel_long_5m.parquet`:** Long-format options table: one row per (instrument, 5m bar), with all OHLC, volume, strike, expiry, call_put, **index_close**, **expiry_bucket**, and **moneyness_proxy**. This is the main options **dataset** for analysis.
- Options are **not** merged into the master 5m panel (which has one row per bar); they stay in a separate long panel because there are many instruments per bar.

---

## 7. Summary table (configuration at a glance)

| Item | Choice |
|------|--------|
| **Source** | Deribit BTC options (REST API v2). |
| **Window** | 2025-10-01 00:00 UTC – 2025-12-31 23:55 UTC. |
| **Instruments** | Live in window (creation ≤ end, expiration ≥ start), base_currency BTC, strike ±10% ATM, up to 10 calls + 10 puts per expiry. |
| **Resolution** | 5 minutes. |
| **Stored** | OHLC + volume; instrument_name, expiry, strike, call_put, source=deribit. |
| **Layout** | `data/options/klines/instrument=<name>/year=Y/month=M/klines_5m.parquet`. |
| **Selected list** | `data/options/selected_instruments.json` (persisted for reproducibility). |
| **Step 1** | Load by window months → join index_close on open_time → add expiry_bucket, moneyness_proxy → write long panel + coverage summary. |

This is how the options data was compiled and configured; you can double-check each step against the code and this overview.
