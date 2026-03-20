# Deribit options v2: quote-based pipeline

This folder implements the **quote-based** Deribit options pipeline: best bid/ask → 5m aligned panel → IV from mid (Black-Scholes).

## Data source limitation

**Deribit’s public API does not provide historical order book.** You can:

1. **Live collection** — Run `run_deribit_quotes_v2.py --live` on a schedule (e.g. every 5 minutes). Each run appends a snapshot to `data/deribit_quotes_v2/raw/`. Over time you get 5m-aligned history.
2. **External ingest** — Use `--ingest <path>` with a pre-downloaded quote file (e.g. from Tardis.dev or another provider) that has columns: `open_time` (or `timestamp`), `instrument_name`, `best_bid_price`, `best_ask_price`, and optionally `strike`, `expiry`, `call_put`, `index_price`.

## Outputs

- **data/deribit_quotes_v2/raw/** — Raw snapshots (one parquet per day when using `--live`).
- **data/deribit_quotes_v2/aligned_5m/deribit_quotes_5m.parquet** — Bid/ask/mid aligned to 5m grid (forward-fill per instrument, no fill before first observation).
- **data/deribit_quotes_v2/aligned_5m/deribit_iv_5m.parquet** — Same grid with `iv_mid` (implied vol from mid price via Black-Scholes).
- **data/deribit_quotes_v2/aligned_5m/deribit_quotes_coverage.json** and **.md** — Per-instrument coverage (bars filled, first/last obs).

## IV model

We use **Black-Scholes** with \(S\) = index price (underlying), \(K\) = strike, \(r = 0\), no dividend. IV is solved from the mid option premium. This matches Deribit’s spot-style options (underlying index = btc_usd).

## Config

In `config.py`:

- `DERIBIT_QUOTES_V2_ROOT` — Base dir for raw and aligned outputs.
- `DERIBIT_QUOTE_MAX_SPREAD_PCT` — Drop rows where (ask−bid)/mid > this (default 50). Set to `None` to disable.

## Run (from repo root)

```bash
# Live snapshot (append to raw store)
python scripts/run_deribit_quotes_v2.py --live

# Align and compute IV from existing raw (or after --live)
python scripts/run_deribit_quotes_v2.py --live --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --overwrite

# Ingest external historical quotes
python scripts/run_deribit_quotes_v2.py --ingest path/to/quotes.parquet --start "2026-01-01 00:00:00" --end "2026-02-15 23:55:00" --overwrite
```

No API auth required for Deribit public endpoints (order book, instruments).
