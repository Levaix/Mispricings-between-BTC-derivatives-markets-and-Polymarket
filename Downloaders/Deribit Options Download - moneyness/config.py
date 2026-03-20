# Options v2 (moneyness-based) pipeline config.
# Window: 2026-01-01 00:00 UTC to 2026-02-28 23:55 UTC. 5-minute resolution.

from pathlib import Path

START_UTC = "2026-01-01T00:00:00"
END_UTC = "2026-02-28T23:55:00"
RESOLUTION_MINUTES = 5

MONEYNESS_MIN = 0.70
MONEYNESS_MAX = 1.30
UNIVERSE_SELECTION_CADENCE_DAYS = 1

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
DERIBIT_INDEX_NAME = "btc_usd"
INDEX_PROXY_INSTRUMENT = "BTC-PERPETUAL"
# Deribit returns at most ~1000-1500 bars per request; 1 day = 288 bars (5m) keeps us under limit
# Use 3-day chunks (864 bars) to reduce calls while staying under the limit.
CANDLE_CHUNK_DAYS = 3
# Keep expiries within a reasonable horizon after window end.
# This avoids pulling far-dated (e.g. Dec 2026) instruments that explode request counts.
MAX_EXPIRY_HORIZON_DAYS = 180
# Only include options with time-to-expiry up to this many days from each selection date.
# This keeps the universe broad but avoids far-dated expiries that explode request counts.
MAX_TTE_DAYS = 90
SLEEP_BETWEEN_CALLS_MS = 250
MAX_RETRIES = 8
BACKOFF_BASE_SEC = 1
BACKOFF_MAX_SEC = 60
REQUEST_TIMEOUT_SEC = 30


def _root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# Unified data layout under "Actual data"
ACTUAL_DATA_DIR = _root() / "Actual data"
DATA_DIR = ACTUAL_DATA_DIR / "Options Deribit"
REPORTS_DIR = ACTUAL_DATA_DIR / "_QA" / "Options Deribit"

INSTRUMENTS_JSON = DATA_DIR / "instruments_btc_options.json"
INDEX_SERIES_PARQUET = DATA_DIR / "index_5m.parquet"
UNIVERSE_PER_DAY_JSON = DATA_DIR / "universe_per_day.json"
UNIVERSE_MASTER_JSON = DATA_DIR / "universe_master.json"
CANDLES_CHECKPOINT_JSON = DATA_DIR / "candles_checkpoint.json"
CANDLES_FAILURES_JSON = DATA_DIR / "candles_failures.json"
RAW_LONG_PARQUET = DATA_DIR / "options_long_5m.parquet"
PARTITION_BY_MONTH = True
ALIGNED_PARQUET_DIR = DATA_DIR / "aligned"

for d in (DATA_DIR, REPORTS_DIR, ALIGNED_PARQUET_DIR):
    d.mkdir(parents=True, exist_ok=True)
