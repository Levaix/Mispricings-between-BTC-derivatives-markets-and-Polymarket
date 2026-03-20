"""
Binance data backfill configuration.
All timestamps UTC. Master interval: 5 minutes.
"""
from datetime import datetime

# --- API base URLs ---
SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
OPTIONS_BASE = "https://eapi.binance.com"
FUTURES_DATA_BASE = "https://fapi.binance.com/futures/data"  # taker ratio, long/short

# --- Timeframe (main analysis window: Jan–Feb 2026 for Deribit options liquidity) ---
START_UTC = "2026-01-01T00:00:00Z"
END_UTC = "2026-02-15T23:55:00Z"
INTERVAL_5M = "5m"

# Parsed for ms (UTC; end is last bar 23:55)
START_DT = datetime(2026, 1, 1, 0, 0, 0)
END_DT = datetime(2026, 2, 15, 23, 55, 0)

# --- Symbols ---
SPOT_SYMBOL = "BTCUSDT"
FUTURES_SYMBOL = "BTCUSDT"
OPTIONS_UNDERLYING = "BTCUSDT"

# --- Rate limit / safety ---
REQUESTS_PER_SECOND = 5
SLEEP_BETWEEN_CALLS_MS = 200  # 150-250ms default
MAX_RETRIES = 8
BACKOFF_BASE_SEC = 1
BACKOFF_MAX_SEC = 60
REQUEST_TIMEOUT_SEC = 30

# --- Kline chunking (5m bars: 1500 bars ≈ 5.2 days) ---
KLINES_LIMIT = 1500
CHUNK_DAYS = 5

# --- Deribit (options only) ---
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
DERIBIT_SLEEP_BETWEEN_CALLS_MS = 250
DERIBIT_MAX_RETRIES = 8
DERIBIT_BACKOFF_BASE_SEC = 1
DERIBIT_BACKOFF_MAX_SEC = 60
DERIBIT_REQUEST_TIMEOUT_SEC = 30
DERIBIT_INDEX_NAME = "btc_usd"

# --- Output (./data relative to Code folder) ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")

# Partition layout
SPOT_PATH = os.path.join(DATA_ROOT, "spot", SPOT_SYMBOL, "interval=5m")
FUTURES_PERP_PATH = os.path.join(DATA_ROOT, "futures", "usdm_perp", FUTURES_SYMBOL)
FUTURES_CONTINUOUS_PATH = os.path.join(DATA_ROOT, "futures", "continuous", FUTURES_SYMBOL)
# Options: pipeline archived at archive/options_legacy_20260214/
OPTIONS_EXCHANGE_INFO_PATH = os.path.join(DATA_ROOT, "_options_archived")
OPTIONS_KLINES_PATH = os.path.join(DATA_ROOT, "_options_archived", "klines")
OPTIONS_SELECTED_INSTRUMENTS_PATH = os.path.join(DATA_ROOT, "_options_archived", "selected_instruments.json")

# Checkpoint / manifest
MANIFEST_DIR = os.path.join(DATA_ROOT, "_manifest")
DONE_CHUNKS_FILE = os.path.join(MANIFEST_DIR, "done_chunks.json")

# --- Options strike selection (Deribit near-ATM) ---
ATM_PERCENT_RANGE = 0.10  # ±10% of ATM
MAX_STRIKES_PER_SIDE = 10
OPTIONS_LIFE_DAYS_BEFORE_EXPIRY = 14
OPTIONS_RESOLUTION_MINUTES = 5
OPTIONS_CANDLE_CHUNK_DAYS = 7  # chunk candle requests by week to avoid limits

# --- Deribit options v2 (quote-based) --- ARCHIVED at archive/options_legacy_20260214/
# DERIBIT_QUOTES_V2_ROOT = os.path.join(DATA_ROOT, "deribit_quotes_v2")
