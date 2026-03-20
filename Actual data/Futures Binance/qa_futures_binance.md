# Futures Binance QA report

- Symbol: BTCUSDT
- Window: 2026-01-01T00:00:00 to 2026-02-28T23:59:59

## Perp OHLCV 5m

- dataset: perp_ohlcv_5m
- rows: 16992
- start_timestamp_utc: 2026-01-01T00:00:00+00:00
- end_timestamp_utc: 2026-02-28T23:55:00+00:00
- expected_bars: 16992
- missing_bars: 0
- duplicate_timestamps: 0
- bad_alignment_count: 0
- bad_open_count: 0
- bad_high_count: 0
- bad_low_count: 0
- bad_close_count: 0
- bad_volume_count: 0
- close_min: 60003.9
- close_max: 97692.5
- status: ok

## Perp Mark 5m

- dataset: perp_mark_5m
- rows: 16992
- start_timestamp_utc: 2026-01-01T00:00:00+00:00
- end_timestamp_utc: 2026-02-28T23:55:00+00:00
- expected_bars: 16992
- missing_bars: 0
- duplicate_timestamps: 0
- bad_alignment_count: 0
- bad_open_count: 0
- bad_high_count: 0
- bad_low_count: 0
- bad_close_count: 0
- close_min: 60020.57731884
- close_max: 97708.98585507
- status: ok

## Funding rate

- dataset: funding_rate
- rows: 177
- start_timestamp_utc: 2026-01-01T00:00:00.008000+00:00
- end_timestamp_utc: 2026-02-28T16:00:00.004000+00:00
- duplicate_fundingTime: 0
- monotonic: True
- gaps_count: 0
- funding_rate_min: -0.00015178
- funding_rate_max: 0.0001
- status: ok
