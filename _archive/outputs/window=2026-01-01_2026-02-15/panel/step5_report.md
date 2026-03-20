# Step 5 report

- Long panel: master_panel_long.parquet (6,703,488 rows)
- Wide panel: master_panel_wide.parquet (13,248 rows)
- QA pass: True

## Schema (long)
- ts_utc: datetime64[ns, UTC]
- open_time_ms: int64 (ms since epoch)
- market_id: int or str
- expiry_date: date
- expiry_ts: datetime64[ns, UTC]
- strike: int
- prob_yes: float64 [0,1]