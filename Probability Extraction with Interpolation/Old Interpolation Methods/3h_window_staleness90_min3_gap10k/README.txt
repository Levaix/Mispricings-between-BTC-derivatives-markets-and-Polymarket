Methodology: 3h window, staleness 90 min, min 3 strikes, max gap 10k (variant 3)

This folder contains the methodology variant 3 script and its outputs.

Rules:
- Rolling lookback: 3 hours [t - 3h, t].
- Per-strike staleness: keep strike only if (t - last_trade_time) <= 90 minutes.
- Minimum unique strikes: >= 3 (after latest-per-strike and staleness).
- Latest trade per strike: one IV per strike = most recent observation only (no VWAP/mean/median).
- Strike gap: interpolate only if local bracket gap <= 10,000; else iv_hat = NaN, out_of_range = True.
- Same expiry switch (Z 08:00 UTC), PCHIP/linear, no extrapolation.

Script: build_options_smiles_3h_stale90_min3_gap10k.py
Outputs: options_smile_snapshot_stats.parquet, options_smile_event_iv.parquet,
         options_smile_report.json, options_iv_timestamp_coverage.json, plots/
