Methodology: 3h window, staleness 120 min, min 3 strikes, max gap 10k

This folder contains the methodology variant with 120-minute staleness (all else as in 3h_stale90_min3_gap10k).

Rules:
- Rolling lookback: 3 hours [t - 3h, t].
- Per-strike staleness: keep strike only if (t - last_trade_time) <= 120 minutes.
- Minimum unique strikes: >= 3 (after latest-per-strike and staleness).
- Latest trade per strike: one IV per strike = most recent observation only (latest_trade_per_strike_rule = true).
- Strike gap: interpolate only if local bracket gap <= 10,000; else iv_hat = NaN, out_of_range = True.
- Same expiry switch (Z 08:00 UTC), PCHIP/linear, no extrapolation.

Script: build_options_smiles_3h_stale120_min3_gap10k.py
Outputs: options_smile_snapshot_stats.parquet, options_smile_event_iv.parquet,
         options_smile_report.json, options_iv_timestamp_coverage.json, plots/
