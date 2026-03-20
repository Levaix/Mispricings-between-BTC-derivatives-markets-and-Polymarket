Methodology: 1h window, no staleness, min 5 strikes (baseline)

This folder contains the current baseline outputs from the options smile pipeline.

Rules:
- Rolling lookback: 1 hour [t - 1h, t] (inclusive).
- Staleness: none — all option bars in the window are used regardless of last-trade age.
- Minimum unique strikes: >= 5 per (bucket, expiry, option_type).
- Strike gap: no constraint on max gap between consecutive strikes.
- Interpolation: PCHIP when available and n >= 3, else linear; no extrapolation.

Outputs:
- options_smile_snapshot_stats.parquet
- options_smile_event_iv.parquet
- options_smile_report.json
- options_iv_timestamp_coverage.json
- plots/ (n_unique_strikes_hist.png, fit_success_rate_by_day.png, example_smiles.png)
