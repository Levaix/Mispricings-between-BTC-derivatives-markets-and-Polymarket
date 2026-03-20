# Robustness comparison: options smile methodology variants 1–4

Comparison built from existing `options_smile_report.json` and `options_iv_timestamp_coverage.json` in each methodology folder. No pipelines were rerun.

## Summary table

| Method | Coverage (usable IV %) | Fit success % | Out-of-range % | Avg usable event rows per successful smile | Main tradeoff |
|--------|-------------------------|---------------|----------------|-------------------------------------------|----------------|
| Method 1: 1h / no staleness / min5 | 16.5201 | 52.1854 | 19.54 | 106.4377 | Most conservative; lowest coverage, many rows without fit. |
| Method 2: 3h / 90m stale / min5 | 20.6679 | 59.5386 | 23.49 | 107.4376 | Better coverage with 3h + staleness; more out-of-range. |
| Method 3: 3h / 90m stale / min3 / gap10k | 22.5596 | 74.2933 | 28.43 | 93.9809 | Latest-trade + gap cap; high coverage, some gap fails. |
| Method 4: 3h / 120m stale / min3 / gap10k | 25.1306 | 80.6697 | 30.04 | 96.4163 | Relaxed staleness; highest coverage, highest out-of-range. |

*Avg usable event rows per successful smile* = usable_iv_count / fit_success_count. Because multiple Polymarket strikes map to the same option smile, several event rows can be produced from one successful snapshot fit.

## Method descriptions

| Method | Lookback | Staleness | Min strikes | Strike gap | Latest trade per strike |
|--------|----------|-----------|-------------|------------|--------------------------|
| Method 1: 1h / no staleness / min5 | 1h | none | 5 | none | no |
| Method 2: 3h / 90m stale / min5 | 3h | 90 min | 5 | none | no |
| Method 3: 3h / 90m stale / min3 / gap10k | 3h | 90 min | 3 | <=10k | yes |
| Method 4: 3h / 120m stale / min3 / gap10k | 3h | 120 min | 3 | <=10k | yes |

## Core outcome metrics

| Method | usable_iv_rows | pct_rows_usable_iv | timestamps_with_iv | pct_timestamps_with_iv | avg_usable_iv_per_ts | median_usable_iv_per_ts |
|--------|----------------|--------------------|--------------------|-------------------------|------------------------|--------------------------|
| Method 1: 1h / no staleness / min5 | 202949 | 16.5201 | 14638 | 86.718 | 12.023 | 11.0 |
| Method 2: 3h / 90m stale / min5 | 253904 | 20.6679 | 15919 | 94.3069 | 15.0417 | 14.0 |
| Method 3: 3h / 90m stale / min3 / gap10k | 277143 | 22.5596 | 16536 | 97.9621 | 16.4184 | 16.0 |
| Method 4: 3h / 120m stale / min3 / gap10k | 308728 | 25.1306 | 16664 | 98.7204 | 18.2896 | 18.0 |

## Smile construction metrics

| Method | total_windows | fit_success_count | fit_success_rate_pct | too_few_strikes | range_too_small | no_iv_data |
|--------|----------------|--------------------|-----------------------|-----------------|----------------|------------|
| Method 1: 1h / no staleness / min5 | 365378 | 190674 | 52.1854 | 172028 | 2676 |  |
| Method 2: 3h / 90m stale / min5 | 396931 | 236327 | 59.5386 | 140805 | 3097 | 16702 |
| Method 3: 3h / 90m stale / min3 / gap10k | 396931 | 294893 | 74.2933 | 65884 | 19452 | 16702 |
| Method 4: 3h / 120m stale / min3 / gap10k | 396931 | 320203 | 80.6697 | 50788 | 17295 | 8645 |

## Failure / tradeoff metrics

| Method | out_of_range_count | pct_rows_out_of_range | no_snapshot_fit_count | pct_rows_no_snapshot_fit |
|--------|--------------------|------------------------|------------------------|---------------------------|
| Method 1: 1h / no staleness / min5 | 240064 | 19.54 | 785482 | 63.94 |
| Method 2: 3h / 90m stale / min5 | 288545 | 23.49 | 686046 | 55.84 |
| Method 3: 3h / 90m stale / min3 / gap10k | 349232 | 28.43 | 602120 | 49.01 |
| Method 4: 3h / 120m stale / min3 / gap10k | 369013 | 30.04 | 550754 | 44.83 |

## Staleness / filtering (methods 2–4)

| Method | total_retained_after_staleness | total_discarded_stale | avg_retained_per_snapshot | avg_discarded_per_snapshot |
|--------|-------------------------------|------------------------|----------------------------|----------------------------|
| Method 1: 1h / no staleness / min5 |  |  |  |  |
| Method 2: 3h / 90m stale / min5 | 2686129 | 1077123 | 6.7672 | 2.7136 |
| Method 3: 3h / 90m stale / min3 / gap10k | 2686129 | 1077123 | 6.7672 | 2.7136 |
| Method 4: 3h / 120m stale / min3 / gap10k | 3107078 | 656174 | 7.8278 | 1.6531 |

## Strike gap diagnostics (methods 3–4)

| Method | in_range_gap_rule_fail_count | avg_local_gap_success | max_local_gap_success |
|--------|------------------------------|------------------------|----------------------|
| Method 1: 1h / no staleness / min5 |  |  |  |
| Method 2: 3h / 90m stale / min5 |  |  |  |
| Method 3: 3h / 90m stale / min3 / gap10k | 4317 | 1016.7441 | 10000.0 |
| Method 4: 3h / 120m stale / min3 / gap10k | 3867 | 900.1645 | 10000.0 |

---

## Interpretation

**Progression from Method 1 to Method 4**

- **Method 1 (1h / no staleness / min5)** is the baseline: shortest lookback, no staleness filter, strict minimum of 5 strikes. It yields the lowest row-level IV coverage (about 16.5% of Polymarket rows with usable IV) but the highest share of rows without any snapshot fit (about 64%), reflecting many timestamps or expiry/type combinations with insufficient option data.

- **Method 2 (3h / 90m stale / min5)** lengthens the lookback to 3 hours and adds a 90-minute staleness filter, keeping min 5 strikes. More snapshots succeed (higher fit success rate), and both row-level and timestamp-level coverage improve. The share of rows with no snapshot fit drops; the share of rows out of range rises slightly as more interpolations are attempted. Staleness diagnostics show a substantial share of candidate strikes discarded, so the filter is active.

- **Method 3 (3h / 90m stale / min3 / gap10k)** relaxes the strike requirement to 3 and adds the latest-trade-per-strike rule plus a local strike gap cap of 10k. Fit success rate and usable IV coverage increase further; fewer rows have no snapshot fit. The gap rule causes a small number of in-range gap failures. Tradeoff: higher coverage but more interpolations that may span wider strike distances (capped at 10k).

- **Method 4 (3h / 120m stale / min3 / gap10k)** keeps the same rules as Method 3 but relaxes staleness to 120 minutes. More strikes are retained after the staleness filter (higher avg retained per snapshot, lower discarded). This yields the highest row-level and timestamp-level coverage and the highest fit success rate, with the lowest no-snapshot-fit share. Out-of-range share is highest, as more marginal interpolations are attempted.

**Main tradeoffs**

- **Coverage vs conservatism:** Moving from Method 1 to 4, usable IV coverage (both by rows and by timestamps) increases monotonically. The cost is more out-of-range outcomes (target strike outside or gap-fail) and, in Methods 3–4, use of thinner strike sets (min 3) and a hard gap cap rather than excluding wide gaps entirely.

- **Staleness:** Tightening staleness (90 min) discards more strikes and reduces coverage; relaxing to 120 min (Method 4) retains more strikes and increases coverage, at the expense of using slightly older quotes.

**Best balance (diagnostics only)**

The diagnostics do not define a single “best” method; they summarize tradeoffs. Method 1 is most conservative (least coverage, strictest rules). Method 4 achieves the highest coverage and lowest no-snapshot-fit rate but the highest out-of-range share. Methods 2 and 3 sit in between. For a thesis, the choice depends on whether the analysis prioritizes coverage (favoring 3 or 4) or methodological conservatism (favoring 1 or 2).
