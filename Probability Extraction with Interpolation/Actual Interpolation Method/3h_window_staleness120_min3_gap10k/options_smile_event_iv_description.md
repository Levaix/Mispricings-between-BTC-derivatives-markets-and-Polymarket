# Options smile event IV — dataset description

**File:** `options_smile_event_iv.parquet` / `options_smile_event_iv.csv`  
**Methodology:** 3h window, 120 min staleness, min 3 strikes, max gap 10k (Method 4).

---

## What this dataset is

- **One row per Polymarket observation:** Each row corresponds to one row in `polymarket_probabilities_5m_long.parquet` (same row count). No aggregation over time or contracts.
- **Purpose:** For each (timestamp, Polymarket contract / event date, strike), the pipeline attempts to build an option IV smile from Deribit 5m option data and to evaluate implied volatility at that strike. This table stores the **evaluation result** (`iv_hat`) and related diagnostics.
- **Key output:** `iv_hat` — implied volatility (decimal, e.g. 0.55 = 55%) interpolated at the Polymarket strike from the option smile, or `NaN` if no fit, out of range, or gap rule failed.

---

## Columns (18)

| Column | Type | Description |
|--------|------|-------------|
| **bucket_5m_utc** | datetime | 5-minute bucket (evaluation time) aligned to Polymarket timestamp; UTC. |
| **event_date** | string | Polymarket event date (expiry date of the market), e.g. `2026-01-15`. |
| **strike_K** | float | Strike level (e.g. BTC price threshold) for this Polymarket contract. |
| **prob_yes** | float | Polymarket “yes” probability if present in the long table; else NaN. |
| **contract_id** | any | Polymarket contract identifier (from long table). |
| **chosen_option_expiry** | string | Option expiry date used for the smile: either event_date or event_date+1, per 08:00 UTC rule. |
| **expiry_utc** | string | Same as chosen_option_expiry (option expiry used). |
| **option_type** | string | Option type used for the smile, e.g. `C` (call). |
| **spot_snapshot** | float | Median index (spot) price used for that snapshot when building the smile; NaN if no fit. |
| **moneyness_event** | float | Strike / spot at evaluation time (strike_K / spot_snapshot); NaN if no valid spot. |
| **iv_hat** | float | **Implied volatility** (decimal) at the Polymarket strike from the option smile; NaN if no fit, out of range, or local strike gap > 10k. |
| **fit_success** | bool | Whether the option smile for this (bucket, expiry, option_type) was fitted successfully. |
| **out_of_range** | bool | True if iv_hat is NaN because the strike was outside the smile range or failed the gap rule. |
| **no_snapshot_fit** | bool | True if there was no successful smile snapshot for this (bucket, chosen_option_expiry, option_type). |
| **n_unique_strikes** | int/float | Number of unique strikes in the snapshot used for the smile (0 if no fit). |
| **moneyness_min** | float | Minimum moneyness (m_min) of the fitted smile; NaN if no fit. |
| **moneyness_max** | float | Maximum moneyness (m_max) of the fitted smile; NaN if no fit. |
| **total_contracts_sum** | float | Sum of contracts (volume) in the option snapshot; 0 if no fit. |

---

## Usage notes

- **Usable IV:** Rows with finite `iv_hat` are “usable” for downstream analysis (e.g. converting to probabilities). Filter with `iv_hat.notna() & np.isfinite(iv_hat)`.
- **Row key:** `(bucket_5m_utc, event_date, strike_K)` is unique and matches the Polymarket long table.
- **Expiry rule:** For event date Z, option expiry is Z if evaluation time ≤ Z 08:00 UTC, else Z+1.
