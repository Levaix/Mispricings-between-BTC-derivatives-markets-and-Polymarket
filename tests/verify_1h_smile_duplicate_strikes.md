# Verification: 1-hour window smile construction — multiple trades per strike

**Source:** `analysis/build_options_smiles.py` (and equivalent logic in `analysis/1h_window_no_staleness_min5/build_options_smiles.py` if present).  
**Scope:** How duplicate trades for the same strike within the 60-minute window are handled; strike counting; aggregation; smile input.

---

## 1. Duplicate trades per strike

**If multiple trades (rows) for the same strike occur within the 60-minute window, how are they handled?**

The code does **not** use the most recent trade, median, or multiple observations. It **aggregates to one IV per strike** as follows:

- **Preferred:** **Volume-weighted average IV** over all rows for that strike in the window.
  - Weight = `contracts_sum` if present and non-null; else `amount_sum`.
  - Formula: `IV_strike = (iv_norm * weight).sum() / weight.sum()` per strike.
- **Fallback:** If no usable weight column, **simple mean** of `iv_norm` per strike.

**Code location:** `fit_smile_for_snapshot`, lines 336–350 (weight choice) and 343–350 (`_agg`):

```python
def _agg(group: pd.Series) -> float:
    if w_col is None:
        return float(group["iv_norm"].mean())
    w = pd.to_numeric(group[w_col], errors="coerce").fillna(0.0)
    v = group["iv_norm"]
    if w.sum() <= 0:
        return float(v.mean())
    return float((v * w).sum() / w.sum())
```

So: **one observation per strike** = weighted average (or mean) of IV across all trades for that strike in the window.

---

## 2. Strike counting logic (minimum unique strikes ≥ 5)

**Does the code count unique strikes or number of trades/rows?**

It counts **unique strikes** (equivalently, unique moneyness values after aggregation).

- The snapshot group `g` is first aggregated by strike (see below), producing one row per strike in `by_strike`.
- The count used for the rule is `n_unique = len(m_unique)`, where `m_unique` comes from `np.unique(m_vals)` and `m_vals = by_strike["m"]` (moneyness = strike / spot).
- So the “minimum 5” rule is applied to **number of distinct strikes** (after aggregation), not to the number of rows/trades in the window.

**Code location:** `fit_smile_for_snapshot`, lines 367–374 and 381–382:

```python
m_unique, idx = np.unique(m_vals, return_index=True)
iv_unique = iv_vals[idx]
n_unique = int(len(m_unique))
# ...
if n_unique < min_points:
    return SmileFit(False, "too_few_strikes", ...)
```

**Conclusion:** Duplicate trades for the same strike **cannot** inflate the strike count; the count is always the number of unique strikes (after aggregation).

---

## 3. Aggregation step (where duplicate strikes become one value)

**Exact code section that reduces multiple rows per strike to one IV per strike:**

**Function:** `fit_smile_for_snapshot` (same file, ~lines 328–393).

**Mechanism:** `groupby("strike")` + `apply(_agg)`:

```python
by_strike = (
    g.groupby("strike", group_keys=False)
    .apply(_agg, include_groups=False)
    .reset_index(name="iv_strike")
    .dropna(subset=["iv_strike"])
)
```

- **Input:** `g` = all option rows in the 1h window for one `(expiry_date, option_type)` (many rows, possibly many per strike).
- **Output:** `by_strike` = one row per distinct `strike` with a single `iv_strike` (weighted average or mean as above).
- There is no `drop_duplicates`, `tail(1)`, or “most recent” selection; aggregation is entirely via this **groupby + _agg**.

So the **exact** place that enforces “one value per strike” is this **groupby("strike").apply(_agg)** block (lines 352–357).

---

## 4. Smile input (one IV per strike?)

**Confirm that the smile fit ultimately receives one IV value per strike.**

Yes.

- `by_strike` has one row per strike with one `iv_strike`.
- Moneyness is computed: `m = strike / snapshot_spot`.
- `m_unique` and `iv_unique` are taken from `by_strike` (with `np.unique(m_vals, return_index=True)` so that the interpolator gets unique m and corresponding iv).
- The `SmileFit` stores `m_nodes = m_unique` and `iv_nodes = iv_unique`, and the interpolator (PCHIP or linear) uses these arrays.
- So the smile fit receives **exactly one IV per unique strike** (one node per unique moneyness).

---

## Summary

| Question | Answer |
|----------|--------|
| **Which observation per strike is used?** | **One value per strike:** volume-weighted average of IV over all trades for that strike in the 1h window (or simple mean if no weight). Not most recent, not median. |
| **Can duplicate trades inflate the strike count?** | **No.** The minimum-5 rule uses `n_unique = len(m_unique)`, i.e. **unique strikes** after aggregation. Multiple trades per strike do not increase the count. |
| **Where is the unique-strike rule enforced?** | (1) **One IV per strike:** `fit_smile_for_snapshot`, **groupby("strike").apply(_agg)** (lines 352–357). (2) **Minimum 5 strikes:** same function, **if n_unique < min_points** (lines 374, 381–382) → fail with `"too_few_strikes"`. |

**File references:** `analysis/build_options_smiles.py` (or the same logic in `analysis/1h_window_no_staleness_min5/build_options_smiles.py`), function `fit_smile_for_snapshot`, and the `build_smiles_and_evaluate` loop that passes window group `g` into it.
