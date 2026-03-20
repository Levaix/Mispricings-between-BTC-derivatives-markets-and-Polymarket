# Step 5c fill report

## Summary
- NaN count before fill: 5745234 (85.71%)
- NaN count after fill: 2437830 (36.37%)
- Number of contracts (market_id): 506

## Sample: first 10 rows around first_obs_ts (3 contracts)
- market_id=1259195:
                   ts_utc  market_id  prob_yes  prob_yes_filled  is_pre_first_trade
2026-01-24 17:15:00+00:00    1259195       NaN              NaN                True
2026-01-24 17:20:00+00:00    1259195       NaN              NaN                True
2026-01-24 17:25:00+00:00    1259195       NaN              NaN                True
2026-01-24 17:30:00+00:00    1259195       0.5              0.5               False
2026-01-24 17:35:00+00:00    1259195       0.5              0.5               False
2026-01-24 17:40:00+00:00    1259195       0.5              0.5               False
2026-01-24 17:45:00+00:00    1259195       0.5              0.5               False
2026-01-24 17:50:00+00:00    1259195       0.5              0.5               False
2026-01-24 17:55:00+00:00    1259195       0.5              0.5               False
2026-01-24 18:00:00+00:00    1259195       0.5              0.5               False

- market_id=1058602:
                   ts_utc  market_id  prob_yes  prob_yes_filled  is_pre_first_trade
2026-01-01 00:00:00+00:00    1058602    0.9865           0.9865               False
2026-01-01 00:05:00+00:00    1058602    0.9865           0.9865               False
2026-01-01 00:10:00+00:00    1058602    0.9865           0.9865               False
2026-01-01 00:15:00+00:00    1058602    0.9865           0.9865               False
2026-01-01 00:20:00+00:00    1058602    0.9860           0.9860               False
2026-01-01 00:25:00+00:00    1058602    0.9870           0.9870               False
2026-01-01 00:30:00+00:00    1058602    0.9860           0.9860               False

- market_id=1294006:
                   ts_utc  market_id  prob_yes  prob_yes_filled  is_pre_first_trade
2026-01-29 17:05:00+00:00    1294006       NaN              NaN                True
2026-01-29 17:10:00+00:00    1294006       NaN              NaN                True
2026-01-29 17:15:00+00:00    1294006       NaN              NaN                True
2026-01-29 17:20:00+00:00    1294006       0.5              0.5               False
2026-01-29 17:25:00+00:00    1294006       NaN              0.5               False
2026-01-29 17:30:00+00:00    1294006       0.5              0.5               False
2026-01-29 17:35:00+00:00    1294006       0.5              0.5               False
2026-01-29 17:40:00+00:00    1294006       0.5              0.5               False
2026-01-29 17:45:00+00:00    1294006       0.5              0.5               False
2026-01-29 17:50:00+00:00    1294006       0.5              0.5               False

QA: ffill semantics OK (prob_yes_filled = last seen in contract at or before ts_utc, or NaN pre-first).
QA: no cross-contract leakage (filled values only from same contract's prob_yes).

## Result
PASS