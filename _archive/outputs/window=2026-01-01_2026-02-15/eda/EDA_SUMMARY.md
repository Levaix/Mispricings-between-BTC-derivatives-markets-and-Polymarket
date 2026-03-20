# EDA Summary (Jan–Feb 2026)

- Window: 2026-01-01 to 2026-02-15 (5m UTC); master panel n_rows=13248.
- Index 5m return std (volatility proxy): 0.001723; daily rolled vol proxy: 0.001592.
- Perp–index basis (bps): median -4.63, p1=-9.58, p99=-0.47.
- Funding rate: median 0.000034, p1=-0.000137, p99=0.000100; 138 unique funding times as expected.
- Spot/index/perp are tightly aligned: spot–index and perp–index spreads small; correlation(spot, index) ~1.
- CCF(perp_ret, index_ret): peak |corr| at lag 0 is 0.9928.
- Options (liquid subset): 37 instruments; top expiries by volume: ['2026-03-27T08:00:00.000Z', '2026-02-27T08:00:00.000Z', '2026-02-20T08:00:00.000Z'].
- Options median trade_bar_share across instruments: 7.27%.
- No major anomalies: funding and basis behave as expected; spot missingness noted in sanity check.