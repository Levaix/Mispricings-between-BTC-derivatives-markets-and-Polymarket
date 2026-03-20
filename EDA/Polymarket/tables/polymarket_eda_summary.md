# Polymarket EDA – Summary

This module summarizes the Polymarket BTC prediction markets used
in the thesis. Prediction market prices are interpreted as
probabilities of specific BTC outcomes (e.g. crossing a strike).

## Dataset description

- **Contracts metadata**: `Actual data/Polymarket/polymarket_contracts_metadata.parquet`
- **Probabilities (long)**: `Actual data/Polymarket/polymarket_probabilities_5m_long.parquet`
- **Probabilities (wide)**: `Actual data/Polymarket/polymarket_probabilities_5m_wide.parquet`
- **Number of markets (rows in metadata)**: 649
- **Number of strikes**: 27
- **Number of event dates**: 59
- **Sample period (5-minute probabilities)**: 2026-01-01T00:00:00+00:00 to 2026-02-28T16:55:00+00:00

## Prediction market probabilities

- Prediction market probabilities (`prob_yes`) represent the market-implied
  probability that a given BTC outcome (e.g. price above a strike on
  an event date) will occur. They reflect the beliefs and risk
  preferences of traders who buy and sell contracts on Polymarket.

## Key probability statistics

- **Mean prob_yes**: 0.4423, **std**: 0.4062
- **5th–95th percentiles**: 0.0005 – 0.9965
- **Range**: 0.0005 – 0.9995

## Event dates and strikes

- `event_date` corresponds to the underlying BTC event date or contract 
  settlement/expiry date; the probabilities describe outcomes on
  these dates.
- `strike_K` corresponds to BTC price thresholds used in the contracts 
  (e.g. "BTC above 80k on event_date").

## Key figures produced

- Probability distribution (`probability_distribution.png`)
- Probability time series for representative strikes (`probability_timeseries_example.png`)
- Probability vs strike cross-section (`probability_vs_strike.png`)
- Strike and event-date distributions
- Probability heatmap across strikes and time (`probability_heatmap.png`)
- Example market evolution for one event date (`example_market_evolution.png`)
