# Options EDA – Summary

This module summarizes the Deribit BTC options dataset used for
constructing the volatility smile and surface in the thesis.

## Dataset description

- **Source**: `Actual data/Options Deribit/agg_5m_merged.parquet`
- **Number of observations**: 342203
- **Sample period**: 2026-01-01T00:00:00+00:00 to 2026-02-28T23:55:00+00:00
- **Number of unique strikes**: 109
- **Number of unique expiries**: 72

## Option type mix

- **Share of calls**: 52.59% (counting labels 'call'/'C')
- **Share of puts**: 47.41% (counting labels 'put'/'P')

## Implied volatility (IV)

- **IV mean**: 55.8676, **std**: 29.8209
- **IV 5th–95th pct**: 35.1200 – 99.3296
- **IV range**: 0.0000 – 999.0000 (raw, before `iv_clean` filter)

Implied volatility reflects the market's view of future uncertainty
embedded in option prices. It is the central object for volatility
smile and surface analysis.

## Moneyness definition

- **Moneyness** is defined as \( K / S \), where \( K \) is the strike
  and \( S \) is the underlying index price. Values near 1
  correspond to at-the-money options; values above/below 1
  correspond to out-of-the-money calls/puts.

## Key figures produced

- IV and moneyness distributions
- IV vs moneyness scatter and IV vs maturity scatter
- Strike and notional/contract volume distributions
- Index price time series
- An example **volatility smile** (IV vs strike at a fixed time)

The volatility smile figure highlights how IV varies across strikes for a given timestamp, which is crucial for understanding departures from the Black–Scholes flat-volatility assumption.
