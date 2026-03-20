# News Events EDA – Summary

This module provides exploratory analysis for the manually curated 
Bitcoin-related news events dataset used in the news event study.

## Dataset description

- **Number of events**: 10 (each row is a distinct news shock).
- **Event date range**: 2026-01-06 to 2026-02-18 (event_date, in UTC calendar terms).
- **Key columns**: `event_id`, `event_timestamp_utc`, `event_date`, `category`, `source`, `event_label`, `headline`, `url`, `short_reason_for_inclusion`.

Events were **manually curated** from high-quality news sources 
(primarily CoinDesk for crypto-specific stories and Bloomberg for macroeconomic announcements). This manual step ensures that only clearly identifiable, market-relevant shocks enter the subsequent event study.

## Category breakdown

- **crypto_bitcoin events**: 5 (crypto-specific news on Bitcoin and the wider crypto market).
- **macro events**: 5 (macro announcements that affect global risk and liquidity conditions).

Distinguishing `crypto_bitcoin` from `macro` events is empirically useful: crypto news tends to move expectations about the crypto ecosystem directly, whereas macro news primarily affects Bitcoin via risk appetite, funding conditions, and cross-asset flows.

## Source breakdown

- **Number of distinct sources**: 2.
- **CoinDesk**: 5 events.
- **Bloomberg**: 5 events.

Using a small set of reputable outlets keeps the event definition consistent and mitigates concerns about noise from low-quality news sources.

## Role of event timestamps in the event study

The `event_timestamp_utc` field records the time at which the news item became public in Coordinated Universal Time. In the event study, this timestamp serves as the **anchor point** for constructing symmetric pre- and post-event windows (e.g. \[-12h, +12h\]). Market data from Bitcoin futures, options-implied probabilities, and Polymarket predictions are aligned relative to this timestamp to quantify how probability spreads and prices respond to the news shock.

## Figures produced

- **news_event_timeline.png** – location of each event on the UTC time line, colour-coded by category.
- **news_category_distribution.png** – bar chart of the number of `crypto_bitcoin` versus `macro` events.
- **news_event_hour_distribution.png** – histogram of `event_hour`, showing when in the day news tends to be released.
- **news_event_weekday_distribution.png** – counts of events by weekday.

Together, these tables and figures document the structure of the news dataset and provide context for interpreting the econometric results of the subsequent event study.