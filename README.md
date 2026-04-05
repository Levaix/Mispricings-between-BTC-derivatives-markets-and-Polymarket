## About

I am an undergraduate **dual-degree student in Business Administration (BBA) and Data Science**. My interests sit at the intersection of **algorithmic programming, statistical modelling, and market microstructure**—including **arbitrage**, **geopolitical and macro developments**, **major asset classes**, and **cross-asset correlations**. I have **hands-on Web3 experience**; prediction markets and digital assets are familiar terrain. I have contributed **research for a leading crypto venture capital firm** and **led a DAO** that served as its **research arm**. In this repository I built the **full empirical stack**—from raw Deribit, Binance, and Polymarket data through option-implied probabilities, exploratory analysis, and econometric work on **mispricings between BTC derivatives and Polymarket**.

# Thesis Objective (Jan–Feb 2026)

This thesis studies **mispricings between BTC derivatives markets (options and futures-implied expectations) and Polymarket**, focusing on how probability spreads behave over time and around curated news events.

## Folder overview (what each part shows)

- `Actual data/`: read-only datasets used in the empirical analysis (spot, futures, options-implied probabilities, Polymarket probabilities, and prepared “empirical master” panels).
- `EDA/`: exploratory data analyses (distributions, event timing structure, and cross-market descriptive patterns).
- `Econometric Analysis/`: econometric workflows and results (event studies, lead–lag systems, spread dynamics, and robustness).
- `Probability Extraction with Interpolation/`: intermediate steps that map option-market information into **option-implied event probabilities** used in the comparison to Polymarket.
- `Downloaders/`: reference scripts used to obtain and aggregate market data into `Actual data/`.
- `tests/`: sanity checks supporting correctness of processing.
- `utils/`: shared helper functions used across pipeline components.
- `_archive/`: legacy/archived experiments and code not required for the January–February 2026 thesis results.
