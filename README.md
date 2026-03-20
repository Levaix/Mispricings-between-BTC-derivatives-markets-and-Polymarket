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
