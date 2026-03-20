# 4.2 Data Collection and Sources

This section describes the primary data sources used in the thesis and the design choices governing their collection and preparation. The discussion is organized by asset class and market type: spot, derivatives, options, prediction markets, and news. Each subsection states the source, the variables of interest, the sampling and filtering rules, and the rationale for inclusion. No empirical results are reported; the aim is to establish a clear and reproducible data foundation for the subsequent construction of probabilities and econometric analysis.

---

## 4.2.1 Cryptocurrency Spot Market Data

Spot market data provide the benchmark price series for Bitcoin and the basis for return and volatility measures used throughout the analysis.

The primary series is the **BTCUSDT spot price**—the USDT-quoted price of Bitcoin on a major centralized exchange. The exchange used is one of the largest by volume and liquidity (e.g., Binance), ensuring that the series is representative of the broad market and less subject to idiosyncratic illiquidity or manipulation than smaller venues. Prices are sampled at **5-minute frequency** in coordinated universal time (UTC), yielding a regular time series of mid or last-trade prices at each interval. This frequency is chosen to match the common grid used for derivatives, options, and prediction markets, so that all series can be aligned without loss of comparability.

**Log returns** are constructed from the 5-minute price series as the first difference of the natural logarithm of price. Log returns are used because they are additive over time and symmetric for gains and losses, and they align with standard assumptions in asset pricing and time-series analysis. **Trading volume** over each 5-minute interval is recorded in base or quote currency, providing a measure of activity and, when combined with price, a simple **liquidity proxy**. Turnover or volume relative to a rolling window can be used to flag periods of unusually low liquidity, though the main analysis relies on the 5-minute grid and the use of a liquid spot pair.

**BTCUSDT** is used as the benchmark spot series for several reasons: it is the dominant trading pair by volume in crypto markets, it is quoted around the clock, and it serves as the reference price for many derivatives and options contracts. Using a single, liquid pair avoids the complexity of aggregating across multiple exchanges and ensures consistency with the pricing of derivatives and options that themselves reference a similar or identical index.

The **sample window** is defined ex ante as a continuous period (e.g., several weeks or months) in coordinated universal time. The start and end dates are chosen to cover a sufficient number of prediction market expiries and option expirations while remaining within the period for which all data sources—spot, derivatives, options, and prediction markets—are available. The window is closed in the sense that no data outside this period are used in the main analysis, ensuring a well-defined population of observations.

---

## 4.2.2 Cryptocurrency Derivatives Data

Derivatives data capture forward-looking expectations and funding conditions that cannot be inferred from spot prices alone.

**Perpetual futures** are contracts that do not have a fixed expiry; they trade continuously and are settled periodically via a funding mechanism. The source used is a major exchange’s perpetual futures market (e.g., Binance USD-margined perpetual futures) for the same underlying (BTC). Prices are sampled at 5-minute frequency in UTC. Because perpetuals trade 24/7, the series has no overnight or weekend gaps, which simplifies alignment with spot and other markets. The **relation to spot price**—the basis or spread between perpetual and spot—is a key variable: it reflects cost of carry, funding expectations, and speculative demand, and it is used later in the analysis as an indicator of sentiment and as a potential predictor or correlate of prediction market probabilities.

The **index price** is a reference price used by the exchange to mark positions and to compute funding. It is typically a volume-weighted or median price across several spot venues, updated at high frequency. The **mark price** is an exchange-computed fair value for the perpetual contract, often a blend of the index price and the perpetual’s own order book, designed to reduce manipulation of the funding rate. Both matter for **pricing and liquidation dynamics**: liquidations and margin calls depend on the mark price, and the funding rate is computed from the difference between mark (or last trade) and index. For the thesis, the index price is used where a clean reference is needed (e.g., basis computation), and the mark price is used where funding or contract-level valuation is relevant.

**Funding rate** data record the periodic payment between long and short positions in the perpetual contract. The rate is typically quoted at fixed intervals (e.g., every 8 hours) and is signed so that positive funding means longs pay shorts. It is interpreted as a **directional positioning signal**: sustained positive funding suggests crowded long positions and can precede mean reversion or corrections; it also reflects the cost of maintaining leveraged exposure. Funding is included as an **explanatory variable** in the empirical design because it summarizes market sentiment and positioning that may be related to prediction market probabilities and to option-implied expectations.

Derivatives are important for the thesis because they embed **forward-looking expectations** and **risk sentiment** that spot prices alone do not reveal. The basis and funding rate provide a direct link between cash and futures markets and allow the analysis to ask whether prediction markets and options are consistent with the information in the derivatives market.

---

## 4.2.3 Options Market Data

Options data are used to derive implied volatilities and option-implied probabilities that serve as a financial-market benchmark for prediction market probabilities.

**BTC options** are sourced from a major crypto options venue (e.g., Deribit). Deribit is used because it offers deep liquidity in Bitcoin options, standardized contract specifications, and a clear index (e.g., BTC spot index) used for settlement. Contracts are **European-style** (exercise only at expiry), which simplifies the interpretation of implied volatility and probability and avoids early-exercise premia. **Expiry selection** is aligned with the prediction market design: where prediction markets reference specific calendar dates (e.g., “Bitcoin above strike on date D”), options with expiries on or near those dates are selected so that implied probabilities from options can be compared directly with prediction market probabilities for the same economic event.

**Contract selection and liquidity filtering** are applied to avoid noise from illiquid series. Minimum volume or open interest thresholds are set so that only contracts with sufficient trading activity are retained. Illiquid or sparsely traded contracts are removed. Sparse strikes are handled by focusing on a strike range around the at-the-money level rather than using every listed strike. These rules ensure that the option-implied measures reflect tradable prices rather than stale or wide quotes.

The **near-at-the-money (ATM) strike focus** is adopted for two reasons. First, near-ATM options have the highest liquidity and the tightest bid-ask spreads, so their prices are the most reliable. Second, **deep in-the-money (ITM) or out-of-the-money (OTM)** options are more sensitive to model assumptions (e.g., tail shape) and to illiquidity; their implied probabilities are therefore less reliable for comparison with prediction markets. By concentrating on near-ATM strikes and liquid expiries, the thesis obtains a stable set of option-implied probabilities that can be used in extraction of implied event probabilities and in tests of consistency with prediction markets. The exact role of these probabilities in the econometric framework is set out in later methodology sections.

---

## 4.2.4 Prediction Market Data (Polymarket)

Prediction market data provide the main object of study: market-implied probabilities over Bitcoin price outcomes.

The contracts used are **daily Bitcoin “above strike” markets** on Polymarket. Each contract is a **binary** outcome: it pays off if the price of Bitcoin is above a specified strike level at a specified expiry time (e.g., end of a given calendar day in UTC), and zero otherwise. The **YES/NO payoff** structure implies that, under standard assumptions, the traded YES price can be interpreted as the market’s implied probability of the event. **Settlement** is typically based on a designated price source (e.g., a spot or index price at the expiry time), which is documented by the platform.

A **market ladder** is constructed **across expiries**: for each calendar day in the sample window, a set of strikes is selected (e.g., a range around the then-prevailing price), and one contract per strike-expiry pair is retained. This yields a **balanced panel** of contracts across the sample window—one row (or one dimension) per contract and one column (or timestamp dimension) per 5-minute interval. The ladder is chosen so that the same type of economic question (“price above strike on date D”) is asked consistently across days and strikes, enabling comparison with option-implied probabilities and with spot and derivatives data.

**Probability extraction** uses the **YES price as the probability proxy**: at each observation time, the mid or last traded price of the YES token is taken as the market-implied probability that the event will occur. Because prediction markets often update only when there is trading, the raw probability series can be **sparse** (many missing values at 5-minute grid points). To obtain a continuous series for econometric use, a **forward-fill within contract** is applied: for each contract, the last observed probability is carried forward until the next observation, and no fill is applied before the first observed trade for that contract. This preserves the contract-specific nature of the series and avoids mixing information across different strikes or expiries. **Liquidity filters** may be applied at the contract-selection stage (e.g., excluding contracts with very low volume or short history) so that the final panel contains only markets that are sufficiently active to be informative.

This **contract family** is chosen because it maps directly onto a well-defined financial variable (Bitcoin price at a point in time) and onto the same type of event that can be approximated with options (e.g., probability that price exceeds a strike at expiry). The daily ladder and strike coverage allow the thesis to study how these probabilities evolve at high frequency and how they relate to spot, derivatives, and option-implied measures over the same horizon.

---

## 4.2.5 News and Event Data

News and event data provide an external information dimension against which market and probability responses can be evaluated.

**Major crypto-related catalysts** include protocol upgrades or hard forks, macroeconomic announcements (e.g., central bank decisions, inflation releases), security incidents or hacks, regulatory news, and large realized volatility or liquidity events. The thesis does not rely on a single narrow definition of “news”; rather, an **event classification framework** is used to identify and **time-stamp** relevant events. Events are **categorized** (e.g., macro vs crypto-native, positive vs negative) so that the analysis can distinguish between types of information and test whether prediction markets and crypto markets react differently to different event types.

**Event windows** are defined around each event (e.g., a window of several 5-minute intervals before and after the event time). These windows are used in **event studies** and in **interaction regressions** where the effect of a variable (e.g., prediction market probability or funding rate) is allowed to differ inside vs outside an event window. The methodological choice is to align events to the same **5-minute grid** used for prices and probabilities, so that event dummies or window indicators can be merged directly into the master panel without additional interpolation.

News and event data are not the primary focus of the thesis, but their inclusion allows the design to speak to **information arrival** and to the robustness of the main results to known catalysts. The exact sources (e.g., public calendars, news feeds, or event datasets) are chosen for transparency and reproducibility; the important point methodologically is that events are defined ex ante or by objective criteria where possible, and that their integration into the tests is clearly specified.

---

The datasets described in this section—spot, derivatives, options, prediction markets, and news—form the **empirical foundation** for the thesis. They are collected over a common sample window, aligned to a common 5-minute grid, and filtered for liquidity and consistency. The next sections of the methodology describe how these raw and aligned series are used to construct probability measures (from options and prediction markets), how the master panel is assembled, and how econometric and arbitrage analyses are applied. No single source is sufficient on its own; the contribution of the data design lies in the integration of these sources into a single, coherent framework for testing information transmission and price discovery across prediction markets and financial markets.
