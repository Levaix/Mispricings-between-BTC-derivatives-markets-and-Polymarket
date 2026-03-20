# 4.1 Overview of the Empirical Research Design

This section outlines the quantitative framework underlying the thesis. It states the rationale for the chosen approach, the joint treatment of prediction markets, crypto markets, and news as an interconnected information system, and the high-level flow of the empirical pipeline from data collection to econometric testing. No results are reported here; the focus is on the finalized research design and methodological choices.

---

## 1. Purpose of the Quantitative Framework

The thesis adopts a quantitative, data-driven approach to study how information is reflected in prediction markets and in crypto and related financial markets. A formal quantitative framework is necessary for three main reasons.

First, *quantification* allows precise measurement of beliefs (via prediction market probabilities), prices (via spot, futures, and options), and their comovement over time. Without a structured quantitative setup, it is difficult to test hypotheses about information transmission, lead-lag relationships, or mispricing in a replicable way. The framework therefore specifies clearly defined variables, sampling schemes, and alignment rules so that all subsequent analysis is consistent and interpretable.

Second, *high-frequency data* are essential for the questions at hand. Prediction market probabilities and crypto prices can react to the same information within minutes or hours. Coarser sampling (e.g., daily) would obscure short-run dynamics, lead-lag effects, and potential arbitrage opportunities that exist only over intraday horizons. The empirical design therefore uses a 5-minute resolution as the common time grid, balancing granularity with data availability and computational feasibility. This choice allows the thesis to speak directly to information efficiency and price discovery at a frequency where many theoretical and empirical studies of prediction markets and crypto markets are still limited.

Third, *prediction markets cannot be evaluated in isolation*. Their interpretation as belief aggregators or forecast devices depends on how they relate to other information-sensitive markets—in this case, crypto spot, futures, options, and funding rates. A structured comparison with these markets is required to assess whether prediction market prices are consistent with no-arbitrage conditions, whether they lead or lag other markets, and how they respond to news. The quantitative framework is therefore built around a unified panel that aligns prediction market probabilities with crypto market and news data, enabling joint econometric analysis rather than isolated case studies.

---

## 2. Joint Study of Prediction Markets, Crypto Markets, and News

The thesis treats prediction markets, crypto markets, and news as parts of a single *interconnected information system*. Prediction markets (here, Polymarket contracts on Bitcoin price outcomes) aggregate beliefs and update in response to information; crypto spot, futures, and options markets reflect expectations and risk premia and are sensitive to the same information; news and events provide external shocks and narrative context. The objective is to study *information transmission* and *price discovery* across these channels—i.e., how information flows between markets and how quickly and consistently it is incorporated into prices and probabilities.

Analyzing prediction markets, crypto markets, or news in isolation would miss important dynamics. Prediction market probabilities may lead or lag option-implied probabilities or spot returns; funding rates and basis may reflect the same underlying expectations as certain prediction contracts; news may move one market first and others with a delay. Only a joint, time-aligned design can reveal these relationships, test for Granger causality or correlation structure, and place prediction markets within the broader information and pricing ecosystem. The methodology is therefore explicitly multi-market and multi-source, with a single master panel as the central object that supports all comparative and causal-style analyses.

---

## 3. High-Level Flow of the Full Analysis Pipeline

The empirical pipeline is organized as a sequence of stages from raw data to econometric inference. Each stage produces well-defined outputs that feed into the next, ensuring consistency and reproducibility.

**Data collection.** Data are collected from multiple sources: (i) crypto spot and futures (e.g., Binance) for prices and volumes; (ii) options markets (e.g., Deribit) for implied volatilities and option-implied probabilities; (iii) prediction markets (Polymarket) for event-level probabilities on Bitcoin price outcomes; (iv) news or event datasets where applicable. Collection is done over a defined sample window (e.g., January–February 2026 for the main analysis) with clear inclusion criteria for instruments and contracts.

**Time alignment at 5-minute resolution.** All series are aligned to a common UTC 5-minute grid. This step is critical: prediction market updates, exchange ticks, and option quotes occur at irregular times. Mapping them to a single grid allows a direct comparison of probabilities, prices, and derived measures at the same timestamps. The grid is defined from the start to the end of the sample window, and missing values are handled explicitly (e.g., forward-fill within contract for prediction markets) so that the panel is consistent for downstream analysis.

**Probability extraction from options.** Where options data are available, option-implied probabilities (e.g., risk-neutral probabilities of price events) are derived from option prices and aligned to the same 5-minute grid. These serve as a financial-market benchmark against which prediction market probabilities can be compared, supporting tests of consistency and arbitrage.

**Master panel construction.** A single master panel is built that combines, at each 5-minute timestamp, (i) prediction market probabilities (raw and, where applicable, forward-filled within contract); (ii) spot and futures prices and returns; (iii) funding rates and basis; (iv) option-implied measures where relevant. The panel is structured so that each row corresponds to a timestamp and each contract or instrument is either a column (wide) or a row identifier (long), as required by the specific econometric methods. This master panel is the central dataset for all subsequent analysis.

**Econometric analysis.** Using the master panel, the thesis applies standard econometric tools: correlation analysis to describe comovement between prediction market probabilities and crypto market variables; Granger causality (or related lead-lag tests) to assess whether one series helps predict another; and, where appropriate, regression-based specifications to control for confounding factors. The exact specifications are detailed in later sections; here the point is that the design enables a coherent set of tests on information transmission and price discovery.

**Arbitrage and mispricing framework.** The aligned panel supports a structured assessment of arbitrage and mispricing. Prediction market probabilities can be compared with option-implied probabilities or with no-arbitrage bounds implied by spot, futures, and options. Deviations are interpreted as potential mispricing or frictions; the framework allows both descriptive evidence and, where suitable, simple tests of statistical significance.

**News integration and event analysis.** Where news or event data are available, they are merged to the master panel by time (and optionally by topic or keyword). This allows event windows around news releases to be defined and the response of prediction markets and crypto markets to be studied in a unified way, linking information arrival to price and probability updates.

The pipeline is thus a single, coherent flow: from raw data to aligned, contract-level and market-level series, to a master panel, to econometric and arbitrage analyses, with news and events integrated when applicable. No single stage is sufficient on its own; the contribution of the design lies in the integration of these stages into one reproducible framework.

---

## 4. Positioning Within the Literature

This framework positions the thesis within the existing literature in three ways. First, it extends the study of prediction markets to a high-frequency, multi-market setting, complementing work that has focused on prediction accuracy or daily data. Second, it provides a clear methodological template for comparing prediction market prices with financial market prices (spot, options, funding) on a level playing field, which is necessary for rigorous statements about efficiency and arbitrage. Third, by treating prediction markets, crypto markets, and news as one information system, the thesis speaks to the broader question of how heterogeneous markets aggregate and transmit information—a question of interest in finance, market design, and information economics. The following sections elaborate each stage of the pipeline in greater detail and state the precise hypotheses and specifications used in the empirical analysis.
