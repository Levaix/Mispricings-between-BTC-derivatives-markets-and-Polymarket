# 4.3 Data Processing and Time Alignment (5-Minute Master Panel Construction)

This section describes how the raw data from the sources outlined in Section 4.2 are cleaned, aligned to a common time grid, and combined into a unified high-frequency dataset—the master panel—that serves as the foundation for all econometric and arbitrage tests in later sections. The discussion covers preprocessing, grid construction, handling of sparse and missing data, and the structure of the final outputs. No empirical findings are reported; the focus is on the finalized processing pipeline.

---

## Data Cleaning and Preprocessing

Before alignment, each data source is subjected to consistent **data cleaning and preprocessing**. Timestamps are **standardized in coordinated universal time (UTC)** so that spot, derivatives, options, and prediction markets refer to the same instant regardless of exchange or platform. Duplicate or obviously erroneous observations (e.g., zero or negative prices where inadmissible) are removed. **Missing observations** are flagged rather than imputed at this stage; the master panel construction step then applies explicit rules for alignment and fill so that missingness is transparent and contract-specific.

**Contract-level filtering** is applied to ensure liquidity and completeness. For options and prediction markets, contracts that fail minimum volume or activity thresholds are excluded. For prediction markets in particular, contracts with no or very few observed probability updates over the sample window are dropped so that the panel does not carry empty or uninformative series. These filters are applied before mapping onto the common grid, yielding a well-defined set of contracts that enter the master panel.

---

## 5-Minute Grid Construction and Alignment

A **unified 5-minute sampling frequency** is used for the entire analysis. This choice balances granularity—capturing intraday variation and lead-lag effects—with data availability and computational tractability. All series are mapped onto a single UTC 5-minute grid that runs from the start to the end of the sample window, so that every timestamp in the panel corresponds to a grid point (e.g., 00:00, 00:05, 00:10, …).

**Irregular market updates**, especially from prediction markets where trading is sparse, do not naturally fall on this grid. **Rounding and alignment rules** are applied: observation times are rounded to the nearest 5-minute boundary (e.g., by floor or standard rounding) so that each observation is assigned to a unique grid point. This ensures that spot, futures, options-implied measures, and prediction market probabilities can be compared at the same instants without ambiguity.

---

## Forward-Filling and Missing Data Handling

Because prediction market probabilities often update only when there is trading, the aligned probability series can contain many missing values at grid points where no trade occurred. **Forward-filling** is applied to obtain a continuous series for econometric use, with strict boundaries to preserve interpretability.

Fill is applied **only within the same contract or market**: the last observed probability for a given contract is carried forward to subsequent grid points until the next observation for that contract. **No cross-strike or cross-expiry contamination** is allowed—each contract (identified by market or contract identifier and, where relevant, strike and expiry) is treated as an independent series. This ensures that the filled value at any timestamp reflects only information that has been revealed for that specific contract, and not from a different strike or expiry. The **motivation** for this rule is the sparse, asynchronous nature of prediction market trading: without it, fills could incorrectly mix beliefs across different events. Pre-first-trade grid points (before the first observed update for a contract) are left as missing rather than filled, so that the series does not impute beliefs before the market has traded.

---

## Master Panel Creation

The **master panel** combines spot prices and returns, futures and funding data, options-implied measures where available, and prediction market probabilities (raw and forward-filled) into a single dataset. Each row corresponds either to a **timestamp** (in the wide format) or to a **timestamp–contract pair** (in the long format). The **long panel** stores one row per timestamp and per contract (e.g., per prediction market or per option contract), with columns for identifiers (market, strike, expiry), prices or probabilities, and any derived variables. The **wide panel** stores one row per timestamp, with separate columns for each contract’s probability or measure. Both formats are produced to support different econometric needs: the long format is suited to panel regressions and contract-level analysis; the wide format is suited to cross-sectional comparisons at a point in time.

Outputs are written as **reproducible parquet datasets** under a standard panel output folder. The principal deliverables of this stage include a long-format master panel (e.g., `panel/master_panel_long.parquet`), a wide-format master panel (e.g., `panel/master_panel_wide.parquet`), and a summary metadata file (e.g., `panel/step5_summary.json`) that records row counts, coverage, and consistency checks. Parquet is used for efficiency and to preserve types (e.g., datetime, integer identifiers) across runs.

---

## Reproducibility and Pipeline Structure

The processing and alignment pipeline is implemented in a **modular, step-based** structure. Each step (e.g., grid construction, alignment of a given source, forward-fill, panel assembly) is clearly separated so that inputs and outputs are well defined. **Output folder conventions** are fixed: all panel outputs reside under a window-specific directory (e.g., keyed by sample start and end dates), with a dedicated subfolder for the panel and its summary files. This allows the same pipeline to be run for different windows or updates without overwriting prior results.

**Sanity checks** are run at the end of the panel construction stage to ensure consistency: verification of timestamp coverage, absence of duplicate timestamp–contract pairs in the long panel, and basic validity of probabilities (e.g., within the unit interval). The summary file records the outcome of these checks. The final output of this stage is a **unified high-frequency dataset**—the master panel—that forms the empirical basis for the correlation, Granger causality, arbitrage, and event analyses described in the remainder of the methodology and in the results chapter.
