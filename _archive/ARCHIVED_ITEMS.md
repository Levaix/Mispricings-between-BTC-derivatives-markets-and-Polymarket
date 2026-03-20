# Archived items

This folder holds data and files that are **not required** for the current empirical analysis (sample window: **2026-01-01 to 2026-02-15** per Methodology Ch. 4 and `config.py`). Everything here can be restored by moving it back if needed.

---

## What was moved (automatically)

| Original location | Archive location | Reason |
|------------------|------------------|--------|
| `data/polymarket/cache/` | `_archive/data/polymarket_cache/` | API response cache; reproducible by re-running Step 4 (Polymarket download). Next Step 4 run will re-create the cache. |
| `data/spot/BTCUSDT/interval=5m/year=2025/` | `_archive/data/spot_year_2025/` | Spot data outside analysis window (Jan–Feb 2026). |
| `data/futures/usdm_perp/BTCUSDT/klines/year=2025/` | `_archive/data/futures_usdm_perp_klines_year_2025/` | Same. |
| `data/futures/usdm_perp/BTCUSDT/mark_klines/year=2025/` | `_archive/data/futures_usdm_perp_mark_klines_year_2025/` | Same. |
| `data/futures/usdm_perp/BTCUSDT/index_klines/year=2025/` | `_archive/data/futures_usdm_perp_index_klines_year_2025/` | Same. |
| `data/futures/usdm_perp/BTCUSDT/funding_rate/year=2025/` | `_archive/data/futures_usdm_perp_funding_rate_year_2025/` | Same. |
| `data/futures/continuous/BTCUSDT/contractType=*/year=2025/` | `_archive/data/futures_continuous_*_year_2025/` | Continuous futures 2025; not used in main pipeline for Jan–Feb 2026. |
| `data/options/` (entire folder) | `_archive/data/options_full_backfill/` | Full options backfill (many instruments, 2025-dated). Master panel uses **Jan–Feb 2026 options** from `data/options_jan_feb_2026/` and the liquid parquet in `outputs_jan_feb_2026/`. Config now points to `options_jan_feb_2026` for options paths. |

---

## Left in place (needed for current analysis)

- **`data/spot/`** — only `year=2026` (Jan–Feb) kept.
- **`data/futures/usdm_perp/`** — only `year=2026` kept.
- **`data/futures/continuous/`** — only `year=2026` kept (if present).
- **`data/options_jan_feb_2026/`** — kept; this is the **only** options data in the main tree now. Used for the liquid panel and by config (OPTIONS_KLINES_PATH etc.). The former `data/options/` is in the archive.
- **`data/polymarket/`** — `markets_selected.csv`, `probabilities_5m.parquet`, `manifest.csv`, `spreads_snapshot.csv` kept; only `cache/` was archived.
- **`data/_manifest/`** — kept (download progress).
- **`outputs/`** — all pipeline outputs (including options panels) go under **`outputs/window=2026-01-01_2026-02-15/`**. The former **`outputs_jan_feb_2026/`** folder is no longer used; if you had files there, move them into `outputs/window=2026-01-01_2026-02-15/` and you can delete the empty `outputs_jan_feb_2026` folder.

---

## Your input needed (optional)

1. **Polymarket cache**  
   If you re-run Step 4 (Polymarket download), the script will create a new `data/polymarket/cache/` and re-fetch from the API. If you prefer to avoid re-downloads, you can copy `_archive/data/polymarket_cache/` back to `data/polymarket/cache/`.

2. **2025 data**  
   If you need 2025 spot/futures again (e.g. for a different window or comparison), move the corresponding folders from `_archive/data/` back to their original paths under `data/`.

3. **Documentation in `data/`**  
   The files `data/DATA_OVERVIEW.md`, `data/DATA_SCHEMA.md`, `data/MASTER_PANEL_BUILD.md`, `data/OPTIONS_DATA_OVERVIEW.md` are reference docs, not inputs to the pipeline. They were **not** moved. If you want them in the archive, they can be moved to `_archive/docs/` (or similar).

4. **Other outputs**  
   If you have old or redundant outputs (e.g. other window folders under `outputs/`) that you want archived, list them and they can be moved into `_archive/outputs/`.

---

## Restoring something

To restore a moved folder, move it back from `_archive/data/` to the path in the “Original location” column above. Example (PowerShell):

```powershell
Move-Item -Path "_archive\data\polymarket_cache" -Destination "data\polymarket\cache" -Force
```

If you restore **options**: move `_archive/data/options_full_backfill` back to `data/options`. Config currently points to `data/options_jan_feb_2026` for the main analysis; if you need the full backfill again, you can point `config.OPTIONS_KLINES_PATH` (and related) back to `data/options`.
