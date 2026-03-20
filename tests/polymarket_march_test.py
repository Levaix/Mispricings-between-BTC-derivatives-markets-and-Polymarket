"""Ad-hoc Polymarket test pull for March 1–2, 2026 (UTC).

Goal
-----
- Re-fetch BTC daily ladder “Bitcoin above $K on [Month] [Day]?” markets
  with expiries in this window: 2026-03-01 to 2026-03-02 (inclusive).
- Download raw YES probabilities from the CLOB, with:
  - No probability filling (raw observations only)
  - Restrict to tradable window: market_open_time_utc <= t <= expiry_utc
  - 5-minute floor and per-bucket deduplication (keep last trade in bucket)
- Save results under ./tests/, *not* under Actual data/ or data/.

Outputs (under ./tests/):
- tests/polymarket_march_2026_probabilities_5m_long.parquet
- tests/polymarket_march_2026_probabilities_5m_long.csv
- tests/polymarket_march_2026_contracts_metadata.parquet
- tests/polymarket_march_2026_qa.json

This script does NOT use any on-disk cache; all data is fetched directly
from the live Gamma and CLOB APIs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# March test window (trading timestamps)
START_UTC = "2026-03-01 00:00:00"
END_UTC = "2026-03-02 23:59:59"
BAR_MS = 5 * 60 * 1000

# Discovery window (endDate filter for Gamma) – expiries on March 1–2, 2026
END_DATE_MIN = "2026-03-01T00:00:00Z"
END_DATE_MAX = "2026-03-02T23:59:59Z"
UNIVERSE_YEAR = 2026

MAX_RETRIES = 6
BACKOFF_BASE = 0.5
BACKOFF_CAP = 15.0

POLY_HISTORY_FIDELITY_SECONDS = 300  # 5 minutes
SLEEP_MS = 350


def _utc_to_ts(utc_str: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS' or ISO to unix seconds."""
    s = utc_str.strip().replace("Z", "+00:00")
    if "T" in s:
        dt = datetime.fromisoformat(s)
    else:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _ms_to_utc_iso(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """Simple retry wrapper without disk cache."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.request(method, url, timeout=30, **kwargs)
            if r.status_code in (429, 500, 502, 503):
                backoff = min(BACKOFF_BASE * (2**attempt), BACKOFF_CAP)
                logger.warning("HTTP %s for %s, backing off %.1fs", r.status_code, url, backoff)
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = min(BACKOFF_BASE * (2**attempt), BACKOFF_CAP)
            logger.warning("Request error %s for %s, backing off %.1fs", e, url, backoff)
            time.sleep(backoff)
    raise last_exc if last_exc is not None else RuntimeError(f"Request failed: {url}")


# ---------- Text parsing helpers (copied from main script) ----------

MARKET_TEXT_KEYS = ["question", "title", "name", "marketQuestion", "description"]


def get_market_text(m: dict) -> str:
    for k in MARKET_TEXT_KEYS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


PAT_CANDIDATE_LADDER = re.compile(
    r"bitcoin.*above\b.*\bon\b.*\b("
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|"
    r"sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\s+\d{1,2}",
    re.IGNORECASE,
)
PAT_STRIKE = re.compile(r"above\s+\$?([\d,]+)\s*([kK])?\b", re.IGNORECASE)
PAT_STRIKE_DOLLAR = re.compile(r"\$([\d,]+)\s*([kK])?\b", re.IGNORECASE)
STRIKE_MIN_PLAUSIBLE = 10_000
STRIKE_MAX_PLAUSIBLE = 250_000
MONTH_PAT = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)"
PAT_DATE_AFTER_ON = re.compile(
    r"\b" + MONTH_PAT + r"\s+(\d{1,2})(?:\s*[,.]?\s*(\d{4}))?\b",
    re.IGNORECASE,
)
MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_date_from_text(text: str, default_year: int = UNIVERSE_YEAR) -> Tuple[str | None, datetime | None]:
    if not text or not text.strip():
        return None, None
    text = text.strip()
    for m in PAT_DATE_AFTER_ON.finditer(text):
        g = m.groups()
        month_str = None
        day_str = None
        year_val = default_year
        for x in g:
            if x is None:
                continue
            x = (x or "").strip().rstrip(".,")
            if not x:
                continue
            if len(x) == 4 and x.isdigit():
                year_val = int(x)
            elif len(x) <= 2 and x.isdigit() and 1 <= int(x) <= 31:
                day_str = x
            elif x.lower() in MONTH_NAMES:
                month_str = x
        if not month_str:
            full = m.group(0)
            for name in MONTH_NAMES:
                if name in full.lower():
                    month_str = name
                    break
        if not day_str:
            for x in g:
                if x and isinstance(x, str) and x.strip().isdigit():
                    d = int(x.strip())
                    if 1 <= d <= 31:
                        day_str = x.strip()
                        break
        if not month_str or not day_str:
            continue
        month = MONTH_NAMES.get(month_str.lower())
        if month is None:
            continue
        try:
            day = int(
                day_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
            )
            if not (1 <= day <= 31):
                continue
            dt = datetime(year_val, month, day, tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d"), dt
        except ValueError:
            continue
    return None, None


def _parse_bitcoin_above_on_date(question: str, year: int = UNIVERSE_YEAR) -> Tuple[int | None, str | None]:
    if not question or not question.strip():
        return None, None
    q = question.strip()
    if not PAT_CANDIDATE_LADDER.search(q):
        return None, None
    strike = None
    strike_m = PAT_STRIKE.search(q)
    if strike_m:
        try:
            strike_str = strike_m.group(1).replace(",", "").strip()
            if strike_str:
                strike = int(strike_str)
                if strike_m.lastindex >= 2 and strike_m.group(2) and strike_m.group(2).lower() == "k":
                    strike *= 1000
        except (ValueError, IndexError):
            strike = None
    if strike is None or not (STRIKE_MIN_PLAUSIBLE <= strike <= STRIKE_MAX_PLAUSIBLE):
        strike_m = PAT_STRIKE_DOLLAR.search(q)
        if strike_m:
            try:
                strike_str = strike_m.group(1).replace(",", "").strip()
                if strike_str:
                    strike = int(strike_str)
                    if strike_m.lastindex >= 2 and strike_m.group(2) and strike_m.group(2).lower() == "k":
                        strike *= 1000
            except (ValueError, IndexError):
                strike = None
    if strike is None or not (STRIKE_MIN_PLAUSIBLE <= strike <= STRIKE_MAX_PLAUSIBLE):
        return None, None
    date_str, _ = _parse_date_from_text(q, year)
    return (strike, date_str) if date_str else (None, None)


def _question_matches_daily_ladder(question: str) -> bool:
    if not question or not question.strip():
        return False
    return PAT_CANDIDATE_LADDER.search(question.strip()) is not None


def _parse_end_date(market: dict) -> Tuple[str | None, int | None]:
    end_date = market.get("endDate") or market.get("end_date") or market.get("resolutionDate")
    if end_date is None:
        return None, None
    if isinstance(end_date, (int, float)):
        if end_date > 1e12:
            end_date = end_date / 1000.0
        dt = datetime.fromtimestamp(end_date, tz=timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        except Exception:
            return None, None
    date_str = dt.strftime("%Y-%m-%d")
    expiry_ms = int(dt.timestamp() * 1000)
    return date_str, expiry_ms


def _parse_start_date(market: dict) -> Tuple[str | None, int | None]:
    start_raw = market.get("startDate") or market.get("start_date") or market.get("createdAt")
    if start_raw is None:
        return None, None
    if isinstance(start_raw, (int, float)):
        if start_raw > 1e12:
            start_raw = start_raw / 1000.0
        dt = datetime.fromtimestamp(start_raw, tz=timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
        except Exception:
            return None, None
    open_ms = int(dt.timestamp() * 1000)
    return _ms_to_utc_iso(open_ms), open_ms


def _parse_clob_token_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            out = json.loads(s)
            return [str(x).strip() for x in out] if isinstance(out, list) else []
        except Exception:
            return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _map_yes_no_tokens(market: dict) -> Tuple[str, str, str, str]:
    ids = _parse_clob_token_ids(market.get("clobTokenIds"))
    yes_id, no_id = "", ""
    method, confidence = "unknown", "unknown"
    outcomes_raw = market.get("outcomes") or market.get("outcomePrices")
    outcomes = None
    if isinstance(outcomes_raw, dict):
        outcomes = outcomes_raw
    elif isinstance(outcomes_raw, str) and outcomes_raw.strip().startswith("["):
        try:
            outcomes = json.loads(outcomes_raw)
        except Exception:
            outcomes = None
    elif isinstance(outcomes_raw, list):
        outcomes = outcomes_raw
    if isinstance(outcomes, dict):
        yes_id = str(outcomes.get("Yes") or outcomes.get("yes") or "").strip()
        no_id = str(outcomes.get("No") or outcomes.get("no") or "").strip()
        if yes_id or no_id:
            method = confidence = "gamma_outcomes"
    if not yes_id and isinstance(outcomes, list) and len(ids) >= len(outcomes):
        for i, o in enumerate(outcomes):
            if i >= len(ids):
                break
            lab = (
                (o.get("outcome") or o.get("title") or str(o) or "").strip().lower()
                if isinstance(o, dict)
                else (str(o) or "").strip().lower()
            )
            if lab == "yes":
                yes_id = ids[i]
            elif lab == "no":
                no_id = ids[i]
        if yes_id or no_id:
            method = confidence = "gamma_outcomes"
    tokens = market.get("tokens") or []
    if (not yes_id or not no_id) and isinstance(tokens, list):
        for t in tokens:
            if isinstance(t, dict):
                lab = (t.get("outcome") or t.get("title") or "").lower()
                tid = (
                    t.get("token_id") or t.get("tokenID") or t.get("clobTokenId") or ""
                ).strip()
                if lab == "yes" and tid:
                    yes_id = str(tid)
                elif lab == "no" and tid:
                    no_id = str(tid)
        if yes_id or no_id:
            method = confidence = "gamma_tokens"
    if not yes_id and len(ids) >= 1:
        yes_id = ids[0]
        if len(ids) >= 2:
            no_id = ids[1]
        method, confidence = "fallback_unknown", "unknown"
    return yes_id, no_id, method, confidence


def discover_markets(session: requests.Session) -> pd.DataFrame:
    """Discover BTC daily ladder markets with expiries on March 1–2, 2026."""
    all_markets: List[dict] = []
    for closed in (True, False):
        offset = 0
        while True:
            url = (
                f"{GAMMA_BASE}/markets?end_date_min={requests.utils.quote(END_DATE_MIN)}"
                f"&end_date_max={requests.utils.quote(END_DATE_MAX)}"
                f"&closed={'true' if closed else 'false'}&limit=500&offset={offset}"
            )
            r = _request(session, "GET", url)
            page_list = r.json()
            if not isinstance(page_list, list) or not page_list:
                break
            all_markets.extend(page_list)
            if len(page_list) < 500:
                break
            offset += 500
    # Deduplicate by id/conditionId
    by_id: Dict[str, dict] = {}
    for m in all_markets:
        kid = m.get("id") or m.get("conditionId")
        if kid and kid not in by_id:
            by_id[str(kid)] = m
    all_markets = list(by_id.values())
    logger.info("Fetched %d unique markets from Gamma", len(all_markets))

    rows: List[Dict[str, Any]] = []
    for m in all_markets:
        question = get_market_text(m)
        if "bitcoin" not in question.lower() or "above" not in question.lower():
            continue
        if not _question_matches_daily_ladder(question):
            continue
        strike, parsed_date = _parse_bitcoin_above_on_date(question)
        if strike is None or parsed_date is None:
            continue
        # Only keep expiries on March 1–2 (just in case)
        date_str, expiry_ms = _parse_end_date(m)
        if date_str is None or expiry_ms is None:
            continue
        if not ("2026-03-01" <= date_str <= "2026-03-02"):
            continue
        yes_id, no_id, map_method, map_confidence = _map_yes_no_tokens(m)
        if not yes_id:
            continue
        open_utc_str, open_ms = _parse_start_date(m)
        if open_utc_str is None or open_ms is None:
            # Fallback: start of thesis window
            open_ms = _utc_to_ts(START_UTC) * 1000
            open_utc_str = _ms_to_utc_iso(open_ms)
        market_id = str(m.get("id") or m.get("conditionId") or "")
        event_id = f"{parsed_date.replace('-', '')}_{strike}"
        slug = (m.get("slug") or "").strip()
        rows.append(
            {
                "event_id": event_id,
                "market_id": market_id,
                "question": question[:500],
                "parsed_date": parsed_date,
                "strike_K": strike,
                "expiry_ms": expiry_ms,
                "expiry_utc": _ms_to_utc_iso(expiry_ms),
                "open_ms": open_ms,
                "market_open_time_utc": open_utc_str,
                "slug": slug,
                "token_yes_id": yes_id,
                "token_no_id": no_id or "",
                "mapping_method": map_method,
                "mapping_confidence": map_confidence,
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Selected %d BTC ladder markets for March 1–2", len(df))
    return df


def fetch_prices_history(
    session: requests.Session,
    token_id: str,
    start_ts: int,
    end_ts: int,
) -> List[Dict[str, Any]]:
    """Fetch CLOB prices-history for a YES token, no caching."""
    token_id = str(token_id).strip()
    url = (
        f"{CLOB_BASE}/prices-history?market={requests.utils.quote(token_id, safe='')}"
        f"&startTs={start_ts}&endTs={end_ts}&fidelity={POLY_HISTORY_FIDELITY_SECONDS//60}"
    )
    time.sleep(SLEEP_MS / 1000.0)
    r = _request(session, "GET", url)
    data = r.json()
    hist = data if isinstance(data, list) else data.get("history", [])
    return hist if isinstance(hist, list) else []


def download_probabilities_for_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """Download raw YES probabilities for all markets in March window."""
    if markets.empty:
        return pd.DataFrame()
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "PolymarketMarchTest/1.0"})

    start_ts = _utc_to_ts(START_UTC)
    end_ts = _utc_to_ts(END_UTC)
    rows: List[Dict[str, Any]] = []
    for _, row in markets.iterrows():
        event_id = row["event_id"]
        strike_k = row["strike_K"]
        expiry_ms = row["expiry_ms"]
        parsed_date = row["parsed_date"]
        token_id = str(row["token_yes_id"]).strip()
        market_id = row["market_id"]
        mapping_confidence = row["mapping_confidence"]
        hist = fetch_prices_history(session, token_id, start_ts, end_ts)
        if not hist:
            logger.info(
                "no_trades_or_no_history event_id=%s market_id=%s token_yes_id=%s",
                event_id,
                market_id,
                token_id[:20] + "..." if len(token_id) > 20 else token_id,
            )
            continue
        for h in hist:
            t = h.get("t") or h.get("timestamp")
            p = h.get("p") or h.get("price")
            if t is None or p is None:
                continue
            t = int(t)
            open_time = t * 1000 if t < 1_000_000_000_000 else t
            rows.append(
                {
                    "open_time": open_time,
                    "event_id": event_id,
                    "strike_K": strike_k,
                    "expiry_ms": expiry_ms,
                    "yes_token_id": token_id,
                    "probability_pm": float(p),
                    "date": parsed_date,
                    "market_id": market_id,
                    "mapping_confidence": mapping_confidence,
                }
            )
    return pd.DataFrame(rows)


def build_long_panel(markets: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """Apply tradable-window filter, 5m floor, and build long-format panel."""
    if markets.empty or probs.empty:
        return pd.DataFrame()

    # Join to get open_ms and expiry_ms per contract
    meta_cols = [
        "event_id",
        "open_ms",
        "expiry_ms",
        "market_id",
        "market_open_time_utc",
        "expiry_utc",
        "parsed_date",
        "strike_K",
        "token_yes_id",
        "mapping_confidence",
        "slug",
    ]
    meta = markets[meta_cols].drop_duplicates("event_id")
    raw = probs.merge(meta, on="event_id", how="left")

    n_before = len(raw)
    raw = raw[(raw["open_time"] >= raw["open_ms"]) & (raw["open_time"] <= raw["expiry_ms"])]
    n_after = len(raw)
    logger.info(
        "Tradable window filter: %d -> %d rows (dropped %d outside open/expiry)",
        n_before,
        n_after,
        n_before - n_after,
    )
    if raw.empty:
        return pd.DataFrame()

    # Floor to 5m and dedupe
    raw["bucket_5m_ms"] = (raw["open_time"] // BAR_MS) * BAR_MS
    raw = raw.sort_values("open_time").drop_duplicates(
        subset=["bucket_5m_ms", "event_id"], keep="last"
    )
    raw["timestamp_utc"] = raw["bucket_5m_ms"].astype("int64").map(_ms_to_utc_iso)

    long_df = pd.DataFrame(
        {
            "timestamp_utc": raw["timestamp_utc"],
            "open_time_ms": raw["bucket_5m_ms"],
            "contract_id": raw["event_id"],
            "market_id": raw["market_id"],
            "expiry_date": raw["parsed_date"],
            "strike": raw["strike_K"],
            "prob_yes": raw["probability_pm"],
            "market_open_time_utc": raw["market_open_time_utc"],
            "expiry_utc": raw["expiry_utc"],
            "token_yes_id": raw["token_yes_id"],
            "mapping_confidence": raw["mapping_confidence"],
            "slug": raw["slug"],
        }
    )
    return long_df


def run() -> None:
    out_dir = ROOT / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover markets
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "PolymarketMarchTest/1.0"})
    markets = discover_markets(session)
    if markets.empty:
        logger.warning("No March 1–2 BTC ladder markets discovered.")
        return

    # 2) Download probabilities (raw)
    probs = download_probabilities_for_markets(markets)
    if probs.empty:
        logger.warning("No probabilities history for discovered markets in March 1–2 window.")

    # 3) Build long panel
    long_df = build_long_panel(markets, probs)

    # 4) Save outputs
    long_path = out_dir / "polymarket_march_2026_probabilities_5m_long.parquet"
    long_df.to_parquet(long_path, index=False)
    logger.info("Wrote %s (%d rows)", long_path, len(long_df))

    long_csv = out_dir / "polymarket_march_2026_probabilities_5m_long.csv"
    long_df.to_csv(long_csv, index=False)
    logger.info("Wrote %s", long_csv)

    meta_out = markets[
        [
            "event_id",
            "question",
            "parsed_date",
            "strike_K",
            "market_id",
            "market_open_time_utc",
            "expiry_utc",
            "open_ms",
            "expiry_ms",
            "token_yes_id",
            "slug",
            "mapping_confidence",
        ]
    ].copy()
    meta_out = meta_out.rename(
        columns={
            "event_id": "contract_id",
            "question": "title",
            "parsed_date": "expiry_date",
            "strike_K": "strike",
        }
    )
    meta_path = out_dir / "polymarket_march_2026_contracts_metadata.parquet"
    meta_out.to_parquet(meta_path, index=False)
    logger.info("Wrote %s", meta_path)

    # 5) Simple QA JSON
    qa = {
        "window_start_utc": START_UTC,
        "window_end_utc": END_UTC,
        "n_contracts": int(len(markets)),
        "n_rows_long": int(len(long_df)),
        "expiry_date_min": meta_out["expiry_date"].min() if not meta_out.empty else None,
        "expiry_date_max": meta_out["expiry_date"].max() if not meta_out.empty else None,
        "prob_bounds_violations": int(
            (((long_df["prob_yes"] < 0) | (long_df["prob_yes"] > 1)).sum())
        )
        if not long_df.empty
        else 0,
    }
    qa_path = out_dir / "polymarket_march_2026_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    logger.info("Wrote %s", qa_path)


if __name__ == "__main__":
    run()

