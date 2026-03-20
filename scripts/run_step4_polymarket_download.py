#!/usr/bin/env python3
"""
Step 4: Polymarket ingestion — daily ladder "Bitcoin above ___ on [Month] [Day]?" only.

Universe: Bitcoin above ___ on [Month] [Day] daily ladder, 2026-01-01 to 2026-02-15 (inclusive).
Only markets matching this exact naming pattern are selected; no broad BTC keyword filtering.

Outputs:
  - data/polymarket/markets_selected.csv   selected markets + token ids + mapping_confidence
  - data/polymarket/manifest.csv           manifest of all processed markets and status (included/excluded + reason)
  - data/polymarket/probabilities_5m.parquet  raw history rows
  - data/polymarket/spreads_snapshot.csv   spread snapshot
  - outputs/.../polymarket/polymarket_panel_5m_aligned.parquet  panel-ready long table (timestamp_utc, date, strike, prob_yes, ...)

Run: python scripts/run_step4_polymarket_download.py --window_dir "outputs/window=2026-01-01_2026-02-15" [--overwrite]

Uses: Gamma API (list-markets), CLOB (prices-history, spreads). Rate-limited with disk cache.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

# Window (UTC) — unified thesis timeframe
START_UTC = "2026-01-01 00:00:00"
END_UTC = "2026-02-28 23:59:59"
BAR_MS = 5 * 60 * 1000
# Universe: daily ladder dates (Jan–Feb 2026)
UNIVERSE_DATE_MIN = "2026-01-01"
UNIVERSE_DATE_MAX = "2026-02-28"
UNIVERSE_YEAR = 2026
# If True: skip markets where YES token cannot be confidently mapped. If False: allow heuristic, set mapping_confidence.
STRICT_YES_MAPPING = True

# API
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
GAMMA_LIST_LIMIT = 500
GAMMA_SLEEP_MS = 250
SLEEP_MS = 350
MAX_RETRIES = 8
BACKOFF_BASE = 0.5
BACKOFF_CAP = 30
SPREAD_BATCH_SIZE = 50
# Discovery window (endDate filter for Gamma)
END_DATE_MIN = "2026-01-01T00:00:00Z"
END_DATE_MAX = "2026-02-28T23:59:59Z"
VOLUME_NUM_MIN = 0
# CLOB prices-history
POLY_HISTORY_FIDELITY_SECONDS = 300
POLY_HISTORY_FIDELITY_MINUTES = POLY_HISTORY_FIDELITY_SECONDS // 60
POLY_HISTORY_CHUNK_DAYS = 14
POLY_HISTORY_CHUNK_SECONDS = POLY_HISTORY_CHUNK_DAYS * 86400


def _utc_to_ts(utc_str: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS' or ISO to unix seconds."""
    s = utc_str.strip().replace("Z", "+00:00")
    if "T" in s:
        dt = datetime.fromisoformat(s)
    else:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _ts_to_ms(ts: int) -> int:
    return int(ts) * 1000


def _ms_to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.request(method, url, timeout=30, **kwargs)
            if r.status_code in (429, 500, 502, 503):
                backoff = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
            time.sleep(backoff)
    return r


def _load_cache(cache_path: Path):
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(cache_path: Path, data) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# Gamma market text: check these keys in order (varies by endpoint/version)
MARKET_TEXT_KEYS = ["question", "title", "name", "marketQuestion", "description"]


def get_market_text(m: dict) -> str:
    """Return the best string field for filtering/parsing. Use only this for all text-based logic."""
    for k in MARKET_TEXT_KEYS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# Permissive candidate: must contain "bitcoin" + "above" (with optional text between, e.g. "be"), " on ", and month+day
# Gamma uses "Will the price of Bitcoin be above $78,000 on January 1?"
PAT_CANDIDATE_LADDER = re.compile(
    r"bitcoin.*above\b.*\bon\b.*\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\s+\d{1,2}",
    re.IGNORECASE,
)
# Strike: number (with $ and commas, optional k suffix) after "above"
PAT_STRIKE = re.compile(r"above\s+\$?([\d,]+)\s*([kK])?\b", re.IGNORECASE)
# Fallback: $ then digits (e.g. "$78,000") in case above pattern matches wrong substring
PAT_STRIKE_DOLLAR = re.compile(r"\$([\d,]+)\s*([kK])?\b", re.IGNORECASE)
# Plausible BTC strike range (reject "1" from "January 1")
STRIKE_MIN_PLAUSIBLE = 10_000
STRIKE_MAX_PLAUSIBLE = 250_000
# Date: (Month|Month_abbrev) (DD) (optional ,? YYYY). Match month name then day then optional year.
MONTH_PAT = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)"
PAT_DATE_AFTER_ON = re.compile(
    r"\b" + MONTH_PAT + r"\s+(\d{1,2})(?:\s*[,.]?\s*(\d{4}))?\b",
    re.IGNORECASE,
)
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date_from_text(text: str, default_year: int = UNIVERSE_YEAR) -> tuple:
    """Parse date from substring like 'February 19', 'Feb 19', 'February 19, 2026', 'Feb 19 2026'.
    Returns (date_YYYY-MM-DD, None) or (None, None) on failure."""
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
            day = int(day_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""))
            if not (1 <= day <= 31):
                continue
            dt = datetime(year_val, month, day, tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d"), dt
        except ValueError:
            continue
    return None, None


def _parse_bitcoin_above_on_date(question: str, year: int = UNIVERSE_YEAR) -> tuple:
    """Parse ladder question -> (strike_int, date_YYYY-MM-DD). Supports $56,000 / 56k / Feb 19 / February 19, 2026."""
    if not question or not question.strip():
        return None, None
    q = question.strip()
    if not PAT_CANDIDATE_LADDER.search(q):
        return None, None
    strike_m = PAT_STRIKE.search(q)
    if not strike_m:
        return None, None
    strike = None
    try:
        strike_str = strike_m.group(1).replace(",", "").strip()
        if strike_str:
            strike = int(strike_str)
            if strike_m.lastindex >= 2 and strike_m.group(2) and strike_m.group(2).lower() == "k":
                strike *= 1000
    except (ValueError, IndexError):
        pass
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
    """True if text looks like ladder: starts with 'Bitcoin above', contains ' on ', and has a month name."""
    if not question or not question.strip():
        return False
    return PAT_CANDIDATE_LADDER.search(question.strip()) is not None


def _test_parse_bitcoin_above() -> None:
    """Inline test: parse strike and date from sample questions."""
    tests = [
        ("Bitcoin above $56,000 on February 19?", 56000, "2026-02-19"),
        ("Bitcoin above 56000 on February 19?", 56000, "2026-02-19"),
        ("Bitcoin above $70,000 on January 1?", 70000, "2026-01-01"),
        ("Bitcoin above 60000 on Dec 31?", 60000, "2026-12-31"),
    ]
    for q, want_strike, want_date in tests:
        strike, date_str = _parse_bitcoin_above_on_date(q)
        assert strike == want_strike and date_str == want_date, "parse %r -> (%s, %s) want (%s, %s)" % (q, strike, date_str, want_strike, want_date)
        assert _question_matches_daily_ladder(q), "match: %r" % q
    assert _parse_bitcoin_above_on_date("Bitcoin above 100k on March 1?") == (100000, "2026-03-01")
    assert not _question_matches_daily_ladder("Will Bitcoin hit $100k?")
    assert _question_matches_daily_ladder("Bitcoin above $56,000 on February 19")
    print("_test_parse_bitcoin_above passed")


def _parse_end_date(market: dict) -> tuple:
    """Get expiry date YYYY-MM-DD and expiry_ms from market endDate. Returns (date_str, expiry_ms) or (None, None)."""
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


def _parse_start_date(market: dict) -> tuple:
    """Get market open time (when tradable) from startDate or createdAt. Returns (open_time_utc_iso, open_ms) or (None, None)."""
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


def _parse_clob_token_ids(raw) -> list:
    """Parse clobTokenIds: JSON string '["id1","id2"]' or comma-separated. Returns list of token id strings."""
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
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def _map_yes_no_tokens(market: dict) -> tuple:
    """Return (token_yes_id, token_no_id, mapping_method, confidence).
    Gamma often returns outcomes/outcomePrices as JSON strings e.g. '[\"Yes\", \"No\"]'; map by position to clobTokenIds."""
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
            pass
    elif isinstance(outcomes_raw, list):
        outcomes = outcomes_raw
    if isinstance(outcomes, dict):
        yes_id = str(outcomes.get("Yes") or outcomes.get("yes") or "").strip()
        no_id = str(outcomes.get("No") or outcomes.get("no") or "").strip()
        if yes_id or no_id:
            method, confidence = "gamma_outcomes", "gamma_outcomes"
    if not yes_id and isinstance(outcomes, list) and len(ids) >= len(outcomes):
        for i, o in enumerate(outcomes):
            if i >= len(ids):
                break
            lab = (o.get("outcome") or o.get("title") or str(o) or "").strip().lower() if isinstance(o, dict) else (str(o) or "").strip().lower()
            if lab == "yes":
                yes_id = ids[i]
            elif lab == "no":
                no_id = ids[i]
        if yes_id or no_id:
            method, confidence = "gamma_outcomes", "gamma_outcomes"
    tokens = market.get("tokens") or []
    if (not yes_id or not no_id) and isinstance(tokens, list):
        for t in tokens:
            if isinstance(t, dict):
                lab = (t.get("outcome") or t.get("title") or "").lower()
                tid = (t.get("token_id") or t.get("tokenID") or t.get("clobTokenId") or "").strip()
                if lab == "yes" and tid:
                    yes_id = str(tid)
                elif lab == "no" and tid:
                    no_id = str(tid)
        if yes_id or no_id:
            method, confidence = "gamma_tokens", "gamma_tokens"
    if not yes_id and len(ids) >= 1:
        yes_id = ids[0]
        if len(ids) >= 2:
            no_id = ids[1]
        method, confidence = "fallback_unknown", "unknown"
    return yes_id, no_id, method, confidence


def _gamma_list_markets(session: requests.Session, cache_dir: Path, end_date_min: str, end_date_max: str, closed: bool, volume_num_min: int = 0) -> list:
    """List markets via Gamma GET /markets with end_date_min/max, closed, limit/offset. Cache each page."""
    all_markets = []
    offset = 0
    closed_str = "true" if closed else "false"
    while True:
        cache_path = cache_dir / ("gamma_markets_enddate_window_closed=%s_offset=%d.json" % (closed_str, offset))
        data = _load_cache(cache_path)
        if data is not None:
            page_list = data if isinstance(data, list) else []
        else:
            time.sleep(GAMMA_SLEEP_MS / 1000.0)
            url = "%s/markets?end_date_min=%s&end_date_max=%s&closed=%s&limit=%d&offset=%d" % (
                GAMMA_BASE, requests.utils.quote(end_date_min), requests.utils.quote(end_date_max), "true" if closed else "false", GAMMA_LIST_LIMIT, offset,
            )
            if volume_num_min > 0:
                url += "&volume_num_min=%d" % volume_num_min
            r = _request(session, "GET", url)
            page_list = r.json()
            if not isinstance(page_list, list):
                page_list = []
            _save_cache(cache_path, page_list)
        if not page_list:
            break
        all_markets.extend(page_list)
        if len(page_list) < GAMMA_LIST_LIMIT:
            break
        offset += GAMMA_LIST_LIMIT
    return all_markets


def _format_end_date(m: dict) -> str:
    """Return endDate as string for debug output."""
    ed = m.get("endDate") or m.get("end_date") or m.get("resolutionDate")
    if ed is None:
        return ""
    if isinstance(ed, (int, float)):
        if ed > 1e12:
            ed = ed / 1000.0
        return datetime.fromtimestamp(ed, tz=timezone.utc).strftime("%Y-%m-%d")
    return str(ed)[:10]


def discover_and_select_markets(session: requests.Session, cache_dir: Path) -> tuple:
    """Discover via Gamma List markets; select ONLY 'Bitcoin above ___ on [Month] [Day]?' in date range.
    Returns (markets_df, manifest_df, exclusion_counts_dict, debug_info)."""
    all_markets = _gamma_list_markets(session, cache_dir, END_DATE_MIN, END_DATE_MAX, closed=True, volume_num_min=VOLUME_NUM_MIN)
    markets_closed_false = _gamma_list_markets(session, cache_dir, END_DATE_MIN, END_DATE_MAX, closed=False, volume_num_min=VOLUME_NUM_MIN)
    by_id = {m.get("id") or m.get("conditionId"): m for m in markets_closed_false}
    for m in all_markets:
        kid = m.get("id") or m.get("conditionId")
        if kid and kid not in by_id:
            by_id[kid] = m
    all_markets = list(by_id.values())
    total_fetched = len(all_markets)

    # ----- DISCOVERY DEBUG MODE: before strict ladder filter -----
    # Collect any market whose text contains "bitcoin above" (case-insensitive) in any plausible field
    candidates_bitcoin_above = []
    for m in all_markets:
        text = get_market_text(m)
        if "bitcoin" in text.lower() and "above" in text.lower():
            candidates_bitcoin_above.append((m, text))
    count_contains_bitcoin_above = len(candidates_bitcoin_above)
    print("\n" + "=" * 60)
    print("DISCOVERY DEBUG MODE")
    print("=" * 60)
    print("total_fetched: %d" % total_fetched)
    print("count_contains_bitcoin_above: %d" % count_contains_bitcoin_above)
    for i, (m, text) in enumerate(candidates_bitcoin_above[:30]):
        mid = str(m.get("id") or m.get("conditionId") or "")
        end_d = _format_end_date(m)
        print("  %2d: market_id=%s endDate=%s text=%s" % (i + 1, mid, end_d, (text[:90] + "..." if len(text) > 90 else text)))
    if candidates_bitcoin_above:
        ex = candidates_bitcoin_above[0][0]
        print("Example market keys (first candidate): %s" % list(ex.keys())[:25])
    if len(candidates_bitcoin_above) > 1:
        ex2 = candidates_bitcoin_above[1][0]
        print("Example market keys (second): %s" % list(ex2.keys())[:25])
    print("=" * 60 + "\n")

    exclusion = {"no_clob_tokens": 0, "parse_fail": 0, "date_out_of_range": 0, "mapping_unknown": 0, "duplicate": 0}
    candidate_count = 0
    manifest_rows = []
    rows = []
    seen = set()  # (parsed_date, strike)

    for m in all_markets:
        question = get_market_text(m)
        if "bitcoin" not in question.lower() or "above" not in question.lower():
            continue
        candidate_count += 1
        market_id = str(m.get("id") or m.get("conditionId") or "")
        if not _question_matches_daily_ladder(question):
            continue
        token_ids = _parse_clob_token_ids(m.get("clobTokenIds"))
        if not token_ids:
            exclusion["no_clob_tokens"] += 1
            manifest_rows.append({"market_id": market_id, "question": question[:200], "status": "excluded", "reason": "no_clob_tokens"})
            continue
        strike, parsed_date = _parse_bitcoin_above_on_date(question)
        if strike is None or parsed_date is None:
            exclusion["parse_fail"] += 1
            manifest_rows.append({"market_id": market_id, "question": question[:200], "status": "excluded", "reason": "parse_fail"})
            print("WARNING: parse_fail market_id=%s question=%s" % (market_id, question[:80]))
            continue
        if not (UNIVERSE_DATE_MIN <= parsed_date <= UNIVERSE_DATE_MAX):
            exclusion["date_out_of_range"] += 1
            manifest_rows.append({"market_id": market_id, "question": question[:200], "parsed_date": parsed_date, "status": "excluded", "reason": "date_out_of_range"})
            continue
        yes_id, no_id, map_method, map_confidence = _map_yes_no_tokens(m)
        if STRICT_YES_MAPPING and map_confidence == "unknown":
            exclusion["mapping_unknown"] += 1
            manifest_rows.append({"market_id": market_id, "question": question[:200], "parsed_date": parsed_date, "strike": strike, "status": "excluded", "reason": "mapping_unknown"})
            continue
        if not yes_id:
            exclusion["mapping_unknown"] += 1
            manifest_rows.append({"market_id": market_id, "question": question[:200], "status": "excluded", "reason": "mapping_unknown"})
            continue
        key = (parsed_date, strike)
        if key in seen:
            exclusion["duplicate"] += 1
            continue
        seen.add(key)
        _, expiry_ms = _parse_end_date(m)
        if expiry_ms is None:
            expiry_ms = int(datetime.strptime(parsed_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        open_utc_str, open_ms = _parse_start_date(m)
        if open_utc_str is None or open_ms is None:
            open_ms = int(datetime.strptime(UNIVERSE_DATE_MIN, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            open_utc_str = _ms_to_utc_iso(open_ms)
        event_id = "%s_%d" % (parsed_date.replace("-", ""), strike)
        slug = (m.get("slug") or m.get("conditionId") or "").strip() or ""
        rows.append({
            "event_id": event_id,
            "market_id": market_id,
            "question": question[:500],
            "parsed_date": parsed_date,
            "strike": strike,
            "strike_K": strike,
            "expiry_ms": expiry_ms,
            "expiry_utc": _ms_to_utc_iso(expiry_ms),
            "open_ms": open_ms,
            "market_open_time_utc": open_utc_str,
            "slug": slug,
            "token_yes_id": yes_id,
            "token_no_id": no_id or "",
            "fallback_token_ids": json.dumps(token_ids),
            "mapping_method": map_method,
            "mapping_confidence": map_confidence,
            "yes_token_id": yes_id,
            "volume": m.get("volume") or m.get("volumeNum") or "",
            "liquidity": m.get("liquidity") or m.get("liquidityNum") or "",
        })
        manifest_rows.append({"market_id": market_id, "question": question[:200], "parsed_date": parsed_date, "strike": strike, "status": "included", "reason": ""})

    df = pd.DataFrame(rows)
    manifest_df = pd.DataFrame(manifest_rows)

    # Group by parsed_date: unique strikes, sort ascending
    if not df.empty:
        by_date = df.groupby("parsed_date").agg({"strike": ["count", "min", "max"]}).reset_index()
        by_date.columns = ["parsed_date", "n_strikes", "strike_min", "strike_max"]
        n_dates = len(by_date)
        avg_strikes = by_date["n_strikes"].mean()
        print("\n--- Ladder summary ---")
        print("number of dates: %d" % n_dates)
        print("avg strikes per date: %.1f" % avg_strikes)
        print("min strike across all: %s  max strike: %s" % (by_date["strike_min"].min(), by_date["strike_max"].max()))
        print("final selected count: %d" % len(df))
    else:
        print("final selected count: 0")

    print("candidate_count (bitcoin above): %d" % candidate_count)
    print("exclusion counts (within candidates): %s" % exclusion)
    debug_info = {
        "total_fetched": total_fetched,
        "count_contains_bitcoin_above": count_contains_bitcoin_above,
        "candidate_count": candidate_count,
        "exclusion": exclusion,
        "final_count": len(df),
    }
    return df, manifest_df, exclusion, debug_info


def _prices_history_request(session: requests.Session, token_id: str, start_ts: int, end_ts: int) -> tuple:
    """Single request to CLOB prices-history (no cache, no retry). Returns (status_code, response_body_dict_or_none, url)."""
    token_id = str(token_id).strip()
    fidelity = POLY_HISTORY_FIDELITY_MINUTES
    url = "%s/prices-history?market=%s&startTs=%d&endTs=%d&fidelity=%d" % (
        CLOB_BASE, requests.utils.quote(token_id, safe=""), start_ts, end_ts, fidelity,
    )
    time.sleep(SLEEP_MS / 1000.0)
    r = session.get(url, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:500]}
    return r.status_code, body, url


def _run_prices_history_smoke_test(session: requests.Session, markets: pd.DataFrame, end_ts: int) -> None:
    """Before bulk download: try first market's YES (and NO) token for last 2 days. Print status, first 5 points, min/max.
    Empty history is logged as no_trades_or_no_history but does not fail the test."""
    if markets.empty:
        return
    row = markets.iloc[0]
    yes_token = str(row.get("token_yes_id") or row.get("yes_token_id") or "").strip()
    no_token = str(row.get("token_no_id") or "").strip()
    start_ts = end_ts - 2 * 86400
    tokens_to_try = [("YES", yes_token)]
    if no_token and no_token != yes_token:
        tokens_to_try.append(("NO", no_token))
    print("\n--- Prices history smoke test (first market, last 2 days) ---")
    any_200 = False
    last_url = None
    for label, tid in tokens_to_try:
        if not tid:
            continue
        status, body, url = _prices_history_request(session, tid, start_ts, end_ts)
        last_url = url
        print("Token %s: HTTP %d" % (label, status))
        if status != 200:
            print("  Response body: %s" % body)
            continue
        any_200 = True
        hist = body if isinstance(body, list) else (body.get("history") or [])
        points = hist[:5]
        print("  First 5 points: %s" % points)
        if not hist:
            print("  (no_trades_or_no_history: empty array)")
        else:
            prices = [float(h.get("p") or h.get("price") or 0) for h in hist if (h.get("p") is not None or h.get("price") is not None)]
            if prices:
                print("  min price: %s  max price: %s" % (min(prices), max(prices)))
    if not any_200 and tokens_to_try:
        raise RuntimeError("Prices history smoke test: no HTTP 200. Last URL: %s" % last_url)
    print("--- Smoke test done ---\n")


def _fetch_prices_history_chunk(session: requests.Session, cache_dir: Path, token_id: str, start_ts: int, end_ts: int) -> list:
    """Single chunk request to CLOB prices-history. Cache key includes token_id, startTs, endTs, fidelity."""
    token_id = str(token_id).strip()
    fidelity = POLY_HISTORY_FIDELITY_MINUTES
    cache_path = cache_dir / "prices_history" / ("%s_%d_%d_f%d.json" % (token_id[:32], start_ts, end_ts, fidelity))
    data = _load_cache(cache_path)
    if data is not None:
        return data if isinstance(data, list) else data.get("history", [])
    time.sleep(SLEEP_MS / 1000.0)
    url = "%s/prices-history?market=%s&startTs=%d&endTs=%d&fidelity=%d" % (
        CLOB_BASE, requests.utils.quote(token_id, safe=""), start_ts, end_ts, fidelity,
    )
    try:
        r = _request(session, "GET", url)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:500]
            print("CLOB prices-history 400 response body: %s" % body)
        raise
    data = r.json()
    hist = data if isinstance(data, list) else data.get("history", [])
    _save_cache(cache_path, hist)
    return hist


def _fetch_prices_history(session: requests.Session, cache_dir: Path, token_id: str, start_ts: int, end_ts: int) -> list:
    """GET CLOB prices-history for a traded token. CLOB param 'market' = token (asset) id.
    Chunks large ranges to avoid 400 (Polymarket recommends e.g. 15-day chunks). Returns list of {t, p}. t in seconds."""
    token_id = str(token_id).strip()
    if end_ts <= start_ts:
        return []
    if end_ts - start_ts <= POLY_HISTORY_CHUNK_SECONDS:
        return _fetch_prices_history_chunk(session, cache_dir, token_id, start_ts, end_ts)
    combined = []
    t = start_ts
    while t < end_ts:
        chunk_end = min(t + POLY_HISTORY_CHUNK_SECONDS, end_ts)
        chunk = _fetch_prices_history_chunk(session, cache_dir, token_id, t, chunk_end)
        combined.extend(chunk)
        t = chunk_end
    combined.sort(key=lambda h: (h.get("t") or h.get("timestamp") or 0))
    return combined


def download_probabilities(session: requests.Session, cache_dir: Path, markets: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Download 5m price history per YES token. If history is empty, log no_trades_or_no_history and write one metadata row."""
    start_ms = start_ts * 1000
    rows = []
    for _, row in markets.iterrows():
        event_id = row["event_id"]
        strike_k = row["strike_K"]
        expiry_ms = row["expiry_ms"]
        parsed_date = row.get("parsed_date") or ""
        token_id = str(row.get("token_yes_id") or row.get("yes_token_id") or "").strip()
        market_id = row.get("market_id") or ""
        mapping_confidence = row.get("mapping_confidence") or ""
        hist = _fetch_prices_history(session, cache_dir, token_id, start_ts, end_ts)
        if not hist:
            # Log and write one row so we can analyze missingness
            print("no_trades_or_no_history event_id=%s market_id=%s token_yes_id=%s" % (event_id, market_id, token_id[:20] + "..." if len(token_id) > 20 else token_id))
            rows.append({
                "open_time": start_ms,
                "open_time_utc": _ms_to_utc_iso(start_ms),
                "event_id": event_id,
                "strike_K": strike_k,
                "expiry_ms": expiry_ms,
                "yes_token_id": token_id,
                "probability_pm": float("nan"),
                "date": parsed_date,
                "market_id": market_id,
                "mapping_confidence": mapping_confidence,
                "history_status": "no_trades_or_no_history",
            })
            continue
        for h in hist:
            t = h.get("t") or h.get("timestamp")
            p = h.get("p") or h.get("price")
            if t is None or p is None:
                continue
            open_time = int(t) * 1000 if int(t) < 1e12 else int(t)
            rows.append({
                "open_time": open_time,
                "open_time_utc": _ms_to_utc_iso(open_time),
                "event_id": event_id,
                "strike_K": strike_k,
                "expiry_ms": expiry_ms,
                "yes_token_id": token_id,
                "probability_pm": float(p),
                "date": parsed_date,
                "market_id": market_id,
                "mapping_confidence": mapping_confidence,
                "history_status": "ok",
            })
    return pd.DataFrame(rows)


def _fetch_spreads(session: requests.Session, cache_dir: Path, token_ids: list) -> dict:
    """POST CLOB /spreads with token ids. Returns map token_id -> {bid, ask, mid, spread} or similar."""
    if len(token_ids) > 100:
        token_ids = token_ids[:100]
    key = "_".join(str(t)[:8] for t in token_ids[:10])
    cache_path = cache_dir / ("spreads_%s_%s.json" % (datetime.now(timezone.utc).strftime("%Y%m%d"), key[:50]))
    data = _load_cache(cache_path)
    if data is not None:
        return data
    time.sleep(SLEEP_MS / 1000.0)
    # POST body: list of token ids per Polymarket docs
    body = [{"token_id": str(tid)} for tid in token_ids]
    r = _request(session, "POST", "%s/spreads" % CLOB_BASE, json=body)
    data = r.json()
    _save_cache(cache_path, data)
    return data


def download_spreads(session: requests.Session, cache_dir: Path, markets: pd.DataFrame) -> pd.DataFrame:
    """Batch fetch spreads for all YES tokens; return snapshot table."""
    token_col = "token_yes_id" if "token_yes_id" in markets.columns else "yes_token_id"
    token_ids = markets[token_col].drop_duplicates().tolist()
    rows = []
    for i in range(0, len(token_ids), SPREAD_BATCH_SIZE):
        batch = token_ids[i : i + SPREAD_BATCH_SIZE]
        resp = _fetch_spreads(session, cache_dir, batch)
        retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(resp, dict):
            for tid, val in resp.items():
                if isinstance(val, (int, float)):
                    rows.append({"yes_token_id": tid, "mid": val, "bid": None, "ask": None, "spread": None, "retrieved_at_utc": retrieved})
                elif isinstance(val, dict):
                    bid = val.get("bid") or val.get("bestBid")
                    ask = val.get("ask") or val.get("bestAsk")
                    mid = val.get("mid")
                    if mid is None and bid is not None and ask is not None:
                        mid = (float(bid) + float(ask)) / 2
                    spread = val.get("spread")
                    if spread is None and bid is not None and ask is not None:
                        spread = float(ask) - float(bid)
                    rows.append({"yes_token_id": tid, "bid": bid, "ask": ask, "mid": mid, "spread": spread, "retrieved_at_utc": retrieved})
        elif isinstance(resp, list):
            for item in resp:
                if isinstance(item, dict):
                    tid = item.get("token_id") or item.get("tokenId")
                    rows.append({
                        "yes_token_id": tid,
                        "bid": item.get("bid"), "ask": item.get("ask"), "mid": item.get("mid"), "spread": item.get("spread"),
                        "retrieved_at_utc": retrieved,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["yes_token_id", "bid", "ask", "mid", "spread", "retrieved_at_utc"])


def build_master_grid(start_ms: int, end_ms: int) -> pd.DataFrame:
    t = start_ms
    times = []
    while t <= end_ms:
        times.append(t)
        t += BAR_MS
    return pd.DataFrame({"open_time": times})


def main():
    _test_parse_bitcoin_above()
    parser = argparse.ArgumentParser(description="Step 4: Polymarket download and 5m panel (daily ladder only)")
    parser.add_argument("--window_dir", type=str, default="outputs/window=2026-01-01_2026-02-28")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    window_dir = Path(args.window_dir)
    if not window_dir.is_absolute():
        window_dir = ROOT / window_dir
    start_ts = _utc_to_ts(START_UTC)
    end_ts = _utc_to_ts(END_UTC)
    start_ms = start_ts * 1000
    end_ms = end_ts * 1000

    data_pm = ROOT / "data" / "polymarket"
    cache_dir = data_pm / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prices_history").mkdir(parents=True, exist_ok=True)
    # Unified thesis outputs (DATA only)
    out_polymarket = ROOT / "Actual data" / "Polymarket"
    out_polymarket.mkdir(parents=True, exist_ok=True)

    discovery_debug_info = None
    manifest_df = pd.DataFrame()
    exclusion_counts = {}
    if not args.overwrite and (data_pm / "markets_selected.csv").exists() and (data_pm / "probabilities_5m.parquet").exists():
        print("Outputs exist and --overwrite not set. Loading from disk.")
        markets = pd.read_csv(data_pm / "markets_selected.csv")
        probs = pd.read_parquet(data_pm / "probabilities_5m.parquet")
        if (data_pm / "manifest.csv").exists():
            manifest_df = pd.read_csv(data_pm / "manifest.csv")
    else:
        session = requests.Session()
        session.headers.update({"Accept": "application/json", "User-Agent": "ThesisPipeline/1.0"})
        print("Discovering markets (Gamma List markets — daily ladder only)...")
        markets, manifest_df, exclusion_counts, discovery_debug_info = discover_and_select_markets(session, cache_dir)
        probs = pd.DataFrame()
        _atomic_write_csv(manifest_df, data_pm / "manifest.csv")
        print("Exclusion counts: %s" % exclusion_counts)
        if markets.empty:
            print("No markets selected. Check pattern 'Bitcoin above ___ on [Month] [Day]?' and date range.")
            _atomic_write_csv(markets, data_pm / "markets_selected.csv")
        else:
            _run_prices_history_smoke_test(session, markets, end_ts)
            print("Downloading price history (CLOB)...")
            probs = download_probabilities(session, cache_dir, markets, start_ts, end_ts)
            _atomic_write_csv(markets, data_pm / "markets_selected.csv")
            if not probs.empty:
                _atomic_write_parquet(probs, data_pm / "probabilities_5m.parquet")
            print("Downloading spreads snapshot...")
            spreads = download_spreads(session, cache_dir, markets)
            try:
                _atomic_write_csv(spreads, data_pm / "spreads_snapshot.csv")
            except Exception as e:
                print("Spread snapshot write failed (non-fatal): %s" % e)

    # Ensure markets has open_ms / market_open_time_utc (e.g. when loaded from old CSV)
    if not markets.empty and "open_ms" not in markets.columns:
        markets["open_ms"] = start_ms
        markets["market_open_time_utc"] = _ms_to_utc_iso(start_ms)
        if "slug" not in markets.columns:
            markets["slug"] = ""

    tradable_filter_qa = {}
    if markets.empty:
        print("No markets; skipping raw probabilities export.")
        long_format = pd.DataFrame()
    else:
        # Raw extraction: only rows from market open until expiry; no forward fill.
        if probs.empty:
            probs = pd.DataFrame(columns=["open_time", "open_time_utc", "event_id", "strike_K", "expiry_ms", "yes_token_id", "probability_pm", "date", "market_id", "mapping_confidence", "history_status"])
        # Keep only rows with valid probability (raw observations; no fill)
        if "probability_pm" not in probs.columns:
            raw = pd.DataFrame()
        else:
            raw = probs[probs["probability_pm"].notna()].copy()
        if raw.empty:
            long_format = pd.DataFrame()
        else:
            # Join with markets to get open_ms and expiry_ms per contract
            mcols = ["event_id", "open_ms", "expiry_ms", "market_id", "market_open_time_utc", "expiry_utc", "parsed_date", "strike_K", "token_yes_id", "mapping_confidence", "slug"]
            mcols = [c for c in mcols if c in markets.columns]
            meta = markets[mcols].drop_duplicates("event_id")
            drop_in_raw = [c for c in meta.columns if c != "event_id" and c in raw.columns]
            if drop_in_raw:
                raw = raw.drop(columns=drop_in_raw)
            raw = raw.merge(meta, on="event_id", how="left")
            # Restrict to tradable window: open_ms <= open_time <= expiry_ms (no rows after expiry)
            n_before_tradable_filter = len(raw)
            raw = raw[(raw["open_time"] >= raw["open_ms"]) & (raw["open_time"] <= raw["expiry_ms"])]
            n_after_tradable_filter = len(raw)
            tradable_filter_qa["n_raw_before_tradable_filter"] = n_before_tradable_filter
            tradable_filter_qa["n_raw_after_tradable_filter"] = n_after_tradable_filter
            tradable_filter_qa["n_dropped_outside_tradable_window"] = n_before_tradable_filter - n_after_tradable_filter
            if raw.empty:
                long_format = pd.DataFrame()
            else:
                # Floor to 5-min boundary; dedupe (bucket, contract) keep last
                raw["bucket_5m_ms"] = (raw["open_time"] // BAR_MS) * BAR_MS
                raw = raw.sort_values("open_time").drop_duplicates(subset=["bucket_5m_ms", "event_id"], keep="last")
                raw["timestamp_utc"] = raw["bucket_5m_ms"].astype("int64").map(_ms_to_utc_iso)
                # Build long format with useful columns for analysis
                long_format = pd.DataFrame({
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
                    "slug": raw["slug"] if "slug" in raw.columns else "",
                })
        if not long_format.empty:
            _atomic_write_parquet(long_format, out_polymarket / "polymarket_probabilities_5m_long.parquet")
            print("Wrote %s (%d rows, raw extraction, tradable window only)" % (out_polymarket / "polymarket_probabilities_5m_long.parquet", len(long_format)))
            try:
                wide = long_format.pivot_table(index="timestamp_utc", columns="contract_id", values="prob_yes")
                _atomic_write_parquet(wide.reset_index(), out_polymarket / "polymarket_probabilities_5m_wide.parquet")
                print("Wrote %s" % (out_polymarket / "polymarket_probabilities_5m_wide.parquet"))
            except Exception as e:
                print("Wide format skip: %s" % e)
            meta_cols = ["event_id", "question", "parsed_date", "strike_K", "market_id", "token_yes_id", "expiry_utc"]
            for c in ["market_open_time_utc", "open_ms", "slug"]:
                if c in markets.columns:
                    meta_cols.append(c)
            meta_out = markets[[c for c in meta_cols if c in markets.columns]].copy()
            meta_out = meta_out.rename(columns={"event_id": "contract_id", "question": "title", "parsed_date": "expiry_date", "strike_K": "strike"})
            _atomic_write_parquet(meta_out, out_polymarket / "polymarket_contracts_metadata.parquet")
            _atomic_write_csv(meta_out, out_polymarket / "polymarket_contracts_metadata.csv")
            print("Wrote %s" % (out_polymarket / "polymarket_contracts_metadata.parquet"))
        else:
            long_format = pd.DataFrame()

    # QA report for Actual data/Polymarket (raw extraction, tradable window only, no fill)
    qa = {
        "window_start_utc": START_UTC,
        "window_end_utc": END_UTC,
        "extraction_mode": "raw_no_fill",
        "tradable_window_only": True,
        "n_contracts_total": len(markets) if not markets.empty else 0,
        "n_rows_long_total": int(len(long_format)) if not long_format.empty else 0,
        "contracts_per_expiry_date": [],
        "strike_distribution": {"min": None, "max": None, "unique_count": None},
        "duplicate_timestamp_contract_id": 0,
        "timestamps_5min_aligned_count_bad": 0,
        "prob_out_of_range_count": 0,
        "observations_per_contract_distribution": {},
        "last_obs_vs_expiry_hours_distribution": {},
        "top10_by_updates": [],
    }
    qa.update(tradable_filter_qa)
    if not markets.empty:
        by_exp = markets.groupby("parsed_date").size().reset_index(name="n_contracts")
        qa["contracts_per_expiry_date"] = by_exp.to_dict(orient="records")
        strikes = markets["strike_K"].dropna()
        if len(strikes):
            qa["strike_distribution"] = {"min": int(strikes.min()), "max": int(strikes.max()), "unique_count": int(strikes.nunique())}
    if not long_format.empty:
        dup = long_format.duplicated(subset=["timestamp_utc", "contract_id"])
        qa["duplicate_timestamp_contract_id"] = int(dup.sum())
        # 5-min alignment: timestamp_utc should have minute % 5 == 0, second == 0
        try:
            ts = pd.to_datetime(long_format["timestamp_utc"], utc=True)
            bad = (ts.dt.minute % 5 != 0) | (ts.dt.second != 0)
            qa["timestamps_5min_aligned_count_bad"] = int(bad.sum())
        except Exception:
            pass
        prob = long_format["prob_yes"].dropna()
        qa["prob_out_of_range_count"] = int(((prob < 0) | (prob > 1)).sum())
        # Observations per contract
        obs_per = long_format.groupby("contract_id").size()
        qa["observations_per_contract_distribution"] = {"min": int(obs_per.min()), "max": int(obs_per.max()), "mean": float(obs_per.mean()), "median": float(obs_per.median())}
        top10 = obs_per.nlargest(10)
        qa["top10_by_updates"] = [{"contract_id": cid, "n_updates": int(n)} for cid, n in top10.items()]
        # Last observation vs expiry (report stats)
        if "expiry_date" in long_format.columns and "expiry_utc" in markets.columns:
            last_obs = long_format.groupby("contract_id").agg({"timestamp_utc": "max"}).reset_index()
            last_obs = last_obs.merge(markets[["event_id", "expiry_utc"]].rename(columns={"event_id": "contract_id"}), on="contract_id", how="left")
            def _hours_to_exp(row):
                try:
                    last_ts = pd.to_datetime(row["timestamp_utc"], utc=True)
                    exp_str = str(row.get("expiry_utc") or "")
                    if not exp_str or exp_str == "nan":
                        return None
                    exp_ts = pd.to_datetime(exp_str.replace("Z", "+00:00"))
                    return (exp_ts - last_ts).total_seconds() / 3600.0
                except Exception:
                    return None
            last_obs["hours_to_expiry"] = last_obs.apply(_hours_to_exp, axis=1)
            h = last_obs["hours_to_expiry"].dropna()
            if len(h):
                qa["last_obs_vs_expiry_hours_distribution"] = {"min": float(h.min()), "max": float(h.max()), "mean": float(h.mean())}
    out_polymarket.mkdir(parents=True, exist_ok=True)
    with open(out_polymarket / "qa_polymarket.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)
    print("Wrote %s" % (out_polymarket / "qa_polymarket.json"))
    with open(out_polymarket / "qa_polymarket.md", "w", encoding="utf-8") as f:
        f.write("# Polymarket QA report\n\n")
        f.write("- Window: %s to %s\n" % (qa["window_start_utc"], qa["window_end_utc"]))
        f.write("- #contracts total: %s\n" % qa["n_contracts_total"])
        f.write("- #rows long total: %s\n" % qa["n_rows_long_total"])
        f.write("- duplicate (timestamp_utc, contract_id): %s\n" % qa["duplicate_timestamp_contract_id"])
        f.write("- timestamps 5min aligned (bad count): %s\n" % qa["timestamps_5min_aligned_count_bad"])
        f.write("- prob out of [0,1] count: %s\n" % qa["prob_out_of_range_count"])
        f.write("- strike_distribution: %s\n" % qa["strike_distribution"])
        f.write("- observations_per_contract: %s\n" % qa.get("observations_per_contract_distribution", {}))
        f.write("- last_obs_vs_expiry_hours: %s\n" % qa.get("last_obs_vs_expiry_hours_distribution", {}))
    print("Wrote %s" % (out_polymarket / "qa_polymarket.md"))

    # Sanity block (daily ladder universe). When no markets selected, do NOT use stale disk data.
    n_markets = len(markets)
    if n_markets == 0:
        probs_df = pd.DataFrame()
        n_rows_probs = 0
    else:
        probs_path = data_pm / "probabilities_5m.parquet"
        probs_df = pd.read_parquet(probs_path) if probs_path.exists() else pd.DataFrame()
        if probs_df.empty and not probs.empty:
            probs_df = probs
        if probs_df.empty:
            probs_df = pd.DataFrame(columns=["event_id", "open_time", "probability_pm", "history_status"])
        n_rows_probs = len(probs_df)
    expiry_min = markets["expiry_utc"].min() if n_markets and "expiry_utc" in markets.columns else ""
    expiry_max = markets["expiry_utc"].max() if n_markets and "expiry_utc" in markets.columns else ""
    bounds_violations = 0
    if "probability_pm" in probs_df.columns:
        valid = probs_df["probability_pm"].notna()
        bounds_violations = ((probs_df.loc[valid, "probability_pm"] < -0.001) | (probs_df.loc[valid, "probability_pm"] > 1.001)).sum()
    # % markets with non-empty history
    with_history = 0
    if "event_id" in probs_df.columns:
        if "history_status" in probs_df.columns:
            with_history = probs_df[probs_df["history_status"] == "ok"]["event_id"].nunique()
        else:
            with_history = probs_df[probs_df["probability_pm"].notna()]["event_id"].nunique() if "probability_pm" in probs_df.columns else 0
    pct_with_history = (100.0 * with_history / n_markets) if n_markets else float("nan")
    # Points per market distribution (only when we have markets and probs from this run)
    points_per_market = []
    if n_markets and "event_id" in probs_df.columns and "probability_pm" in probs_df.columns:
        for eid in probs_df["event_id"].drop_duplicates():
            n_pts = probs_df[probs_df["event_id"] == eid]["probability_pm"].notna().sum()
            points_per_market.append(n_pts)
    n_pts_min = min(points_per_market) if points_per_market else None
    n_pts_max = max(points_per_market) if points_per_market else None
    n_pts_avg = sum(points_per_market) / len(points_per_market) if points_per_market else None
    # Per-date: fraction of strikes with any data
    per_date_coverage = []
    if n_markets and "parsed_date" in markets.columns and "event_id" in probs_df.columns:
        if "history_status" in probs_df.columns:
            with_data_ids = set(probs_df[probs_df["history_status"] == "ok"]["event_id"].unique())
        else:
            with_data_ids = set(probs_df[probs_df["probability_pm"].notna()]["event_id"].unique()) if "probability_pm" in probs_df.columns else set()
        for d in markets["parsed_date"].drop_duplicates():
            strikes_date = set(markets[markets["parsed_date"] == d]["event_id"])
            n_strikes = len(strikes_date)
            n_with = len(strikes_date & with_data_ids)
            per_date_coverage.append({"date": d, "n_strikes": n_strikes, "n_with_data": n_with, "pct": 100.0 * n_with / n_strikes if n_strikes else 0})
    top_volume = "N/A"
    if n_markets and "volume" in markets.columns:
        v = pd.to_numeric(markets["volume"], errors="coerce").fillna(0)
        if v.gt(0).any():
            top_volume = markets.assign(_v=v).nlargest(10, "_v")[["event_id", "volume"]].to_string()
    spreads_path = data_pm / "spreads_snapshot.csv"
    if spreads_path.exists():
        try:
            spr = pd.read_csv(spreads_path)
            if "spread" in spr.columns and spr["spread"].notna().any():
                spr = spr.dropna(subset=["spread"]).nsmallest(10, "spread")
                top_spread = spr[["yes_token_id", "spread"]].to_string()
            else:
                top_spread = "N/A (no spread column or all NaN)"
        except Exception:
            top_spread = "N/A (read error)"
    else:
        top_spread = "N/A (no snapshot)"
    warnings = []
    if n_markets and "probability_pm" in probs_df.columns and "event_id" in probs_df.columns:
        for eid in probs_df["event_id"].drop_duplicates():
            sub = probs_df[probs_df["event_id"] == eid]["probability_pm"].dropna()
            if len(sub) > 10 and sub.std() < 0.01:
                warnings.append("nearly constant probability (std<0.01): %s" % eid)

    print("\n" + "=" * 60)
    print("STEP 4 POLYMARKET SANITY BLOCK (daily ladder)")
    print("=" * 60)
    print("n_markets_selected: %d" % n_markets)
    print("date range of expiries: min=%s max=%s" % (expiry_min, expiry_max))
    print("n_rows_probs_total: %d" % n_rows_probs)
    if n_markets:
        print("%% markets with non-empty history: %.1f%% (%d / %d)" % (pct_with_history, with_history, n_markets))
    else:
        print("%% markets with non-empty history: N/A (no markets selected)")
    if n_pts_min is not None and n_pts_max is not None and n_pts_avg is not None:
        print("points per market: min=%s max=%s avg=%.1f" % (n_pts_min, n_pts_max, n_pts_avg))
    else:
        print("points per market: N/A")
    if per_date_coverage:
        print("per-date strike coverage (sample):")
        for r in per_date_coverage[:5]:
            print("  %s: %d/%d strikes (%.0f%%)" % (r["date"], r["n_with_data"], r["n_strikes"], r["pct"]))
    print("probability bounds violations count: %d" % bounds_violations)
    print("top 10 markets by volume:")
    print(top_volume)
    print("top 10 by smallest spread:")
    print(top_spread)
    print("exclusion counts: %s" % exclusion_counts)
    if n_markets == 0:
        print("warnings: N/A (no markets selected)")
    else:
        print("warnings: %s" % (warnings if warnings else "none"))
    print("=" * 60)

    if discovery_debug_info:
        d = discovery_debug_info
        print("\nDISCOVERY SUMMARY: total_fetched=%d exclusion=%s final_count=%d" % (d.get("total_fetched", 0), d.get("exclusion", {}), d.get("final_count", 0)))
    else:
        print("\nDISCOVERY: not run (outputs loaded from disk).")

    print("\nRun this (full refresh for 2026-01-01 to 2026-02-28):")
    print("python scripts/run_step4_polymarket_download.py --window_dir \"outputs/window=2026-01-01_2026-02-28\" --overwrite")


if __name__ == "__main__":
    main()
