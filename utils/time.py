"""
Time utilities: ms conversions, range chunking for 5m klines.
Config START_DT/END_DT are intended as UTC; naive datetimes are treated as UTC.
"""
from datetime import datetime, timedelta, timezone
from typing import Iterator, Tuple

import config


def utc_to_ms(dt: datetime) -> int:
    """Convert UTC datetime to milliseconds since epoch. Naive datetimes are treated as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_utc(ms: int) -> datetime:
    """Convert milliseconds since epoch to UTC datetime."""
    return datetime.utcfromtimestamp(ms / 1000.0)


def ms_to_utc_iso(ms: int) -> str:
    """Convert ms to ISO string (UTC)."""
    return ms_to_utc(ms).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_start_end_ms() -> Tuple[int, int]:
    """Return (start_ms, end_ms) from config."""
    return utc_to_ms(config.START_DT), utc_to_ms(config.END_DT)


def parse_window_start_end(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    """Parse 'YYYY-MM-DD HH:MM:SS' (UTC) to naive datetimes (treated as UTC)."""
    def parse(s: str) -> datetime:
        s = s.strip()
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        parts = s.split()
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else "00:00:00"
        return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
    return parse(start_str), parse(end_str)


def window_output_suffix(start_ms: int, end_ms: int) -> str:
    """Return folder suffix for windowed outputs, e.g. window=2026-01-01_2026-02-15."""
    s = ms_to_utc(start_ms).strftime("%Y-%m-%d")
    e = ms_to_utc(end_ms).strftime("%Y-%m-%d")
    return f"window={s}_{e}"


def chunk_range_5d(start_ms: int, end_ms: int) -> Iterator[Tuple[int, int]]:
    """
    Yield (chunk_start_ms, chunk_end_ms) in ~5-day chunks.
    5 days = 5 * 24 * 60 * 60 * 1000 ms = 432_000_000 ms
    """
    chunk_ms = config.CHUNK_DAYS * 24 * 60 * 60 * 1000
    current = start_ms
    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        yield current, chunk_end
        current = chunk_end
