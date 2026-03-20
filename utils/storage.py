"""
Write parquet/csv, de-dup by open_time, schema helpers, checkpointing.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _partition_path(base: str, year: int, month: int) -> str:
    return os.path.join(base, f"year={year}", f"month={month:02d}")


def write_parquet(
    df: pd.DataFrame,
    base_path: str,
    year: int,
    month: int,
    filename: str,
    dedup_col: Optional[str] = "open_time",
) -> str:
    """Write DataFrame to partitioned parquet; de-dup by dedup_col if set."""
    if df.empty:
        return ""
    if dedup_col and dedup_col in df.columns:
        df = df.drop_duplicates(subset=[dedup_col], keep="last").sort_values(dedup_col)
    out_dir = _partition_path(base_path, year, month)
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, filename)
    df.to_parquet(out_file, index=False)
    logger.info("wrote %s rows to %s", len(df), out_file)
    return out_file


def load_done_chunks() -> Dict[str, List[Any]]:
    """Load manifest of done chunks for resume."""
    ensure_dir(config.MANIFEST_DIR)
    path = config.DONE_CHUNKS_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("could not load done_chunks: %s", e)
        return {}


def save_done_chunk(key: str, chunk_id: Any) -> None:
    """Append a done chunk to manifest."""
    ensure_dir(config.MANIFEST_DIR)
    data = load_done_chunks()
    data.setdefault(key, []).append(chunk_id)
    with open(config.DONE_CHUNKS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_chunk_done(key: str, chunk_id: Any) -> bool:
    done = load_done_chunks()
    return chunk_id in done.get(key, [])


def add_utc_columns(df: pd.DataFrame, time_col: str = "open_time") -> pd.DataFrame:
    """Add open_time_utc ISO string from open_time (ms)."""
    if time_col not in df.columns or df.empty:
        return df
    df = df.copy()
    df["open_time_utc"] = pd.to_datetime(df[time_col], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    return df
