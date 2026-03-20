#!/usr/bin/env python3
"""Clear spot_klines from done_chunks.json so spot download will re-fetch all chunks."""
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

def main():
    path = config.DONE_CHUNKS_FILE
    if not os.path.exists(path):
        print("No done_chunks.json found.")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["spot_klines"] = []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Cleared spot_klines from done_chunks. Re-run backfill to re-download spot.")


if __name__ == "__main__":
    main()
