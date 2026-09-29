"""Runs exported as data/raw/<name>/*.json in our simple {"messages", "system", "meta"} form
(e.g. Docent exports via scripts/data/fetch_docent_runs.py)."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

from ..schema import RAW


def iter_runs(name: str) -> Iterator[tuple[Path, list[dict], dict]]:
    for f in sorted(glob.glob(str(RAW / name / "*.json"))):
        d = json.load(open(f))
        msgs = d.get("messages") or []
        if len(msgs) < 4:
            continue
        meta = dict(d.get("meta") or {})
        meta["system"] = (d.get("system") or "")[:4000]
        meta["session"] = Path(f).stem
        yield Path(f), msgs, meta
