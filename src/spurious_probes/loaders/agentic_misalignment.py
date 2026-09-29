"""Anthropic 'Agentic misalignment, summer 2026' transcripts (public viewer at aenguslynch.com/portfolio-transcript-viewer;
files in data/raw/align/agentic_misalignment/<section>__<id>.json). We take the target model's view (`target_messages`):
system prompt (kept in meta), user turns, assistant text (reasoning is redacted in the release) and tool outputs, which
arrive as plain 'tool' messages without structured calls, so they are rendered as text observations in a user turn
(the same treatment as Terminus-2 traces)."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "align" / "agentic_misalignment"
REDACTED = re.compile(r"^\s*\[reasoning redacted\]\s*", re.I)


def iter_transcripts() -> Iterator[tuple[Path, list[dict], dict]]:
    for f in sorted(glob.glob(str(RAW / "*__*.json"))):
        d = json.load(open(f))
        tm = d.get("target_messages") or []
        msgs: list[dict] = []; system = None
        for m in tm:
            role, content = m.get("role"), str(m.get("content") or "").replace("\u2028", "\n")
            if role == "system":
                system = content
            elif role == "user":
                msgs.append({"role": "user", "content": [{"type": "text", "text": content}]})
            elif role == "assistant":
                text = REDACTED.sub("", content).strip()
                if text:
                    msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            elif role == "tool":
                msgs.append({"role": "user", "content": [{"type": "text", "text": "Tool output:\n" + content}]})
        meta = d.get("metadata") or {}
        yield Path(f), msgs, {"session": d.get("id"), "section": d.get("section"), "target_model": meta.get("target_model"),
                              "auditor_model": meta.get("auditor_model"), "system": system, "models": [meta.get("target_model") or "?"]}
