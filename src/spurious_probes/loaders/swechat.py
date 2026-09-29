"""SWE-chat (HF SALT-NLP/SWE-chat): real coding-agent sessions from many open-source developers, raw agent-native
transcripts. The subset we used is listed in sources/swechat_subset.json (session_id -> agent / transcript path); the transcripts
themselves go under data/raw/swechat/transcripts/ (scripts/data/download.py).
Claude Code sessions go through the claude_code loader, Codex rollouts through the codex loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..schema import iter_jsonl
from . import claude_code, codex

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "swechat"


def iter_sessions(agent: str = "Claude Code") -> Iterator[tuple[Path, list[dict], dict]]:
    meta = json.load(open(ROOT / "sources" / "swechat_subset.json"))
    for sid, m in sorted(meta.items()):
        if m["agent"] != agent:
            continue
        f = RAW / m["transcript_path"]
        if not f.exists():
            continue
        lines = list(iter_jsonl(f))
        if agent == "Claude Code":
            path = claude_code.main_path(lines)
            if len(path) < 6:
                continue
            msgs, info = claude_code.to_messages(path)
            models = info["models"]
        else:
            msgs, info = codex.to_messages(lines)
            models = sorted(info.get("models") or [])
        yield f, msgs, {"session": sid, "agent": m["agent"], "models": models,
                        "cli_version": m.get("cli_version"), "api_call_count": m.get("api_call_count")}
