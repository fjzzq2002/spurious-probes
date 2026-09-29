"""Claude Code session transcripts: ~/.claude/projects/<project>/<uuid>.jsonl.

Lines form a DAG via parentUuid (rewinds create branches). We walk back from the
last conversation line to the root to get the active path. One API response is
split across several lines that share message.id; those are merged. Assistant
lines with model "<synthetic>" are harness-generated, not model output, and dropped.
The system prompt is not stored in these files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from ..schema import iter_jsonl

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
EXCLUDE_PROJECT_RE = re.compile(r"spurious-probes", re.I)


def session_files() -> list[Path]:
    files = []
    for proj in sorted(CLAUDE_PROJECTS.iterdir()):
        if not proj.is_dir() or EXCLUDE_PROJECT_RE.search(proj.name):
            continue
        files.extend(sorted(proj.glob("*.jsonl")))
    return files


def main_path(lines: list[dict]) -> list[dict]:
    conv = [l for l in lines if l.get("type") in ("user", "assistant") and isinstance(l.get("message"), dict)]
    if not conv:
        return []
    by_uuid = {l["uuid"]: l for l in conv if l.get("uuid")}
    leaf = conv[-1]
    path = []
    seen = set()
    cur = leaf
    while cur is not None and cur.get("uuid") not in seen:
        seen.add(cur.get("uuid"))
        path.append(cur)
        cur = by_uuid.get(cur.get("parentUuid"))
    path.reverse()
    # if the DAG walk lost most of the file (e.g. missing parent links), fall back to file order
    if len(path) < 0.5 * len(conv):
        path = conv
    return path


def to_messages(path: list[dict]) -> tuple[list[dict], dict]:
    msgs: list[dict] = []
    models: set[str] = set()
    last_msg_id = None
    for l in path:
        msg = l["message"]
        role = msg.get("role")
        if role == "assistant":
            if msg.get("model") == "<synthetic>":
                continue
            if msg.get("model"):
                models.add(msg["model"])
            content = msg.get("content") or []
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            content = [dict(b) for b in content]
            for b in content:
                b.pop("caller", None)
            mid = msg.get("id")
            if msgs and msgs[-1]["role"] == "assistant" and mid and mid == last_msg_id:
                msgs[-1]["content"].extend(content)
            else:
                msgs.append({"role": "assistant", "content": content})
            last_msg_id = mid
        elif role == "user":
            msgs.append({"role": "user", "content": msg.get("content")})
            last_msg_id = None
    return msgs, {"models": sorted(models)}


def iter_sessions() -> Iterator[tuple[Path, list[dict], dict]]:
    for f in session_files():
        lines = list(iter_jsonl(f))
        path = main_path(lines)
        if len(path) < 6:
            continue
        msgs, info = to_messages(path)
        first = next((l for l in path if l.get("timestamp")), {})
        meta = {
            "project": f.parent.name,
            "session": f.stem,
            "cc_version": first.get("version"),
            "started": first.get("timestamp"),
            "models": info["models"],
            "n_lines": len(lines),
        }
        yield f, msgs, meta
