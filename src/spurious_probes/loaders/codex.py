"""Codex CLI rollouts: ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.

Every line is {timestamp, ordinal, type, payload}. Only type == "response_item"
payloads are model I/O (OpenAI Responses-API items). The system/developer
instructions are not stored (base_instructions: null); the first user item is an
<environment_context> block, kept as-is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..schema import iter_jsonl

CODEX_SESSIONS = Path.home() / ".codex" / "sessions"


def session_files() -> list[Path]:
    return sorted(CODEX_SESSIONS.rglob("rollout-*.jsonl"))


def _parts_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for p in content or []:
        if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text"):
            out.append(p.get("text", ""))
        elif isinstance(p, dict) and "image" in str(p.get("type", "")):
            out.append("[image omitted]")
    return "\n".join(t for t in out if t)


def to_messages(lines: list[dict]) -> tuple[list[dict], dict]:
    msgs: list[dict] = []
    meta: dict = {"models": set(), "cwd": None, "cli_version": None}
    for l in lines:
        t = l.get("type")
        p = l.get("payload") or {}
        if t == "session_meta":
            meta["cwd"] = p.get("cwd"); meta["forked_from"] = p.get("forked_from_id")
            meta["cli_version"] = p.get("cli_version")
            continue
        if t == "turn_context":
            if p.get("model"):
                meta["models"].add(p["model"])
            continue
        if t != "response_item":
            continue
        pt = p.get("type")
        if pt == "message":
            role = p.get("role")
            text = _parts_text(p.get("content"))
            if role in ("user", "assistant") and text.strip():
                msgs.append({"role": role, "content": [{"type": "text", "text": text}]})
        elif pt == "function_call":
            try:
                args = json.loads(p.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"arguments": args}
            except json.JSONDecodeError:
                args = {"arguments": p.get("arguments")}
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": p.get("call_id"), "name": p.get("name") or "tool", "input": args}]})
        elif pt == "function_call_output":
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": p.get("call_id"), "content": str(p.get("output", ""))}]})
        elif pt == "custom_tool_call":
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": p.get("call_id"), "name": p.get("name") or "custom_tool", "input": {"input": p.get("input", "")}}]})
        elif pt == "custom_tool_call_output":
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": p.get("call_id"), "content": str(p.get("output", ""))}]})
        # reasoning, web_search_call etc. are dropped
    meta["models"] = sorted(meta["models"])
    return msgs, meta


_PROBE_DIRS = {"codex-luna-probe", "luna-probe"}


def _probe_texts() -> set[str]:
    """Texts of every probe in the repo's probes*.yaml, so live-Codex probe turns never enter the corpus."""
    import yaml
    root = Path(__file__).resolve().parents[3]
    out: set[str] = set()
    for f in [*root.glob("probes/*.yaml"), *root.glob("chameleon/probes/*.yaml")]:
        try:
            d = yaml.safe_load(f.read_text()) or []
            for p in (d.get("probes") or []) if isinstance(d, dict) else d:
                if isinstance(p, dict) and p.get("text"):
                    out.add(str(p["text"]).strip())
        except Exception:
            continue
    return out


def _strip_probe_turns(msgs: list[dict], probe_texts: set[str]) -> list[dict]:
    """Cut a session at the first user turn that is one of our questions (a session where you asked one live, as for figure 1)."""
    for i, m in enumerate(msgs):
        if m["role"] == "user" and _parts_text(m["content"]).strip() in probe_texts:
            return msgs[:i]
    return msgs


def iter_sessions() -> Iterator[tuple[Path, list[dict], dict]]:
    probe_texts = _probe_texts()
    for f in session_files():
        lines = list(iter_jsonl(f))
        msgs, meta = to_messages(lines)
        if meta.get("cwd") and Path(meta["cwd"]).name in _PROBE_DIRS:
            continue
        if meta.get("forked_from"):  # forks re-cover the parent's history; keep the parent only
            continue
        msgs = _strip_probe_turns(msgs, probe_texts)
        if len(msgs) < 6:
            continue
        meta.update({"session": f.stem, "n_lines": len(lines)})
        yield f, msgs, meta
