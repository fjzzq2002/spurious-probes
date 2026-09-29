"""Shared preprocessing: block normalisation, same-role merging, tool-pair repair,
text-only cut, tool stubs, token estimate, offline validator."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

CHARS_PER_TOKEN = 3.5  # calibrated against the provider's token counts
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------- blocks

def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def normalize_result_content(content: Any) -> str | list[dict]:
    """tool_result.content may be a string or a list of text/image blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[dict] = []
        for b in content:
            if not isinstance(b, dict):
                out.append(_text_block(str(b)))
            elif b.get("type") == "text":
                if b.get("text", "").strip():
                    out.append(_text_block(b["text"]))
            elif b.get("type") == "image":
                out.append(_text_block("[image omitted]"))
            else:
                out.append(_text_block(f"[{b.get('type', 'block')} omitted]"))
        if not out:
            return ""
        if len(out) == 1:
            return out[0]["text"]
        return out
    return json.dumps(content)


def normalize_blocks(content: Any) -> list[dict]:
    """Return a clean list of Anthropic content blocks (text / tool_use / tool_result)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [_text_block(content)] if content.strip() else []
    blocks: list[dict] = []
    for b in content:
        if not isinstance(b, dict):
            if str(b).strip():
                blocks.append(_text_block(str(b)))
            continue
        t = b.get("type")
        if t == "text":
            if isinstance(b.get("text"), str) and b["text"].strip():
                blocks.append(_text_block(b["text"]))
        elif t == "tool_use":
            inp = b.get("input")
            if not isinstance(inp, dict):
                inp = {"input": inp}
            blocks.append({"type": "tool_use", "id": str(b.get("id")), "name": str(b.get("name")), "input": inp})
        elif t == "tool_result":
            blk = {"type": "tool_result", "tool_use_id": str(b.get("tool_use_id")),
                   "content": normalize_result_content(b.get("content"))}
            if b.get("is_error"):
                blk["is_error"] = True
            blocks.append(blk)
        elif t in ("thinking", "redacted_thinking", "server_tool_use", "web_search_tool_result"):
            continue
        elif t == "image":
            blocks.append(_text_block("[image omitted]"))
        elif t == "document":
            blocks.append(_text_block("[document omitted]"))
        # anything else is dropped silently
    return blocks


# ---------------------------------------------------------------- messages

def clean_messages(messages: list[dict]) -> list[dict]:
    """Normalise blocks, drop empty turns, merge consecutive same-role turns,
    ensure the first turn is a user turn."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        blocks = normalize_blocks(m.get("content"))
        if not blocks:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def repair_tool_pairs(messages: list[dict]) -> tuple[list[dict], dict]:
    """Every tool_use in an assistant turn must have a tool_result at the start of the
    next user turn, and vice versa. Insert synthetic results / drop orphans; report counts."""
    messages = copy.deepcopy(messages)
    stats = {"synthetic_results": 0, "orphan_results_dropped": 0}
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant":
            uses = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if uses:
                if i + 1 >= len(messages):
                    messages.append({"role": "user", "content": []})
                nxt = messages[i + 1]
                have = {b["tool_use_id"] for b in nxt["content"] if b["type"] == "tool_result"}
                missing = [u for u in uses if u not in have]
                if missing:
                    synth = [{"type": "tool_result", "tool_use_id": u, "content": "[no result recorded]"} for u in missing]
                    nxt["content"] = synth + nxt["content"]
                    stats["synthetic_results"] += len(missing)
                # results must precede other blocks in the user turn
                res = [b for b in nxt["content"] if b["type"] == "tool_result"]
                rest = [b for b in nxt["content"] if b["type"] != "tool_result"]
                nxt["content"] = res + rest
        if m["role"] == "user":
            prev_uses = set()
            if i > 0 and messages[i - 1]["role"] == "assistant":
                prev_uses = {b["id"] for b in messages[i - 1]["content"] if b["type"] == "tool_use"}
            kept = []
            for b in m["content"]:
                if b["type"] == "tool_result" and b["tool_use_id"] not in prev_uses:
                    stats["orphan_results_dropped"] += 1
                    continue
                kept.append(b)
            m["content"] = kept
        i += 1
    # a repair can leave empty user turns; re-clean (merges and drops empties)
    return clean_messages(messages), stats


_CPT = [CHARS_PER_TOKEN]


def set_chars_per_token(v: float) -> None:
    _CPT[0] = v


def estimate_tokens(obj: Any) -> int:
    if isinstance(obj, str):
        return int(len(obj) / _CPT[0]) + 1
    return int(len(json.dumps(obj, ensure_ascii=False)) / _CPT[0]) + 1


def is_text_only_assistant(m: dict) -> bool:
    return m["role"] == "assistant" and all(b["type"] == "text" for b in m["content"])


def is_tool_result_only_user(m: dict) -> bool:
    return m["role"] == "user" and bool(m["content"]) and all(b["type"] == "tool_result" for b in m["content"])


def cut_prefix(messages: list[dict], min_tokens: int, max_tokens: int, policy: str = "prefer_text",
               text_observations: bool = False) -> tuple[list[dict] | None, int, str | None]:
    """Choose a prefix boundary inside the token window.

    Boundary kinds:
      text: prefix ends with a text-only assistant turn; the probe becomes a new user turn.
      tool: prefix ends with a user turn made only of tool_result blocks (a complete
            exchange); the probe is appended as a text block after those results.
            With text_observations=True (text-rendered agent traces such as Terminus 2),
            any user turn after the first counts as an observation and is a tool boundary.
    policy: "text" | "tool" | "prefer_text" (last text boundary if any, else last tool boundary).
    Returns (prefix or None, estimated tokens, kind).
    """
    total = 0
    last_text: tuple[int, int] | None = None
    last_tool: tuple[int, int] | None = None
    for i, m in enumerate(messages):
        total += estimate_tokens(m["content"])
        if total > max_tokens:
            break
        if total < min_tokens:
            continue
        if is_text_only_assistant(m) and (not text_observations or i == len(messages) - 1):
            # in text-rendered agent traces every assistant turn is text; only the final
            # (task-complete) turn counts as a text boundary there
            last_text = (i, total)
        elif i > 0 and (is_tool_result_only_user(m) or (text_observations and m["role"] == "user")):
            last_tool = (i, total)
    choice = None
    if policy in ("text", "prefer_text") and last_text is not None:
        choice = (last_text, "text")
    elif policy in ("tool", "prefer_text") and last_tool is not None:
        choice = (last_tool, "tool")
    if choice is None:
        return None, 0, None
    (i, total), kind = choice
    return messages[: i + 1], total, kind


def cut_text_only(messages: list[dict], min_tokens: int, max_tokens: int) -> tuple[list[dict] | None, int]:
    cut, est, _ = cut_prefix(messages, min_tokens, max_tokens, policy="text")
    return cut, est


def tool_stubs(messages: list[dict]) -> list[dict]:
    names = sorted({b["name"] for m in messages if m["role"] == "assistant"
                    for b in m["content"] if b["type"] == "tool_use"})
    return [{"name": n, "description": f"{n} tool", "input_schema": {"type": "object", "additionalProperties": True}}
            for n in names]


def validate(messages: list[dict], text_observations: bool = False) -> list[str]:
    """Offline structural check; returns a list of problems (empty = ok)."""
    issues: list[str] = []
    if not messages:
        return ["empty"]
    if messages[0]["role"] != "user":
        issues.append("first turn not user")
    for i, m in enumerate(messages):
        if i > 0 and messages[i - 1]["role"] == m["role"]:
            issues.append(f"consecutive {m['role']} at {i}")
        if not m["content"]:
            issues.append(f"empty content at {i}")
        for b in m["content"]:
            if b["type"] == "text" and not b["text"].strip():
                issues.append(f"empty text block at {i}")
            if b["type"] == "tool_use" and not TOOL_NAME_RE.match(b["name"]):
                issues.append(f"bad tool name {b['name']!r} at {i}")
        if m["role"] == "assistant":
            uses = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if uses:
                if i + 1 >= len(messages):
                    issues.append(f"dangling tool_use at {i}")
                else:
                    have = [b["tool_use_id"] for b in messages[i + 1]["content"] if b["type"] == "tool_result"]
                    if sorted(have) != sorted(uses):
                        issues.append(f"tool pairing mismatch at {i}")
        if m["role"] == "user":
            res_idx = [k for k, b in enumerate(m["content"]) if b["type"] == "tool_result"]
            if res_idx and res_idx != list(range(len(res_idx))):
                issues.append(f"tool_result not leading at {i}")
            if res_idx:
                prev_uses = {b["id"] for b in messages[i - 1]["content"] if b["type"] == "tool_use"} if i > 0 else set()
                if any(m["content"][k]["tool_use_id"] not in prev_uses for k in res_idx):
                    issues.append(f"orphan tool_result at {i}")
    last = messages[-1]
    if not (is_text_only_assistant(last) or is_tool_result_only_user(last) or (text_observations and last["role"] == "user" and len(messages) > 1)):
        issues.append("last turn is neither a text-only assistant turn nor a tool-result-only user turn")
    return issues


def finalize(messages: list[dict], min_tokens: int, max_tokens: int, policy: str = "prefer_text",
             text_observations: bool = False) -> tuple[list[dict] | None, dict]:
    """clean -> repair -> cut -> validate. Returns (messages or None, info)."""
    msgs = clean_messages(messages)
    repairs = {"synthetic_results": 0, "orphan_results_dropped": 0}
    for _ in range(6):
        before = json.dumps(msgs, sort_keys=True)
        msgs, r = repair_tool_pairs(msgs)
        repairs = {k: repairs[k] + r[k] for k in repairs}
        if json.dumps(msgs, sort_keys=True) == before:
            break
    cut, est, kind = cut_prefix(msgs, min_tokens, max_tokens, policy, text_observations)
    info: dict[str, Any] = {"repairs": repairs, "n_messages_full": len(msgs), "tokens_full_est": estimate_tokens([m["content"] for m in msgs])}
    if cut is None:
        info["reason"] = f"no {policy} boundary in window"
        return None, info
    info["cut_kind"] = kind
    issues = validate(cut, text_observations)
    if issues:
        info["reason"] = "validation: " + "; ".join(issues[:5])
        return None, info
    info["prefix_tokens_est"] = est
    info["n_messages"] = len(cut)
    return cut, info
