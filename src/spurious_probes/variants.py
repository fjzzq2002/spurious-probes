"""Transcript variants: *_text renders tool calls / tool results as plain text inside the turns, for chat templates without a tool
format (Gemma-2, used for the Neural Chameleon test)."""
from __future__ import annotations

import copy
import json

from .schema import Transcript


def tools_as_text(t: Transcript, condition: str) -> Transcript:
    """Every message becomes one text block: tool_use -> '<tool_call>\\nname({json args})\\n</tool_call>', tool_result ->
    '<tool_result>\\n...\\n</tool_result>' (in the user turn that carried it); the tool stubs are dropped. Roles still alternate, so the
    result renders through templates that only know user/assistant text (Gemma-2 raises on 'tool' and 'system' roles)."""
    out = copy.deepcopy(t)
    for m in out.messages:
        parts = []
        for b in m["content"]:
            if b["type"] == "text":
                parts.append(b["text"])
            elif b["type"] == "tool_use":
                parts.append(f"<tool_call>\n{b['name']}({json.dumps(b['input'], ensure_ascii=False)})\n</tool_call>")
            elif b["type"] == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                parts.append(f"<tool_result>\n{str(c).strip() or '(empty)'}\n</tool_result>")
        m["content"] = [{"type": "text", "text": "\n\n".join(p for p in parts if p and p.strip()) or "(empty)"}]
    out.tools = []
    out.condition = condition
    out.meta = {**t.meta, "ablation": "tools_as_text", "base_condition": t.condition}
    return out
