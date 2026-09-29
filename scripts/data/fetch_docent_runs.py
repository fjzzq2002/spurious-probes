"""Export a subsample of agent runs from a Docent collection into data/raw/<name>/<run_id>.json
(our simple {"messages": [...], "system": str|None, "meta": {...}} form).

usage: uv run scripts/data/fetch_docent_runs.py --collection b038912e-0133-4594-b093-92806f8ffb17 --name swebench_mini \
           --model-like "%claude-opus-4-5%" --n 70     (the SWE-bench Verified runs in the post)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from docent.sdk.client import Docent

ROOT = Path(__file__).resolve().parents[2]


def content_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts = []
    for p in c:
        t = getattr(p, "text", None) if not isinstance(p, dict) else p.get("text")
        if t:
            parts.append(t)
        elif getattr(p, "type", None) == "image" or (isinstance(p, dict) and p.get("type") == "image"):
            parts.append("[image omitted]")
    return "\n".join(parts)


def tool_call_fields(tc) -> tuple[str, str, dict]:
    g = (lambda k: tc.get(k) if isinstance(tc, dict) else getattr(tc, k, None))
    tid = g("id") or g("tool_call_id") or ""
    name = g("function") or g("name") or "tool"
    args = g("arguments") or g("input") or {}
    if isinstance(name, dict):  # OpenAI-style {"name":..., "arguments":...}
        args = name.get("arguments", args); name = name.get("name", "tool")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"arguments": args}
    if not isinstance(args, dict):
        args = {"arguments": args}
    return str(tid), str(name), args


def convert(run) -> tuple[list[dict], str | None]:
    msgs: list[dict] = []
    system = None
    for tr in run.transcripts:
        for m in tr.messages:
            role = m.role
            if role == "system":
                system = (system or "") + content_text(m.content)
            elif role == "user":
                msgs.append({"role": "user", "content": [{"type": "text", "text": content_text(m.content)}]})
            elif role == "assistant":
                blocks: list[dict] = []
                text = content_text(m.content)
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in getattr(m, "tool_calls", None) or []:
                    tid, name, args = tool_call_fields(tc)
                    blocks.append({"type": "tool_use", "id": tid, "name": name, "input": args})
                msgs.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                err = getattr(m, "error", None)
                text = content_text(m.content)
                if err:
                    text = f"[error] {err}\n{text}" if text else f"[error] {err}"
                blk = {"type": "tool_result", "tool_use_id": str(getattr(m, "tool_call_id", "") or ""), "content": text}
                if err:
                    blk["is_error"] = True
                msgs.append({"role": "user", "content": [blk]})
    return msgs, system


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--model-like", required=True); ap.add_argument("--n", type=int, default=70); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-path", default="metadata_json->>'model_name'", help="DQL expression for the model name")
    ap.add_argument("--resolved-path", default="metadata_json->>'resolved'"); ap.add_argument("--turns-path", default="metadata_json->>'turns'")
    a = ap.parse_args()
    client = Docent()
    res = client.execute_dql(a.collection, f"SELECT id, {a.model_path} AS model, metadata_json->>'instance_id' AS instance, "
                                           f"{a.resolved_path} AS resolved, {a.turns_path} AS turns "
                                           f"FROM agent_runs WHERE {a.model_path} LIKE '{a.model_like}' LIMIT 2000")
    rows = client.dql_result_to_dicts(res)
    print(f"{len(rows)} runs match model_like={a.model_like!r}", file=sys.stderr)
    random.Random(a.seed).shuffle(rows)
    out = ROOT / "data" / "raw" / a.name
    out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for r in rows[: a.n]:
        run = client.get_agent_run(a.collection, r["id"])
        if run is None:
            continue
        msgs, system = convert(run)
        json.dump({"messages": msgs, "system": system, "meta": {"docent_collection": a.collection, "agent_run_id": r["id"], "model_name": r["model"],
                                                               "instance_id": r["instance"], "resolved": r["resolved"], "turns": r["turns"]}},
                  open(out / f"{r['id']}.json", "w"))
        n_ok += 1
    print(f"wrote {n_ok} runs to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
