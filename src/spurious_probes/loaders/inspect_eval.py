"""Inspect AI .eval logs: ImpossibleBench SWE-bench runs (condition swebench_inspect), put under data/raw/impossiblebench/.

Only logs whose model is a Claude model are used. The system prompt is dropped
from `messages` for parity with the deployment conditions but kept in meta.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..schema import RAW

SWEBENCH_DIR = RAW / "impossiblebench"


def _tool_text(m) -> str:
    err = getattr(m, "error", None)
    if err is not None and getattr(err, "message", None):
        return f"[error] {err.message}"
    return m.text or ""


def sample_to_messages(sample) -> tuple[list[dict], str | None]:
    msgs: list[dict] = []
    system = None
    for m in sample.messages:
        if m.role == "system":
            system = (system or "") + m.text
        elif m.role == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": m.text}]})
        elif m.role == "assistant":
            blocks: list[dict] = []
            if m.text and m.text.strip():
                blocks.append({"type": "text", "text": m.text})
            for tc in m.tool_calls or []:
                args = tc.arguments if isinstance(tc.arguments, dict) else {"arguments": tc.arguments}
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function, "input": args})
            msgs.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            blk = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": _tool_text(m)}
            if getattr(m, "error", None) is not None:
                blk["is_error"] = True
            msgs.append({"role": "user", "content": [blk]})
    return msgs, system


def iter_samples(log_dir: Path = SWEBENCH_DIR, model_substr: str = "claude") -> Iterator[tuple[Path, list[dict], dict]]:
    from inspect_ai.log import read_eval_log   # optional dependency: uv sync --extra data

    for f in sorted(log_dir.glob("*.eval")):
        header = read_eval_log(str(f), header_only=True)
        if model_substr not in (header.eval.model or ""):
            continue
        log = read_eval_log(str(f))
        for s in log.samples or []:
            msgs, system = sample_to_messages(s)
            if len(msgs) < 4:
                continue
            meta = {"eval_model": header.eval.model, "task": header.eval.task, "sample_id": str(s.id),
                    "epoch": s.epoch, "system": (system or "")[:4000], "log": f.name}
            yield f, msgs, meta
