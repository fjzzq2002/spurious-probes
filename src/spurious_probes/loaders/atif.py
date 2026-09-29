"""Terminal-Bench 2.0 leaderboard trajectories (ATIF v1.6) from the Terminus 2 agent.

Terminus 2 is text-based: the harness instructions are the first *user* message, the
model replies with a JSON object {analysis, plan, commands, task_complete}, and the
terminal output comes back as the next user message. ATIF stores the parsed pieces
(message = "Analysis: ...\nPlan: ...", tool_calls = commands, observation = terminal
output), so we reconstruct the JSON the model emitted and render everything as text.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Iterator

from ..schema import RAW

TBENCH_DIR = RAW / "tbench"
AP_RE = re.compile(r"^Analysis:\s*(.*?)\s*(?:\nPlan:\s*(.*))?$", re.S)


def trajectory_files(root: Path = TBENCH_DIR) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "**" / "trajectory.json"), recursive=True))


def agent_step_text(step: dict, is_last: bool) -> str:
    msg = step.get("message") or ""
    m = AP_RE.match(msg.strip()) if isinstance(msg, str) else None
    analysis, plan = (m.group(1), m.group(2) or "") if m else (msg, "")
    commands = []
    for tc in step.get("tool_calls") or []:
        args = tc.get("arguments") or {}
        cmd = {"keystrokes": args.get("keystrokes", "")}
        if "duration" in args:
            cmd["duration"] = args["duration"]
        commands.append(cmd)
    obj = {"analysis": analysis, "plan": plan, "commands": commands, "task_complete": bool(is_last and not commands)}
    return json.dumps(obj, indent=2, ensure_ascii=False)


def to_messages_tools(traj: dict) -> list[dict]:
    """ATIF trace from a tool-calling agent (e.g. Claude Code): agent steps -> assistant text + tool_use blocks,
    observations -> tool_result blocks keyed by source_call_id."""
    msgs: list[dict] = []
    for s in traj["steps"]:
        if s["source"] == "user":
            msgs.append({"role": "user", "content": str(s.get("message") or "")})
        elif s["source"] == "agent":
            blocks: list[dict] = []
            if isinstance(s.get("message"), str) and s["message"].strip():
                blocks.append({"type": "text", "text": s["message"]})
            for tc in s.get("tool_calls") or []:
                args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {"input": tc.get("arguments")}
                blocks.append({"type": "tool_use", "id": str(tc.get("tool_call_id")), "name": str(tc.get("function_name") or "tool"), "input": args})
            if blocks:
                msgs.append({"role": "assistant", "content": blocks})
            results = (s.get("observation") or {}).get("results") or []
            if results:
                msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(r.get("source_call_id")), "content": str(r.get("content") or "")}
                                                          for r in results]})
        elif s["source"] == "tool":
            msgs.append({"role": "user", "content": str(s.get("message") or "")})
    return msgs


def to_messages(traj: dict) -> list[dict]:
    steps = traj["steps"]
    msgs: list[dict] = []
    agent_idx = [i for i, s in enumerate(steps) if s["source"] == "agent"]
    last_agent = agent_idx[-1] if agent_idx else -1
    for i, s in enumerate(steps):
        if s["source"] == "user":
            msgs.append({"role": "user", "content": str(s.get("message") or "")})
        elif s["source"] == "agent":
            msgs.append({"role": "assistant", "content": agent_step_text(s, i == last_agent)})
            obs = s.get("observation") or {}
            texts = [r.get("content", "") for r in obs.get("results") or [] if r.get("content")]
            if texts:
                msgs.append({"role": "user", "content": "\n".join(texts)})
        elif s["source"] == "tool":
            msgs.append({"role": "user", "content": str(s.get("message") or "")})
    return msgs


def iter_trajectories(root: Path = TBENCH_DIR, tool_mode: bool = False) -> Iterator[tuple[Path, list[dict], dict]]:
    for f in trajectory_files(root):
        traj = json.load(open(f))
        msgs = to_messages_tools(traj) if tool_mode else to_messages(traj)
        if len(msgs) < 4:
            continue
        trial = f.parent.parent.name
        result = f.parent.parent / "result.json"
        reward = None
        if result.exists():
            try:
                r = json.load(open(result))
                reward = (r.get("verifier_result") or {}).get("rewards") or r.get("reward")
            except Exception:
                pass
        meta = {"agent": traj.get("agent", {}).get("name"), "agent_model": traj.get("agent", {}).get("model_name"),
                "trial": trial, "task": trial.rsplit("__", 1)[0], "n_steps": len(traj["steps"]), "reward": reward,
                "text_observations": not tool_mode}
        yield f, msgs, meta
