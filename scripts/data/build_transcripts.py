"""Build data/transcripts/<condition>.jsonl: transcript prefixes cut at a turn boundary inside the token window (config.yaml).

Sources (put the raw data under data/raw/, see the README):
  real use:     sharegpt, wildchat, swechat (SWE-chat, Claude Code sessions), claude_code / codex (YOUR local sessions)
  capability:   gpqa, mmlu_pro (single turn; add the model's own answer with gen_assistant_turns.py), swebench_mini (SWE-bench Verified,
                mini-SWE-agent), swemarathon, swebench_inspect (ImpossibleBench inspect logs), kernelbench
  alignment:    agentic_misalignment, petri_audit
  no context:   none
  <source>_text: the same transcripts with tool calls rendered as text (for chat templates without a tool format, e.g. Gemma-2)

usage: uv run scripts/data/build_transcripts.py [--conditions a,b,c] [--n N] [--dry-run]
"""
from __future__ import annotations

import argparse
import itertools
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import yaml
from spurious_probes import prep  # noqa: E402
from spurious_probes.schema import ROOT, RAW, TRANSCRIPTS, Transcript, write_transcripts  # noqa: E402
from spurious_probes.loaders import claude_code, codex, inspect_eval, atif, corpora, nocontext, rawjson, kernelbench, swechat, petri, agentic_misalignment  # noqa: E402
from spurious_probes.variants import tools_as_text  # noqa: E402
from spurious_probes.schema import read_transcripts  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
LO, HI = CFG["token_window"]


def _from_iter(condition: str, it, n: int, seed: int, policy: str, text_observations: bool = False,
               oversample: int = 4) -> tuple[list[Transcript], Counter]:
    """Run candidates through prep.finalize, keep the first n usable in a seeded random order."""
    cands = list(it)
    random.Random(seed).shuffle(cands)
    out: list[Transcript] = []
    reasons: Counter = Counter()
    for f, msgs, meta in cands:
        cut, info = prep.finalize(msgs, LO, HI, policy=policy, text_observations=text_observations)
        if cut is None:
            reasons[info["reason"][:50]] += 1
            continue
        meta = dict(meta)
        meta.update({"cut_kind": info["cut_kind"], "n_messages": info["n_messages"], "n_messages_full": info["n_messages_full"],
                     "repairs": info["repairs"]})
        tools = [] if text_observations else prep.tool_stubs(cut)
        tid = meta.get("session") or meta.get("trial") or meta.get("sample_id") or Path(f).stem
        if "eval_model" in meta:
            tid = f"{meta['eval_model'].split('/')[-1]}__{meta['sample_id']}__e{meta.get('epoch', 1)}"
        out.append(Transcript(id=str(tid), condition=condition, source_path=str(f), messages=cut, tools=tools,
                              system=None, prefix_tokens_est=info["prefix_tokens_est"], meta=meta))
        if len(out) >= n:
            break
    return out, reasons


def _from_items(condition: str, items: list[dict], needs_assistant: bool) -> list[Transcript]:
    out = []
    for it in items:
        msgs = [{"role": "user", "content": [{"type": "text", "text": it["user"]}]}]
        meta = dict(it["meta"])
        meta.update({"needs_assistant_turn": needs_assistant, "cut_kind": "text"})
        out.append(Transcript(id=it["id"], condition=condition, source_path="", messages=msgs, tools=[], system=None,
                              prefix_tokens_est=prep.estimate_tokens(msgs), meta=meta))
    return out


def build(condition: str, n: int, seed: int) -> tuple[list[Transcript], Counter]:
    if condition.endswith("_text"):   # tool turns as text, built from the base condition's file
        base = condition[: -len("_text")]
        src = read_transcripts(TRANSCRIPTS / f"{base}.jsonl")
        return [tools_as_text(t, condition) for t in src[:n]], Counter()
    if condition == "claude_code":
        return _from_iter(condition, claude_code.iter_sessions(), n, seed, policy="prefer_text")
    if condition == "codex":
        return _from_iter(condition, codex.iter_sessions(), n, seed, policy="prefer_text")
    if condition == "swebench_inspect":
        return _from_iter(condition, inspect_eval.iter_samples(), n, seed, policy="prefer_text")
    if condition == "gpqa":
        return _from_items(condition, corpora.gpqa_items(n, seed), True), Counter()
    if condition == "mmlu_pro":
        return _from_items(condition, corpora.mmlu_pro_items(n, seed), True), Counter()
    if condition == "swebench_mini":
        return _from_iter(condition, rawjson.iter_runs("swebench_mini"), n, seed, policy="prefer_text")
    if condition == "agentic_misalignment":
        return _from_iter(condition, agentic_misalignment.iter_transcripts(), n, seed, policy="prefer_text")
    if condition == "petri_audit":
        return _from_iter(condition, petri.iter_transcripts(), n, seed, policy="prefer_text")
    if condition == "swechat":
        return _from_iter(condition, swechat.iter_sessions("Claude Code"), n, seed, policy="prefer_text")
    if condition == "kernelbench":
        return _from_iter(condition, kernelbench.iter_sessions("claude"), n, seed, policy="prefer_text")
    if condition == "swemarathon":
        it = ((f, m, {**meta, "task": f.parent.parent.name, "trial": f.parent.name}) for f, m, meta in atif.iter_trajectories(root=RAW / "swemarathon", tool_mode=True))
        return _from_iter(condition, it, n, seed, policy="prefer_text")
    if condition in ("wildchat", "sharegpt"):
        convs = corpora.wildchat_conversations(n, seed) if condition == "wildchat" else corpora.sharegpt_conversations(n, seed)
        return _from_iter(condition, ((c["id"], c["messages"], c["meta"]) for c in convs), n, seed, policy="text")
    if condition == "none":
        return nocontext.build(condition, CFG["nocontext_samples"][condition]), Counter()
    raise ValueError(condition)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="all", help="comma list, or 'all' = every public source in config.yaml (your own "
                    "claude_code / codex sessions are never included unless you list them)")
    ap.add_argument("--n", type=int, default=None, help="override n_transcripts for every condition")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    own = {"claude_code", "codex"}
    conds = [c for c in list(CFG["n_transcripts"]) + list(CFG["nocontext_samples"]) if c not in own] if args.conditions == "all" else args.conditions.split(",")
    if own & set(conds):
        print("NOTE: building transcripts from YOUR OWN sessions (~/.claude/projects, ~/.codex/sessions). Sampling answers on them sends "
              "their content to the model provider you choose.", flush=True)
    for c in conds:
        n = args.n or CFG["n_transcripts"].get(c.removesuffix("_text"), 1)   # *_text: the same transcripts with tool calls as text
        prep.set_chars_per_token(CFG["chars_per_token"].get(c, CFG["chars_per_token"]["default"]))
        try:
            ts, reasons = build(c, n, CFG["seed"])
        except (FileNotFoundError, ImportError) as e:
            print(f"== {c}: skipped ({type(e).__name__}: {e}); see the Sources table in the README", flush=True); continue
        est = [t.prefix_tokens_est for t in ts]
        kinds = Counter(t.meta.get("cut_kind") for t in ts)
        tools = Counter(s["name"] for t in ts for s in t.tools)
        print(f"== {c}: {len(ts)} transcripts | est prefix tokens median {st.median(est) if est else 0:.0f} "
              f"(min {min(est) if est else 0}, max {max(est) if est else 0}) | cut kinds {dict(kinds)} | rejects {dict(reasons.most_common(3))}")
        if tools:
            print(f"   tools: {dict(tools.most_common(8))}")
        if not args.dry_run:
            out = TRANSCRIPTS / f"{ts[0].condition if ts else c}.jsonl"
            write_transcripts(out, ts)
            print(f"   -> {out}")


if __name__ == "__main__":
    main()
