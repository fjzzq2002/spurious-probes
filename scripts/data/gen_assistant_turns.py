"""Generate the assistant turn for single-turn transcripts (gpqa / mmlu_pro / mwe_airisk / filler)
with one model (--model; the post used Claude Opus 5, thinking disabled). The same answer turn is then shared by every model
that is probed on these transcripts.

usage: uv run scripts/data/gen_assistant_turns.py [--conditions gpqa,mwe_airisk,filler] [--dry-run] [--n N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from spurious_probes import client as api, prep  # noqa: E402
from spurious_probes.schema import ROOT, TRANSCRIPTS, read_transcripts, write_transcripts  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
MAX_TOKENS = {"gpqa": 6000, "mmlu_pro": 6000, "mwe_airisk": 1500, "filler": 1024, "moralchoice": 200, "dailydilemmas": 400, "bbq": 300, "stereoset": 300, "sycophancy": 800, "dolci": 2000}
ASSUMED_OUT = {"gpqa": 900, "mmlu_pro": 900, "mwe_airisk": 60, "filler": 40, "moralchoice": 20, "dailydilemmas": 60, "bbq": 20, "stereoset": 20, "sycophancy": 250, "dolci": 500}
BACKEND = {"name": "anthropic", "model": None}


async def gen(condition: str, n: int | None, dry: bool, concurrency: int) -> None:
    path = TRANSCRIPTS / f"{condition}.jsonl"
    ts = read_transcripts(path)
    todo = [t for t in ts if t.meta.get("needs_assistant_turn")]
    if n:
        todo = todo[:n]
    model = BACKEND["model"] or CFG["model"]   # --model applies to both backends (before 2026-09-22 the Anthropic path silently used config.yaml's model)
    if dry:
        if BACKEND["name"] == "openai":
            from spurious_probes import client_openrouter as orapi
            p_in, _, p_out = orapi.OR_PRICES[model]
        else:
            p_in, _, _, p_out = api.PRICES[model]
        usd = sum((t.prefix_tokens_est * p_in + ASSUMED_OUT.get(condition, ASSUMED_OUT.get(condition.split('_')[0], 400)) * p_out) / 1e6 for t in todo)
        print(f"{condition}: {len(todo)} assistant turns to generate, projected ≈ ${usd:.2f}", file=sys.stderr)
        return
    sem = asyncio.Semaphore(concurrency)
    spend = api.Spend()
    if BACKEND["name"] == "openai":   # one asyncio loop per condition: rebuild the async client each time
        from spurious_probes import client_openrouter as orapi
        orapi._client = None

    async def one(t) -> None:
        if BACKEND["name"] == "openai":
            from spurious_probes import client_openrouter as orapi
            orapi.PROVIDER = "openai"
            async with sem:
                r = await orapi.create(BACKEND["model"], orapi.to_openai_messages(t.messages, None), [], MAX_TOKENS.get(condition, MAX_TOKENS.get(condition.split('_')[0], 400)), "none")
            text = r["text"].strip(); stop = {"stop": "end_turn", "length": "max_tokens"}.get(r["finish_reason"], r["finish_reason"])
            spend.usd += orapi.usage_cost(BACKEND["model"], r["usage"]); spend.calls += 1
            if stop != "end_turn" or not text:
                t.meta["assistant_gen_failed"] = stop
                return
            t.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            t.meta["needs_assistant_turn"] = False; t.meta["assistant_stop_reason"] = stop; t.meta["assistant_model"] = BACKEND["model"]
            t.prefix_tokens_est = prep.estimate_tokens(t.messages)
            return
        async with sem:
            msg = await api.create_message(model=model, max_tokens=MAX_TOKENS.get(condition, MAX_TOKENS.get(condition.split('_')[0], 400)), messages=t.messages,
                                           thinking={"type": "disabled"})
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        spend.add(model, msg.usage.model_dump())
        if msg.stop_reason != "end_turn" or not text:
            t.meta["assistant_gen_failed"] = msg.stop_reason
            return
        t.meta["assistant_model"] = model
        t.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        t.meta["needs_assistant_turn"] = False
        t.meta["assistant_stop_reason"] = msg.stop_reason
        t.prefix_tokens_est = prep.estimate_tokens(t.messages)

    await asyncio.gather(*(one(t) for t in todo))
    keep = [t for t in ts if not t.meta.get("needs_assistant_turn") and not t.meta.get("assistant_gen_failed")]
    dropped = len(ts) - len(keep)
    write_transcripts(path, keep)
    print(f"{condition}: generated {len(todo)} turns, dropped {dropped} (truncated/empty), ${spend.usd:.2f}, usage={spend.usage}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="gpqa,mwe_airisk,filler")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=CFG["concurrency"])
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"]); ap.add_argument("--model", default=None)
    args = ap.parse_args()
    BACKEND.update({"name": args.backend, "model": args.model or CFG["model"]})
    for c in args.conditions.split(","):
        if (TRANSCRIPTS / f"{c}.jsonl").exists():
            asyncio.run(gen(c, args.n, args.dry_run, args.concurrency))


if __name__ == "__main__":
    main()
