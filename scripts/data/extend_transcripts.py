"""Add more transcripts to an existing condition without disturbing the ones already there (their order and ids are what every
samples file and every 'seen' split refers to). `extend` rebuilds the condition with a large n under the config seed, drops every id
already present, and appends the first --add new ones; for conditions that need an assistant turn (gpqa) it writes them to a staging
condition instead (--staging), to be filled by scripts/data/gen_assistant_turns.py. `strip-thinking` then derives the
gpqa_nothinking rows from the staged answers (whole visible <thinking>/<think>/<reasoning> blocks removed, the recipe of the
original gpqa_nothinking.jsonl) and appends them.

usage: uv run scripts/data/extend_transcripts.py extend --condition sharegpt --add 60 [--build-n 200]
       uv run scripts/data/extend_transcripts.py extend --condition gpqa --add 60 --staging gpqa_new --also-exclude gpqa_nothinking
       uv run scripts/data/gen_assistant_turns.py --conditions gpqa_new [--dry-run]
       uv run scripts/data/extend_transcripts.py strip-thinking --from gpqa_new --into gpqa_nothinking
"""
from __future__ import annotations

import argparse, importlib.util, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from spurious_probes import prep  # noqa: E402
from spurious_probes.schema import TRANSCRIPTS, read_transcripts, write_transcripts  # noqa: E402

THINK_BLOCK = re.compile(r"<(thinking|think|reasoning)>.*?</\1>\s*", re.S | re.I)
NOTE = "derived from gpqa.jsonl: whole visible <thinking>/<think>/<reasoning> blocks (content included) removed from Opus 5's answers"


def builder():
    spec = importlib.util.spec_from_file_location("bt", ROOT / "scripts" / "data" / "build_transcripts.py"); bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
    return bt


def extend(a) -> None:
    bt = builder(); cfg = bt.CFG
    prep.set_chars_per_token(cfg["chars_per_token"].get(a.condition, cfg["chars_per_token"]["default"]))
    have = set()
    for c in [a.condition] + ([a.staging] if a.staging else []) + [x for x in a.also_exclude.split(",") if x]:
        p = TRANSCRIPTS / f"{c}.jsonl"
        if p.exists():
            have |= {t.id for t in read_transcripts(p)}
    built, reasons = bt.build(a.condition, a.build_n, cfg["seed"])
    new = [t for t in built if t.id not in have][: a.add]
    target = a.staging or a.condition
    for t in new:
        t.condition = target; t.meta["extension"] = f"added {__import__('datetime').date.today()} beyond the original set (extend_transcripts.py)"
    out = TRANSCRIPTS / f"{target}.jsonl"
    cur = read_transcripts(out) if out.exists() else []
    write_transcripts(out, cur + new)
    print(f"{a.condition}: built {len(built)} candidates, {len(built) - len([t for t in built if t.id not in have])} already present, appended {len(new)} -> {out.name} now {len(cur) + len(new)} transcripts"
          + (f" (staging; run gen_assistant_turns.py --conditions {target})" if a.staging else "") + (f"; rejects {dict(reasons.most_common(2))}" if reasons else ""))


def strip_thinking(a) -> None:
    src = read_transcripts(TRANSCRIPTS / f"{a.src}.jsonl"); dst_path = TRANSCRIPTS / f"{a.into}.jsonl"
    dst = read_transcripts(dst_path) if dst_path.exists() else []; have = {t.id for t in dst}
    added, stripped = [], 0
    for t in src:
        if t.id in have or t.meta.get("needs_assistant_turn") or t.meta.get("assistant_gen_failed"):
            continue
        for m in t.messages:
            if m["role"] != "assistant":
                continue
            for b in m["content"]:
                if b["type"] == "text":
                    new = THINK_BLOCK.sub("", b["text"]).strip()
                    if new != b["text"].strip():
                        stripped += 1
                    b["text"] = new
        t.condition = a.into; t.meta.update({"thinking_stripped": True, "note": NOTE}); t.prefix_tokens_est = prep.estimate_tokens(t.messages)
        added.append(t)
    write_transcripts(dst_path, dst + added)
    print(f"{a.src} -> {a.into}: appended {len(added)} ({stripped} answers had thinking blocks removed); {a.into} now {len(dst) + len(added)} transcripts")


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extend"); e.add_argument("--condition", required=True); e.add_argument("--add", type=int, required=True); e.add_argument("--build-n", type=int, default=200)
    e.add_argument("--staging", default=None); e.add_argument("--also-exclude", default="")
    s = sub.add_parser("strip-thinking"); s.add_argument("--from", dest="src", required=True); s.add_argument("--into", required=True)
    a = ap.parse_args(); (extend if a.cmd == "extend" else strip_thinking)(a)


if __name__ == "__main__":
    main()
