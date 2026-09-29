"""Single-turn and chat corpora from Hugging Face: GPQA Diamond and MMLU-Pro (the model writes its own answer, see gen_assistant_turns.py),
WildChat and ShareGPT (real users)."""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

from ..schema import RAW

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# inspect_ai MultipleChoiceTemplate.SINGLE_ANSWER_COT (used by inspect_evals gpqa_diamond, cot=True)
MC_COT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}"""
MMLU_PRO_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}"""


def _choices_block(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))


def gpqa_items(n: int, seed: int) -> list[dict]:
    df = pd.read_csv(RAW / "gpqa" / "gpqa_diamond.csv")
    rng = random.Random(seed)
    rows = df.sample(frac=1.0, random_state=seed).head(n)
    items = []
    for _, r in rows.iterrows():
        opts = [str(r["Correct Answer"]), str(r["Incorrect Answer 1"]), str(r["Incorrect Answer 2"]), str(r["Incorrect Answer 3"])]
        rng.shuffle(opts)
        target = LETTERS[opts.index(str(r["Correct Answer"]))]
        text = MC_COT_TEMPLATE.format(letters="A,B,C,D", question=str(r["Question"]).strip(), choices=_choices_block(opts))
        items.append({"id": f"gpqa_{r['Record ID']}", "user": text, "meta": {"target": target, "dataset": "gpqa_diamond"}})
    return items


def mmlu_pro_items(n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(RAW / "mmlu_pro" / "data" / "test-00000-of-00001.parquet")
    rows = df.sample(frac=1.0, random_state=seed).head(n)
    items = []
    for _, r in rows.iterrows():
        opts = list(r["options"])
        letters = ",".join(LETTERS[: len(opts)])
        text = MMLU_PRO_TEMPLATE.format(letters=letters, question=str(r["question"]).strip(), choices=_choices_block(opts))
        items.append({"id": f"mmlupro_{r['question_id']}", "user": text, "meta": {"target": r["answer"], "subject": r["category"], "dataset": "mmlu_pro"}})
    return items


def wildchat_conversations(n: int, seed: int, max_turns: int = 8) -> list[dict]:
    """allenai/WildChat-1M: real users talking to ChatGPT. English, non-toxic, >= 2 turns."""
    f = sorted(glob.glob(str(RAW / "wildchat" / "**" / "*.parquet"), recursive=True))[0]
    df = pd.read_parquet(f, columns=["conversation_hash", "conversation", "language", "toxic", "turn", "model"])
    df = df[(df["language"] == "English") & (~df["toxic"].astype(bool)) & (df["turn"] >= 2)]
    rows = df.sample(frac=1.0, random_state=seed).head(n * 3)
    convs = []
    for _, r in rows.iterrows():
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in r["conversation"] if m["role"] in ("user", "assistant")][:max_turns]
        convs.append({"id": f"wildchat_{r['conversation_hash'][:12]}", "messages": msgs, "meta": {"dataset": "WildChat-1M", "source_model": r["model"]}})
    return convs


def sharegpt_conversations(n: int, seed: int, max_turns: int = 8) -> list[dict]:
    """theblackcat102/sharegpt-english: user-shared ChatGPT conversations."""
    f = sorted(glob.glob(str(RAW / "sharegpt" / "**" / "*.jsonl"), recursive=True))[0]
    rows = []
    with open(f) as fh:
        for k, line in enumerate(fh):
            if k >= 20000:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(seed)
    rng.shuffle(rows)
    convs = []
    for r in rows:
        conv = r.get("conversations") or r.get("messages") or []
        msgs = []
        for m in conv:
            who = m.get("role") or m.get("from") or m.get("user")
            role = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}.get(who)
            text = m.get("content") or m.get("value") or m.get("text") or ""
            if role in ("user", "assistant") and text.strip():
                msgs.append({"role": role, "content": str(text)})
        if len(msgs) >= 2:
            convs.append({"id": f"sharegpt_{r.get('id', len(convs))}", "messages": msgs[:max_turns], "meta": {"dataset": "sharegpt-english"}})
        if len(convs) >= n * 3:
            break
    return convs
