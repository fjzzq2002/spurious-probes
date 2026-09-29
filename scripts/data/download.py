"""Download the public sources the post uses into data/raw/ (idempotent). Needs a Hugging Face login (`huggingface-cli login`) and
accepting the terms of the gated datasets (GPQA, WildChat).

  gpqa          Idavidrein/gpqa (gpqa_diamond.csv)
  mmlu_pro      TIGER-Lab/MMLU-Pro
  sharegpt      theblackcat102/sharegpt-english
  wildchat      allenai/WildChat-1M (first shard)
  swechat       SALT-NLP/SWE-chat (only the sessions listed in sources/swechat_subset.json)
  kernelbench   Infatoshi/kernelbench-hard-traces, Infatoshi/kernelbench-mega-traces
  petri_audit   auditing-agents/petri-transcripts-top50-llama70b (the *_transcripts_adv_high files)

Not covered here (see the README): SWE-Marathon (scripts/data/fetch_swemarathon.py), SWE-bench Verified mini-SWE-agent runs
(scripts/data/fetch_docent_runs.py), ImpossibleBench inspect logs, Anthropic's agentic-misalignment transcripts, and your own
Claude Code / Codex sessions (read in place from ~/.claude/projects and ~/.codex/sessions).

usage: uv run scripts/data/download.py [--only gpqa,sharegpt,...]
"""
from __future__ import annotations

import argparse, json

from huggingface_hub import hf_hub_download, snapshot_download

from spurious_probes.schema import ROOT, RAW


def gpqa():
    hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", local_dir=RAW / "gpqa")


def mmlu_pro():
    snapshot_download("TIGER-Lab/MMLU-Pro", repo_type="dataset", local_dir=RAW / "mmlu_pro", allow_patterns=["*.parquet"])


def sharegpt():
    snapshot_download("theblackcat102/sharegpt-english", repo_type="dataset", local_dir=RAW / "sharegpt", allow_patterns=["*.jsonl"])


def wildchat():
    hf_hub_download("allenai/WildChat-1M", "data/train-00000-of-00014.parquet", repo_type="dataset", local_dir=RAW / "wildchat")


def swechat():
    subset = json.load(open(ROOT / "sources" / "swechat_subset.json"))
    for m in subset.values():
        hf_hub_download("SALT-NLP/SWE-chat", m["transcript_path"], repo_type="dataset", local_dir=RAW / "swechat")


def kernelbench():
    for ds in ("hard", "mega"):
        snapshot_download(f"Infatoshi/kernelbench-{ds}-traces", repo_type="dataset", local_dir=RAW / "kernelbench" / ds)


def petri_audit():
    snapshot_download("auditing-agents/petri-transcripts-top50-llama70b", repo_type="dataset", local_dir=RAW / "align" / "petri_top50",
                      allow_patterns=["data/*_transcripts_adv_high-*"])


SOURCES = {"gpqa": gpqa, "mmlu_pro": mmlu_pro, "sharegpt": sharegpt, "wildchat": wildchat, "swechat": swechat, "kernelbench": kernelbench, "petri_audit": petri_audit}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=",".join(SOURCES)); a = ap.parse_args()
    for name in a.only.split(","):
        print("downloading", name, flush=True); SOURCES[name]()
