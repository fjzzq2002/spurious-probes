"""Subsample SWE-Marathon trajectories from the public logs bucket (Harbor/ATIF layout).

Reads AWS credentials from the environment / project .env (never prints them).

usage:
  uv run scripts/data/fetch_swemarathon.py list                  # per (task, agent, model) trial counts under v1.1
  uv run scripts/data/fetch_swemarathon.py fetch --n 60 [--model-substr claude] [--seed 7]
                                                            # downloads agent/trajectory.json + config.json for a spread of trials
Output: data/raw/swemarathon/<task>/<trial>/{trajectory.json,config.json}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
import boto3  # noqa: E402

BUCKET = "ralphbench-logs"
PREFIX = "v1.1/"
OUT = ROOT / "data" / "raw" / "swemarathon"


def s3():
    return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))


def list_trajectories(client) -> list[str]:
    """All keys ending in agent/trajectory.json under the suite prefix, skipping trash/."""
    keys = []
    pag = client.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith("agent/trajectory.json") and "/trash/" not in k and not k.startswith("trash/"):
                keys.append(k)
    return keys


def trial_config(client, traj_key: str) -> dict:
    cfg_key = traj_key[: -len("agent/trajectory.json")] + "config.json"
    try:
        return json.loads(client.get_object(Bucket=BUCKET, Key=cfg_key)["Body"].read())
    except Exception:
        return {}


def agent_model(cfg: dict) -> tuple[str, str]:
    a = cfg.get("agent") or {}
    if isinstance(a, dict):
        return str(a.get("name") or a.get("import_path") or "?"), str(a.get("model_name") or a.get("model") or "?")
    return str(a), str(cfg.get("model_name") or cfg.get("model") or "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=60)
    f.add_argument("--model-substr", default="claude")
    f.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    client = s3()
    keys = list_trajectories(client)
    print(f"{len(keys)} trajectories under s3://{BUCKET}/{PREFIX}", file=sys.stderr)
    by_task: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_task[k[len(PREFIX):].split("/")[0]].append(k)
    print(f"{len(by_task)} tasks; trials per task: median {sorted(len(v) for v in by_task.values())[len(by_task)//2]}", file=sys.stderr)

    # read configs for a bounded sample of trials per task to learn agent/model
    rng = random.Random(args.seed if args.cmd == "fetch" else 0)
    sampled = []
    for task, ks in sorted(by_task.items()):
        ks = sorted(ks)
        rng.shuffle(ks)
        sampled.extend(ks[:25])
    info = {k: agent_model(trial_config(client, k)) for k in sampled}
    counts = Counter((k[len(PREFIX):].split("/")[0], *info[k]) for k in sampled)
    if args.cmd == "list":
        print("task | agent | model | sampled trials")
        for (task, agent, model), n in sorted(counts.items()):
            print(f"{task} | {agent} | {model} | {n}")
        print("\nmodels seen:", Counter(m for _, m in info.values()).most_common(12))
        return

    # fetch: spread across tasks, prefer the requested model substring
    pref = [k for k in sampled if args.model_substr.lower() in info[k][1].lower()] or sampled
    rng.shuffle(pref)
    per_task: dict[str, int] = Counter()
    chosen = []
    for k in pref:
        task = k[len(PREFIX):].split("/")[0]
        if per_task[task] >= 3:
            continue
        chosen.append(k); per_task[task] += 1
        if len(chosen) >= args.n:
            break
    print(f"fetching {len(chosen)} trials across {len(per_task)} tasks (model filter '{args.model_substr}')", file=sys.stderr)
    for k in chosen:
        task, trial = k[len(PREFIX):].split("/")[0], k.split("/")[-3]
        d = OUT / task / trial
        d.mkdir(parents=True, exist_ok=True)
        client.download_file(BUCKET, k, str(d / "trajectory.json"))
        with open(d / "config.json", "w") as fh:
            json.dump(trial_config(client, k), fh)
    print(f"done -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
