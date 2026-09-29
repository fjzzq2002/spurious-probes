"""Do good questions transfer between models? Spearman correlation of question accuracies (best one-answer rule, short answers) between
every pair of models screened on the same 2,426 questions: the five Qwen3.5 sizes and Gemma-2-9B (base and Neural Chameleon, a fine-tune
of the same base). Reads the screen tables in results/.

usage: uv run scripts/evaluate/transfer.py
"""
from __future__ import annotations

import argparse, itertools
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TABLES = {**{f"Qwen3.5-{s.upper()}": f"results/scaling/qwen3.5-{s}/probes.csv" for s in ("0.8b", "2b", "4b", "9b", "27b")},
          "Gemma-2-9B base": "results/chameleon/screen_base/probes.csv", "Neural Chameleon": "results/chameleon/screen_cham/probes.csv"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    acc = {m: pd.read_csv(ROOT / f).set_index("probe_id")["acc_short"] for m, f in TABLES.items()}
    M = pd.DataFrame(index=list(acc), columns=list(acc), dtype=float)
    for a, b in itertools.combinations(acc, 2):
        q = acc[a].index.intersection(acc[b].index); M.loc[a, b] = M.loc[b, a] = spearmanr(acc[a][q], acc[b][q])[0]
    pd.set_option("display.width", 200)
    print("Spearman correlation of short-answer screen accuracy across models (2,426 questions):")
    print(M.round(2).to_string(na_rep=""))


if __name__ == "__main__":
    main()
