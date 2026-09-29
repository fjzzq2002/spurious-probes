"""Data records shared by loaders, sampler and analysis."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
TRANSCRIPTS = DATA / "transcripts"
RESULTS = ROOT / "results"


@dataclass
class Transcript:
    """A conversation prefix ready to have a probe appended.

    `messages` is Anthropic Messages-API format, starts with a user turn and ends
    with an assistant turn that contains no tool_use blocks. `tools` are stub
    definitions for every tool name that appears in `messages` (may be empty).
    `system` is only used by the no-context label conditions.
    """
    id: str
    condition: str
    source_path: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    system: str | None = None
    prefix_tokens_est: int = 0
    prefix_tokens: int | None = None  # exact, from count_tokens
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Transcript":
        return cls(**json.loads(line))


def write_transcripts(path: Path, transcripts: list[Transcript]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in transcripts:
            f.write(t.to_json() + "\n")


def read_transcripts(path: Path) -> list[Transcript]:
    with open(path) as f:
        return [Transcript.from_json(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
