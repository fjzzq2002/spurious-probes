"""Anthropic client, retry loop, price table and usage accounting."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv

from .schema import ROOT

load_dotenv(ROOT / ".env")

# $ per million tokens: input, cache write (5m), cache read, output
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-4-8": (5.0, 6.25, 0.50, 25.0),
    "claude-sonnet-5": (2.0, 2.50, 0.20, 10.0),
    "claude-haiku-4-5": (1.0, 1.25, 0.10, 5.0),
}

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set (expected in .env)")
        _client = anthropic.AsyncAnthropic(max_retries=4, timeout=120.0)
    return _client


def usage_cost(model: str, usage: dict[str, Any]) -> float:
    p_in, p_w, p_r, p_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (usage.get("input_tokens", 0) * p_in
            + usage.get("cache_creation_input_tokens", 0) * p_w
            + usage.get("cache_read_input_tokens", 0) * p_r
            + usage.get("output_tokens", 0) * p_out) / 1e6


CACHE_MIN = {"claude-opus-5": 512, "claude-sonnet-5": 1024, "claude-opus-4-8": 1024, "claude-haiku-4-5": 4096}


def project_cost(model: str, prefix_tokens: int, n_calls: int, probe_tokens: int = 40, out_tokens: int = 12) -> float:
    """One cache write of the prefix, then n_calls-1 cache reads; every call pays probe + output."""
    p_in, p_w, p_r, p_out = PRICES.get(model, PRICES["claude-opus-5"])
    cacheable = prefix_tokens >= CACHE_MIN.get(model, 1024)
    if n_calls <= 0:
        return 0.0
    if cacheable:
        prefix_cost = prefix_tokens * p_w + (n_calls - 1) * prefix_tokens * p_r
    else:
        prefix_cost = n_calls * prefix_tokens * p_in
    return (prefix_cost + n_calls * probe_tokens * p_in + n_calls * out_tokens * p_out) / 1e6


@dataclass
class Spend:
    calls: int = 0
    usd: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "cache_creation_input_tokens": 0,
                                                            "cache_read_input_tokens": 0, "output_tokens": 0})

    def add(self, model: str, usage: dict[str, Any]) -> None:
        self.calls += 1
        self.usd += usage_cost(model, usage)
        for k in self.usage:
            self.usage[k] += usage.get(k, 0) or 0


RETRY_DELAYS = [10, 30, 120]


async def create_message(**kwargs) -> anthropic.types.Message:
    """messages.create with the SDK's own retries plus a slow outer loop for connection/rate-limit errors."""
    client = get_client()
    last: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            print(f"  retry {attempt}/{len(RETRY_DELAYS)} after {delay}s: {type(last).__name__}", file=sys.stderr, flush=True)
            await asyncio.sleep(delay)
        try:
            return await client.messages.create(**kwargs)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError,
                anthropic.InternalServerError) as e:
            last = e
    assert last is not None
    raise last


async def count_tokens(**kwargs) -> int:
    client = get_client()
    r = await client.messages.count_tokens(**kwargs)
    return r.input_tokens
