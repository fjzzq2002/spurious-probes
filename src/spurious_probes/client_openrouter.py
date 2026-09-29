"""OpenRouter backend (OpenAI-compatible chat completions) for probe sampling.
Converts our Anthropic-style message blocks (text / tool_use / tool_result) to OpenAI chat format."""
from __future__ import annotations

import asyncio
import re
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from .schema import ROOT

load_dotenv(ROOT / ".env")

# $ per million tokens: input, cached input (OpenAI auto prompt caching ≈ 50% off), output
OR_PRICES: dict[str, tuple[float, float, float]] = {
    "openai/gpt-5.6-luna": (0.20, 0.10, 1.20),
    "openai/gpt-5.6-sol": (2.00, 0.20, 10.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),      # direct OpenAI
    "gpt-5.6-sol": (2.00, 0.20, 10.00),
    "gpt-4o": (2.50, 1.25, 10.00),           # direct OpenAI, non-reasoning (reasoning_effort must not be sent)
    "gemini-3-flash-preview": (0.50, 0.05, 3.00),   # direct Gemini API (PROVIDER "google", GEMINI_API_KEY); OpenRouter stalls at high concurrency for it
}
_client: AsyncOpenAI | None = None


PROVIDER = "openrouter"   # "openai" (direct, OPENAI_API_KEY) or "vllm" (local OpenAI-compatible server at VLLM_BASE_URL)
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
TOP_LOGPROBS: int | None = None   # vllm: also record the top-k logprobs of the first generated token
LOGIT_BIAS: dict[str, float] = {}   # vllm: e.g. ban the <tool_call> token to emulate tool_choice=none


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if PROVIDER == "openai":
            _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=4, timeout=120.0)
        elif PROVIDER == "vllm":
            _client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL, max_retries=4, timeout=900.0)
        else:
            _client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", max_retries=4, timeout=120.0,
                                  default_headers={"X-Title": "spurious-probes"})
    return _client


def to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m["role"] == "user":
            texts = [b["text"] for b in m["content"] if b["type"] == "text"]
            results = [b for b in m["content"] if b["type"] == "tool_result"]
            for r in results:  # tool results must directly follow the assistant tool_calls
                c = r["content"]
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": str(c) or "(empty)"})
            if texts:
                out.append({"role": "user", "content": "\n\n".join(texts)})
        elif m["role"] == "assistant":
            text = "\n\n".join(b["text"] for b in m["content"] if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in m["content"] if b["type"] == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {"type": "object"})}} for t in tools]


def usage_cost(model: str, usage: dict[str, Any]) -> float:
    if PROVIDER == "vllm":
        return 0.0
    p_in, p_cached, p_out = OR_PRICES.get(model, (0.2, 0.1, 1.2))
    cached = usage.get("cached_tokens", 0) or 0
    return ((usage.get("prompt_tokens", 0) - cached) * p_in + cached * p_cached + usage.get("completion_tokens", 0) * p_out) / 1e6


RETRY_DELAYS = [10, 30, 120]


def add_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """OpenAI explicit prompt caching: mark the end of the transcript, i.e. the last content block of the message just
    before the final (probe) message, so requests that differ only in the probe share the cached prefix. With implicit
    caching the breakpoint sits at the end of the probe message itself and nothing is ever reused across probes."""
    if len(messages) < 2 or messages[-2].get("role") in ("system", "developer"):
        return messages
    msgs = list(messages); m = dict(msgs[-2]); c = m.get("content")
    if isinstance(c, str) and c:
        m["content"] = [{"type": "text", "text": c, "prompt_cache_breakpoint": {"mode": "explicit"}}]
    elif isinstance(c, list) and c and c[-1].get("type") == "text":
        blocks = [dict(b) for b in c]; blocks[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}; m["content"] = blocks
    else:
        return messages
    msgs[-2] = m
    return msgs


_google_client = None


def _google() -> "httpx.AsyncClient":
    """Direct Gemini API client (generateContent); thinking off via thinkingBudget 0, vanilla sampling pinned per request."""
    global _google_client
    import httpx
    if _google_client is None:
        _google_client = httpx.AsyncClient(base_url="https://generativelanguage.googleapis.com/v1beta", timeout=180.0,
                                           headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
                                           limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
    return _google_client


async def _create_google(model: str, messages: list[dict], max_tokens: int) -> dict[str, Any]:
    import json as _json
    import httpx
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages if m["role"] in ("user", "assistant")]
    body: dict[str, Any] = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1.0, "topP": 1.0, "thinkingConfig": {"thinkingBudget": 0}}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    last: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            print(f"  retry {attempt}/{len(RETRY_DELAYS)} after {delay}s: {type(last).__name__}: {str(last)[:100]}", file=sys.stderr, flush=True)
            await asyncio.sleep(delay)
        try:
            resp = await _google().post(f"/models/{model}:generateContent", content=_json.dumps(body))
            if resp.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}"); continue
            resp.raise_for_status(); d = resp.json()
            cand = (d.get("candidates") or [{}])[0]; um = d.get("usageMetadata", {})
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            reason = cand.get("finishReason", "?")
            finish = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "refusal", "PROHIBITED_CONTENT": "refusal", "BLOCKLIST": "refusal"}.get(reason, reason.lower())
            if not cand and d.get("promptFeedback", {}).get("blockReason"):
                finish = "refusal"
            return {"text": text, "finish_reason": finish, "tool_calls": False,
                    "usage": {"prompt_tokens": um.get("promptTokenCount", 0), "cached_tokens": um.get("cachedContentTokenCount", 0) or 0, "completion_tokens": um.get("candidatesTokenCount", 0)},
                    "id": d.get("responseId")}
        except httpx.TransportError as e:
            last = e
    raise last  # type: ignore[misc]


async def create(model: str, messages: list[dict], tools: list[dict], max_tokens: int, reasoning_effort: str | None = "none") -> dict[str, Any]:
    if PROVIDER == "google":
        return await _create_google(model, messages, max_tokens)
    client = get_client()
    if PROVIDER == "openai":
        messages = add_cache_breakpoint(messages)
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "extra_body": {}}
    if PROVIDER == "openai" and not re.match(r"gpt-(3|4)", model):
        kwargs["extra_body"]["prompt_cache_options"] = {"mode": "explicit"}   # gpt-4o rejects the option (400)
    kwargs["max_completion_tokens" if PROVIDER == "openai" else "max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools; kwargs["tool_choice"] = "none"
    if PROVIDER == "vllm":
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}   # Qwen3-style hybrid thinking: off
        # vanilla sampling, pinned explicitly (checkpoints ship generation_config.json with T=0.6/top_k=20/top_p=0.95, which vLLM would apply)
        kwargs["temperature"] = 1.0; kwargs["top_p"] = 1.0; kwargs["presence_penalty"] = 0.0; kwargs["frequency_penalty"] = 0.0
        kwargs["extra_body"].update({"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0})
        if LOGIT_BIAS:
            kwargs["logit_bias"] = LOGIT_BIAS
        if TOP_LOGPROBS:
            kwargs["logprobs"] = True; kwargs["top_logprobs"] = TOP_LOGPROBS
        reasoning_effort = None
    if reasoning_effort and PROVIDER == "openai" and re.match(r"gpt-(3|4)", model):
        reasoning_effort = None   # non-reasoning models reject the parameter
    if reasoning_effort:
        if PROVIDER == "openai":
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["extra_body"]["reasoning"] = {"effort": reasoning_effort}
    last: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            print(f"  retry {attempt}/{len(RETRY_DELAYS)} after {delay}s: {type(last).__name__}", file=sys.stderr, flush=True)
            await asyncio.sleep(delay)
        try:
            r = await client.chat.completions.create(**kwargs)
            ch = r.choices[0]
            u = r.usage.model_dump() if r.usage else {}
            cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
            finish = "refusal" if (ch.finish_reason == "content_filter" or getattr(ch.message, "refusal", None)) else ch.finish_reason
            out = {"text": ch.message.content or "", "finish_reason": finish, "tool_calls": bool(ch.message.tool_calls),
                   "usage": {"prompt_tokens": u.get("prompt_tokens", 0), "cached_tokens": cached, "completion_tokens": u.get("completion_tokens", 0)},
                   "id": r.id}
            if TOP_LOGPROBS and ch.logprobs and ch.logprobs.content:
                out["top_logprobs"] = [[t.token, round(t.logprob, 4)] for t in ch.logprobs.content[0].top_logprobs]
            return out
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
            last = e
    assert last is not None
    raise last
