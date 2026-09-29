"""Brainstorm categories ("name one member of a category") with many cheap models via OpenRouter, using the brief in brief.md; two
sequential batches per model, the second told to avoid the first. Needs OPENROUTER_API_KEY.

usage: uv run scripts/questions/gen_categories.py      (writes data/brainstorm/cat_<model>_<1|2>.md; then run merge_categories.py)
"""
import argparse, json, os, re, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]; load_dotenv(ROOT / ".env")
def headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
MODELS = ["qwen/qwen3.8-flash", "qwen/qwen3.7-plus", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro-0813", "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.7-code",
          "meituan/longcat-2.0", "mistralai/mistral-small-2603", "mistralai/mistral-large-2512", "openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-5.4-mini", "openai/gpt-5.6-luna",
          "google/gemini-3.8-flash", "google/gemma-4-31b-it", "z-ai/glm-5.3-flash", "z-ai/glm-5", "minimax/minimax-m3", "nvidia/nemotron-3-super-120b-a12b", "xiaomi/mimo-v2.5",
          "tencent/hy3", "stepfun/step-3.7-flash", "bytedance-seed/seed-2.0-lite", "meta-llama/llama-4-maverick", "inception/mercury-2.5", "cohere/command-r-08-2024"]
brief = open(Path(__file__).with_name("brief.md")).read()
OUT = ROOT / "data" / "brainstorm"; OUT.mkdir(parents=True, exist_ok=True)
LINE = re.compile(r"^\s*\d+\.\s*[^|\n]+\|[^|\n]+\|", re.M)
def ask(model, prompt):
    for reasoning_off in (True, False):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 12000, "temperature": 1.0}
        if reasoning_off: payload["reasoning"] = {"enabled": False}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", headers=headers(), data=json.dumps(payload).encode())
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900)); msg = r["choices"][0]["message"]
            text = msg.get("content") or ""
            return text if LINE.search(text) else (msg.get("reasoning") or text)
        except urllib.error.HTTPError as e:
            if reasoning_off and e.code in (400, 404, 422): continue
            raise
    return ""
def one(model):
    tag = model.replace("/", "_").replace(":", "_"); res = []
    try:
        t1 = ask(model, brief.format(avoid="")); (OUT / f"cat_{tag}_1.md").write_text(t1); res.append(len(LINE.findall(t1)))
        prev = "\n".join(l.strip() for l in t1.splitlines() if LINE.match(l))[:20000]
        t2 = ask(model, brief.format(avoid=" Do not repeat any of these categories you listed before:\n" + prev)); (OUT / f"cat_{tag}_2.md").write_text(t2); res.append(len(LINE.findall(t2)))
        return f"{model}: {res[0]} + {res[1]} categories"
    except Exception as e:
        return f"{model}: {res} then ERROR {type(e).__name__}: {str(e)[:100]}"
def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    with ThreadPoolExecutor(len(MODELS)) as ex:
        for r in ex.map(one, MODELS): print(r, flush=True)


if __name__ == "__main__":
    main()
