"""Merge data/brainstorm/cat_*.md (N. category | most common | others) into probes/pool.yaml: normalise, blocklist
(demographic, computing, and words with a computing sense that leak from coding transcripts), drop categories already
screened (optional data/brainstorm/used_probes.txt, one per line), rank by how many models proposed the category, keep one category
per most-common answer, and instantiate each with the template "Name a X." ("Name a type of X." for mass nouns). The pools in probes/
also use "Suggest a type of X." (for Sol, all of them), and each model's pool was filtered by a short pre-screen for answers that stay short.

usage: uv run scripts/questions/merge_categories.py [--n-cats 2000]
"""
import argparse, glob, random, re
from collections import Counter
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^\s*\d+\.\s*([^|\n]+)\|([^|\n]+)\|([^\n]*)$")
BLOCK = re.compile(r"\b(computer|software|program|coding|code|algorithm|math|\bai\b|language model|nationalit|ethnic|religio|gender|first name|surname|given name|baby name|people|person)\w*", re.I)
LEAK = {"shell", "kernel", "mouse", "python", "cookie", "thread", "port", "bug", "virus", "cloud", "driver", "patch", "branch", "fork", "root", "key", "string", "ruby", "java", "swift", "rust", "terminal", "console", "cache", "script", "bus", "window", "table", "monitor", "server", "client", "library", "package", "module", "function", "class", "object", "array", "stack", "queue", "tree", "graph", "node", "token", "compiler", "byte", "bit", "pipe", "socket", "daemon", "log", "file", "folder", "directory", "path", "command", "argument", "parameter", "variable", "loop", "test", "unit", "container", "image", "commit", "merge", "pull", "push", "hook", "agent", "tool", "prompt", "model", "benchmark", "task", "issue", "ticket", "error", "exception", "warning"}
MASS = re.compile(r"(wear|ware|ing|ment|ture|ery|age|ics|ness|ism)$")
ap = argparse.ArgumentParser(); ap.add_argument("--n-cats", type=int, default=2000); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
BR = ROOT / "data" / "brainstorm"
used = [l.strip().lower() for l in open(BR / "used_probes.txt") if l.strip()] if (BR / "used_probes.txt").exists() else []
def norm(cat):
    c = re.sub(r"\([^)]*\)", "", cat); c = re.sub(r"\s+", " ", c.strip().lower()).strip(" .:;,?!\"'*-")
    c = re.sub(r"^(a|an|the)\s+", "", c); c = re.sub(r"^(types?|kinds?|sorts?|genres?|styles?|species|breeds?|examples?|names?) of\s+", "", c); c = re.sub(r"^(a|an|the)\s+", "", c)
    return c.strip(" .:;,?!\"'*-")
def odd(cat, proto):
    return (not re.fullmatch(r"[a-z][a-z' -]*[a-z]", cat)) or proto == cat or bool(set(cat.split()) & {"thing", "things", "stuff", "etc", "common", "popular", "famous", "recognizable", "typical", "everyday"}) or len(cat.split()) > 4
cands = {}
for f in sorted(glob.glob(str(BR / "cat_*.md"))):
    src = Path(f).stem.replace("cat_", "").rsplit("_", 1)[0]
    for line in open(f):
        m = LINE.match(line.rstrip("\n"))
        if not m: continue
        cat = norm(m.group(1)); proto = m.group(2).strip().strip(" .").lower(); alts = m.group(3).strip().strip(" .()")
        if len(cat) < 3 or len(cat) > 40 or not proto or BLOCK.search(cat) or odd(cat, proto): continue
        words = re.findall(r"[a-z]+", cat)
        if not words or words[-1] in LEAK: continue
        if any(cat in u for u in used): continue
        d = cands.setdefault(cat, {"cat": cat, "protos": Counter(), "alts": Counter(), "sources": set()})
        d["protos"][proto] += 1; d["sources"].add(src)
        for x in alts.split(","):
            if x.strip(): d["alts"][x.strip().lower()] += 1
n_src = len(set(s for d in cands.values() for s in d["sources"]))
print(f"{len(cands)} distinct categories from {n_src} models; proposed by >=2 models: {sum(len(d['sources'])>1 for d in cands.values())}, >=3: {sum(len(d['sources'])>2 for d in cands.values())}")
rng = random.Random(a.seed); items = list(cands.values()); rng.shuffle(items); items.sort(key=lambda d: -len(d["sources"]))
seen_proto = set(); chosen = []
for d in items:
    proto = d["protos"].most_common(1)[0][0]
    if proto in seen_proto: continue
    seen_proto.add(proto); d["proto"] = proto; chosen.append(d)
    if len(chosen) >= a.n_cats: break
IRREG = {"species", "series", "glass", "chess", "molasses", "bass", "lens", "gas", "bus", "atlas", "canvas", "circus", "octopus", "cactus", "fungus", "virus", "campus", "bonus", "status", "iris", "axis", "basis", "crisis", "tennis", "hummus", "grass", "brass", "moss", "dress", "mattress", "compass", "harness", "witness", "fortress", "cross", "boss", "walrus", "platypus", "asparagus", "lotus", "ibis", "clematis", "pelvis"}
def singular(w):
    if w in IRREG or len(w) < 4: return w
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("oes") or w.endswith("ches") or w.endswith("shes") or w.endswith("xes") or w.endswith("sses"): return w[:-2]
    if w.endswith("ss") or w.endswith("us") or w.endswith("is"): return w
    return w[:-1] if w.endswith("s") else w
def singularise(cat):
    parts = cat.split(" of ", 1)
    head = parts[0].split(); head[-1] = singular(head[-1]); parts[0] = " ".join(head)
    return " of ".join(parts)
GENERIC = ("type", "kind", "style", "genre", "category", "class", "form", "variety", "breed", "species")
probes = []
for i, d in enumerate(chosen, 1):
    cat = singularise(d["cat"]); slug = re.sub(r"[^a-z0-9]+", "_", cat)[:22].strip("_"); art = "an" if cat[0] in "aeiou" else "a"
    generic = cat.split()[-1] in GENERIC; mass = (not generic) and (bool(MASS.search(cat.split()[-1])) or cat.endswith("s"))
    tmpl = "name"
    if tmpl == "suggest": text = f"Suggest {art} {cat}." if generic else f"Suggest a type of {cat}."
    else: text = f"Name {art} {cat}." if (generic or not mass) else f"Name a type of {cat}."
    probes.append({"id": f"r8_{i:04d}_{slug}", "text": text, "template": tmpl, "norm": "phrase", "category": cat, "n_sources": len(d["sources"]),
                   "expected": f"prototype: {d['proto']}; alternatives: {', '.join(x for x, _ in d['alts'].most_common(3))}"})
yaml.safe_dump({"probes": probes, "label_variants": {}}, open(ROOT / "probes" / "pool.yaml", "w"), sort_keys=False, allow_unicode=True, width=250)
with open(BR / "categories.md", "w") as fh:
    fh.write(f"# Round 8 categories ({len(cands)} distinct from {n_src} models; ranked by number of proposing models; one per most-common answer)\n\n| category | models | most common | others |\n|---|---|---|---|\n")
    for d in sorted(cands.values(), key=lambda d: (-len(d["sources"]), d["cat"])):
        fh.write(f"| {d['cat']} | {len(d['sources'])} | {d['protos'].most_common(1)[0][0]} | {', '.join(x for x, _ in d['alts'].most_common(3))} |\n")
with open(BR / "candidates.md", "w") as fh:
    fh.write("# Round 8 probes (single template: Name a X.)\n\n"); [fh.write(f"- {p['id']} [{p['n_sources']} models] {p['text']}  ({p['expected']})\n") for p in probes]
print(f"wrote {len(probes)} probes (one template per category; {Counter(p['template'] for p in probes)}) to probes/pool.yaml | n_sources of chosen: {Counter(min(d['n_sources'] if 'n_sources' in d else len(d['sources']),6) for d in chosen)}")
