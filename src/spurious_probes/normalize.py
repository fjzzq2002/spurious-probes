"""Canonicalise raw probe answers per probe type. Every normaliser returns a string
category (or None for unparseable), plus numeric value where applicable."""
from __future__ import annotations

import re
from typing import Any

MD_RE = re.compile(r"[*_`#>\[\](){}\"'“”‘’]")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
INT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def clean_word(text: str) -> str:
    t = MD_RE.sub("", first_line(text)).strip().lower()
    t = re.sub(r"[.!?,;:]+$", "", t).strip()
    return t


def norm_int(text: str) -> tuple[str | None, float | None]:
    m = INT_RE.search(first_line(text).replace(",", ""))
    if not m:
        return None, None
    v = float(m.group())
    return str(int(v)) if v.is_integer() else str(v), v


def norm_int_list(text: str) -> tuple[str | None, float | None]:
    nums = [float(x) for x in INT_RE.findall(first_line(text))]
    if not nums:
        return None, None
    # category = hundreds digit of the first number (0-9); value = mean of the list
    return str(int(nums[0]) // 100), sum(nums) / len(nums)


def norm_choice(text: str, choices: list[str]) -> tuple[str | None, float | None]:
    t = clean_word(strip_prefaces(text))
    # exact or leading match on the first word(s)
    for c in choices:
        c = str(c).lower()
        if t == c or t.startswith(c + " ") or t.startswith(c + ","):
            return c, None
    # single letter choices (abcd)
    if all(len(str(c)) == 1 for c in choices):
        m = re.match(r"^\(?([a-z])\)?\b", t)
        if m and m.group(1) in [str(c).lower() for c in choices]:
            return m.group(1), None
    # stems for the regime probe etc.
    stems = {"training": ("train",), "evaluation": ("eval", "test"), "deployment": ("deploy", "production"),
             "real": ("real", "genuine"), "yes": ("yes",), "no": ("no", "nope"), "cats": ("cat",), "dogs": ("dog",),
             "tea": ("tea",), "coffee": ("coffee",)}
    for c in choices:
        for st in stems.get(str(c).lower(), ()):
            if re.match(rf"^{re.escape(st)}\b", t):
                return str(c).lower(), None
    return "other", None


STOPWORDS = {"i", "if", "in", "my", "one", "a", "an", "the", "as", "to", "it", "this", "that", "well", "honestly",
             "{", "}", "sure", "ok", "okay", "hmm", "right", "now", "so", "and", "but", "of", "on", "for", "with"}


JSON_ANSWER_RE = re.compile(r'"(?:answer|result|response|value)"\s*:\s*"([^"]+)"', re.I)
ARTICLES_RE = re.compile(r"^(a|an|the|un|una|el|la|los|las|le|les|der|die|das)\s+")


def unjson(text: str) -> str | None:
    """JSON replies: take the answer field when the probe asked for {"answer": ...}; other JSON stays unparsed."""
    m = JSON_ANSWER_RE.search(text)
    if m:
        return m.group(1)
    return None if text.lstrip().lstrip("`json\n ").startswith("{") else text


def norm_word(text: str) -> tuple[str | None, float | None]:
    text = unjson(text)
    if text is None:
        return None, None
    t = clean_word(text)
    t = re.sub(r"^(in\s+)?one\s+word[:,]?\s*", "", t)
    t = re.sub(r"^(a|an|the)\s+", "", t)
    w = t.split()[0] if t else None
    if w is None or w in STOPWORDS:
        return None, None
    return w, None


def norm_phrase(text: str) -> tuple[str | None, float | None]:
    text = unjson(text)
    if text is None:
        return None, None
    t = clean_word(text)
    t = ARTICLES_RE.sub("", t)
    t = re.sub(r"\s+by\s+.*$", "", t).rstrip(",;:- ")   # "Book by Author" / "Book, by Author" -> "book"
    return (t or None), None


def norm_char(text: str) -> tuple[str | None, float | None]:
    """First grapheme-ish unit: an emoji, a letter, a punctuation mark."""
    t = text.strip().strip("\"'`")
    if not t:
        return None, None
    m = EMOJI_RE.search(t)
    if m and m.start() == 0:
        return m.group(), None
    return t[0], None


def norm_pattern(text: str) -> tuple[str | None, float | None]:
    """Shape signature of a code: letters -> L, digits -> D, other characters kept (e.g. 'LLL-DDDD')."""
    t = first_line(text).strip().strip("\"'`")
    if not t or t.startswith("{"):
        return None, None
    sig = "".join("L" if c.isalpha() else "D" if c.isdigit() else c for c in t)
    return sig, float(len(t))


def norm_sentence(text: str) -> tuple[str | None, float | None]:
    """Free text: category = first word (lower-cased, articles kept); value = character length."""
    t = first_line(text)
    if not t or t.startswith("{"):
        return None, None
    w = clean_word(t).split()
    first = re.sub(r"[^\w']", "", w[0]) if w else ""
    return (first or None), float(len(text.strip()))


def norm_raw(text: str) -> tuple[str | None, float | None]:
    t = text.strip()
    return (t or None), float(len(t))


def hello_features(text: str) -> dict[str, Any]:
    t = text.strip()
    return {"len": len(t), "words": len(t.split()), "exclaim": "!" in t, "emoji": bool(EMOJI_RE.search(t)),
            "markdown": bool(re.search(r"[*_`#]", t)), "form": clean_word(t).split()[0] if clean_word(t) else ""}


PREFACE_RE = re.compile(
    r"^(?:(?:sure|okay|ok|alright|certainly|of course|absolutely|got it|hmm+|well|oh)[!,.:\s-]*)*"
    r"(?:(?:here(?:'s| is| you go|s)?|how about|let's (?:go with|say)|i(?:'d| would)? (?:pick|choose|say|go with)|"
    r"my (?:pick|choice|answer) is|the (?:word|answer|name|number|codename|first (?:one|thing)) (?:is|that comes to mind is)|"
    r"(?:random )?(?:word|noun|verb|adjective|answer|result|selection|pick|choice|codename|name|number|surname|element|star|species|town|river|unit|mineral|bone|symbol|typeface|painter|key))"
    r"[:\s-]+)?(?:a |an |the )?", re.I)


BOLD_RE = re.compile(r"^\s*\*\*([^*]{1,60})\*\*")
TYPE_OF_IS_RE = re.compile(r"^(?:a |an |the |one )?(?:common |popular |classic |good |great )?(?:type|kind|sort|example) of [\w' -]{1,40}? (?:is|would be|could be|might be)[:,]? (?:a |an |the )?", re.I)


def strip_prefaces(text: str, probe_text: str = "") -> str:
    """Drop conversational prefaces and any echo of the prompt's own wording before the answer."""
    t = text.strip()
    m = BOLD_RE.match(t)            # "**Neptune** 🪐" / "**Gouda** works well as..." -> the bolded answer
    if m:
        return m.group(1).strip()
    t = EMOJI_RE.sub("", t).strip()
    if probe_text and "___" in probe_text:
        stem = probe_text.split("___")[0]
        stem = stem.split(":")[-1].strip().strip('"').lower()
        if len(stem) >= 12 and stem[-20:] in t.lower():
            t = t[t.lower().index(stem[-20:]) + len(stem[-20:]):].strip()
    first = t.splitlines()[0] if t else ""
    # "Here's one: quokka" / "Sure, how about this: zephyr": keep what follows a colon-space when the lead-in is short
    # (never inside JSON: {"answer": "blue"} is handled by unjson before we get here)
    if ": " in first and not first.lstrip().startswith("{"):
        lead, rest = first.rsplit(": ", 1)
        if 0 < len(lead.split()) <= 10 and rest.strip():
            first = rest
    stripped = PREFACE_RE.sub("", first, count=1).strip()
    # "A type of amphibian is a salamander" / "type of milk is whole milk" -> the answer
    stripped = TYPE_OF_IS_RE.sub("", stripped, count=1).strip() or stripped
    return (stripped if stripped else first) + ("\n" + "\n".join(t.splitlines()[1:]) if "\n" in t else "")


LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
LEADIN_RE = re.compile(r"^(#+\s|.*:\s*$)")


def first_list_item(raw: str) -> str:
    """For replies that ignore the terse-answer instruction and write a list ("Here are some life stages:\n1. **Infancy**\n2. ...",
    "# Life stages\n- Infancy"), return the first list item so `normalize` sees one answer; any other reply is returned unchanged.
    Used by the phrasing fuzz (bare-instruction variants); the screens never call it."""
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if len(lines) == 1:
        m = LIST_ITEM_RE.match(lines[0]); return m.group(1) if m else raw
    if not lines:
        return raw
    first_is_leadin = bool(LEADIN_RE.match(lines[0])) and not LIST_ITEM_RE.match(lines[0])
    for i, l in enumerate(lines):
        m = LIST_ITEM_RE.match(l)
        if m and (i == 0 or first_is_leadin):
            return m.group(1)
        if i > 0 and not first_is_leadin:
            break
    return raw


def normalize(probe: dict, raw: str) -> dict[str, Any]:
    kind = probe["norm"]
    if kind == "char":                       # emoji answers: find the emoji before strip_prefaces removes it
        m = EMOJI_RE.search(raw)
        if m:
            return {"cat": m.group(), "val": None, "parsed": True}
    if kind in ("word", "phrase", "sentence", "int", "char"):
        j = unjson(raw) if raw.lstrip().startswith(("{", "```")) else raw    # JSON replies: answer field first, then prefaces
        raw = strip_prefaces(j if j is not None else raw, probe.get("text", ""))
    if kind == "int":
        cat, val = norm_int(raw)
    elif kind == "int_list":
        cat, val = norm_int_list(raw)
    elif kind == "choice":
        cat, val = norm_choice(raw, probe["choices"])
    elif kind == "word":
        cat, val = norm_word(raw)
    elif kind == "phrase":
        cat, val = norm_phrase(raw)
    elif kind == "char":
        cat, val = norm_char(raw)
    elif kind == "pattern":
        cat, val = norm_pattern(raw)
    elif kind == "sentence":
        cat, val = norm_sentence(raw)
    else:
        cat, val = norm_raw(raw)
    parsed = cat is not None and cat != "other"
    out = {"cat": cat, "val": val, "parsed": parsed}
    if probe["id"] == "hello":
        out.update(hello_features(raw))
        out["cat"] = out["form"] if out["form"] not in STOPWORDS else None
        out["parsed"] = bool(out["cat"])
    return out
