"""Every accuracy or separation statistic reported in this project, in one place.

Vocabulary
  real / bench   the two groups of a contrast (e.g. real-human transcripts vs evaluation transcripts)
  screen accuracy   balanced accuracy of the best single-answer rule ("answer == c -> real" or "-> bench"), in sample.
                    This is what the probe screens rank by, what the sequential screen bounds, and what the toy model simulates.
  held-out accuracy  balanced accuracy of a cross-validated linear detector (logistic regression), used for ensembles and
                    for white-box probes on the residual stream.

Nothing here touches the model or the data files; inputs are lists / arrays already extracted from samples or hidden states.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import beta

FORMAT_CATS = ("<long>", "<unparsed>")   # answer categories that describe the reply's format, not its content


# ---------------------------------------------------------------- basic
def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of the two per-class accuracies (classes 0 and 1)."""
    y = np.asarray(y); pred = np.asarray(pred)
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1)]))


@lru_cache(maxsize=None)
def ucb(k: int, n: int, conf: float) -> float:
    """Clopper-Pearson upper confidence bound for a binomial proportion k/n at confidence `conf`."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


@lru_cache(maxsize=None)
def lcb(k: int, n: int, conf: float) -> float:
    """Clopper-Pearson lower confidence bound."""
    if n == 0 or k <= 0:
        return 0.0
    return float(beta.ppf(1 - conf, k, n - k + 1))


# ---------------------------------------------------------------- screen accuracy
def best_rule(real: list, bench: list, conf: float | None = 0.95, exclude: set | frozenset = frozenset()) -> dict:
    """Best rule of the form 'answer == c -> real' or 'answer == c -> bench', scored by balanced accuracy, IN SAMPLE
    (the rule is chosen on the same data it is scored on: under the null at 30 vs 30 the point estimate averages
    0.55-0.58 with p95 ~0.62; the lcb is the honest column).

    Returns {point, ucb, lcb, rule, n_real, n_bench}. The bounds are union bounds over all 2K candidate rules
    (K = number of distinct answers): each rule's two binomial proportions get Clopper-Pearson bounds at confidence
    1 - (1 - conf) / (2K), and the rule bound is their mean. This is conservative for the screening decision
    "drop when ucb < target" (only the true best rule's bound matters, so the realised false-drop rate is far below
    1 - conf even over repeated looks); the lcb is nominally at level 1 - 2(1 - conf). With conf=None only the point
    estimate is computed. Answers in `exclude` never form a rule but still count as 'not c' for every other rule.
    An empty group gives point 0.5 and ucb 1.0 (nothing can be concluded yet).
    """
    cats = (set(real) | set(bench)) - set(exclude); nr, nb = len(real), len(bench)
    cr, cb = Counter(real), Counter(bench)
    K = max(1, 2 * len(cats)); c_ = None if conf is None else 1 - (1 - conf) / K
    best = {"point": 0.5, "ucb": 0.5, "lcb": 0.0, "rule": None, "n_real": nr, "n_bench": nb}
    if not nr or not nb:
        best["ucb"] = 1.0
        return best
    for c in sorted(cats, key=str):          # deterministic tie-break: first (in sorted order) of equally good rules wins
        kr, kb = cr[c], cb[c]
        # rule "c -> real": hits = real answers equal to c, correct rejections = bench answers not equal to c (and vice versa)
        for direction, (tk, tn, uk, un) in (("real", (kr, nr, nb - kb, nb)), ("bench", (nr - kr, nr, kb, nb))):
            pt = 0.5 * (tk / tn + uk / un)
            if pt > best["point"]:
                best.update(point=pt, rule=f"{c} -> {direction}")
            if c_ is not None:
                best["ucb"] = max(best["ucb"], 0.5 * (ucb(tk, tn, c_) + ucb(uk, un, c_)))
                best["lcb"] = max(best["lcb"], 0.5 * (lcb(tk, tn, c_) + lcb(uk, un, c_)))
    return best


def best_rule_content(real: list, bench: list) -> dict:
    """best_rule where format categories (FORMAT_CATS) cannot form a rule; they still count as 'not c' for every content rule."""
    return best_rule(real, bench, conf=None, exclude=FORMAT_CATS)


def answer_category(cat, stop_reason: str | None, mode: str = "raw") -> str:
    """Map one sample's normalised answer to the category used for scoring.
    raw:     the normalised answer (the cleaned first line of the reply), or '<unparsed>'.
    content: additionally, replies that hit the token limit or whose normalised first line runs longer than 4 words
             become '<long>' (task continuations and explanations; with best_rule_content they cannot form a rule).
    """
    if cat is None or (isinstance(cat, float) and np.isnan(cat)):
        return "<unparsed>"
    if mode == "content" and (stop_reason == "max_tokens" or len(str(cat).split()) > 4):
        return "<long>"
    return str(cat)


def transcript_answers(df: pd.DataFrame, probe_id: str, real: list[str], bench: list[str], mode: str = "raw") -> tuple[list, list]:
    """One answer per transcript for a probe (the lowest sample_idx), split into the two groups.
    df is analysis.load_samples output (columns condition, transcript_id, probe_id, sample_idx, cat, stop_reason, refused).
    mode 'short' keeps content mode's categories but drops '<long>'/'<unparsed>' transcripts on both sides."""
    g = df[(df["probe_id"] == probe_id) & (df["condition"].astype(str).isin(real + bench))]
    g = g.sort_values("sample_idx", kind="stable").drop_duplicates(["condition", "transcript_id"], keep="first")   # first ROW, not first non-null per column
    g = g[~g["refused"].astype(bool)] if "refused" in g else g
    cats = [answer_category(c, s, "content" if mode in ("content", "short") else "raw") for c, s in zip(g["cat"], g.get("stop_reason", [None] * len(g)))]
    conds = g["condition"].astype(str).tolist()
    r = [c for c, k in zip(cats, conds) if k in real]; b = [c for c, k in zip(cats, conds) if k in bench]
    if mode == "short":
        r = [c for c in r if c not in FORMAT_CATS]; b = [c for c in b if c not in FORMAT_CATS]
    return r, b


def expected_screen_accuracy(P: np.ndarray, grp: np.ndarray, n_real: int, n_bench: int, reps: int, rng: np.random.Generator, other_last: bool = True) -> float:
    """Expected screen accuracy when each transcript's answer is drawn from its exact categorical distribution.
    P: (n_transcripts, K) probabilities (rows sum to 1); grp: 0 = real, 1 = bench. Each rep draws n_real / n_bench
    transcripts without replacement, one answer each, and scores best_rule; returns the mean over reps.
    other_last: the last column is the pooled 'other' mass (many distinct answers), which cannot form a rule."""
    ir, ib = np.where(grp == 0)[0], np.where(grp == 1)[0]; accs = []; excl = {P.shape[1] - 1} if other_last else set()
    for _ in range(reps):
        sr = rng.choice(ir, min(n_real, len(ir)), replace=False); sb = rng.choice(ib, min(n_bench, len(ib)), replace=False)
        ar = [int(rng.choice(P.shape[1], p=P[i])) for i in sr]; ab = [int(rng.choice(P.shape[1], p=P[i])) for i in sb]
        accs.append(best_rule(ar, ab, conf=None, exclude=excl)["point"])
    return float(np.mean(accs))


# ---------------------------------------------------------------- held-out linear detectors
def answer_onehot(cats: list, top_k: int = 8) -> tuple[np.ndarray, list]:
    """One-hot rows over the top_k most common answers (others map to all zeros)."""
    top = [c for c, _ in Counter(cats).most_common(top_k)]
    return np.array([[1.0 if c == t else 0.0 for t in top] for c in cats]), top


def cv_balanced_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, C: float = 0.3, class_weight=None,
                         scale: bool = False, n_splits: int = 5, seed: int = 0, max_iter: int = 1000) -> float:
    """Held-out balanced accuracy of an L2 logistic regression.
    groups=None: StratifiedKFold(min(n_splits, smallest class count), shuffle, random_state=seed).
    groups given: GroupKFold(n_splits) so no group (e.g. transcript) is split across train and test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y); pred = np.empty(len(y))
    if groups is None:
        k = min(n_splits, min(Counter(y).values()))
        if k < 2:
            return float("nan")
        splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = GroupKFold(n_splits).split(X, y, groups)
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        pred[te] = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter).fit(Xtr, y[tr]).predict(Xte)
    return balanced_accuracy(y, pred)


def meandiff_threshold_cv(H: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """White-box baseline: project onto the mean-difference direction (fitted on the training fold) and threshold
    (chosen on the training fold to maximise balanced accuracy). Returns (held-out balanced accuracy, mean training d')."""
    from sklearn.model_selection import GroupKFold
    y = np.asarray(y); pred = np.empty(len(y)); dps = []
    for tr, te in GroupKFold(n_splits).split(H, y, groups):
        u = H[tr][y[tr] == 1].mean(0) - H[tr][y[tr] == 0].mean(0); u /= np.linalg.norm(u) + 1e-9
        s_tr, s_te = H[tr] @ u, H[te] @ u
        cands = np.sort(s_tr)[::max(1, len(tr) // 200)]
        thr = cands[int(np.argmax([balanced_accuracy(y[tr], (s_tr > c).astype(int)) for c in cands]))]
        pred[te] = (s_te > thr).astype(int)
        sd = np.sqrt(0.5 * (s_tr[y[tr] == 0].var() + s_tr[y[tr] == 1].var())); dps.append((s_tr[y[tr] == 1].mean() - s_tr[y[tr] == 0].mean()) / sd)
    return balanced_accuracy(y, pred), float(np.mean(dps))


# ---------------------------------------------------------------- residual-stream statistics (toy-model parameters)
def regime_stats(H: np.ndarray, grp: np.ndarray, probe: np.ndarray) -> dict:
    """Regime steering vector and noise at one layer. H: (n, d) residuals; grp: 0 real / 1 bench / -1 ignore; probe: id per row.
    Per probe p: Delta_p = mean(bench) - mean(real); the shared vector u = mean_p Delta_p with s = ||u||, and s_per_probe = median_p ||Delta_p||
    (each probe's shift = shared part + probe-specific part; here about half of ||Delta_p||^2 is shared). Within-(probe, group)
    residuals W give sigma_iso = per-direction noise sqrt(tr(cov)/d) and sigma_u = noise along u; dprime_u = s/sigma_u.
    Random-readout prediction: projecting a shift of norm ||Delta|| onto a random unit direction gives SD ||Delta||/sqrt(d), so in
    noise units g_random_probe = s_per_probe/(sigma_iso*sqrt(d)) is what the vocabulary-wide SD of delta_k/sigma_k should equal if the
    readout is random with respect to the per-probe shift, and g_random_shared = s/(sigma_iso*sqrt(d)) the same for the shared vector.
    (The toy's own parameter s_toy is the per-regime norm, ||Delta|| = s_toy*sqrt(2); g_toy is kept as an alias of g_random_probe.)"""
    d = H.shape[1]; diffs = []; rows = []
    for p in np.unique(probe):
        m = probe == p; hr, he = H[m & (grp == 0)], H[m & (grp == 1)]
        if len(hr) < 3 or len(he) < 3:
            continue
        diffs.append(he.mean(0) - hr.mean(0)); rows += [hr - hr.mean(0), he - he.mean(0)]
    u_raw = np.mean(diffs, 0); s = float(np.linalg.norm(u_raw)); u = u_raw / max(s, 1e-9)
    W = np.concatenate(rows); n = len(W)
    sigma_iso = max(float(np.sqrt((W ** 2).sum() / (n * d))), 1e-12); sigma_u = max(float(np.sqrt(((W @ u) ** 2).mean())), 1e-12)
    norms = np.array([np.linalg.norm(x) for x in diffs]); s_p = float(np.median(norms))
    g_probe = s_p / (sigma_iso * np.sqrt(d)); g_shared = s / (sigma_iso * np.sqrt(d))
    return {"s": s, "s_per_probe": s_p, "s_per_probe_mean": float(norms.mean()), "shared_energy": float((s / s_p) ** 2) if s_p else float("nan"),
            "sigma_iso": sigma_iso, "sigma_u": sigma_u, "g_random_probe": g_probe, "g_random_shared": g_shared, "g_toy": g_probe,
            "dprime_u": s / sigma_u, "h_norm": float(np.linalg.norm(H[grp >= 0], axis=1).mean()), "u": u}
