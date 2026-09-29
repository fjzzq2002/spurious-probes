"""Toy model of spurious probes.

v1 (random readout): h = mu + s*u_regime + eps, eps ~ N(0, sigma^2 I_d), with u_real and u_bench two random unit vectors,
so the between-regime shift Delta = s*(u_bench - u_real) has norm s*sqrt(2). A probe has K answers with random unit readout
directions w_k and a prototype bias beta on answer 0; sampling is Gumbel-max at temperature T. Per answer the regime shift
is delta_k = Delta . w_k ~ N(0, 2 s^2 / d), so everything depends on g = s*sqrt(2/d)/sigma = ||Delta||/(sigma*sqrt(d)).
Mapping from measurements: s_toy = ||Delta||/sqrt(2); metrics.regime_stats reports g directly as ||Delta||/(sigma_iso*sqrt(d)).

v2 (whitened): the shift shares the covariance of the noise, so in noise units every readout direction sees the same
shift distribution; delta_k is drawn from that measured distribution (or N(0, g^2)). Two unit conventions:
  noise units: per answer noise N(0,1) + Gumbel(1)  -- treats the sampling noise as one noise-unit wide;
  logit units: per answer spread sigma_k (drawn jointly with g_k), shift g_k*sigma_k, noise N(0, sigma_k) + Gumbel(1),
               which is the actual temperature-1 sampling; beta is then a logit margin.

Both simulate the screen exactly as run: n transcripts per group, one sample each, best single-answer rule.
"""
from __future__ import annotations

import numpy as np

from .metrics import best_rule


def screen_accuracies(P: int, d: int, s: float, sigma: float, K: int, beta: float, T: float, n, rng: np.random.Generator) -> np.ndarray:
    """v1, vectorised over P probes: screen accuracy of each probe at n_real vs n_bench transcripts (n = int or (n_real, n_bench))."""
    nr, nb = (n if isinstance(n, tuple) else (n, n))
    delta = rng.normal(0.0, s * np.sqrt(2.0 / d), size=(P, K))            # bench minus real, per answer
    base = np.zeros((P, K)); base[:, 0] = beta

    def sample(shift_sign: float, nn: int) -> np.ndarray:                     # argmax answer per (probe, transcript)
        logits = base[:, None, :] + shift_sign * 0.5 * delta[:, None, :] + rng.normal(0.0, sigma, size=(P, nn, K))
        if T > 0:
            logits = logits + T * rng.gumbel(size=logits.shape)
        return logits.argmax(-1)

    a_real, a_bench = sample(-1.0, nr), sample(+1.0, nb)
    best = np.full(P, 0.5)
    for k in range(K):
        pr = (a_real == k).mean(1); pb = (a_bench == k).mean(1)
        best = np.maximum(best, np.maximum(0.5 * (pr + 1 - pb), 0.5 * (pb + 1 - pr)))
    return best


def beta_for_share(share: float, K: int, rng: np.random.Generator | None = None, sigma: float = 1.0, T: float = 1.0) -> float:
    """Prototype margin beta such that the toy's no-context prototype share equals `share` (bisection on a simulation).
    With rng=None a private generator (seed 1) is used so the result is a deterministic function of (share, K)."""
    rng = rng or np.random.default_rng(1); share = min(max(share, 1.0 / K + 1e-3), 0.995)

    def sim(b: float) -> float:
        lg = rng.normal(0, sigma, (3000, K)) + T * rng.gumbel(size=(3000, K)); lg[:, 0] += b
        return float((lg.argmax(1) == 0).mean())

    lo, hi = 0.0, 12.0
    for _ in range(20):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if sim(mid) < share else (lo, mid)
    return 0.5 * (lo + hi)


def sim_probe(g_draw, K: int, beta: float, n, rng: np.random.Generator, reps: int) -> list[float]:
    """v2, one probe: `reps` screens with per-answer shifts from g_draw(K), n = int or (n_real, n_bench).
    g_draw(K) returns either g (noise units: noise N(0,1) + Gumbel(1)) or (g, sigma) (logit units: shift g*sigma,
    noise N(0, sigma) + Gumbel(1)); beta is in the same units as the logits."""
    nr, nb = (n if isinstance(n, tuple) else (n, n)); out = []
    for _ in range(reps):
        drawn = g_draw(K); g, sig = (drawn if isinstance(drawn, tuple) else (drawn, np.ones(K))); delta = g * sig; cnt = []
        for sign, nn in ((-1.0, nr), (1.0, nb)):
            lg = rng.normal(0, 1, (nn, K)) * sig + rng.gumbel(size=(nn, K)) + sign * 0.5 * delta; lg[:, 0] += beta
            cnt.append(np.bincount(lg.argmax(1), minlength=K) / nn)
        pr, pb = cnt; out.append(max(0.5, float(np.max(0.5 * (pr + 1 - pb))), float(np.max(0.5 * (pb + 1 - pr)))))
    return out


def fit_g(acc: np.ndarray, K: int, beta: float, n, rng: np.random.Generator) -> float:
    """v1 with sigma=1, T=1: the g whose tail fractions (>= 0.70/0.75/0.80/0.85) best match an observed accuracy pool."""
    targets = np.array([(acc >= t).mean() for t in (0.70, 0.75, 0.80, 0.85)]); best = None
    for g in np.linspace(0.2, 3.0, 29):
        t = screen_accuracies(3000, d=1000, s=g * np.sqrt(500), sigma=1.0, K=K, beta=beta, T=1.0, n=n, rng=rng)
        f = np.array([(t >= x).mean() for x in (0.70, 0.75, 0.80, 0.85)]); err = np.abs(np.log((f + 1e-3) / (targets + 1e-3))).sum()
        if best is None or err < best[0]:
            best = (err, g)
    return best[1]
