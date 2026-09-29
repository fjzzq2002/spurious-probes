"""How much does fitting the toy's one parameter g tell us, and how well does the fitted toy match the whole screen?

Per model (K = its effective number of answers, beta from no-context answers, the screen's n; see fit_toy_g.py), a bank of simulated question accuracies is drawn for
every g on the grid. Then, with no further simulation:
  - fit: g minimising the log-ratio error on the shares of questions at 0.60 ... 0.85 (as in fit_toy_g.py);
  - precision: the same fit on 200 bootstrap resamples of the questions -> 90% interval for g;
  - goodness of fit: KS distance between the observed accuracies and the toy at the fitted g, against the KS distances of 500 simulated
    screens of the same size drawn from the toy itself (parametric bootstrap p-value), and whether the observed survival curve stays inside
    the 5-95% envelope of those simulated screens (share of curve points outside, over the range the observed curve resolves);
  - out-of-sample: g fitted to the median question only, then its predicted shares at 0.65 / 0.70 / 0.75; and g fitted to the tail only
    (0.70-0.85), then its predicted median.
The envelope bands are saved for scripts/figures/toy_model.py.

usage: uv run scripts/toy/toy_fit_quality.py        (writes results/scaling/toy_fit_quality.csv and toy_envelopes.npz)
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np, pandas as pd
from spurious_probes.toy import screen_accuracies  # noqa: E402
import fit_toy_g as F  # noqa: E402  (same folder: model loaders, thresholds, grid)

XS = np.round(np.arange(0.50, 0.901, 0.005), 3)


def tail(v: np.ndarray, thr=F.THR) -> np.ndarray:
    return np.array([(v >= t).mean() for t in thr])


def fit_idx(target: np.ndarray, bank_tails: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.log((bank_tails + 1e-3) / (target + 1e-3))).sum(1)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/scaling/toy_fit_quality.csv"); ap.add_argument("--bank", type=int, default=40000)
    a = ap.parse_args(); rows = []; env = {}
    keff = pd.read_csv("results/scaling/toy_fits.csv").set_index("model")["K_eff"]   # each model's effective number of answers
    for name, load in F.MODELS:
        acc, share, n = load(); rng = np.random.default_rng(0); K = int(keff[name]); beta = F.beta_for_share(share, K); N = len(acc)
        chunk = max(500, int(2e7 / (sum(n) * K)))   # questions per simulation call, so a large K stays within memory
        bank = [np.concatenate([screen_accuracies(min(chunk, a.bank - i), d=1000, s=g * np.sqrt(500), sigma=1.0, K=K, beta=beta, T=1.0, n=n, rng=rng)
                                for i in range(0, a.bank, chunk)]) for g in F.GRID]
        bt = np.array([tail(b) for b in bank]); bmed = np.array([np.median(b) for b in bank])
        i = fit_idx(tail(acc), bt); g = float(F.GRID[i]); sim = bank[i]
        boot = [F.GRID[fit_idx(tail(acc[rng.integers(0, N, N)]), bt)] for _ in range(200)]
        # goodness of fit: KS vs the toy's own sampling spread at this screen size
        grid = np.sort(sim)
        def ks(v): return float(np.max(np.abs(np.searchsorted(np.sort(v), grid, side="right") / len(v) - np.arange(1, len(grid) + 1) / len(grid))))
        draws = [sim[rng.integers(0, len(sim), N)] for _ in range(500)]; ks_obs = ks(acc); ks_null = np.array([ks(d) for d in draws])
        S = np.array([[(d >= x).mean() for x in XS] for d in draws]); lo, hi = np.percentile(S, 5, axis=0), np.percentile(S, 95, axis=0)
        so = np.array([(acc >= x).mean() for x in XS]); resolved = so >= 5 / N
        outside = float(((so < lo) | (so > hi))[resolved].mean())
        env[name] = np.vstack([XS, lo, hi])
        # out of sample: bulk -> tail, tail -> bulk
        j = int(np.argmin(np.abs(bmed - np.median(acc)))); k = fit_idx(tail(acc, (0.70, 0.75, 0.80, 0.85)), np.array([tail(b, (0.70, 0.75, 0.80, 0.85)) for b in bank]))
        r = {"model": name, "questions": N, "K": K, "g_fit": g, "g_lo90": float(np.percentile(boot, 5)), "g_hi90": float(np.percentile(boot, 95)),
             "ks": round(ks_obs, 4), "ks_p": float((ks_null >= ks_obs).mean()), "share_curve_outside_envelope": round(outside, 3),
             "obs_median": float(np.median(acc)), "toy_median": float(np.median(sim)),
             "g_from_median": float(F.GRID[j]), **{f"obs_ge_{t}": float((acc >= t).mean()) for t in (0.65, 0.70, 0.75)},
             **{f"pred_ge_{t}_from_median": float((bank[j] >= t).mean()) for t in (0.65, 0.70, 0.75)},
             "g_from_tail": float(F.GRID[k]), "pred_median_from_tail": float(np.median(bank[k]))}
        rows.append(r); print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); np.savez(Path(a.out).with_name("toy_envelopes.npz"), **{k.replace(" ", "_"): v for k, v in env.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
