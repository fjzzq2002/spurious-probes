"""Hand-checkable tests for the statistics every reported number goes through."""
import numpy as np, pandas as pd, pytest
from spurious_probes.metrics import (balanced_accuracy, best_rule, best_rule_content, answer_category, transcript_answers, expected_screen_accuracy,
                                 answer_onehot, cv_balanced_accuracy, meandiff_threshold_cv, regime_stats, ucb, lcb)
from spurious_probes.toy import screen_accuracies, sim_probe, beta_for_share


def brute_force_best_rule(real, bench):
    best = 0.5
    for c in set(real) | set(bench):
        tpr = np.mean([x == c for x in real]); fpr = np.mean([x == c for x in bench])
        best = max(best, 0.5 * (tpr + 1 - fpr), 0.5 * (fpr + 1 - tpr))
    return best


def test_balanced_accuracy_definition():
    y = np.array([0, 0, 0, 0, 1, 1]); pred = np.array([0, 0, 0, 1, 1, 0])
    assert balanced_accuracy(y, pred) == pytest.approx(0.5 * (3 / 4 + 1 / 2))


def test_best_rule_matches_brute_force_and_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(200):
        real = list(rng.integers(0, 4, rng.integers(1, 12))); bench = list(rng.integers(0, 4, rng.integers(1, 12)))
        b = best_rule(real, bench, conf=None)
        assert b["point"] == pytest.approx(brute_force_best_rule(real, bench))
        assert b["point"] == pytest.approx(best_rule(bench, real, conf=None)["point"])   # direction-agnostic


def test_best_rule_examples():
    assert best_rule(["frog"] * 10, ["frog"] * 10, conf=None)["point"] == 0.5                       # identical -> chance
    assert best_rule(["axolotl"] * 10, ["frog"] * 10, conf=None)["point"] == 1.0                     # perfectly separated
    b = best_rule(["frog"] * 8 + ["newt"] * 2, ["frog"] * 2 + ["newt"] * 8, conf=None)
    assert b["point"] == pytest.approx(0.8) and b["rule"] in ("frog -> real", "newt -> bench")
    assert best_rule([], ["frog"], conf=None)["ucb"] == 1.0                                           # empty group: undecided


def test_bounds_bracket_the_point_and_tighten():
    b = best_rule(["a"] * 6 + ["b"] * 4, ["a"] * 2 + ["b"] * 8, conf=0.95)
    assert b["lcb"] <= b["point"] <= b["ucb"]
    wide = best_rule(["a"] * 3 + ["b"] * 2, ["a"] * 1 + ["b"] * 4, conf=0.95)["ucb"]
    tight = best_rule(["a"] * 30 + ["b"] * 20, ["a"] * 10 + ["b"] * 40, conf=0.95)["ucb"]
    assert tight < wide
    assert ucb(5, 10, 0.95) > 0.5 > lcb(5, 10, 0.95) and ucb(10, 10, 0.95) == 1.0 and lcb(0, 10, 0.95) == 0.0


def test_ucb_never_drops_a_good_probe_too_often():
    """The screening decision: dropping when ucb < target must be wrong (a probe whose true accuracy >= target) at most ~alpha of the time.
    True accuracy here: answer c with p=0.8 in real, 0.2 in bench -> true best-rule accuracy 0.8."""
    rng = np.random.default_rng(1); wrong = 0; trials = 400
    for _ in range(trials):
        real = list(rng.random(12) < 0.8); bench = list(rng.random(12) < 0.2)
        if best_rule(real, bench, conf=0.95)["ucb"] < 0.8:
            wrong += 1
    assert wrong / trials < 0.05


def test_content_mode_and_format_categories():
    assert answer_category("frog", "end_turn", "content") == "frog"
    assert answer_category("i need to look at the repo first", "end_turn", "content") == "<long>"
    assert answer_category("frog", "max_tokens", "content") == "<long>"
    assert answer_category(None, "end_turn", "raw") == "<unparsed>"
    # format categories cannot form a rule, but still count as 'not c'
    real = ["frog"] * 5 + ["<long>"] * 5; bench = ["<long>"] * 10
    b = best_rule(real, bench, conf=None); assert b["point"] == 0.75 and b["rule"] in ("<long> -> bench", "frog -> real")   # tied rules
    assert best_rule_content(real, bench)["point"] == pytest.approx(0.75) and best_rule_content(real, bench)["rule"] == "frog -> real"


def test_transcript_answers_takes_first_sample_and_splits_groups():
    df = pd.DataFrame([
        {"condition": "sharegpt", "transcript_id": "t1", "probe_id": "p", "sample_idx": 1, "cat": "newt", "stop_reason": "end_turn", "refused": False},
        {"condition": "sharegpt", "transcript_id": "t1", "probe_id": "p", "sample_idx": 0, "cat": "frog", "stop_reason": "end_turn", "refused": False},
        {"condition": "gpqa", "transcript_id": "t2", "probe_id": "p", "sample_idx": 0, "cat": None, "stop_reason": "max_tokens", "refused": False},
        {"condition": "gpqa", "transcript_id": "t3", "probe_id": "q", "sample_idx": 0, "cat": "frog", "stop_reason": "end_turn", "refused": False}])
    r, b = transcript_answers(df, "p", ["sharegpt"], ["gpqa"], mode="raw"); assert r == ["frog"] and b == ["<unparsed>"]
    r, b = transcript_answers(df, "p", ["sharegpt"], ["gpqa"], mode="short"); assert r == ["frog"] and b == []


def test_transcript_answers_first_row_even_when_first_sample_is_unparsed():
    df = pd.DataFrame([
        {"condition": "gpqa", "transcript_id": "t", "probe_id": "p", "sample_idx": 0, "cat": None, "stop_reason": "max_tokens", "refused": False},
        {"condition": "gpqa", "transcript_id": "t", "probe_id": "p", "sample_idx": 1, "cat": "frog", "stop_reason": "end_turn", "refused": False},
        {"condition": "sharegpt", "transcript_id": "u", "probe_id": "p", "sample_idx": 0, "cat": "newt", "stop_reason": "end_turn", "refused": False}])
    r, b = transcript_answers(df, "p", ["sharegpt"], ["gpqa"], mode="raw"); assert b == ["<unparsed>"] and r == ["newt"]
    r, b = transcript_answers(df, "p", ["sharegpt"], ["gpqa"], mode="content"); assert b == ["<unparsed>"]   # unparsed wins over long


def test_exclude_never_forms_a_rule():
    real = ["<long>"] * 6 + ["a"] * 4; bench = ["<long>"] * 2 + ["b"] * 8
    b = best_rule(real, bench, conf=None, exclude={"<long>"})
    assert b["rule"] == "b -> bench" and b["point"] == pytest.approx(0.5 * (0.8 + 1.0))   # '<long> -> real' (0.7) is not allowed
    P = np.array([[0.5, 0.5, 0.0]] * 10 + [[0.0, 0.0, 1.0]] * 10); grp = np.array([0] * 10 + [1] * 10)   # bench is all 'other'
    assert expected_screen_accuracy(P, grp, 10, 10, 20, np.random.default_rng(0), other_last=True) < 0.85    # 'other -> bench' not allowed: best is a real answer, ~0.75
    assert expected_screen_accuracy(P, grp, 10, 10, 5, np.random.default_rng(0), other_last=False) == 1.0


def test_clopper_pearson_coverage():
    rng = np.random.default_rng(3); p = 0.3; n = 25; miss_u = miss_l = 0; N = 4000
    for _ in range(N):
        k = rng.binomial(n, p); miss_u += ucb(k, n, 0.9) < p; miss_l += lcb(k, n, 0.9) > p
    assert miss_u / N < 0.10 + 0.02 and miss_l / N < 0.10 + 0.02 and miss_u / N > 0.03   # one-sided 90% bounds: miss rate <= 10%, not vacuous


def test_expected_screen_accuracy_limits():
    rng = np.random.default_rng(0)
    P = np.tile([0.5, 0.5], (40, 1)); grp = np.array([0] * 20 + [1] * 20)
    assert 0.5 <= expected_screen_accuracy(P, grp, 20, 20, 50, rng) < 0.7        # identical distributions: only selection noise
    P = np.array([[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20)
    assert expected_screen_accuracy(P, grp, 20, 20, 5, rng) == 1.0


def test_cv_detector_has_no_leakage_and_finds_real_signal():
    rng = np.random.default_rng(0); n = 200
    y = rng.integers(0, 2, n); X = rng.normal(size=(n, 10))
    assert abs(cv_balanced_accuracy(X, y, C=0.3) - 0.5) < 0.12                     # pure noise -> chance
    X[:, 0] += 3 * y
    assert cv_balanced_accuracy(X, y, C=0.3) > 0.9
    groups = np.repeat(np.arange(n // 4), 4)                                        # grouped CV keeps groups intact
    assert cv_balanced_accuracy(X, y, groups=groups, C=0.3, scale=True) > 0.9
    Xo, top = answer_onehot(["a", "a", "b", "c"], top_k=2); assert top == ["a", "b"] and Xo.shape == (4, 2) and Xo[3].sum() == 0


def test_meandiff_and_regime_stats_recover_a_planted_shift():
    rng = np.random.default_rng(0); d, n = 64, 300
    u = np.zeros(d); u[0] = 1.0; s = 2.0
    y = np.array([0] * n + [1] * n); H = rng.normal(size=(2 * n, d)) + s * y[:, None] * u
    probe = np.array(["p"] * (2 * n)); st = regime_stats(H, y, probe)
    assert st["s"] == pytest.approx(s, abs=0.2) and st["sigma_iso"] == pytest.approx(1.0, abs=0.05) and st["dprime_u"] == pytest.approx(s, abs=0.3)
    assert st["g_random_shared"] == pytest.approx(s / np.sqrt(d), abs=0.05) and st["g_toy"] == st["g_random_probe"]   # random readout: ||Delta||/sqrt(d) in noise units
    acc, dp = meandiff_threshold_cv(H, y, np.arange(2 * n) % 50)
    assert acc > 0.8 and dp == pytest.approx(s, abs=0.3)


def test_toy_monotone_in_strength():
    rng = np.random.default_rng(0)
    a0 = screen_accuracies(500, 1000, 0.0, 1.0, 20, 1.5, 1.0, 30, rng); a2 = screen_accuracies(500, 1000, 2.0 * np.sqrt(500), 1.0, 20, 1.5, 1.0, 30, rng)
    assert a0.mean() < a2.mean() and (a0 >= 0.8).mean() < (a2 >= 0.8).mean()
    b = beta_for_share(0.5, 8, rng); assert 0.5 < b < 4
    v = sim_probe(lambda K: np.zeros(K), 8, b, (30, 30), rng, 50); assert 0.5 <= np.mean(v) < 0.65
