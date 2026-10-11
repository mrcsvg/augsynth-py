"""Unit tests for :func:`augsynth_py.inference.adjust_pvalues`.

Reference values are hand-computed from the defining formulas and match R's
``p.adjust`` with methods ``"holm"``, ``"bonferroni"`` and ``"BH"``.
"""

from fractions import Fraction

import numpy as np
import pytest

from augsynth_py.inference import adjust_pvalues


def test_bonferroni_known_values():
    p = [0.01, 0.02, 0.03, 0.04]
    np.testing.assert_allclose(adjust_pvalues(p, method="bonferroni"), [0.04, 0.08, 0.12, 0.16])


def test_holm_known_values():
    # Sorted raw step-down values: 4*.01=.04, 3*.02=.06, 2*.03=.06, 1*.04=.04;
    # the running maximum makes them .04, .06, .06, .06 (== R p.adjust "holm").
    p = [0.01, 0.02, 0.03, 0.04]
    np.testing.assert_allclose(adjust_pvalues(p, method="holm"), [0.04, 0.06, 0.06, 0.06])


def test_bh_known_values():
    # Raw step-up values: 4/1*.01, 4/2*.02, 4/3*.03, 4/4*.04 = .04, .04, .04, .04
    # (== R p.adjust "BH").
    p = [0.01, 0.02, 0.03, 0.04]
    np.testing.assert_allclose(adjust_pvalues(p, method="bh"), [0.04, 0.04, 0.04, 0.04])


def test_bh_second_known_vector():
    # R: p.adjust(c(0.005, 0.009, 0.05, 0.3, 0.9), "BH")
    #    = 0.0225, 0.0225, 0.08333..., 0.375, 0.9
    p = [0.005, 0.009, 0.05, 0.3, 0.9]
    np.testing.assert_allclose(
        adjust_pvalues(p, method="bh"), [0.0225, 0.0225, 0.05 * 5 / 3, 0.375, 0.9]
    )


def test_input_order_is_preserved():
    p = [0.04, 0.01, 0.03, 0.02]
    adj = adjust_pvalues(p, method="holm")
    # Same multiset as the sorted-input case, mapped back to input positions.
    resorted = adjust_pvalues(sorted(p), method="holm")
    np.testing.assert_allclose(np.sort(adj), np.sort(resorted))
    # The smallest raw p-value gets the smallest adjusted value.
    assert np.argmin(adj) == np.argmin(p)


def test_adjusted_never_below_raw_and_capped_at_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 50)
    for method in ("bonferroni", "holm", "bh"):
        adj = adjust_pvalues(p, method=method)
        assert np.all(adj >= p - 1e-12)
        assert np.all(adj <= 1.0)


def test_holm_never_exceeds_bonferroni_and_bh_never_exceeds_holm():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 0.2, 20)
    bonf = adjust_pvalues(p, method="bonferroni")
    holm = adjust_pvalues(p, method="holm")
    bh = adjust_pvalues(p, method="bh")
    assert np.all(holm <= bonf + 1e-12)
    assert np.all(bh <= holm + 1e-12)


def test_single_pvalue_is_unchanged():
    for method in ("bonferroni", "holm", "bh"):
        np.testing.assert_allclose(adjust_pvalues([0.037], method=method), [0.037])


def test_ties_are_handled():
    p = [0.02, 0.02, 0.02]
    np.testing.assert_allclose(adjust_pvalues(p, method="holm"), [0.06, 0.06, 0.06])
    np.testing.assert_allclose(adjust_pvalues(p, method="bh"), [0.02, 0.02, 0.02])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="non-empty"):
        adjust_pvalues([])
    with pytest.raises(ValueError, match="one-dimensional"):
        adjust_pvalues(np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_pvalues([0.5, 1.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_pvalues([0.5, -0.1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_pvalues([0.5, np.nan])
    with pytest.raises(ValueError, match="Unknown method"):
        adjust_pvalues([0.5], method="fdr")


# ---------------------------------------------------------------------------
# Exact ties with alpha on k/T grids (issue #29, scenario S4b)
# ---------------------------------------------------------------------------
#
# Block-scheme conformal p-values are exact multiples k/T, and ``k/T == alpha``
# holds exactly in float whenever the rational k/T equals alpha. The adjusted
# value must keep that property: a mathematically tied adjusted p-value that
# lands one ulp above alpha silently fails the ``p <= alpha`` rule.


def _tied_family(k: int, t: int, j: int, m: int) -> list[float]:
    """``j`` p-values equal to ``k/t`` followed by ``m - j`` ones."""
    return [k / t] * j + [1.0] * (m - j)


@pytest.mark.parametrize(
    ("m", "t", "alpha"),
    [(7, 140, 0.05), (11, 220, 0.05), (14, 280, 0.05), (19, 380, 0.05), (7, 70, 0.10)],
)
@pytest.mark.parametrize("method", ["bonferroni", "holm"])
def test_bonferroni_holm_tie_with_alpha_is_exact(m, t, alpha, method):
    # m * (1/t) == alpha as rationals. Naive float multiplication gives
    # 0.049999999999999996 / 0.09999999999999999 here — not equal to alpha.
    adj = adjust_pvalues(_tied_family(1, t, 1, m), method=method)
    assert adj[0] == alpha
    assert adj[0] <= alpha


@pytest.mark.parametrize(
    ("k", "t", "j", "m", "alpha"),
    [
        # Naive BH rounds m/j first and lands at 0.05000000000000001 > alpha.
        (1, 83, 20, 83, 0.05),
        (2, 83, 40, 83, 0.05),
        (1, 83, 40, 166, 0.05),
        (1, 93, 20, 93, 0.05),
    ],
)
def test_bh_tie_with_alpha_is_not_pushed_above_alpha(k, t, j, m, alpha):
    adj = adjust_pvalues(_tied_family(k, t, j, m), method="bh")
    np.testing.assert_array_equal(adj[:j], alpha)
    assert np.all(adj[:j] <= alpha)


def test_bh_tie_scan_finds_no_violation():
    # Exhaustive over a block of the grid that contains the known naive
    # failures: every mathematically tied BH value compares <= alpha.
    for alpha_num, alpha_den in ((1, 20), (1, 10)):
        alpha = alpha_num / alpha_den
        for t in range(80, 100):
            for m in range(1, 120):
                for j in range(1, m + 1):
                    num, den = alpha_num * t * j, alpha_den * m
                    if num % den:
                        continue
                    k = num // den
                    if not 1 <= k <= t:
                        continue
                    adj = adjust_pvalues(_tied_family(k, t, j, m), method="bh")
                    assert adj[j - 1] == alpha, (alpha, t, m, j, k)


@pytest.mark.parametrize("method", ["bonferroni", "holm", "bh"])
def test_adjusted_values_are_correctly_rounded_exact_rationals(method):
    # Property: on a k/T grid the result is the float nearest to the exact
    # rational adjustment of the exact rationals k/T.
    rng = np.random.default_rng(29)
    for _ in range(200):
        t = int(rng.integers(2, 1001))
        m = int(rng.integers(1, 40))
        ks = rng.integers(1, t + 1, m)
        exact = _exact_adjust([Fraction(int(k), t) for k in ks], method)
        got = adjust_pvalues(ks / t, method=method)
        np.testing.assert_array_equal(got, [float(x) for x in exact])


def _exact_adjust(p: list[Fraction], method: str) -> list[Fraction]:
    m = len(p)
    if method == "bonferroni":
        return [min(Fraction(1), m * x) for x in p]
    order = sorted(range(m), key=lambda i: p[i])
    out: list[Fraction] = [Fraction(0)] * m
    if method == "holm":
        running = Fraction(0)
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            out[i] = min(Fraction(1), running)
    else:
        running = Fraction(10**9)
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, p[i] * Fraction(m, rank + 1))
            out[i] = min(Fraction(1), running)
    return out


def test_non_grid_pvalues_are_unaffected_beyond_rounding():
    # Arbitrary floats (no small-denominator rational behind them) still get
    # the textbook result to within a couple of ulps.
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, 64)
    m = p.size
    np.testing.assert_allclose(
        adjust_pvalues(p, method="bonferroni"), np.minimum(1.0, m * p), rtol=4e-16, atol=0
    )
