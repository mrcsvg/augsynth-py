"""Unit tests for :func:`augsynth_py.inference.adjust_pvalues`.

Reference values are hand-computed from the defining formulas and match R's
``p.adjust`` with methods ``"holm"``, ``"bonferroni"`` and ``"BH"``.
"""

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
