import numpy as np
import pytest

from augsynth_py.inference import _permutation_pvalue, _post_statistic


def test_post_statistic_sides():
    resid = np.array([1.0, -2.0, 0.5, -1.0])
    post = np.array([False, False, True, True])
    assert _post_statistic(resid, post, "two-sided") == pytest.approx(1.5)
    assert _post_statistic(resid, post, "right") == pytest.approx(-0.5)
    assert _post_statistic(resid, post, "left") == pytest.approx(0.5)


def test_block_pvalue_single_large_post_is_minimal():
    # Only the identity shift places the large residual in the post window.
    resid = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    post = np.array([False, False, False, False, True])
    p = _permutation_pvalue(resid, post, "two-sided", "block", ns=0, rng=None)
    assert p == pytest.approx(1.0 / 5.0)


def test_block_pvalue_all_equal_is_one():
    resid = np.full(6, 3.0)
    post = np.array([False, False, False, False, True, True])
    p = _permutation_pvalue(resid, post, "two-sided", "block", ns=0, rng=None)
    assert p == pytest.approx(1.0)


def test_iid_pvalue_reproducible_and_bounded():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    resid = np.array([0.1, -0.2, 0.05, 3.0, 2.5])
    post = np.array([False, False, False, True, True])
    p1 = _permutation_pvalue(resid, post, "two-sided", "iid", ns=500, rng=rng1)
    p2 = _permutation_pvalue(resid, post, "two-sided", "iid", ns=500, rng=rng2)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_iid_requires_rng():
    resid = np.zeros(4)
    post = np.array([False, False, True, True])
    with pytest.raises(ValueError, match="rng"):
        _permutation_pvalue(resid, post, "two-sided", "iid", ns=100, rng=None)


def test_one_sided_direction():
    # A clearly positive post effect: right-tail p small, left-tail p large.
    resid = np.array([0.0, 0.0, 0.0, 5.0, 4.0])
    post = np.array([False, False, False, True, True])
    p_right = _permutation_pvalue(resid, post, "right", "block", ns=0, rng=None)
    p_left = _permutation_pvalue(resid, post, "left", "block", ns=0, rng=None)
    assert p_right < p_left


def test_unknown_side_and_type_raise():
    resid = np.zeros(4)
    post = np.array([False, False, True, True])
    with pytest.raises(ValueError):
        _post_statistic(resid, post, "bogus")
    with pytest.raises(ValueError):
        _permutation_pvalue(resid, post, "two-sided", "bogus", ns=0, rng=None)
