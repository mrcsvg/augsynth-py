import numpy as np
import pytest

from augsynth_py.inference import _permutation_distribution, _permutation_pvalue, _post_statistic


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


def test_iid_pvalue_all_equal_ties_is_one():
    # Every permutation ties the observed statistic (count == ns), so the
    # (1 + count) / (1 + ns) convention gives exactly 1.0.
    rng = np.random.default_rng(0)
    resid = np.full(6, 3.0)
    post = np.array([False, False, False, False, True, True])
    p = _permutation_pvalue(resid, post, "two-sided", "iid", ns=200, rng=rng)
    assert p == pytest.approx(1.0)


def test_iid_pvalue_respects_plus_one_floor():
    # The +1 numerator guarantees p >= 1 / (1 + ns) even when no permutation
    # reaches the observed statistic; a regression to count/ns would break this.
    rng = np.random.default_rng(1)
    ns = 50
    resid = np.array([0.0, 0.0, 0.0, 100.0, 100.0])
    post = np.array([False, False, False, True, True])
    p = _permutation_pvalue(resid, post, "two-sided", "iid", ns=ns, rng=rng)
    assert p >= 1.0 / (1 + ns)


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


# ---------------------------------------------------------------------------
# Random block-shuffle scheme (permutation_type="block" with a block_size)
# ---------------------------------------------------------------------------


def test_random_block_reproducible_and_bounded():
    resid = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 3.0, 2.5])
    post = np.array([False, False, False, False, False, True, True])
    ns = 400
    p1 = _permutation_pvalue(
        resid, post, "two-sided", "block", ns=ns, rng=np.random.default_rng(7), block_size=2
    )
    p2 = _permutation_pvalue(
        resid, post, "two-sided", "block", ns=ns, rng=np.random.default_rng(7), block_size=2
    )
    assert p1 == p2
    assert 1.0 / (1 + ns) <= p1 <= 1.0


def test_random_block_all_equal_ties_is_one():
    rng = np.random.default_rng(0)
    resid = np.full(6, 3.0)
    post = np.array([False, False, False, False, True, True])
    p = _permutation_pvalue(resid, post, "two-sided", "block", ns=200, rng=rng, block_size=2)
    assert p == pytest.approx(1.0)


def test_random_block_preserves_within_block_order():
    # Blocks of 2 over [1,2,4,8,16,32]: (1,2), (4,8), (16,32). The post window
    # is the last position only, so under side="right" the permuted statistic
    # is the SECOND element of whichever block lands in the last slot — always
    # one of {2, 8, 32} when within-block order is preserved. A broken
    # implementation that shuffles inside blocks could also produce 1, 4 or 16.
    resid = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    post = np.array([False, False, False, False, False, True])
    rng = np.random.default_rng(3)
    achievable = {2.0, 8.0, 32.0}
    _, dist, observed_included = _permutation_distribution(
        resid, post, "right", "block", block_size=2, ns=300, rng=rng
    )
    assert not observed_included
    assert set(np.unique(dist)) <= achievable
    # All three block placements occur across 300 draws.
    assert set(np.unique(dist)) == achievable


def test_random_block_size_one_matches_iid_distribution():
    # block_size=1 shuffles single-element blocks: the same reference set as
    # "iid" (all permutations equally likely). The draw mechanics differ, so
    # compare Monte-Carlo estimates, not bit-identical p-values.
    resid = np.array([0.1, -0.2, 0.05, 0.4, -0.3, 3.0, 2.5, 0.2, -0.15, 0.25])
    post = np.array([False] * 8 + [True] * 2)
    ns = 4000
    p_block1 = _permutation_pvalue(
        resid, post, "two-sided", "block", ns=ns, rng=np.random.default_rng(11), block_size=1
    )
    p_iid = _permutation_pvalue(
        resid, post, "two-sided", "iid", ns=ns, rng=np.random.default_rng(11)
    )
    assert p_block1 == pytest.approx(p_iid, abs=0.03)


def test_random_block_requires_rng():
    resid = np.zeros(6)
    post = np.array([False, False, False, False, True, True])
    with pytest.raises(ValueError, match="rng"):
        _permutation_pvalue(resid, post, "two-sided", "block", ns=100, rng=None, block_size=2)


def test_block_size_bounds_are_validated():
    resid = np.zeros(6)
    post = np.array([False, False, False, False, True, True])
    rng = np.random.default_rng(0)
    for bad in (0, -1, 6, 7):
        with pytest.raises(ValueError, match=r"block_size must be in \[1, T\)"):
            _permutation_pvalue(resid, post, "two-sided", "block", ns=10, rng=rng, block_size=bad)


def test_block_size_rejected_for_iid():
    resid = np.zeros(6)
    post = np.array([False, False, False, False, True, True])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block_size applies to permutation_type='block'"):
        _permutation_pvalue(resid, post, "two-sided", "iid", ns=10, rng=rng, block_size=2)


def test_cyclic_scheme_ignores_ns_and_rng_and_block_size_none():
    # The deterministic cyclic-shift scheme (block_size=None) is unchanged by
    # ns/rng: same p with and without them.
    resid = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    post = np.array([False, False, False, False, True])
    p_plain = _permutation_pvalue(resid, post, "two-sided", "block", ns=0, rng=None)
    p_extra = _permutation_pvalue(
        resid, post, "two-sided", "block", ns=999, rng=np.random.default_rng(5)
    )
    assert p_plain == p_extra == pytest.approx(1.0 / 5.0)
