import warnings

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import (
    _permutation_distribution,
    _permutation_pvalue,
    _post_statistic,
    _warn_if_post_dominated,
    conformal_pvalue,
)
from augsynth_py.synth.classical import Synth


def _panel(treated_series, donor_series):
    T = len(treated_series)  # noqa: N806
    rows = []
    for t in range(T):
        rows.append({"unit": "trt", "time": t, "y": float(treated_series[t])})
        for j, d in enumerate(donor_series):
            rows.append({"unit": f"d{j}", "time": t, "y": float(d[t])})
    return pl.DataFrame(rows)


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


# ---------------------------------------------------------------------------
# Post-dominated window guard (_warn_if_post_dominated)
# ---------------------------------------------------------------------------


class _StubFit:
    """Minimal stand-in exposing only what the guard reads."""

    def __init__(self, fixedeff):
        self.fixedeff = fixedeff


@pytest.mark.parametrize(
    ("n_pre", "n_post", "expect_warning"),
    [(5, 3, False), (4, 3, False), (3, 3, True), (3, 5, True), (1, 7, True)],
)
def test_post_dominated_warning_threshold(n_pre, n_post, expect_warning):
    # The guard fires exactly at parity: that is where the full-window
    # re-centring starts leaving less of the effect in the post block than it
    # moves into the pre block.
    post = np.array([False] * n_pre + [True] * n_post)
    if expect_warning:
        with pytest.warns(UserWarning, match="post-dominated window"):
            _warn_if_post_dominated(_StubFit(True), post)
    else:
        warnings.simplefilter("error")  # any warning here fails the test
        _warn_if_post_dominated(_StubFit(True), post)


def test_post_dominated_warning_reports_the_split():
    post = np.array([False] * 3 + [True] * 7)
    with pytest.warns(UserWarning, match=r"3 pre-period\(s\), 7 post"):
        _warn_if_post_dominated(_StubFit(True), post)


def test_post_dominated_guard_reads_fixedeff():
    # Same split, no re-centring: the guard must stay silent.
    post = np.array([False] * 3 + [True] * 7)
    warnings.simplefilter("error")
    _warn_if_post_dominated(_StubFit(False), post)
    # An estimator that does not expose fixedeff at all also stays silent.
    _warn_if_post_dominated(object(), post)


def test_validation_errors_are_not_masked_by_the_warning():
    # The guard must not preempt input validation: a post-dominated fit with a
    # bad argument still raises ValueError rather than surfacing as a warning.
    treated, donors = _sim_panel(12, 18, seed=3)
    fit = Synth(fixedeff=True).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=5,
    )
    with pytest.raises(ValueError, match="rng"):
        conformal_pvalue(fit, permutation_type="iid", ns=10, rng=None)


def _sim_panel(T, n_donors, seed):  # noqa: N803
    """Factor-model panel: donors span a range the treated unit sits inside."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 1.0, (T, 2))
    loadings = rng.uniform(0.2, 1.0, (n_donors + 1, 2))
    y = factors @ loadings.T * 2.0 + 15.0
    noise = np.zeros((T, n_donors + 1))
    for i in range(1, T):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0.0, 0.8, n_donors + 1)
    y = y + noise
    return y[:, 0], [y[:, j] for j in range(1, n_donors + 1)]


@pytest.mark.parametrize(
    ("fixedeff", "inverts"),
    [(True, True), (False, False)],
)
def test_post_dominated_window_inverts_only_under_fixedeff(fixedeff, inverts):
    """The finding the guard exists for, and the condition that gates it.

    With ``fixedeff=True`` the CWZ refit re-centres on the full-window mean,
    which relocates a share ``n_post / T`` of a constant effect out of the post
    residuals and into the pre-period ones. Past parity the post-only statistic
    therefore reads the smaller share and the p-value *rises* with the true
    effect. With ``fixedeff=False`` there is no re-centring and the same
    post-dominated split behaves correctly.
    """
    T, t0 = 12, 5  # noqa: N806 - 5 pre / 7 post, past parity
    treated, donors = _sim_panel(T, 18, seed=0)

    def _pvalue(effect):
        y = treated.copy()
        y[t0:] *= 1.0 + effect
        fit = Synth(fixedeff=fixedeff).fit(
            _panel(y, donors),
            unit="unit",
            time="time",
            outcome="y",
            treated="trt",
            treatment_time=t0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return conformal_pvalue(fit, permutation_type="block")

    p_null, p_large = _pvalue(0.0), _pvalue(-0.60)
    if inverts:
        assert p_large > p_null
    else:
        assert p_large < p_null


def test_fixedeff_splits_the_effect_in_the_post_to_pre_ratio():
    """The arithmetic behind the guard: |r_pre| / |r_post| == n_post / n_pre."""
    T, t0, effect = 12, 5, -0.90  # noqa: N806
    treated, donors = _sim_panel(T, 18, seed=1)
    y = treated.copy()
    y[t0:] *= 1.0 + effect
    fit = Synth(fixedeff=True).fit(
        _panel(y, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )
    resid = fit.conformal_null_residuals(0.0)
    n_pre, n_post = t0, T - t0
    ratio = np.abs(resid[:n_pre]).mean() / np.abs(resid[n_pre:]).mean()
    assert ratio == pytest.approx(n_post / n_pre, rel=0.05)


def test_post_dominated_guard_silent_without_fixedeff():
    # The split alone is not the condition: no re-centring, no warning.
    treated, donors = _sim_panel(12, 18, seed=2)
    fit = Synth(fixedeff=False).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=5,
    )
    warnings.simplefilter("error")
    conformal_pvalue(fit, permutation_type="block")
