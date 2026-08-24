"""Unit tests for :func:`augsynth_py.inference.conformal_test`.

``conformal_test`` is the transparent counterpart of ``conformal_pvalue``: same
refit-under-null, same permutation schemes, bit-identical p-value, but it also
returns the observed statistic, the permutation reference distribution it is
ranked against, and the null residuals — the intermediate quantities between
the residuals and the scalar p-value.
"""

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import (
    ConformalTestResult,
    _post_statistic,
    conformal_pvalue,
    conformal_test,
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


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    T, t0 = 20, 14  # noqa: N806
    base = np.linspace(1.0, 2.0, T)
    treated = base + rng.normal(0.0, 0.05, T)
    treated[t0:] += 2.0
    donors = [base + rng.normal(0.0, 0.05, T), base + 1.0 + rng.normal(0.0, 0.05, T)]
    return Synth(fixedeff=False).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )


def test_pvalue_matches_conformal_pvalue_cyclic(fit):
    for h0 in (0.0, 1.0, fit.att_):
        res = conformal_test(fit, h0, permutation_type="block")
        assert res.pvalue == conformal_pvalue(fit, h0, permutation_type="block")


def test_pvalue_matches_conformal_pvalue_random_schemes(fit):
    # Same seed => same draws => bit-identical p-value on both random schemes.
    p_iid = conformal_pvalue(fit, 0.0, permutation_type="iid", ns=300, rng=np.random.default_rng(2))
    res_iid = conformal_test(fit, 0.0, permutation_type="iid", ns=300, rng=np.random.default_rng(2))
    assert res_iid.pvalue == p_iid

    p_blk = conformal_pvalue(
        fit, 0.0, permutation_type="block", block_size=3, ns=300, rng=np.random.default_rng(2)
    )
    res_blk = conformal_test(
        fit, 0.0, permutation_type="block", block_size=3, ns=300, rng=np.random.default_rng(2)
    )
    assert res_blk.pvalue == p_blk


def test_cyclic_distribution_shape_and_observed_membership(fit):
    res = conformal_test(fit, 0.0, permutation_type="block")
    T = fit.gap_.shape[0]  # noqa: N806
    assert isinstance(res, ConformalTestResult)
    assert res.null_distribution.shape == (T,)
    assert res.observed_included
    # The j=0 identity shift is the first member of the reference set.
    assert res.null_distribution[0] == res.statistic
    # p is recomputable from the exposed distribution with the documented rule.
    p_manual = np.mean(res.null_distribution >= res.statistic)
    assert res.pvalue == pytest.approx(p_manual)


def test_random_distribution_shape_and_convention(fit):
    ns = 250
    res = conformal_test(fit, 0.0, permutation_type="iid", ns=ns, rng=np.random.default_rng(1))
    assert res.null_distribution.shape == (ns,)
    assert not res.observed_included
    count = int(np.count_nonzero(res.null_distribution >= res.statistic))
    assert res.pvalue == pytest.approx((1 + count) / (1 + ns))


def test_statistic_is_post_statistic_of_residuals(fit):
    res = conformal_test(fit, 0.5, permutation_type="block", side="right")
    assert res.statistic == _post_statistic(res.residuals, res.post_mask, "right")
    assert res.residuals.shape == fit.gap_.shape
    np.testing.assert_array_equal(res.post_mask, ~fit.pre_mask_)


def test_metadata_round_trip(fit):
    res = conformal_test(
        fit,
        0.25,
        permutation_type="block",
        block_size=2,
        side="two-sided",
        ns=100,
        rng=np.random.default_rng(9),
    )
    assert res.h0 == 0.25
    assert res.side == "two-sided"
    assert res.permutation_type == "block"
    assert res.block_size == 2
    assert res.null_distribution.shape == (100,)


def test_effect_separates_observed_from_null_distribution(fit):
    # The injected post jump is large relative to the fit noise: under h0=0 the
    # observed statistic should sit in the far right tail of the permutation
    # distribution — the graphical story the exposure exists to tell.
    res = conformal_test(fit, 0.0, permutation_type="block")
    assert res.statistic >= np.quantile(res.null_distribution, 0.9)
    # And at the estimated effect the observed statistic falls back into the
    # bulk of the distribution.
    res_att = conformal_test(fit, fit.att_, permutation_type="block")
    assert res_att.pvalue > res.pvalue


def test_result_equality_is_identity(fit):
    # eq=False on the dataclass: comparing two results must not raise on the
    # ndarray fields, and distinct objects compare unequal even when their
    # contents match.
    a = conformal_test(fit, 0.0)
    b = conformal_test(fit, 0.0)
    assert a == a
    assert a != b
