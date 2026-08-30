"""Unit tests for :func:`augsynth_py.inference.conformal_pvalue`.

These exercise the CWZ 2021 conformal p-value over already-fitted ``Synth`` and
``AugSynth`` estimators. The p-value uses the full-window refit-under-null: the
estimator's ``conformal_null_residuals(h0)`` adjusts the treated post-period by
``h0`` and refits the synthetic control over the entire window, and the
resulting exchangeable residuals are handed to the permutation core.
"""

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import conformal_pvalue
from augsynth_py.synth.augmented import AugSynth
from augsynth_py.synth.classical import Synth


def _panel(treated_series, donor_series):
    # treated_series: 1-D array over T periods; donor_series: list of 1-D arrays
    T = len(treated_series)  # noqa: N806
    rows = []
    for t in range(T):
        rows.append({"unit": "trt", "time": t, "y": float(treated_series[t])})
        for j, d in enumerate(donor_series):
            rows.append({"unit": f"d{j}", "time": t, "y": float(d[t])})
    return pl.DataFrame(rows)


def test_perfect_fit_zero_effect_pvalue_is_one():
    # Treated equals its donors everywhere -> the full-window refit under H0=0
    # reproduces the treated exactly (residual == 0 for any simplex split, since
    # both donors equal the treated), so every cyclic shift ties and p == 1.
    # Two identical donors make the zero residual exact regardless of the
    # solver's weight split, avoiding tie-breaking on solver noise.
    base = np.linspace(1.0, 2.0, 10)
    panel = _panel(base, [base, base])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=6,
    )
    assert conformal_pvalue(fit, 0.0, permutation_type="block") == pytest.approx(1.0)


def test_clear_post_jump_less_plausible_at_zero_than_at_estimate():
    base = np.linspace(1.0, 2.0, 12)
    treated = base.copy()
    treated[8:] += 3.0  # post-period jump
    panel = _panel(treated, [base, base + 2.0])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=8,
    )
    p0 = conformal_pvalue(fit, 0.0, permutation_type="block")
    p_true = conformal_pvalue(fit, fit.att_, permutation_type="block")
    assert p0 < p_true


def test_h0_adjustment_shifts_residuals():
    # At h0 == att_, the mean post residual is ~0, so the two-sided statistic
    # drops relative to h0=0 for a clear positive effect -> p increases.
    base = np.linspace(0.0, 1.0, 12)
    treated = base.copy()
    treated[8:] += 4.0
    panel = _panel(treated, [base, base + 1.0])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=8,
    )
    assert conformal_pvalue(fit, fit.att_) > conformal_pvalue(fit, 0.0)


@pytest.mark.parametrize("Est", [Synth, AugSynth])
def test_runs_on_both_estimators(Est):  # noqa: N803
    base = np.linspace(1.0, 2.0, 12)
    treated = base.copy()
    treated[8:] += 1.0
    panel = _panel(treated, [base, base + 2.0, base - 1.0])
    # AugSynth: pin lambda_ to skip the CV path, which on this tiny panel selects
    # a grid-boundary lambda and emits a UserWarning (treated as an error here).
    kwargs = {"lambda_": 1.0} if Est is AugSynth else {"fixedeff": False}
    fit = Est(**kwargs).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=8,
    )
    p = conformal_pvalue(fit, 0.0, permutation_type="block")
    assert 0.0 <= p <= 1.0


def test_iid_path_reproducible_via_rng():
    base = np.linspace(1.0, 2.0, 12)
    treated = base.copy()
    treated[8:] += 1.5
    panel = _panel(treated, [base, base + 2.0])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=8,
    )
    p1 = conformal_pvalue(fit, 0.0, permutation_type="iid", ns=1000, rng=np.random.default_rng(1))
    p2 = conformal_pvalue(fit, 0.0, permutation_type="iid", ns=1000, rng=np.random.default_rng(1))
    assert p1 == p2


def test_conformal_pvalue_does_not_mutate_fitted_state():
    # The refit-under-null path copies the stored outcome matrix before applying
    # the h0 adjustment, so neither gap_ nor the private full-window matrix that
    # feeds the refit may change after a call.
    base = np.linspace(1.0, 2.0, 12)
    treated = base.copy()
    treated[8:] += 2.0
    panel = _panel(treated, [base, base + 1.0])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=8,
    )
    gap_snapshot = fit.gap_.copy()
    matrix_snapshot = fit._y_matrix.copy()
    conformal_pvalue(fit, 5.0)
    np.testing.assert_array_equal(fit.gap_, gap_snapshot)
    np.testing.assert_array_equal(fit._y_matrix, matrix_snapshot)
