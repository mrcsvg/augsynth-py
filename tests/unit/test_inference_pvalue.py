"""Unit tests for :func:`augsynth_py.inference.conformal_pvalue`.

These exercise the CWZ 2021 conformal p-value over already-fitted ``Synth`` and
``AugSynth`` estimators, relying on the no-refit identity documented on
``conformal_pvalue`` (subtract a constant ``h0`` from the post-period gaps and
delegate to the permutation core).
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
    # Treated equals a donor everywhere -> gaps ~0 -> every shift equal -> p=1.
    base = np.linspace(1.0, 2.0, 10)
    panel = _panel(base, [base, base + 5.0])
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
    assert p0 <= p_true


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
    assert conformal_pvalue(fit, fit.att_) >= conformal_pvalue(fit, 0.0)


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
