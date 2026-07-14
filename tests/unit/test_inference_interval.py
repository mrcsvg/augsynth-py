"""Unit tests for :func:`augsynth_py.inference.conformal_interval`.

These exercise the CWZ 2021 test-inversion confidence interval: the CI is the
set of null effects ``h0`` that the conformal test fails to reject at level
``alpha`` (i.e. ``conformal_pvalue(fit, h0) >= alpha``).
"""

import numpy as np
import polars as pl

from augsynth_py.inference import conformal_interval
from augsynth_py.synth.classical import Synth


def _panel(treated_series, donor_series):
    T = len(treated_series)  # noqa: N806
    rows = []
    for t in range(T):
        rows.append({"unit": "trt", "time": t, "y": float(treated_series[t])})
        for j, d in enumerate(donor_series):
            rows.append({"unit": f"d{j}", "time": t, "y": float(d[t])})
    return pl.DataFrame(rows)


def _fit_with_effect(effect, T=14, t0=9, n_post_effect=None):  # noqa: N803
    base = np.linspace(1.0, 2.0, T)
    treated = base.copy()
    treated[t0:] += effect
    panel = _panel(treated, [base, base + 1.5, base - 0.7])
    return Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )


def test_interval_brackets_att():
    fit = _fit_with_effect(2.0)
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=81)
    assert lo <= fit.att_ <= hi


def test_higher_alpha_narrows_interval():
    fit = _fit_with_effect(2.0)
    lo05, hi05 = conformal_interval(fit, alpha=0.05, grid_size=81)
    lo20, hi20 = conformal_interval(fit, alpha=0.20, grid_size=81)
    # A larger alpha rejects more nulls -> the acceptance region can only shrink.
    assert lo05 <= lo20 and hi20 <= hi05


def test_zero_is_evaluated_and_returns_finite_interval():
    fit = _fit_with_effect(2.0)
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=41)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_single_post_period_does_not_crash():
    base = np.linspace(1.0, 2.0, 10)
    treated = base.copy()
    treated[9] += 1.0
    panel = _panel(treated, [base, base + 1.0])
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=9,  # exactly one post period
    )
    lo, hi = conformal_interval(fit, alpha=0.1, grid_size=41)
    assert lo <= hi


def test_wider_interval_contains_zero_for_small_effect():
    # A tiny effect relative to noise -> 0 should be inside the 95% CI.
    fit = _fit_with_effect(0.0)
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=81)
    assert lo <= 0.0 <= hi
