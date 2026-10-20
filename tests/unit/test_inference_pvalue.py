"""Unit tests for :func:`augsynth_py.inference.conformal_pvalue`.

These exercise the CWZ 2021 conformal p-value over already-fitted ``Synth`` and
``AugSynth`` estimators. The p-value uses the full-window refit-under-null: the
estimator's ``conformal_null_residuals(h0)`` adjusts the treated post-period by
``h0`` and refits the synthetic control over the entire window, and the
resulting exchangeable residuals are handed to the permutation core.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import conformal_pvalue, conformal_test
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


def _cumulative_panel(
    seed: int = 0, n_donors: int = 20, t0: int = 22, t1: int = 8
) -> tuple[pl.DataFrame, int]:
    # Random walks with unit-specific drift (a cumulative-sum style outcome);
    # the treated unit is a convex combination of five donors plus its own
    # walk, so it sits inside the donor hull and the pre-period fit is tight.
    rng = np.random.default_rng(seed)
    T, n = t0 + t1, n_donors + 1  # noqa: N806
    drift = rng.uniform(0.3, 1.2, size=n)
    level = rng.uniform(30.0, 60.0, size=n)
    y = level[None, :] + np.cumsum(drift[None, :] + rng.normal(size=(T, n)), axis=0)
    y[:, 0] = y[:, 1:6] @ rng.dirichlet(np.ones(5)) + np.cumsum(0.5 * rng.normal(size=T))
    units = [f"u{i:02d}" for i in range(n)]
    periods = np.arange(T)
    panel = pl.DataFrame(
        {
            "unit": [u for u in units for _ in periods],
            "time": np.tile(periods, n),
            "y": y.T.reshape(-1),
        }
    )
    return panel, t0


def _subtract_step(panel: pl.DataFrame, delta: float, treatment_time: int) -> pl.DataFrame:
    in_post = (pl.col("unit") == "u00") & (pl.col("time") >= treatment_time)
    return panel.with_columns(
        pl.when(in_post).then(pl.col("y") - delta).otherwise(pl.col("y")).alias("y")
    )


@pytest.mark.parametrize("delta", [0.5, 5.0, 20.0])
def test_injected_step_equals_h0_shift_of_unshifted_panel(delta):
    # Subtracting `delta` from the treated post-period and testing h0=0 hands
    # conformal_null_residuals the same adjusted matrix as the unshifted panel
    # at h0=delta (it subtracts h0 from exactly those cells), so the residuals
    # and p-value are bit-identical. This is the lemma behind methodology.md
    # §5.9: a sweep of injected effects *is* the p(h0) test-inversion curve of
    # §5.5, non-monotonicity included.
    panel, t0 = _cumulative_panel()
    kw = dict(unit="unit", time="time", outcome="y", treated="u00", treatment_time=t0)
    base = Synth(fixedeff=True).fit(panel, **kw)
    shifted = Synth(fixedeff=True).fit(_subtract_step(panel, delta, t0), **kw)

    # Both calls sit in the absorbed-share regime this panel was built for;
    # the warning is covered by its own test below, not by this identity.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="conformal_test: the refit under H0")
        via_injection = conformal_test(shifted, 0.0)
        via_h0 = conformal_test(base, delta)

    assert np.array_equal(via_injection.residuals, via_h0.residuals)
    assert via_injection.pvalue == via_h0.pvalue


def test_null_refit_explains_most_of_an_injected_step_on_a_cumulative_panel():
    # Characterises the power gap of methodology.md §5.9 / known-issues I-5:
    # on a trending outcome with a rich donor pool the full-window refit under
    # H0 follows the shifted post-period (re-weighting plus the full-window
    # fixed effect), so most of the estimated gap is explained away before the
    # statistic reads the post residuals. The share is far above the
    # fixed-effect share T1/T alone (§5.6); on this panel it is ~0.89.
    panel, t0 = _cumulative_panel()
    delta = 5.0
    fit = Synth(fixedeff=True).fit(
        _subtract_step(panel, delta, t0),
        unit="unit",
        time="time",
        outcome="y",
        treated="u00",
        treatment_time=t0,
    )
    post = ~fit.pre_mask_
    pre_rmspe = float(np.sqrt(np.mean(fit.gap_[fit.pre_mask_] ** 2)))
    assert delta > 10 * pre_rmspe  # the step is unmistakable on the pre-fit scale
    assert fit.att_ < -delta / 2  # and the point estimate sees it

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="conformal_test: the refit under H0")
        res = conformal_test(fit, 0.0)
    absorbed = 1.0 - float(res.residuals[post].mean()) / fit.att_
    fixed_effect_share = post.sum() / post.shape[0]
    assert absorbed > 0.5
    assert absorbed > fixed_effect_share + 0.2
    # ... and the test does not reject at 10% despite a step > 10 pre-RMSPEs.
    assert res.pvalue > 0.1


def test_absorbed_share_is_the_documented_ratio_and_warns_on_the_cumulative_panel():
    # The field is 1 - mean(post residuals) / (att_ - h0) (methodology §5.9);
    # on this panel the refit explains ~0.89 of a 5-unit step, past the 0.75
    # warning threshold, and the step dwarfs the pre-period RMSPE, so
    # conformal_test warns.
    panel, t0 = _cumulative_panel()
    fit = Synth(fixedeff=True).fit(
        _subtract_step(panel, 5.0, t0),
        unit="unit",
        time="time",
        outcome="y",
        treated="u00",
        treatment_time=t0,
    )
    with pytest.warns(UserWarning, match=r"explains \d+% of the post-period gap") as record:
        res = conformal_test(fit, 0.0)
    assert len(record) == 1
    post = ~fit.pre_mask_
    manual = 1.0 - float(res.residuals[post].mean()) / fit.att_
    assert res.absorbed_share == pytest.approx(manual)
    assert res.absorbed_share > 0.75
    # conformal_pvalue is the per-refit primitive and stays silent by design
    # (filterwarnings=error would turn a stray warning into a failure here).
    assert conformal_pvalue(fit, 0.0) == res.pvalue


def test_absorbed_share_does_not_warn_at_a_well_specified_null():
    # At h0 == att_ there is no gap to absorb: the share is nan and no warning
    # is emitted (filterwarnings=error makes a stray one fail the test).
    panel, t0 = _cumulative_panel()
    fit = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="y", treated="u00", treatment_time=t0
    )
    res = conformal_test(fit, fit.att_)
    assert np.isnan(res.absorbed_share)


def test_absorbed_share_gate_ignores_gaps_inside_the_pre_period_noise():
    # A gap no larger than the pre-period RMSPE makes the share a ratio of two
    # small numbers; the gate suppresses the warning there whatever the share
    # is (filterwarnings=error turns a stray warning into a failure).
    panel, t0 = _cumulative_panel()
    fit = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="y", treated="u00", treatment_time=t0
    )
    pre_rmspe = float(np.sqrt(np.mean(fit.gap_[fit.pre_mask_] ** 2)))
    assert pre_rmspe > 0
    res = conformal_test(fit, fit.att_ - 0.5 * pre_rmspe)
    assert np.isfinite(res.absorbed_share)
