"""Unit tests for :func:`augsynth_py.inference.conformal_interval`.

These exercise the CWZ 2021 test-inversion confidence interval: the CI is the
set of null effects ``h0`` that the conformal test fails to reject at level
``alpha`` (i.e. ``conformal_pvalue(fit, h0) >= alpha``).
"""

import numpy as np
import polars as pl
import pytest

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


def _fit_with_effect(effect, T=30, t0=20):  # noqa: N803
    # Powered, imperfect-fit regime. Two properties matter for the CI to be a
    # genuine bounded interval rather than an artifact:
    #   * T >= 20 so the block p-value floor 1/T (~0.033) is below alpha=0.05 --
    #     the conformal test can actually reject, so the acceptance region is
    #     bounded. With the old T=14 the floor was 1/14 ~ 0.071 > 0.05, making
    #     the region structurally unbounded (the truncation guard now flags it).
    #   * Mild noise gives the pre-fit real residual dispersion (sd > 0), so the
    #     inverted interval has width rather than collapsing to a point.
    rng = np.random.default_rng(0)
    base = np.linspace(1.0, 2.0, T)
    treated = base + rng.normal(0.0, 0.05, T)
    treated[t0:] += effect
    donors = [
        base + rng.normal(0.0, 0.05, T),
        base + 1.5 + rng.normal(0.0, 0.05, T),
        base - 0.7 + rng.normal(0.0, 0.05, T),
    ]
    return Synth(fixedeff=False).fit(
        _panel(treated, donors),
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
    # At alpha=0.20 this panel's acceptance region is genuinely non-contiguous:
    # block p-values are quantized to k/T (T=30) and a k=5 dip (p ~ 0.167)
    # sits strictly inside the accepted envelope, so the D-6 diagnostic fires.
    # Deterministic (block scheme, fixed DGP seed) -- a real-panel counterpart
    # to test_non_contiguous_acceptance_region_warns.
    with pytest.warns(UserWarning, match="non-contiguous"):
        lo20, hi20 = conformal_interval(fit, alpha=0.20, grid_size=81)
    # A larger alpha rejects more nulls -> the acceptance region can only shrink.
    assert lo05 <= lo20 and hi20 <= hi05


def test_zero_is_evaluated_and_returns_finite_interval():
    fit = _fit_with_effect(2.0)
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=41)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_single_post_period_does_not_crash():
    # Exactly one post period (post_gap.size == 1 -> sd == 0, center-scaled grid
    # fallback). T = 25 keeps the block floor 1/25 = 0.04 below alpha = 0.1 so
    # the region is bounded; mild noise avoids a perfect fit.
    rng = np.random.default_rng(0)
    T = 25  # noqa: N806
    t0 = 24  # one post period
    base = np.linspace(1.0, 2.0, T)
    treated = base + rng.normal(0.0, 0.05, T)
    treated[t0:] += 1.0
    donors = [base + rng.normal(0.0, 0.05, T), base + 1.0 + rng.normal(0.0, 0.05, T)]
    fit = Synth(fixedeff=False).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )
    lo, hi = conformal_interval(fit, alpha=0.1, grid_size=41)
    assert lo <= hi


def test_wider_interval_contains_zero_for_small_effect():
    # A tiny effect relative to noise -> 0 should be inside the 95% CI.
    fit = _fit_with_effect(0.0)
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=81)
    assert lo <= 0.0 <= hi


def test_empty_acceptance_region_returns_nan_nan():
    # (nan, nan) requires the PEAK of the p(h0) curve to fall below alpha, which
    # is driven by residual NON-EXCHANGEABILITY (a poor fit) and is only possible
    # when 1/T < alpha (here T=60 -> 1/T ~ 0.017 < 0.05). A steep post-period
    # RAMP divergence that no *constant* h0 can flatten, tracked only by flat
    # donors, keeps the post-window statistic atypically large at every h0, so
    # the peak p(h0) = 2/60 ~ 0.033 < 0.05 and NO grid point (nor 0.0) is
    # accepted -> the acceptance region is empty.
    #
    # This is the opposite regime from the small-T (T <= 1/alpha) case, where the
    # 1/T floor keeps p(h0) >= alpha at every h0 -> an *unbounded* region and the
    # truncation guard, never (nan, nan). Deterministic (no RNG): peak p is fixed
    # by the block ranking and sits well below alpha, so the empty result is
    # stable across grid sizes.
    T = 60  # noqa: N806
    t0 = 30
    base = np.linspace(1.0, 2.0, T)
    idx = np.arange(T) - t0
    ramp = np.where(idx >= 0, idx * 50.0, 0.0)  # steep, non-constant post divergence
    treated = base + ramp
    donors = [base, base + 1.0, base - 1.0]  # flat donors cannot reproduce a ramp
    fit = Synth(fixedeff=False).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )
    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=201)
    assert np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("bad_side", ["right", "left"])
def test_conformal_interval_rejects_one_sided(bad_side):
    # A one-sided acceptance region is a half-line, not an interval.
    fit = _fit_with_effect(2.0)
    with pytest.raises(ValueError):
        conformal_interval(fit, alpha=0.05, grid_size=41, side=bad_side)


def test_interval_strictly_interior_with_residual_spread():
    # Non-degenerate regime: imperfect pre-fit -> post gaps have real variance
    # (sd > 0), long enough panel that the block peak p exceeds alpha=0.05, so
    # the inverted interval must be strictly interior to the +/-6*sd grid span
    # rather than pinned to the fallback grid edge.
    #
    # The grid is placed from the point-estimate gap_ post-period sd, but each
    # grid point is scored by the full-window refit (CWZ 2021), whose acceptance
    # region is wider than the no-refit one. This DGP is tuned so the gap_-based
    # +/-6*sd span comfortably contains the refit acceptance region (verified:
    # generous margins on both bounds), keeping the interior check meaningful
    # under the refit semantics.
    T = 70  # noqa: N806
    t0 = 55
    rng = np.random.default_rng(0)
    trend = np.linspace(0.0, 3.0, T)
    treated = trend + rng.normal(0.0, 1.2, T)
    treated[t0:] += 1.0
    donors = [trend + 1.0, trend - 0.5, np.linspace(0.5, 2.0, T)]
    panel = _panel(treated, donors)
    fit = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )

    post_gap = fit.gap_[~fit.pre_mask_]
    sd = float(np.std(post_gap, ddof=1))
    center = float(fit.att_)
    spread = 6.0 * sd
    assert sd > 0.0  # genuine residual spread, not a degenerate perfect fit

    lo, hi = conformal_interval(fit, alpha=0.05, grid_size=201)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert center - spread < lo
    assert hi < center + spread


def test_low_power_interval_expands_past_initial_grid_and_warns():
    # Truncation guard (I1): the grid span starts at +/-6*sd(post gap) from the
    # POINT-ESTIMATE gap_, but each point is scored by the full-window refit
    # acceptance region. Here the donors fit the pre-period almost perfectly
    # (tiny sd -> a sub-unit initial span) yet two of them carry large, opposite
    # post-only level jumps, so the refit can absorb almost any constant h0 ->
    # the conformal test has little power and accepts h0 across a very wide
    # range. Without the guard, min/max would clip to the tiny initial grid and
    # silently understate the interval. The guard must widen the grid (far past
    # the initial span) and, since the region exceeds even the doubling cap
    # here, emit a UserWarning.
    T = 40  # noqa: N806
    t0 = 30
    rng = np.random.default_rng(0)
    base = np.linspace(10.0, 12.0, T)
    post = np.zeros(T)
    post[t0:] = 1.0
    treated = base + rng.normal(0.0, 0.005, T)
    treated[t0:] += 5.0
    donors = [
        base + rng.normal(0.0, 0.005, T) + 40.0 * post,  # large +40 post jump
        base + rng.normal(0.0, 0.005, T) - 40.0 * post,  # large -40 post jump
        base + rng.normal(0.0, 0.005, T),
    ]
    fit = Synth(fixedeff=False).fit(
        _panel(treated, donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=t0,
    )

    post_gap = fit.gap_[~fit.pre_mask_]
    sd = float(np.std(post_gap, ddof=1))
    center = float(fit.att_)
    initial_half_span = 6.0 * sd
    initial_grid_max = center + initial_half_span
    assert sd > 0.0
    # The initial +/-6*sd span is sub-unit; the true acceptance region is orders
    # of magnitude wider, so any correct expansion must blow past it.
    assert 2.0 * initial_half_span < 1.0

    with pytest.warns(UserWarning, match="truncated"):
        lo, hi = conformal_interval(fit, alpha=0.05, grid_size=101)

    assert np.isfinite(lo) and np.isfinite(hi)
    # Expansion happened: the upper bound moved well past the initial grid edge,
    # and the interval is vastly wider than the initial +/-6*sd span.
    assert hi > initial_grid_max + 1.0
    assert (hi - lo) > 20.0 * (2.0 * initial_half_span)


class _StubFit:
    # Minimal _FittedSC surface for exercising the interval-extraction layer
    # with a controlled p(h0) curve. gap_/pre_mask_/att_ place the grid;
    # _conformal_null_residuals is never reached because conformal_pvalue is
    # monkeypatched in the tests below.
    def __init__(self, T=30, t0=20):  # noqa: N803
        gap = np.zeros(T)
        gap[t0:] = np.linspace(-1.0, 1.0, T - t0)  # sd ~ 0.6 -> spread ~ 3.7
        self.gap_ = gap
        self.pre_mask_ = np.arange(T) < t0
        self.att_ = float(gap[t0:].mean())

    def _conformal_null_residuals(self, h0):
        raise AssertionError("must not be called when conformal_pvalue is stubbed")


def test_non_contiguous_acceptance_region_warns(monkeypatch):
    # D-6 (clean-room audit 2026-07-07, Recommendation 4): the acceptance
    # region of the refit-under-null test can have a rejected gap strictly
    # inside the accepted envelope (demonstrated on the Basque panel, where
    # the gap contains att_ itself -- methodology.md 5.5). min/max extraction
    # must then WARN rather than silently present the envelope as a connected
    # interval. The p(h0) curve here is stubbed: accepted on [-2,-1] u [1,2],
    # rejected in between and outside -- the curve itself is pinned against R
    # by test_basque_pvalue_curve_matches_r_exact.
    import augsynth_py.inference as inference

    def fake_pvalue(fit, h0=0.0, **kwargs):
        return 1.0 if 1.0 <= abs(h0) <= 2.0 else 0.0

    monkeypatch.setattr(inference, "conformal_pvalue", fake_pvalue)
    with pytest.warns(UserWarning, match="non-contiguous"):
        lo, hi = inference.conformal_interval(_StubFit(), alpha=0.05, grid_size=41)
    # The envelope brackets both accepted islands (grid resolution ~0.19).
    assert lo == pytest.approx(-2.0, abs=0.2)
    assert hi == pytest.approx(2.0, abs=0.2)


def test_contiguous_acceptance_region_does_not_warn(monkeypatch):
    # Complement of the non-contiguity test: a connected accepted run must
    # return silently (filterwarnings=error in pytest.ini makes any stray
    # warning fail this test).
    import augsynth_py.inference as inference

    def fake_pvalue(fit, h0=0.0, **kwargs):
        return 1.0 if abs(h0) <= 2.0 else 0.0

    monkeypatch.setattr(inference, "conformal_pvalue", fake_pvalue)
    lo, hi = inference.conformal_interval(_StubFit(), alpha=0.05, grid_size=41)
    assert lo == pytest.approx(-2.0, abs=0.2)
    assert hi == pytest.approx(2.0, abs=0.2)
