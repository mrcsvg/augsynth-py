"""Unit tests for the absorbed-null (interpolating refit) diagnostic.

CWZ 2021 refits the synthetic control on all ``T`` periods under each null
``h0``. A ridge-augmented refit with a donor pool as large as the window
(roughly ``J >= T``) and a tiny CV-selected ``lambda_`` interpolates the
``h0``-adjusted treated series, so the null residuals are solver noise for
*every* ``h0`` and the test has no power: the p-value stops responding to
``h0`` and ``conformal_interval`` runs to its search bounds. The classical
simplex fit cannot interpolate and is unaffected. See
``docs/known-issues.md`` I-5 and ``docs/methodology.md`` §5.9.

These tests reproduce the regime synthetically (80 donors on 60 periods) and
assert the ``UserWarning`` from ``_warn_if_null_absorbed``; they change no
estimate. ``filterwarnings = error`` in ``pyproject.toml`` turns any stray
warning in the negative controls into a failure.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import augsynth_py.inference as inference
from augsynth_py import AugSynth, Synth, conformal_interval, conformal_pvalue, conformal_test
from augsynth_py.inference import _ABSORBED_SHIFT_FRACTION, _warn_if_null_absorbed

T = 60
T0 = 45
EFFECT = 10.0


def _panel(n_donors: int, seed: int = 0) -> pl.DataFrame:
    # Outcome scale ~100 (the CAGED-like panel that motivated I-5), a smooth
    # common component, unit level shifts and iid noise of sd 1 on every
    # unit; the treated unit carries a constant post-period effect of 10,
    # i.e. ~10x the noise, so a powered test must reject h0 = 0.
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 100.0 + 5.0 * np.sin(t / 6.0) + 0.1 * t
    treated = base + rng.normal(0.0, 1.0, T)
    treated[T0:] += EFFECT
    rows = [{"unit": "trt", "time": int(k), "y": float(treated[k])} for k in range(T)]
    for j in range(n_donors):
        donor = base + rng.normal(0.0, 3.0) + rng.normal(0.0, 1.0, T) * rng.uniform(0.5, 2.0)
        rows.extend({"unit": f"d{j}", "time": int(k), "y": float(donor[k])} for k in range(T))
    return pl.DataFrame(rows)


def _fit(estimator, n_donors: int):
    return estimator.fit(
        _panel(n_donors),
        unit="unit",
        time="time",
        outcome="y",
        treated="trt",
        treatment_time=T0,
    )


@pytest.fixture(scope="module")
def interpolating_fit() -> AugSynth:
    # J = 80 > T = 60. LOO-CV drives lambda to the auto-grid floor (1e-4 *
    # var(y)) -- the existing boundary warning fires -- and the fit
    # interpolates the pre-period.
    with pytest.warns(UserWarning, match="left boundary"):
        fit = _fit(AugSynth(), n_donors=80)
    return fit


def test_regime_is_reproduced(interpolating_fit: AugSynth) -> None:
    fit = interpolating_fit
    # Point estimate is fine (the effect is 10); it is the inference that
    # degenerates. Pre-period RMSPE is at the solver-noise floor, and the
    # refit under h0 = 0 absorbs the whole 10-unit shift: its residual RMS is
    # orders of magnitude below the sqrt(T0*T1)/T ~ 0.43 share a
    # non-absorbing refit would leave.
    assert fit.att_ == pytest.approx(EFFECT, abs=1.0)
    assert fit.rmspe_pre_ < 1e-4
    resid = fit.conformal_null_residuals(0.0)
    rms = float(np.sqrt(np.mean(resid**2)))
    assert rms < 1e-3 * abs(fit.att_)


def test_conformal_pvalue_warns_when_refit_absorbs_the_null(interpolating_fit: AugSynth) -> None:
    with pytest.warns(UserWarning, match="absorbed by the refit") as rec:
        p = conformal_pvalue(interpolating_fit, 0.0)
    assert 0.0 <= p <= 1.0
    msg = str(rec[0].message)
    assert "J >= T" in msg
    assert "80 donors" in msg
    assert "known-issues.md I-5" in msg


def test_conformal_test_warns_and_exposes_near_zero_residuals(
    interpolating_fit: AugSynth,
) -> None:
    with pytest.warns(UserWarning, match="absorbed by the refit"):
        result = conformal_test(interpolating_fit, 0.0)
    # The residual vector on the result object is what the guard inspected.
    assert float(np.abs(result.residuals).max()) < 1e-2


def test_conformal_interval_warns_absorbed_and_runs_to_search_bounds(
    interpolating_fit: AugSynth,
) -> None:
    fit = interpolating_fit
    with pytest.warns(UserWarning) as rec:
        lo, hi = conformal_interval(fit, alpha=0.05, grid_size=11)
    messages = [str(w.message) for w in rec]
    assert any("absorbed by the refit" in m for m in messages)
    truncated = [m for m in messages if "may be truncated" in m]
    assert truncated, messages
    # The truncation message now names the regime.
    assert "J >= T" in truncated[0]
    # Every h0 is accepted, so the bounds are the search bounds: the span
    # doubled 8 times from 6*sd(post gap) -- wildly wider than the effect.
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < 0.0 < fit.att_ < hi
    assert (hi - lo) > 50.0 * abs(fit.att_)


def test_classical_synth_on_the_same_panel_is_unaffected() -> None:
    # The simplex fit cannot interpolate: the 10-unit shift under h0 = 0
    # surfaces in the residuals, no warning fires (filterwarnings = error),
    # and the test rejects at the 1/T floor.
    fit = _fit(Synth(), n_donors=80)
    p = conformal_pvalue(fit, 0.0)
    assert p <= 0.05
    assert p == pytest.approx(1.0 / T)


def test_small_donor_pool_does_not_warn() -> None:
    # J = 30 < T = 60: CV picks an interior lambda, the refit cannot absorb
    # the shift and the test keeps its power. No warning at fit or test time.
    fit = _fit(AugSynth(), n_donors=30)
    p = conformal_pvalue(fit, 0.0)
    assert p <= 0.05


# ---------------------------------------------------------------------------
# The guard itself, on controlled residual vectors.
# ---------------------------------------------------------------------------


class _StubFit:
    def __init__(self, att: float, n_donors: int | None = None) -> None:
        self.att_ = att
        if n_donors is not None:
            self.weights_ = {f"d{j}": 0.0 for j in range(n_donors)}


def _mask(t: int = 60, t0: int = 45) -> np.ndarray:
    return np.arange(t) >= t0


def test_guard_fires_below_the_fraction_of_the_no_absorption_bound() -> None:
    post = _mask()
    t = post.shape[0]
    n_post = int(post.sum())
    shift = 10.0
    rms_min = shift * np.sqrt((t - n_post) * n_post) / t
    resid = np.full(t, 0.5 * _ABSORBED_SHIFT_FRACTION * rms_min)
    with pytest.warns(UserWarning, match="absorbed by the refit") as rec:
        _warn_if_null_absorbed(_StubFit(att=0.0, n_donors=80), resid, post, h0=shift)
    assert "T0=45, T1=15, 80 donors" in str(rec[0].message)


def test_guard_is_silent_at_or_above_the_bound() -> None:
    post = _mask()
    t = post.shape[0]
    n_post = int(post.sum())
    shift = 10.0
    rms_min = shift * np.sqrt((t - n_post) * n_post) / t
    # Exactly the non-absorbing split of methodology.md 5.6: no warning.
    resid = np.where(post, shift * (t - n_post) / t, -shift * n_post / t)
    assert float(np.sqrt(np.mean(resid**2))) == pytest.approx(rms_min)
    _warn_if_null_absorbed(_StubFit(att=0.0), resid, post, h0=shift)
    # Just above the threshold: still silent.
    resid = np.full(t, 1.01 * _ABSORBED_SHIFT_FRACTION * rms_min)
    _warn_if_null_absorbed(_StubFit(att=0.0), resid, post, h0=shift)


def test_guard_is_silent_when_there_is_nothing_to_absorb() -> None:
    # h0 == att_: a perfect fit at the point estimate is legitimate, not
    # interpolation, so near-zero residuals are not flagged.
    post = _mask()
    resid = np.full(post.shape[0], 1e-12)
    _warn_if_null_absorbed(_StubFit(att=2.5), resid, post, h0=2.5)
    # Degenerate windows (no pre or no post period) are not scored either.
    _warn_if_null_absorbed(_StubFit(att=0.0), resid, np.ones(5, dtype=bool), h0=1.0)
    _warn_if_null_absorbed(_StubFit(att=0.0), resid, np.zeros(5, dtype=bool), h0=1.0)


def test_guard_message_omits_donor_count_without_weights() -> None:
    post = _mask()
    resid = np.zeros(post.shape[0])
    with pytest.warns(UserWarning, match="absorbed by the refit") as rec:
        _warn_if_null_absorbed(_StubFit(att=0.0), resid, post, h0=1.0)
    assert "donors" not in str(rec[0].message).split("here T0=")[1].split(")")[0]


def test_guard_is_wired_into_pvalue_and_test(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both public entry points hand the null residuals to the guard; an
    # estimator whose refit returns zeros under a non-trivial h0 is flagged
    # by each of them (and only once per call).
    class _ZeroRefit:
        gap_ = np.zeros(60)
        pre_mask_ = ~_mask()
        att_ = 3.0

        def conformal_null_residuals(self, h0: float) -> np.ndarray:
            return np.zeros(60)

    fit = _ZeroRefit()
    with pytest.warns(UserWarning, match="absorbed by the refit") as rec:
        conformal_pvalue(fit, 0.0)
    assert len(rec) == 1
    with pytest.warns(UserWarning, match="absorbed by the refit") as rec:
        inference.conformal_test(fit, 0.0)
    assert len(rec) == 1
