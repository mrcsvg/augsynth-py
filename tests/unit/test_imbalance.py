"""Unit tests for L2 imbalance (no R)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from augsynth_py import AugSynth, Synth
from augsynth_py.synth._panel import imbalance


def test_imbalance_matches_manual_norm() -> None:
    rng = np.random.default_rng(1)
    y1 = rng.normal(size=10)
    y0 = rng.normal(size=(10, 4))
    # Non-uniform weights so the scaled assertion distinguishes the fitted
    # numerator from the uniform-baseline denominator.
    w = np.array([0.7, 0.1, 0.1, 0.1])
    l2, scaled = imbalance(y1, y0, w)

    resid = y1 - y0 @ w
    w_unif = np.full(4, 0.25)
    resid_unif = y1 - y0 @ w_unif
    np.testing.assert_allclose(l2, np.linalg.norm(resid))
    np.testing.assert_allclose(scaled, np.linalg.norm(resid) / np.linalg.norm(resid_unif))


def test_scaled_imbalance_is_one_for_uniform_weights() -> None:
    rng = np.random.default_rng(2)
    y1 = rng.normal(size=8)
    y0 = rng.normal(size=(8, 3))
    _, scaled = imbalance(y1, y0, np.full(3, 1 / 3))
    np.testing.assert_allclose(scaled, 1.0)


def test_zero_denominator_follows_ieee_convention() -> None:
    # Identical donor columns equal to y1: the uniform 1/J baseline fits
    # exactly, so the denominator is 0. Convention matches unguarded R
    # division: x/0 = Inf, 0/0 = NaN.
    y1 = np.array([1.0, 2.0, 3.0])
    y0 = np.column_stack([y1, y1])

    # Weights sum to 2 -> fitted residual is -y1, so l2 > 0: scaled is inf.
    l2, scaled = imbalance(y1, y0, np.array([1.0, 1.0]))
    assert l2 > 0.0
    assert scaled == float("inf")

    # Uniform weights also fit exactly -> 0/0: scaled is nan.
    l2, scaled = imbalance(y1, y0, np.array([0.5, 0.5]))
    np.testing.assert_allclose(l2, 0.0, atol=1e-15)
    assert np.isnan(scaled)


def test_synth_exposes_imbalance_attributes() -> None:
    rng = np.random.default_rng(3)
    names = [f"u{i}" for i in range(5)]
    rows = []
    for j, name in enumerate(names):
        for t in range(10):
            rows.append({"unit": name, "time": t, "Y": float(rng.normal() + j)})
    panel = pl.DataFrame(rows)
    est = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=7
    )
    assert isinstance(est.l2_imbalance_, float)
    assert isinstance(est.scaled_l2_imbalance_, float)
    assert est.l2_imbalance_ >= 0.0
    # A fitted SCM should not be worse than uniform weights.
    assert est.scaled_l2_imbalance_ <= 1.0 + 1e-9


def test_augsynth_exposes_imbalance_attributes() -> None:
    rng = np.random.default_rng(4)
    names = [f"u{i}" for i in range(5)]
    rows = []
    for j, name in enumerate(names):
        for t in range(12):
            rows.append({"unit": name, "time": t, "Y": float(rng.normal() + j)})
    panel = pl.DataFrame(rows)
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=8
    )
    assert isinstance(est.l2_imbalance_, float)
    assert isinstance(est.scaled_l2_imbalance_, float)
    assert math.isfinite(est.l2_imbalance_)
    assert math.isfinite(est.scaled_l2_imbalance_)
    assert est.l2_imbalance_ >= 0.0
    # Provably <= 1: uniform 1/J is simplex-feasible, so the SCM optimum
    # satisfies l2_scm <= l2_unif; and gamma = 0 is ridge-feasible, so the
    # augmented optimum satisfies l2_aug <= l2_scm <= l2_unif.
    assert est.scaled_l2_imbalance_ <= 1.0 + 1e-9


def test_multi_treated_imbalance_intersection() -> None:
    """Intersection of the two P0.1 features: a multi-treated fit must still
    expose sound L2 imbalance attributes.

    The per-feature suites exercise multi-treated pooling and imbalance
    separately; nothing pins down their composition -- that the imbalance
    reported for a fit on a *group-mean* treated series is the imbalance of
    that pooled series. This guards it for both estimators, and for Synth
    independently recomputes ``l2_imbalance_`` from the reconstructed group
    mean in the estimator's own fixedeff-demeaned fit space using its reported
    simplex weights, so a regression in either the pooling map or the imbalance
    plumbing would surface here.
    """
    rng = np.random.default_rng(11)
    names = [f"u{i}" for i in range(5)]
    n_time = 12
    # Pre-draw the full (T, N) outcome matrix so the manual recompute below can
    # reconstruct the exact fit space rather than replaying the RNG cell by cell.
    wide = rng.normal(size=(n_time, len(names))) + np.arange(len(names))[None, :]
    rows = [
        {"unit": name, "time": t, "Y": float(wide[t, j])}
        for j, name in enumerate(names)
        for t in range(n_time)
    ]
    panel = pl.DataFrame(rows)

    treated = ["u0", "u1"]
    treatment_time = 8

    synth = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=treated,
        treatment_time=treatment_time,
    )
    aug = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=treated,
        treatment_time=treatment_time,
    )

    for est in (synth, aug):
        assert math.isfinite(est.l2_imbalance_)
        assert est.l2_imbalance_ >= 0.0
        assert math.isfinite(est.scaled_l2_imbalance_)
        # Uniform 1/J is simplex-feasible (and gamma = 0 is ridge-feasible for
        # AugSynth), so the fitted scaled imbalance cannot exceed the uniform
        # baseline.
        assert est.scaled_l2_imbalance_ <= 1.0 + 1e-9

    # Random data + ridge augmentation -> strictly positive residual.
    assert aug.l2_imbalance_ > 0.0

    # Strong guard for Synth: rebuild the group-mean treated series and the
    # fixedeff-demeaned pre-period fit space by hand, then reproduce
    # l2_imbalance_ from the estimator's own reported simplex weights. (For
    # AugSynth the fit space is period-demeaned with effective omega + gamma
    # weights, so the finiteness + <= 1 bound + l2 > 0 checks above stand in.)
    periods = synth.periods_
    pre = periods < treatment_time
    unit_cols = {name: wide[:, j] for j, name in enumerate(names)}
    # long_to_wide pools ["u0", "u1"] into their elementwise mean at column 0.
    treated_series = np.mean([unit_cols["u0"], unit_cols["u1"]], axis=0)
    donor_names = synth.units_[1:]  # donor pool, treated group is units_[0]
    donor_matrix = np.column_stack([unit_cols[name] for name in donor_names])

    # Same transform the estimator applies: subtract each column's pre mean.
    treated_dm = treated_series - treated_series[pre].mean()
    donor_dm = donor_matrix - donor_matrix[pre].mean(axis=0)

    w = np.array([synth.weights_[name] for name in donor_names])
    manual_l2 = float(np.linalg.norm(treated_dm[pre] - donor_dm[pre] @ w))
    np.testing.assert_allclose(synth.l2_imbalance_, manual_l2)
