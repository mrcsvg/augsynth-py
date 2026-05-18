"""Unit tests for the ridge-augmented synthetic control estimator (M1 scope).

These tests exercise the private primitives in
:mod:`augsynth_py.synth.augmented`. The public ``AugSynth`` class is M3 and
not yet implemented; U9 (input-validation surface) is therefore deferred to
M3 as well.

Test ids U1..U8 map to the inventory in
``docs/plans/2026-05-16-augsynth-v0.2-design.md`` "Unit-test inventory".
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from augsynth_py import Synth
from augsynth_py.synth.augmented import _fit_at_lambda, _ridge_augment


def _long_panel(y_matrix: np.ndarray, units: list[str]) -> pl.DataFrame:
    """Build a long polars panel from a ``(T, N)`` outcome matrix.

    Local copy of the helper used in ``test_classical_synth.py`` -- the two
    test files are kept independent so renames in one don't perturb the
    other.
    """
    n_periods, n_units = y_matrix.shape
    assert n_units == len(units)
    rows = []
    for t in range(n_periods):
        for j, name in enumerate(units):
            rows.append({"unit": name, "day": t, "y": float(y_matrix[t, j])})
    return pl.DataFrame(rows)


def test_ridge_augment_matches_closed_form_on_tiny_case() -> None:
    """gamma must equal (Y0.T @ Y0 + lambda I)^-1 Y0.T (Y1 - Y0 omega) on a tiny case."""
    # T0 = 3 pre-periods, J = 2 donors. omega chosen so the residual is non-trivial.
    y_pre_donors = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    y_pre_treated = np.array([1.5, 2.0, 4.0], dtype=np.float64)
    scm_weights = np.array([0.5, 0.5], dtype=np.float64)  # not on simplex; that's fine
    lambda_ = 1.0

    residual = y_pre_treated - y_pre_donors @ scm_weights
    expected_gamma = np.linalg.solve(
        y_pre_donors.T @ y_pre_donors + lambda_ * np.eye(2),
        y_pre_donors.T @ residual,
    )
    expected_pre_correction = y_pre_donors @ expected_gamma

    gamma, pre_correction = _ridge_augment(scm_weights, y_pre_donors, y_pre_treated, lambda_)

    assert gamma.shape == (2,)
    assert pre_correction.shape == (3,)
    np.testing.assert_allclose(gamma, expected_gamma, atol=1e-12, rtol=0)
    np.testing.assert_allclose(pre_correction, expected_pre_correction, atol=1e-12, rtol=0)


def test_ridge_augment_rejects_negative_lambda() -> None:
    """lambda_ < 0 has no statistical meaning; must raise ValueError."""
    y0 = np.eye(2, dtype=np.float64)
    y1 = np.array([1.0, 1.0], dtype=np.float64)
    omega = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="lambda_ must be non-negative"):
        _ridge_augment(omega, y0, y1, lambda_=-1.0)


def test_ridge_augment_shrinks_gamma_to_zero_at_large_lambda(
    rng: np.random.Generator,
) -> None:
    """For huge lambda, gamma ~ (Y0.T @ r) / lambda, which vanishes; primitive-level check."""
    t0, n_donors = 12, 5
    y_pre_donors = rng.normal(0.0, 1.0, (t0, n_donors))
    y_pre_treated = rng.normal(0.0, 1.0, t0)
    omega = np.full(n_donors, 1.0 / n_donors, dtype=np.float64)

    gamma, pre_correction = _ridge_augment(omega, y_pre_donors, y_pre_treated, lambda_=1e15)

    assert np.max(np.abs(gamma)) < 1e-10
    assert np.max(np.abs(pre_correction)) < 1e-9


def test_u3_weight_decomposition_holds(rng: np.random.Generator) -> None:
    """U3: weights_ == scm_weights_ + augmentation_weights_ to numerical precision.

    BFR 2021 Section 2.4 Lemma 1 -- this is the identity the rest of the API
    rests on. Tolerance is tight because the decomposition is exact in
    floating-point arithmetic (no nonlinear solver).
    """
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])

    result = _fit_at_lambda(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=25,
        lambda_=1.0,
    )

    for donor in result.scm_weights:
        decomposed = result.scm_weights[donor] + result.augmentation_weights[donor]
        assert abs(result.weights[donor] - decomposed) < 1e-10


def test_u1_scm_weights_lie_on_simplex(rng: np.random.Generator) -> None:
    """U1: even with ridge augmentation, the inner SCM weights stay on the simplex.

    Augmentation operates on a *separate* parameter gamma -- it must not
    perturb omega. This is mostly a regression test that we keep omega and
    gamma disjoint in storage.
    """
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])

    result = _fit_at_lambda(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=25,
        lambda_=1.0,
    )

    omega = np.array(list(result.scm_weights.values()))
    assert (omega >= -1e-9).all()
    assert abs(omega.sum() - 1.0) < 1e-6


def test_u2_effective_weights_can_be_negative(rng: np.random.Generator) -> None:
    """U2: the API must not clip omega + gamma back onto the simplex.

    Ridge can -- and on most non-degenerate panels at moderate lambda_ does
    -- produce negative entries in gamma. The effective weights omega +
    gamma then have negative entries as well. Verify the dict carries them
    through unchanged.
    """
    # Wide donor pool + low lambda_ -> near-OLS gamma, which generically has
    # signs of both kinds on noisy data.
    y_matrix = rng.normal(100.0, 10.0, (40, 12))
    units = ["t", *(f"d{i}" for i in range(11))]
    panel = _long_panel(y_matrix, units=units)

    result = _fit_at_lambda(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=25,
        lambda_=1e-3,
    )

    effective = np.array(list(result.weights.values()))
    assert effective.min() < -1e-6, (
        f"Expected at least one negative effective weight at small lambda_; "
        f"got min={effective.min():.6f}. If this is flaky on this seed, "
        f"investigate whether the implementation silently clips gamma."
    )


def test_u4_large_lambda_reproduces_synth(rng: np.random.Generator) -> None:
    """U4: as lambda_ -> infinity, gamma -> 0 and the augmented fit collapses onto Synth.

    Verifies all three forms of the limit: gamma vanishing, ridge_correction
    vanishing, and counterfactual agreement with a vanilla Synth fit on the
    same panel.
    """
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])
    common_kwargs: dict[str, object] = {
        "unit": "unit",
        "time": "day",
        "outcome": "y",
        "treated": "t",
        "treatment_time": 25,
    }

    synth = Synth(fixedeff=True).fit(panel, **common_kwargs)
    aug = _fit_at_lambda(panel, lambda_=1e15, fixedeff=True, **common_kwargs)

    # gamma vanishes.
    gamma = np.array(list(aug.augmentation_weights.values()))
    assert np.max(np.abs(gamma)) < 1e-9

    # ridge_correction vanishes.
    assert np.max(np.abs(aug.ridge_correction)) < 1e-6

    # Counterfactual matches Synth's, on both the SCM-only and augmented
    # paths (they collapse together in this limit).
    np.testing.assert_allclose(aug.synthetic_scm, synth.synthetic_, atol=1e-9)
    np.testing.assert_allclose(aug.synthetic, synth.synthetic_, atol=1e-6)

    # Effective weights collapse onto the simplex SCM weights.
    for donor, w_synth in synth.weights_.items():
        assert abs(aug.weights[donor] - w_synth) < 1e-9


def test_u5_small_lambda_drives_pre_gap_to_zero(rng: np.random.Generator) -> None:
    """U5: at lambda_ -> 0, gamma -> OLS and (when T0 <= J) interpolates the residual.

    The pre-period correction Y0_pre @ gamma then equals the projection of
    the residual onto the column space of Y0_pre. When that span is all of
    R^T0, the residual is its own projection and the pre-period gap is
    machine-zero.
    """
    # T0 = 8, J = 12: more donors than pre-periods -> exact interpolation.
    t_total, t0, n_donors = 16, 8, 12
    y_matrix = rng.normal(100.0, 10.0, (t_total, n_donors + 1))
    units = ["t", *(f"d{i}" for i in range(n_donors))]
    panel = _long_panel(y_matrix, units=units)

    result = _fit_at_lambda(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=t0,
        lambda_=1e-10,
    )

    pre_gap = result.gap[result.pre_mask]
    assert pre_gap.shape == (t0,)
    assert np.max(np.abs(pre_gap)) < 1e-5, (
        f"Expected near-zero pre-period gap in interpolation regime; "
        f"got max|gap_pre| = {np.max(np.abs(pre_gap)):.3e}."
    )


def test_u8_perfect_donor_dominates_with_small_correction(
    rng: np.random.Generator,
) -> None:
    """U8: when one donor matches the treated pre-period, ridge stays well-behaved.

    SCM concentrates omega on the matching donor; the residual is ~0; gamma
    is therefore ~0; the effective weight on the matching donor stays close
    to 1, and no other donor blows up.

    Mirrors ``test_perfect_donor_gets_full_weight`` from the Synth suite.
    """
    n_periods = 30
    treatment_time = 20
    treated = np.linspace(100.0, 130.0, n_periods) + rng.normal(0.0, 0.5, n_periods)

    match = treated.copy()
    match[treatment_time:] += rng.normal(0.0, 5.0, n_periods - treatment_time)

    other_c = np.linspace(50.0, 40.0, n_periods) + rng.normal(0.0, 1.0, n_periods)
    other_d = np.full(n_periods, 80.0) + rng.normal(0.0, 1.0, n_periods)
    other_e = np.linspace(200.0, 210.0, n_periods) + rng.normal(0.0, 1.0, n_periods)

    y_matrix = np.column_stack([treated, match, other_c, other_d, other_e])
    panel = _long_panel(y_matrix, units=["A_treated", "B_match", "C", "D", "E"])

    result = _fit_at_lambda(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="A_treated",
        treatment_time=treatment_time,
        lambda_=1.0,
    )

    assert result.scm_weights["B_match"] > 0.95
    assert abs(result.weights["B_match"] - 1.0) < 0.1
    for other in ("C", "D", "E"):
        assert abs(result.weights[other]) < 0.1
    assert np.max(np.abs(result.ridge_correction)) < 5.0  # not exploding
