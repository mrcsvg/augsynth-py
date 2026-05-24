"""Unit tests for the ridge-augmented synthetic control estimator (M1 scope).

These tests exercise the private primitives in
:mod:`augsynth_py.synth.augmented`. The public ``AugSynth`` class is M3 and
not yet implemented; U9 (input-validation surface) is therefore deferred to
M3 as well.

Test ids U1..U8 map to the inventory in
``docs/plans/2026-05-16-augsynth-v0.2-design.md`` "Unit-test inventory".
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from augsynth_py import Synth
from augsynth_py.synth.augmented import (
    AugSynth,
    _build_auto_lambda_grid,
    _fit_at_lambda,
    _loo_cv_lambda,
    _ridge_augment,
    _warn_if_lambda_at_boundary,
)


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


def test_auto_grid_scales_with_variance() -> None:
    """Auto-grid is logspace(-4, 4, 50) scaled by var(y_pre_treated) (D6)."""
    y_pre_treated = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64)
    sigma_sq = float(np.var(y_pre_treated, ddof=0))  # 8.0

    grid = _build_auto_lambda_grid(y_pre_treated)

    assert grid.shape == (50,)
    assert grid.dtype == np.float64
    # Strictly increasing.
    assert (np.diff(grid) > 0).all()
    # End points are 10^-4 * sigma_sq and 10^4 * sigma_sq.
    np.testing.assert_allclose(grid[0], 1e-4 * sigma_sq, rtol=1e-12)
    np.testing.assert_allclose(grid[-1], 1e4 * sigma_sq, rtol=1e-12)


def test_auto_grid_rejects_constant_y() -> None:
    """Constant treated series has zero variance -- auto-grid is meaningless."""
    y = np.full(10, 42.0, dtype=np.float64)
    with pytest.raises(ValueError, match="zero variance"):
        _build_auto_lambda_grid(y)


def test_warn_if_lambda_at_boundary_left_edge() -> None:
    """Best index == 0 -> warns about left boundary."""
    grid = np.logspace(-4, 4, 50)
    with pytest.warns(UserWarning, match="left"):
        _warn_if_lambda_at_boundary(best_idx=0, lambda_grid=grid)


def test_warn_if_lambda_at_boundary_right_edge() -> None:
    """Best index == n-1 -> warns about right boundary."""
    grid = np.logspace(-4, 4, 50)
    with pytest.warns(UserWarning, match="right"):
        _warn_if_lambda_at_boundary(best_idx=len(grid) - 1, lambda_grid=grid)


def test_warn_if_lambda_at_boundary_interior_silent() -> None:
    """Interior index must not warn.

    pytest's ``filterwarnings='error'`` (set in pyproject) turns any
    UserWarning into a test failure, so a plain call is the assertion.
    """
    grid = np.logspace(-4, 4, 50)
    _warn_if_lambda_at_boundary(best_idx=10, lambda_grid=grid)
    _warn_if_lambda_at_boundary(best_idx=25, lambda_grid=grid)
    _warn_if_lambda_at_boundary(best_idx=len(grid) - 2, lambda_grid=grid)


def test_loo_cv_matches_manual_computation_on_tiny_case() -> None:
    """For T0=4, J=2, two lambdas, cv_path matches a hand-rolled LOO."""
    from augsynth_py.synth.classical import Synth

    # Deterministic inputs -- no rng. T0=4 pre-periods, J=2 donors.
    y_pre_donors = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ],
        dtype=np.float64,
    )
    y_pre_treated = np.array([1.5, 2.0, 4.0, 3.5], dtype=np.float64)
    lambda_grid = np.array([0.1, 10.0], dtype=np.float64)

    # Manual LOO: for each (lambda, t), drop period t, refit omega + gamma
    # on the remaining 3 periods, predict y_treated[t], record squared error.
    expected_cv_loss = np.zeros(2, dtype=np.float64)
    for lam_idx, lam in enumerate(lambda_grid):
        for t in range(4):
            keep = [i for i in range(4) if i != t]
            y0_lo = y_pre_donors[keep]
            y1_lo = y_pre_treated[keep]
            omega_lo = Synth._solve_simplex_qp(y1_lo, y0_lo)
            gamma_lo, _ = _ridge_augment(omega_lo, y0_lo, y1_lo, lambda_=lam)
            pred = y_pre_donors[t] @ (omega_lo + gamma_lo)
            expected_cv_loss[lam_idx] += (y_pre_treated[t] - pred) ** 2

    best_lambda, cv_path = _loo_cv_lambda(y_pre_donors, y_pre_treated, lambda_grid)

    assert cv_path.shape == (2, 2)
    np.testing.assert_allclose(cv_path[:, 0], lambda_grid, atol=0, rtol=0)
    np.testing.assert_allclose(cv_path[:, 1], expected_cv_loss, atol=1e-10, rtol=1e-10)
    expected_best = lambda_grid[int(expected_cv_loss.argmin())]
    assert best_lambda == pytest.approx(expected_best, abs=1e-12)


def test_loo_cv_path_shape_and_finite(rng: np.random.Generator) -> None:
    """cv_path has the right shape and contains only finite, non-negative losses."""
    y_pre_donors = rng.normal(0.0, 1.0, (10, 4))
    y_pre_treated = rng.normal(0.0, 1.0, 10)
    grid = np.logspace(-3, 3, 7, dtype=np.float64)

    best_lambda, cv_path = _loo_cv_lambda(y_pre_donors, y_pre_treated, grid)

    assert cv_path.shape == (7, 2)
    np.testing.assert_array_equal(cv_path[:, 0], grid)
    assert np.isfinite(cv_path[:, 1]).all()
    assert (cv_path[:, 1] >= 0.0).all()
    assert best_lambda in grid


def test_loo_cv_returns_argmin_of_grid(rng: np.random.Generator) -> None:
    """best_lambda must equal the grid entry with the smallest cv_loss."""
    y_pre_donors = rng.normal(0.0, 1.0, (12, 5))
    y_pre_treated = rng.normal(0.0, 1.0, 12)
    grid = np.logspace(-3, 3, 9, dtype=np.float64)

    best_lambda, cv_path = _loo_cv_lambda(y_pre_donors, y_pre_treated, grid)

    expected = float(cv_path[int(cv_path[:, 1].argmin()), 0])
    assert best_lambda == pytest.approx(expected, abs=1e-12)


def test_loo_cv_rejects_empty_grid(rng: np.random.Generator) -> None:
    """Empty lambda_grid -> ValueError before any QP runs."""
    y_pre_donors = rng.normal(0.0, 1.0, (5, 3))
    y_pre_treated = rng.normal(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="at least one candidate"):
        _loo_cv_lambda(y_pre_donors, y_pre_treated, np.array([], dtype=np.float64))


def test_loo_cv_rejects_t0_below_three(rng: np.random.Generator) -> None:
    """T0 < 3 leaves <2 periods after one is dropped -- meaningless LOO."""
    y_pre_donors = rng.normal(0.0, 1.0, (2, 3))
    y_pre_treated = rng.normal(0.0, 1.0, 2)
    with pytest.raises(ValueError, match="at least T0=3"):
        _loo_cv_lambda(
            y_pre_donors,
            y_pre_treated,
            np.array([1.0], dtype=np.float64),
        )


def test_u6_boundary_warning_fires(rng: np.random.Generator) -> None:
    """U6: when CV picks the largest (or smallest) grid entry, UserWarning fires.

    Construct a panel with very high noise relative to signal; the
    CV-optimal lambda is large. Provide a grid whose largest entry is still
    small enough that the optimum is beyond it -> argmin lands at index
    n-1 -> warning fires.
    """
    # Strong noise on a small panel: the SCM residual carries little signal,
    # so the best ridge does is to flatten gamma -> 0, i.e. pick large lambda.
    y_pre_donors = rng.normal(0.0, 1.0, (12, 5))
    y_pre_treated = rng.normal(0.0, 50.0, 12)  # noise dwarfs donors

    # Grid that tops out before the true optimum.
    grid = np.array([1e-6, 1e-4, 1e-2, 1e0], dtype=np.float64)

    _, cv_path = _loo_cv_lambda(y_pre_donors, y_pre_treated, grid)
    best_idx = int(cv_path[:, 1].argmin())

    # Pre-condition for the U6 assertion: the construction actually pushed
    # the argmin to the right boundary. If this assertion fires, the test
    # setup is wrong (e.g. seed change), not the code.
    assert best_idx == len(grid) - 1, (
        f"U6 setup failed: expected argmin at right boundary, got idx={best_idx}; "
        f"cv_path losses={cv_path[:, 1].tolist()}"
    )

    with pytest.warns(UserWarning, match="right boundary"):
        _warn_if_lambda_at_boundary(best_idx, grid)


def test_u10_cv_picks_interior_lambda_better_than_extremes(
    rng: np.random.Generator,
) -> None:
    """U10: on a structurally-mismatched panel, CV picks an interior lambda whose
    CV loss is strictly less than the loss at either grid extreme.

    Operationalises "CV is doing its job" without overclaiming statistical
    optimality. The mechanical correctness of argmin is covered by
    test_loo_cv_returns_argmin_of_grid; this test additionally checks the
    *content* of the loss surface: it has a non-trivial interior minimum,
    which is what CV is supposed to find.

    DGP. The true treated outcome is donors @ ``true_w`` with ``true_w``
    deliberately outside the simplex (contains a negative entry; sum > 1).
    Simplex SCM cannot represent this combination -> structural residual.
    Ridge augmentation can recover the missing direction. With T0 only
    slightly larger than J, OLS-style augmentation (lambda -> 0) overfits
    the noise badly. The two failure modes pin a U-shape:

    - lambda -> 0  : gamma overfits noise   -> high held-out error
    - lambda -> inf: gamma -> 0, structural residual remains -> high error
    - interior     : bias-variance sweet spot               -> low error
    """
    t0 = 10
    # Eight donors with linearly trending levels and i.i.d. noise.
    donors = np.column_stack(
        [
            np.linspace(10.0, 20.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(8.0, 18.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(12.0, 22.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(15.0, 25.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(11.0, 21.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(13.0, 23.0, t0) + rng.normal(0.0, 1.0, t0),
            np.full(t0, 30.0) + rng.normal(0.0, 1.0, t0),
            np.linspace(0.0, 50.0, t0) + rng.normal(0.0, 2.0, t0),
        ]
    )
    # True combination: extra-simplex (negative coefficient on donor 1, sum > 1).
    # SCM cannot represent this; gamma must supply the missing direction.
    true_w = np.array([1.2, -0.3, 0.4, 0.2, 0.1, 0.2, 0.1, 0.1], dtype=np.float64)
    treated = donors @ true_w + rng.normal(0.0, 1.5, t0)

    grid = np.logspace(-4, 4, 50, dtype=np.float64) * float(np.var(treated, ddof=0))

    _, cv_path = _loo_cv_lambda(donors, treated, grid)
    best_idx = int(cv_path[:, 1].argmin())

    # Interior: not at either end. If this fires, the DGP is no longer
    # showing a U-shaped CV surface on this seed -- inspect the loss path
    # in the failure message.
    assert 0 < best_idx < len(grid) - 1, (
        f"U10: expected interior CV minimum, got idx={best_idx} "
        f"(grid extremes 0 and {len(grid) - 1}). "
        f"Loss at left={cv_path[0, 1]:.3e}, "
        f"right={cv_path[-1, 1]:.3e}, "
        f"argmin={cv_path[best_idx, 1]:.3e}."
    )

    # The interior loss is strictly less than either extreme.
    assert cv_path[best_idx, 1] < cv_path[0, 1]
    assert cv_path[best_idx, 1] < cv_path[-1, 1]


def test_augsynth_fixed_lambda_matches_fit_at_lambda(rng: np.random.Generator) -> None:
    """AugSynth(lambda_=L).fit() must produce the same arrays as _fit_at_lambda(L).

    Sanity check that the new class wires the M1 primitive correctly. If
    these match, all the M1-level property tests on _fit_at_lambda
    transitively apply to AugSynth too.
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

    primitive = _fit_at_lambda(panel, lambda_=1.5, fixedeff=True, **common_kwargs)
    est = AugSynth(fixedeff=True, lambda_=1.5).fit(panel, **common_kwargs)

    # Path quantities are bit-for-bit equal (same arithmetic).
    np.testing.assert_array_equal(est.actual_, primitive.actual)
    np.testing.assert_array_equal(est.synthetic_, primitive.synthetic)
    np.testing.assert_array_equal(est.synthetic_scm_, primitive.synthetic_scm)
    np.testing.assert_array_equal(est.ridge_correction_, primitive.ridge_correction)
    np.testing.assert_array_equal(est.gap_, primitive.gap)
    np.testing.assert_array_equal(est.pre_mask_, primitive.pre_mask)
    assert est.units_ == primitive.units
    np.testing.assert_array_equal(est.periods_, primitive.periods)

    # Weight dicts are equal entry-for-entry.
    assert est.weights_ == primitive.weights
    assert est.scm_weights_ == primitive.scm_weights
    assert est.augmentation_weights_ == primitive.augmentation_weights

    # AugSynth-specific attributes.
    assert est.lambda_ == 1.5
    assert est.lambda_cv_path_ is None  # fixed-lambda path
    assert est.treated_ == "t"
    assert est.treatment_time_ == 25


def test_augsynth_att_pct_rmspe_attributes(rng: np.random.Generator) -> None:
    """Synth-parallel scalar attributes are computed from the augmented gap.

    Recomputes the values inline from the public arrays and asserts they
    match. Catches any silent regression in the att_ / att_pct_ /
    rmspe_pre_ arithmetic.
    """
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=25,
    )

    pre = est.pre_mask_
    pre_actual_mean = float(est.actual_[pre].mean())
    pre_actual_scale = abs(pre_actual_mean) if pre_actual_mean != 0 else 1.0
    expected_rmspe = float(np.sqrt(np.mean(est.gap_[pre] ** 2)) / pre_actual_scale)
    expected_att = float(est.gap_[~pre].mean())
    expected_att_pct = expected_att / pre_actual_scale * float(np.sign(pre_actual_mean or 1.0))

    assert est.rmspe_pre_ == pytest.approx(expected_rmspe, abs=1e-12)
    assert est.att_ == pytest.approx(expected_att, abs=1e-12)
    assert est.att_pct_ == pytest.approx(expected_att_pct, abs=1e-12)


def test_augsynth_cv_auto_grid_populates_lambda_cv_path(
    rng: np.random.Generator,
) -> None:
    """lambda_=None, lambda_grid=None -> build auto-grid, run CV, expose path.

    Uses the same U-shaped DGP as U10 so we know CV picks a non-trivial
    lambda.
    """
    t0 = 10
    donors = np.column_stack(
        [
            np.linspace(10.0, 20.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(8.0, 18.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(12.0, 22.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(15.0, 25.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(11.0, 21.0, t0) + rng.normal(0.0, 1.0, t0),
            np.linspace(13.0, 23.0, t0) + rng.normal(0.0, 1.0, t0),
            np.full(t0, 30.0) + rng.normal(0.0, 1.0, t0),
            np.linspace(0.0, 50.0, t0) + rng.normal(0.0, 2.0, t0),
        ]
    )
    true_w = np.array([1.2, -0.3, 0.4, 0.2, 0.1, 0.2, 0.1, 0.1], dtype=np.float64)
    treated = donors @ true_w + rng.normal(0.0, 1.5, t0)

    # treatment_time=8 leaves 2 post-period periods (indices 8, 9). T0=8.
    y_matrix = np.column_stack([treated, donors])
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(donors.shape[1]))])

    # If CV picks the smallest or largest grid entry on this seed,
    # _warn_if_lambda_at_boundary emits a UserWarning. pyproject's
    # filterwarnings='error' would then fail the test for a reason
    # unrelated to what we're asserting (path shape + auto-grid formula).
    # Silence only that warning -- the conflict-warning path is tested in B3.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="CV-selected lambda",
        )
        est = AugSynth().fit(  # all defaults: fixedeff=True, lambda_=None, lambda_grid=None
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="t",
            treatment_time=8,
        )

    # CV path was taken.
    assert est.lambda_cv_path_ is not None
    assert est.lambda_cv_path_.shape == (50, 2)

    # The chosen lambda equals the argmin of the path.
    best_idx = int(est.lambda_cv_path_[:, 1].argmin())
    assert est.lambda_ == pytest.approx(float(est.lambda_cv_path_[best_idx, 0]), abs=1e-12)

    # The first column is the auto-grid: logspace(-4, 4, 50) * var(y_pre_treated).
    # var() is shift-invariant, so this equals var of the *demeaned* y1_pre that
    # _build_auto_lambda_grid actually sees inside fit() with fixedeff=True.
    pre_treated = est.actual_[est.pre_mask_]
    expected_grid = np.logspace(-4, 4, 50, dtype=np.float64) * float(np.var(pre_treated, ddof=0))
    np.testing.assert_allclose(est.lambda_cv_path_[:, 0], expected_grid, rtol=1e-12)


def test_augsynth_cv_with_user_grid(rng: np.random.Generator) -> None:
    """lambda_=None, lambda_grid=<array> -> CV on user grid, no auto-grid."""
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])
    user_grid = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float64)

    # Random panel + 4-point grid: CV best may land at a boundary, firing
    # _warn_if_lambda_at_boundary. Test is about path shape + grid identity,
    # not warning behavior, so suppress only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="CV-selected lambda",
        )
        est = AugSynth(lambda_=None, lambda_grid=user_grid).fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="t",
            treatment_time=25,
        )

    assert est.lambda_cv_path_ is not None
    assert est.lambda_cv_path_.shape == (4, 2)
    # First column is exactly the user grid (not the auto-grid).
    np.testing.assert_array_equal(est.lambda_cv_path_[:, 0], user_grid)
    # Chosen lambda is one of the four candidates.
    assert est.lambda_ in user_grid.tolist()


def test_augsynth_warns_when_lambda_and_grid_both_supplied(
    rng: np.random.Generator,
) -> None:
    """lambda_=<float> + lambda_grid=<array> -> warn that grid is ignored.

    The float wins; CV is skipped; lambda_cv_path_ is None.
    """
    y_matrix = rng.normal(100.0, 10.0, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])
    grid = np.array([0.1, 1.0, 10.0], dtype=np.float64)

    with pytest.warns(UserWarning, match="lambda_grid was supplied"):
        est = AugSynth(lambda_=2.5, lambda_grid=grid).fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="t",
            treatment_time=25,
        )

    assert est.lambda_ == 2.5
    assert est.lambda_cv_path_ is None
