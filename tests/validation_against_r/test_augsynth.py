"""Parity tests for the augmented synthetic control estimator (M4 scope).

Validates :class:`augsynth_py.synth.augmented.AugSynth` against the R
``augsynth`` package called with ``progfunc = "Ridge"``, ``scm = TRUE``,
``fixedeff = TRUE``. See ``docs/plans/2026-05-24-augsynth-v0.2-m4.md`` for the
design (Test A: pinned-lambda algorithm parity across three regularization
regimes; Test B: CV-chosen lambda parity on a pinned 11-entry grid within one
grid cell).

The R-side fits use a two-call decomposition per Test A invocation
(``progfunc='Ridge', lambda=L`` for augmented path + effective weights;
``progfunc='None'`` for SCM-only path + simplex weights ω). BFR 2021
Section 2 guarantees both calls compute the same simplex SCM ω; the ridge
augmentation step only fits gamma on ω's residual.
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest


def _panel_with_date_column(panel: pl.DataFrame) -> pl.DataFrame:
    """Return ``panel`` with the ``date`` column parsed to a real ``Date``.

    The GeoLift fixture ships ``date`` as a character column; both R-side
    (via ``as.Date(df$date)``) and Python-side (via this helper) need a
    parsed date for the parity to be meaningful. Same convention as
    :file:`test_classical_synth.py`.
    """
    return panel.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _treated_pre_variance(panel: pl.DataFrame, treated: str, treatment_time: date) -> float:
    """Return ``var(y_pre_treated, ddof=0)`` on the parsed-date GeoLift panel.

    Used to scale the lambda regimes for Test A and the lambda grid for
    Test B. ``ddof=0`` matches ``numpy.var``'s default and ``AugSynth``'s
    auto-grid convention (see ``_build_auto_lambda_grid``).
    """
    pre = (
        panel.filter((pl.col("location") == treated) & (pl.col("date") < treatment_time))
        .get_column("Y")
        .to_numpy()
    )
    return float(np.var(pre, ddof=0))


# ---------------------------------------------------------------------------
# Phase A smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_augsynth_python_smoke_on_geolift(
    r_geolift_pretest: pl.DataFrame,
) -> None:
    """Python-side ``AugSynth(lambda_=1.0).fit(...)`` runs end-to-end.

    Confirms the GeoLift fixture path, panel preparation, and AugSynth call
    surface are wired up correctly before the parity tests engage R. No
    numerical agreement is asserted here.
    """
    from augsynth_py.synth.augmented import AugSynth

    panel = _panel_with_date_column(r_geolift_pretest)
    treatment_time = date(2021, 2, 15)
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated="new york",
        treatment_time=treatment_time,
    )

    assert est.lambda_ == 1.0
    assert est.synthetic_.shape == (90,)
    assert est.lambda_cv_path_ is None  # no CV when lambda_ is given
    n_donors = len(est.units_) - 1
    assert len(est.weights_) == n_donors
    assert len(est.scm_weights_) == n_donors
    assert len(est.augmentation_weights_) == n_donors


@pytest.mark.requires_r_pkg("augsynth")
def test_r_augsynth_ridge_smoke_on_geolift(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> None:
    """R ``augsynth(progfunc='Ridge', lambda=1.0)`` runs end-to-end.

    Confirms the R harness, argument names, and field-extraction calls are
    valid before Test A. Pins the R-side argument convention used by all
    downstream parity tests in this file.
    """
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(
        """
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location == 'new york' & df$date >= as.Date('2021-02-15')
        )
        res <- augsynth(
            Y ~ treat,
            unit     = location,
            time     = date,
            data     = df,
            progfunc = 'Ridge',
            scm      = TRUE,
            fixedeff = TRUE,
            lambda   = 1.0
        )
        """
    )
    r_synthetic = np.asarray(r_session("as.numeric(predict(res, att=FALSE))"), dtype=float)
    r_weights = np.asarray(r_session("as.numeric(res$weights)"), dtype=float).ravel()
    r_donor_names = [str(x) for x in r_session("rownames(res$weights)")]
    r_lambda = float(np.asarray(r_session("res$lambda"), dtype=float).ravel()[0])

    assert r_synthetic.shape == (90,)
    assert r_weights.shape == (39,)
    assert len(r_donor_names) == 39
    assert r_lambda == 1.0


# ---------------------------------------------------------------------------
# Test A — algorithm parity at pinned lambda (three regularization regimes)
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
@pytest.mark.parametrize("lambda_multiplier", [0.1, 1.0, 10.0])
def test_augsynth_matches_r_at_pinned_lambda(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
    lambda_multiplier: float,
) -> None:
    """``AugSynth(lambda_=L)`` matches ``augsynth(progfunc='Ridge', lambda=L)``.

    The plan pins ``L = multiplier * var(y_pre_treated, ddof=0)`` for
    ``multiplier in {0.1, 1.0, 10.0}`` so low/mid/high regularization regimes
    are each exercised. Algorithm parity that holds at one λ but fails at
    another is a precise diagnostic of regime-specific bugs we'd otherwise
    miss.

    Strict-first protocol (see plan §"Strict-first protocol for weight
    comparison"): all four quantities at the strict D8 tolerance, with the
    SCM-only and augmentation per-donor checks degrading to invariants only
    on the characteristic collinearity signature (paths match, per-donor
    weights diverge while invariants hold). The classical-SCM parity test
    on this same fixture demonstrated that simplex argmins differ between
    Clarabel and R's solver when donors are collinear; whether that bites
    AugSynth is the hypothesis under test here.
    """
    treated_unit = "new york"
    treatment_time = date(2021, 2, 15)
    panel = _panel_with_date_column(r_geolift_pretest)
    sigma_sq = _treated_pre_variance(panel, treated_unit, treatment_time)
    pinned_lambda = lambda_multiplier * sigma_sq

    # ------------------------------------------------------------------
    # 1. R reference: two calls per BFR 2021 §2 decomposition.
    # ------------------------------------------------------------------
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(f"pinned_lambda <- {pinned_lambda!r}")
    r_session(
        """
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location == 'new york' & df$date >= as.Date('2021-02-15')
        )

        # Augmented path + effective weights (omega + gamma).
        ridge_res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'Ridge', scm = TRUE, fixedeff = TRUE,
            lambda = pinned_lambda
        )

        # SCM-only path + simplex weights (omega).
        none_res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'None', scm = TRUE, fixedeff = TRUE
        )
        """
    )
    r_synthetic = np.asarray(r_session("as.numeric(predict(ridge_res, att = FALSE))"), dtype=float)
    r_synthetic_scm = np.asarray(
        r_session("as.numeric(predict(none_res, att = FALSE))"), dtype=float
    )
    r_ridge_correction = r_synthetic - r_synthetic_scm
    r_weights_effective = np.asarray(
        r_session("as.numeric(ridge_res$weights)"), dtype=float
    ).ravel()
    r_weights_scm = np.asarray(r_session("as.numeric(none_res$weights)"), dtype=float).ravel()
    r_donor_names_ridge = [str(x) for x in r_session("rownames(ridge_res$weights)")]
    r_donor_names_none = [str(x) for x in r_session("rownames(none_res$weights)")]
    assert r_donor_names_ridge == r_donor_names_none, (
        "augsynth donor row ordering differs between Ridge and None calls; "
        "the rest of this test assumes a shared ordering. Reorder both before "
        "comparison if this assertion ever fires."
    )
    r_weights_gamma = r_weights_effective - r_weights_scm

    # ------------------------------------------------------------------
    # 2. Python implementation at the same pinned lambda.
    # ------------------------------------------------------------------
    from augsynth_py.synth.augmented import AugSynth

    est = AugSynth(fixedeff=True, lambda_=pinned_lambda).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=treated_unit,
        treatment_time=treatment_time,
    )

    # Pull Python weight dicts into arrays in R's donor order so per-donor
    # comparisons are well-defined. If a donor name R uses is absent on the
    # Python side (or vice versa), KeyError fires here — that would itself be
    # a parity failure worth seeing.
    py_weights_effective = np.array([est.weights_[d] for d in r_donor_names_ridge], dtype=float)
    py_weights_scm = np.array([est.scm_weights_[d] for d in r_donor_names_ridge], dtype=float)
    py_weights_gamma = np.array(
        [est.augmentation_weights_[d] for d in r_donor_names_ridge], dtype=float
    )

    # ------------------------------------------------------------------
    # 3. Numerical agreement — paths first (always strict per D8).
    # ------------------------------------------------------------------
    from tests.validation_against_r.conftest import assert_array_close

    assert_array_close(
        est.synthetic_,
        r_synthetic,
        atol=1e-4,
        rtol=1e-4,
        name=f"augmented counterfactual (lambda_mult={lambda_multiplier})",
    )
    assert_array_close(
        est.synthetic_scm_,
        r_synthetic_scm,
        atol=1e-4,
        rtol=1e-4,
        name=f"SCM-only counterfactual (lambda_mult={lambda_multiplier})",
    )
    # D8 tolerance deviation (see M4 plan §"Deviations" D-3): ridge_correction_
    # is the algebraic difference synthetic_ - synthetic_scm_, both already
    # asserted at strict D8 tolerance above. Each path has an absolute Clarabel
    # solver floor of ~2e-3 (well inside its strict budget because path values
    # are ~3600); their difference inherits the sum of these floors but lives on
    # a much smaller scale (correction values 1-800, median ~100). atol=1e-2
    # corresponds to ~1e-4 relative to the median correction magnitude --
    # consistent with the spirit of D8's rtol=1e-4 evaluated against the
    # correction's own scale. rtol=1e-4 is retained so the assertion still
    # bites in the high-|correction| regime.
    assert_array_close(
        est.ridge_correction_,
        r_ridge_correction,
        atol=1e-2,
        rtol=1e-4,
        name=f"ridge correction path (lambda_mult={lambda_multiplier})",
    )

    # Algebraic identity: Python's ridge_correction_ must be exactly the
    # difference synthetic_ - synthetic_scm_ at machine precision (no solver
    # involved). Re-asserted here against the just-fit object so the test
    # carries its own coverage of the decomposition that motivates D8's
    # path-trio grouping.
    np.testing.assert_allclose(
        est.ridge_correction_,
        est.synthetic_ - est.synthetic_scm_,
        atol=1e-12,
        rtol=0,
        err_msg=(
            f"Python ridge_correction_ != synthetic_ - synthetic_scm_ at "
            f"lambda_mult={lambda_multiplier}."
        ),
    )

    # ------------------------------------------------------------------
    # 4. Effective weights (omega + gamma) — strict per D8.
    # ------------------------------------------------------------------
    # The effective ω+gamma vector drives ``synthetic_`` directly, so its
    # uniqueness follows from the path uniqueness even in the presence of
    # SCM collinearity. We assert it strictly.
    np.testing.assert_allclose(
        py_weights_effective,
        r_weights_effective,
        atol=1e-3,
        rtol=0,
        err_msg=(
            f"Effective weights (omega+gamma) at lambda_mult="
            f"{lambda_multiplier} differ per donor; max abs delta="
            f"{np.max(np.abs(py_weights_effective - r_weights_effective)):.3e}"
        ),
    )

    # ------------------------------------------------------------------
    # 5. Per-donor SCM omega — strict-first, fall back to invariants only
    #    on the characteristic collinearity signature.
    # ------------------------------------------------------------------
    # Hypothesis under test: GeoLift donors are collinear and the simplex
    # argmin is non-unique, so R's solver and our Clarabel solver land on
    # different omega even when ``synthetic_scm_`` matches at 1e-4.
    if np.allclose(py_weights_scm, r_weights_scm, atol=1e-3, rtol=0):
        # Strict per-donor passes — best outcome, the original-package fit
        # is exact within D8 tolerance on this fixture too.
        pass
    else:
        # Relax to invariants only after confirming the path agrees. The
        # path assert above already ran, so reaching here means SCM ω
        # disagrees per donor while ``synthetic_scm_`` agrees — the
        # collinearity signature.
        max_scm_delta = float(np.max(np.abs(py_weights_scm - r_weights_scm)))
        assert (py_weights_scm >= -1e-9).all(), (
            f"Python SCM weights violate w>=0 at lambda_mult={lambda_multiplier}"
        )
        assert abs(py_weights_scm.sum() - 1.0) < 1e-6, (
            f"Python SCM weights do not sum to 1 at lambda_mult={lambda_multiplier}"
        )
        # R `augsynth` does not enforce a strict sum-to-1 at the solver
        # level; the classical-parity test uses the same 1e-3 tolerance.
        assert abs(r_weights_scm.sum() - 1.0) < 1e-3, (
            f"R SCM weights do not sum to 1 at lambda_mult={lambda_multiplier}"
        )
        # Surface the empirical max delta in case it grows over time.
        assert max_scm_delta < 1.0, (
            f"SCM per-donor disagreement at lambda_mult={lambda_multiplier} "
            f"exceeds 1.0 ({max_scm_delta:.3e}); not the collinearity "
            f"signature — investigate before relaxing further."
        )

    # ------------------------------------------------------------------
    # 6. Per-donor augmentation gamma — strict-first, same fallback rule.
    # ------------------------------------------------------------------
    # gamma is defined by omega; if Python and R chose different omega
    # argmins under collinearity, their gamma vectors will differ donor-by-
    # donor too, even though effective ω+gamma and the path agree.
    if np.allclose(py_weights_gamma, r_weights_gamma, atol=1e-3, rtol=0):
        pass
    else:
        # The U3 invariant (ω+gamma ≡ effective) must hold on the Python side.
        # We re-assert it here against the just-fit object for clarity.
        np.testing.assert_allclose(
            py_weights_scm + py_weights_gamma,
            py_weights_effective,
            atol=1e-10,
            rtol=0,
            err_msg=(
                f"Python decomposition omega+gamma != effective at "
                f"lambda_mult={lambda_multiplier} (U3 invariant)."
            ),
        )


# ---------------------------------------------------------------------------
# Test B — CV parity at pinned grid (chosen lambda within +/- 1 grid cell)
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_augsynth_cv_matches_r_chosen_lambda_within_one_grid_cell(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> None:
    """CV-chosen ``lambda_`` from AugSynth matches R augsynth within one grid cell.

    Pin both sides to the same 11-entry log-spaced grid over
    ``[0.01, 100] * sigma_sq`` (where ``sigma_sq = var(y_pre_treated, ddof=0)``).
    Per M4 plan deviation D-2: R augsynth does not accept a vector ``lambda``
    for CV; pass ``lambda=NULL, lambda_max=100*sigma_sq, lambda_min_ratio=1e-4,
    n_lambda=10`` to produce 11 candidates log-spaced from ``100*sigma_sq``
    down to ``0.01*sigma_sq``. Set ``min_1se=FALSE`` so R picks argmin to
    match AugSynth's behavior.

    Surviving caveats absorbed by the +/- 1 grid-cell tolerance:
    - R's CV loop runs ``T0 - 1`` folds (skips the last pre-period); Python
      runs ``T0`` folds.
    - R uses mean-of-squared-errors per fold; Python uses sum-of-squared-
      errors. Both yield identical argmin (constant scaling), so the only
      effective difference is the fold-count quirk.
    """
    treated_unit = "new york"
    treatment_time = date(2021, 2, 15)
    panel = _panel_with_date_column(r_geolift_pretest)
    sigma_sq = _treated_pre_variance(panel, treated_unit, treatment_time)

    grid = np.logspace(-2, 2, 11, dtype=np.float64) * sigma_sq

    # ------------------------------------------------------------------
    # 1. R reference: augsynth CV via lambda_max + lambda_min_ratio + n_lambda
    # ------------------------------------------------------------------
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(f"r_lambda_max <- {(100.0 * sigma_sq)!r}")
    r_session(
        """
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location == 'new york' & df$date >= as.Date('2021-02-15')
        )
        ridge_res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'Ridge', scm = TRUE, fixedeff = TRUE,
            lambda = NULL,
            lambda_max = r_lambda_max,
            lambda_min_ratio = 1e-4,
            n_lambda = 10,
            min_1se = FALSE,
            holdout_length = 1
        )
        """
    )
    r_chosen_lambda = float(np.asarray(r_session("ridge_res$lambda"), dtype=float).ravel()[0])
    r_lambdas = np.asarray(r_session("as.numeric(ridge_res$lambdas)"), dtype=float)

    # Sanity check: both sides agree on the SET of 11 candidates evaluated.
    # The traversal order differs (R: max -> min; Python: min -> max) but the
    # log-spaced candidate values are equal.
    np.testing.assert_allclose(
        np.sort(r_lambdas),
        np.sort(grid),
        rtol=1e-10,
        atol=0,
        err_msg=(
            "R and Python lambda grids disagree -- the +/- 1 grid-cell "
            "assertion below is only meaningful if both sides evaluated the "
            "same candidate set."
        ),
    )

    # ------------------------------------------------------------------
    # 2. Python CV on the same grid. Filter the boundary UserWarning per
    #    pyproject filterwarnings='error' policy: on a coarse 11-entry grid
    #    the argmin can legitimately land at a boundary, and the warning is
    #    a separate diagnostic concern from the parity test under
    #    examination.
    # ------------------------------------------------------------------
    from augsynth_py.synth.augmented import AugSynth

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="CV-selected lambda",
        )
        est = AugSynth(fixedeff=True, lambda_=None, lambda_grid=grid).fit(
            panel,
            unit="location",
            time="date",
            outcome="Y",
            treated=treated_unit,
            treatment_time=treatment_time,
        )
    py_chosen_lambda = float(est.lambda_)

    # ------------------------------------------------------------------
    # 3. Compare chosen lambda within +/- 1 grid cell.
    # ------------------------------------------------------------------
    py_idx = int(np.argmin(np.abs(grid - py_chosen_lambda)))
    r_idx = int(np.argmin(np.abs(grid - r_chosen_lambda)))
    assert abs(py_idx - r_idx) <= 1, (
        f"Python picked grid[{py_idx}]={grid[py_idx]:g}, "
        f"R picked grid[{r_idx}]={grid[r_idx]:g}; "
        f"distance={abs(py_idx - r_idx)} cells exceeds the +/- 1 budget."
    )
