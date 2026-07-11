"""Augmented synthetic control with ridge augmentation (BFR 2021 Section 2).

This module implements the ``progfunc = "Ridge"`` form of augsynth: classical
simplex-SCM weights ``omega`` are computed by reusing
:class:`augsynth_py.synth.classical.Synth`, then a ridge regression of the
pre-period SCM residual on the donor pre-period outcomes produces an
augmentation correction ``gamma``. Per BFR 2021 Section 2.4 Lemma 1, the
effective AugSynth weights are ``omega + gamma``; they are real-valued and
need not lie on the simplex.

M1 (this module's current scope) implements the augmentation at a *fixed*
``lambda_``. Cross-validated ``lambda_`` selection and the public
``AugSynth`` class are M2 and M3 respectively (see
``docs/plans/2026-05-16-augsynth-v0.2-design.md``).

References
----------
Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic
    Control Method. *JASA*, 116(536), 1789-1803. Section 2.3 (ridge outcome
    model); Section 2.4 Lemma 1 (decomposition of effective weights).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from augsynth_py.synth._panel import (
    apply_unit_fixedeff,
    long_to_wide,
    pre_period_mask,
)
from augsynth_py.synth.classical import Synth


@dataclass(frozen=True)
class _RidgeAugmentResult:
    """Carrier object for the M1 fixed-lambda augmented synthetic-control fit.

    This is *internal*: M3 will replace it with the public ``AugSynth``
    class. The field shape anticipates the public attribute surface in
    design-doc Section D7 so M3 is a thin wrapper rather than a redesign.

    Attributes
    ----------
    units, periods, pre_mask, actual
        Panel scaffolding, same semantics as on ``Synth``.
    synthetic
        Augmented counterfactual ``y0_full @ (omega + gamma) + offset``.
    synthetic_scm
        SCM-only counterfactual ``y0_full @ omega + offset``.
    ridge_correction
        ``synthetic - synthetic_scm`` = ``y0_full @ gamma``.
    gap
        ``actual - synthetic``.
    scm_weights
        Inner SCM weights ``omega`` (on the simplex).
    augmentation_weights
        Ridge augmentation ``gamma`` (real-valued, may be negative).
    weights
        Effective AugSynth weights ``omega + gamma`` (real-valued, may be
        negative, need not sum to 1). Per BFR 2021 Section 2.4 Lemma 1 this
        is the decomposition the paper and R ``augsynth`` expose.
    lambda_
        The ridge penalty at which this fit was produced.
    """

    units: list[str]
    periods: NDArray[Any]
    pre_mask: NDArray[np.bool_]
    actual: NDArray[np.float64]
    synthetic: NDArray[np.float64]
    synthetic_scm: NDArray[np.float64]
    ridge_correction: NDArray[np.float64]
    gap: NDArray[np.float64]
    scm_weights: dict[str, float]
    augmentation_weights: dict[str, float]
    weights: dict[str, float]
    lambda_: float


def _ridge_augment(
    scm_weights: NDArray[np.float64],
    y_pre_donors: NDArray[np.float64],
    y_pre_treated: NDArray[np.float64],
    lambda_: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Solve the BFR 2021 Section 2.3 ridge augmentation step.

    Given SCM weights :math:`\omega`, fit :math:`\gamma` minimizing

    .. math::

        \sum_{t=1}^{T_0} \bigl(Y_1(t) - Y_0(t)^\top \omega - Y_0(t)^\top
        \gamma\bigr)^2 + \lambda \|\gamma\|^2,

    i.e. ridge-regress the SCM pre-period residual on the donor pre-period
    outcome matrix. The closed form is

    .. math::

        \gamma = \bigl(Y_{0,\text{pre}}^\top Y_{0,\text{pre}} + \lambda
        I_J\bigr)^{-1} Y_{0,\text{pre}}^\top \bigl(Y_{1,\text{pre}} -
        Y_{0,\text{pre}} \omega\bigr).

    Parameters
    ----------
    scm_weights
        Length-``J`` SCM weight vector :math:`\omega`. Need not lie on the
        simplex; this routine only multiplies it.
    y_pre_donors
        Pre-period donor outcome matrix :math:`Y_{0,\text{pre}}`, shape
        ``(T0, J)``. Caller is responsible for any unit fixed-effect
        demeaning and period demeaning before invocation; see
        :func:`_period_demean_pre` for the latter, which matches the R
        ``augsynth`` reference at machine precision when applied.
    y_pre_treated
        Pre-period treated outcome vector :math:`Y_{1,\text{pre}}`, shape
        ``(T0,)``. Same demeaning convention as ``y_pre_donors``.
    lambda_
        Ridge penalty :math:`\lambda \geq 0`.

    Returns
    -------
    gamma
        Length-``J`` augmentation correction vector :math:`\gamma`. Real-valued.
    pre_correction
        Length-``T0`` in-sample correction :math:`Y_{0,\text{pre}} \gamma`.
        Returned for caller convenience; the full-period correction at time
        ``t`` is ``Y_0(t) @ gamma`` and is computed by the composition layer.

    Raises
    ------
    ValueError
        If ``lambda_`` is negative.

    Notes
    -----
    See BFR 2021 Section 2.4 Lemma 1 for the interpretation as an additive
    correction to the SCM weights: ``omega + gamma`` are the effective
    AugSynth weights.
    """
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative; got {lambda_}.")

    n_donors = y_pre_donors.shape[1]
    residual = y_pre_treated - y_pre_donors @ scm_weights
    gram = y_pre_donors.T @ y_pre_donors
    lhs = gram + lambda_ * np.eye(n_donors, dtype=np.float64)
    rhs = y_pre_donors.T @ residual
    gamma = np.linalg.solve(lhs, rhs).astype(np.float64, copy=False)
    pre_correction: NDArray[np.float64] = y_pre_donors @ gamma
    return gamma, pre_correction


def _period_demean_pre(
    y_pre_donors: NDArray[np.float64],
    y_pre_treated: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Subtract donor-mean-per-pre-period from donor and treated matrices.

    Implements the period-demeaning step R ``augsynth`` applies inside
    ``fit_ridgeaug_formatted`` (donor mean across donor units, computed
    separately at each pre-period column, subtracted from every unit's value
    at that column). Equivalent to absorbing time fixed effects in the ridge
    augmentation step.

    The simplex SCM solve is invariant to this subtraction (since :math:`\omega`
    sums to 1, the period-mean shift cancels in the objective), so applying
    period demeaning before solving for :math:`\omega` does not change
    ``scm_weights_``. The ridge :math:`\gamma` solve **is** affected; this is
    the step required for parity with the R reference at machine precision
    (verified against ``augsynth(progfunc='Ridge', fixedeff=TRUE)`` on the
    GeoLift fixture).

    Parameters
    ----------
    y_pre_donors
        Pre-period donor matrix shape ``(T0, J)``, already unit-fixedeff
        demeaned by the caller. Not modified in place.
    y_pre_treated
        Pre-period treated vector shape ``(T0,)``, same demeaning
        convention. Not modified in place.

    Returns
    -------
    y_pre_donors_pdem
        ``y_pre_donors - period_means[:, None]`` where ``period_means`` is
        the donor mean at each pre-period row.
    y_pre_treated_pdem
        ``y_pre_treated - period_means``, using the same donor means.
    """
    period_means = y_pre_donors.mean(axis=1)
    return (
        y_pre_donors - period_means[:, None],
        y_pre_treated - period_means,
    )


def _fit_at_lambda(
    panel: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treated: Any,
    treatment_time: Any,
    lambda_: float,
    fixedeff: bool = True,
) -> _RidgeAugmentResult:
    """Fit Ridge ASCM at fixed ``lambda_`` -- M1 composition of Synth + ridge augmentation.

    Private. The public ``AugSynth`` class (M3) will wrap this together with
    the LOO-CV layer from M2 and the input-validation surface from D7. Use
    this function only from tests and from M2/M3.

    ``treated`` may be a unit value or an iterable of unit values, with the
    same treated-group semantics as :meth:`Synth.fit` (see there for the
    full documentation); for an iterable, the ``actual``, ``synthetic``,
    and ``gap`` fields of the result refer to the treated-group mean.

    The flow:

    1. Reshape the panel via :func:`augsynth_py.synth._panel.long_to_wide`.
    2. Optionally demean by per-unit pre-period mean (``fixedeff=True``).
    3. Fit ``omega`` with :meth:`Synth._solve_simplex_qp` on the demeaned matrices.
    4. Fit ``gamma`` with :func:`_ridge_augment` on the same demeaned matrices.
    5. Project ``omega`` and ``gamma`` across the full timespan, re-add the
       treated unit's pre-period offset, and pack the result.

    See ``docs/plans/2026-05-16-augsynth-v0.2-design.md`` "Implementation
    milestones / M1" for the milestone scope.
    """
    y_matrix, units, periods, treated_idx = long_to_wide(
        panel, unit=unit, time=time, outcome=outcome, treated=treated
    )
    pre_mask = pre_period_mask(periods, treatment_time)

    if fixedeff:
        y_fit, offsets = apply_unit_fixedeff(y_matrix, pre_mask)
    else:
        y_fit = y_matrix
        offsets = np.zeros(y_matrix.shape[1], dtype=np.float64)

    donor_idx = np.array([i for i in range(y_matrix.shape[1]) if i != treated_idx])
    y1_pre = y_fit[pre_mask, treated_idx]
    y0_pre = y_fit[np.ix_(pre_mask, donor_idx)]
    y0_full = y_fit[:, donor_idx]

    # Period-demean donor and treated matrices before solving for omega and
    # gamma. Matches R augsynth's ridge step at machine precision; see
    # _period_demean_pre for the math motivation and the parity verification
    # context. The unit-fixedeff-demeaned y0_full (without period demeaning)
    # is the correct projection target for the final synthetic, since R's
    # predict uses `treated_pre_mean + y_donors_unit_demeaned @ (omega+gamma)`
    # and the period-mean shift cancels in this combination only at the solve.
    y0_pre_pdem, y1_pre_pdem = _period_demean_pre(y0_pre, y1_pre)
    omega = Synth._solve_simplex_qp(y1_pre_pdem, y0_pre_pdem)
    gamma, _ = _ridge_augment(omega, y0_pre_pdem, y1_pre_pdem, lambda_=lambda_)

    full_correction = y0_full @ gamma  # ridge correction at every period
    treated_offset = offsets[treated_idx]
    synthetic_scm = y0_full @ omega + treated_offset
    synthetic = synthetic_scm + full_correction
    actual = y_matrix[:, treated_idx]
    gap = actual - synthetic

    donor_names = [units[i] for i in donor_idx]
    scm_weights = {name: float(w) for name, w in zip(donor_names, omega, strict=True)}
    augmentation_weights = {name: float(w) for name, w in zip(donor_names, gamma, strict=True)}
    effective = omega + gamma
    weights = {name: float(w) for name, w in zip(donor_names, effective, strict=True)}

    return _RidgeAugmentResult(
        units=units,
        periods=periods,
        pre_mask=pre_mask,
        actual=actual,
        synthetic=synthetic,
        synthetic_scm=synthetic_scm,
        ridge_correction=full_correction,
        gap=gap,
        scm_weights=scm_weights,
        augmentation_weights=augmentation_weights,
        weights=weights,
        lambda_=float(lambda_),
    )


def _build_auto_lambda_grid(
    y_pre_treated: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the default lambda grid per design-doc D6.

    Returns ``logspace(-4, 4, 50) * var(y_pre_treated)``. The variance
    scaling defuses the same outcome-scale pathology that the
    ``Synth._solve_simplex_qp`` normalisation step exists to handle: a grid
    fixed in raw units misbehaves on series whose natural scale differs by
    orders of magnitude.

    Parameters
    ----------
    y_pre_treated
        Length-``T0`` pre-period treated outcome vector. Demeaning by
        ``fixedeff`` is irrelevant here -- the variance is invariant to an
        additive shift.

    Returns
    -------
    grid
        Strictly increasing length-50 float64 array.

    Raises
    ------
    ValueError
        If ``y_pre_treated`` has zero variance (the auto-grid would collapse
        to all zeros, killing the regularization).
    """
    sigma_sq = float(np.var(y_pre_treated, ddof=0))
    if sigma_sq == 0.0:
        raise ValueError(
            "Cannot build auto lambda grid: y_pre_treated has zero variance. "
            "Supply lambda_grid= explicitly or pre-process the input."
        )
    return np.logspace(-4, 4, 50, dtype=np.float64) * sigma_sq


def _warn_if_lambda_at_boundary(
    best_idx: int,
    lambda_grid: NDArray[np.float64],
) -> None:
    """Emit a UserWarning when the CV-best lambda is at either grid extreme.

    Standard diagnostic per design-doc D6: if the optimizer picks the
    smallest or largest grid entry, the grid is almost certainly too narrow
    -- the actual optimum lies outside it. Cheap to ignore, valuable when
    needed.

    Parameters
    ----------
    best_idx
        Index of the chosen lambda in ``lambda_grid``.
    lambda_grid
        The grid that was searched. Used only for the warning message (no
        validation of monotonicity here).
    """
    n = len(lambda_grid)
    if best_idx == 0:
        warnings.warn(
            f"CV-selected lambda ({float(lambda_grid[0]):.3e}) is at the "
            f"left boundary of the grid (length {n}). Refit with a wider "
            f"lambda_grid covering smaller lambda values.",
            UserWarning,
            stacklevel=2,
        )
    elif best_idx == n - 1:
        warnings.warn(
            f"CV-selected lambda ({float(lambda_grid[-1]):.3e}) is at the "
            f"right boundary of the grid (length {n}). Refit with a wider "
            f"lambda_grid covering larger lambda values.",
            UserWarning,
            stacklevel=2,
        )


def _loo_cv_lambda(
    y_pre_donors: NDArray[np.float64],
    y_pre_treated: NDArray[np.float64],
    lambda_grid: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Leave-one-out CV over pre-treatment periods (BFR 2021 Section 3.2).

    For each lambda in ``lambda_grid`` and each pre-period ``t``:

    1. Drop row ``t`` from the donor and treated matrices.
    2. Refit ``omega`` via ``Synth._solve_simplex_qp`` on the T0-1 remaining
       rows.
    3. Refit ``gamma`` via :func:`_ridge_augment` on the same rows at this
       lambda.
    4. Predict the held-out outcome: ``y_pre_donors[t] @ (omega + gamma)``.
    5. Accumulate the squared prediction error.

    Sum across ``t`` to get the CV loss for this lambda. Return the argmin
    lambda and the full path.

    Parameters
    ----------
    y_pre_donors
        Pre-period donor matrix, shape ``(T0, J)``. Caller is responsible
        for any unit fixed-effect demeaning, matching the
        :func:`_ridge_augment` convention.
    y_pre_treated
        Pre-period treated vector, shape ``(T0,)``.
    lambda_grid
        Candidate ``lambda`` values to evaluate. Must be 1-D with
        ``len >= 1``.

    Returns
    -------
    best_lambda
        Entry of ``lambda_grid`` that minimizes the CV loss.
    cv_path
        Shape ``(len(lambda_grid), 2)`` float64 array. Column 0 is
        ``lambda_grid``; column 1 is the corresponding total LOO squared
        error.

    Raises
    ------
    ValueError
        If ``lambda_grid`` is empty, or if ``T0 < 3`` (LOO needs at least
        two periods of training data after dropping one).
    """
    t0 = y_pre_donors.shape[0]
    n_grid = len(lambda_grid)
    if n_grid == 0:
        raise ValueError("lambda_grid must contain at least one candidate.")
    if t0 < 3:
        raise ValueError(
            f"LOO needs at least T0=3 pre-periods (got {t0}); after dropping "
            f"one fold, the leave-out fit still requires two periods."
        )

    cv_path = np.empty((n_grid, 2), dtype=np.float64)
    cv_path[:, 0] = np.asarray(lambda_grid, dtype=np.float64)

    # Precompute the (T0-1, J) leave-out matrices once -- saves T0 * n_grid
    # rebuilds of the same arrays.
    leave_one_out_donors = [np.delete(y_pre_donors, t, axis=0) for t in range(t0)]
    leave_one_out_treated = [np.delete(y_pre_treated, t, axis=0) for t in range(t0)]

    for lam_idx in range(n_grid):
        lam = float(lambda_grid[lam_idx])
        total_sq_err = 0.0
        for t in range(t0):
            y0_lo = leave_one_out_donors[t]
            y1_lo = leave_one_out_treated[t]
            try:
                omega_lo = Synth._solve_simplex_qp(y1_lo, y0_lo)
                gamma_lo, _ = _ridge_augment(omega_lo, y0_lo, y1_lo, lambda_=lam)
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(
                    f"LOO-CV fit failed at lambda={lam:g}, held-out t={t}: {exc}"
                ) from exc
            pred = float(y_pre_donors[t] @ (omega_lo + gamma_lo))
            total_sq_err += (float(y_pre_treated[t]) - pred) ** 2
        cv_path[lam_idx, 1] = total_sq_err

    best_idx = int(cv_path[:, 1].argmin())
    best_lambda = float(cv_path[best_idx, 0])
    return best_lambda, cv_path


class AugSynth:
    r"""Augmented synthetic control via ridge augmentation (BFR 2021).

    Composes simplex-SCM weights :math:`\omega` (via
    :class:`augsynth_py.synth.classical.Synth`) with a ridge augmentation
    correction :math:`\gamma` fit on the pre-period SCM residual. The
    effective weights are :math:`\omega + \gamma` (BFR 2021 Section 2.4
    Lemma 1), real-valued and not constrained to the simplex.

    Parameters
    ----------
    fixedeff
        If True (default), subtract each unit's pre-period mean from its
        outcome trajectory before fitting weights, then add back the treated
        unit's pre-period mean when reporting the counterfactual. Same
        semantics as ``Synth.fixedeff``.
    lambda_
        Ridge penalty. If a float, that value is used as-is and CV is
        skipped (``lambda_cv_path_`` will be ``None`` after ``fit``). If
        ``None`` (default), the value is chosen by leave-one-out CV over
        the pre-treatment period; see ``lambda_grid``.
    lambda_grid
        Candidate values for CV. Ignored when ``lambda_`` is a float (a
        ``UserWarning`` is emitted in that case). If ``None`` and
        ``lambda_`` is also ``None``, an auto-grid is built per design-doc
        D6: ``logspace(-4, 4, 50) * var(y_pre_treated)``.

    Attributes
    ----------
    weights_ : dict[str, float]
        Effective weights :math:`\omega + \gamma`. **May be negative and
        need not sum to 1.** This is the primary weight attribute and
        differs from ``Synth.weights_`` (which is on the simplex). Per
        D5 / BFR 2021 Section 2.4 Lemma 1.
    scm_weights_ : dict[str, float]
        Inner SCM weights :math:`\omega` (simplex).
    augmentation_weights_ : dict[str, float]
        Ridge augmentation :math:`\gamma` (real-valued).
    lambda_ : float
        The penalty value actually used (CV-chosen or user-supplied).
    lambda_cv_path_ : numpy.ndarray or None
        Shape ``(n_grid, 2)`` array of ``(lambda, cv_loss)`` rows. ``None``
        when ``lambda_`` was supplied as a float.
    synthetic_scm_ : numpy.ndarray
        SCM-only counterfactual (before ridge correction).
    ridge_correction_ : numpy.ndarray
        ``synthetic_ - synthetic_scm_``.
    units_, periods_, actual_, synthetic_, gap_, att_, att_pct_,
    rmspe_pre_, pre_mask_, treated_, treatment_time_
        Mirror the corresponding ``Synth`` attributes; ``synthetic_``,
        ``gap_``, ``att_``, ``att_pct_``, ``rmspe_pre_`` refer to the
        augmented counterfactual rather than the pure SCM one. As on
        ``Synth``, for a multi-treated fit (``treated`` passed as an
        iterable) ``units_[0]`` is the comma-joined group label rather than
        a single panel unit name, and ``att_`` / ``att_pct_`` refer to the
        treated-group *mean*: for flow outcomes (where summing across units
        and periods is meaningful, e.g. sales or conversions), the total
        incremental effect is ``att_ * n_treated * n_post_periods``
        (downstream consumers such as geolift-py do this multiplication —
        do not double-count).
    """

    def __init__(
        self,
        *,
        fixedeff: bool = True,
        lambda_: float | None = None,
        lambda_grid: NDArray[np.float64] | None = None,
    ) -> None:
        self.fixedeff = fixedeff
        self._lambda_init = lambda_
        self._lambda_grid_init = lambda_grid

    def fit(
        self,
        panel: pl.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        treated: Any,
        treatment_time: Any,
    ) -> AugSynth:
        """Fit AugSynth on a long-format panel.

        Parameters mirror :meth:`Synth.fit`, including the ``treated``
        semantics: a unit value, or an iterable of unit values designating a
        treated *group* that is collapsed to its elementwise mean and
        excluded from the donor pool. See :meth:`Synth.fit` for the full
        parameter documentation and the class docstring for the attribute
        surface populated by this call.

        Returns
        -------
        self
        """
        # Panel prep -- mirrors _fit_at_lambda. Duplicated rather than
        # extracted per design-doc D2 (extraction waits for estimator #3).
        y_matrix, units, periods, treated_idx = long_to_wide(
            panel, unit=unit, time=time, outcome=outcome, treated=treated
        )
        pre_mask = pre_period_mask(periods, treatment_time)

        if self.fixedeff:
            y_fit, offsets = apply_unit_fixedeff(y_matrix, pre_mask)
        else:
            y_fit = y_matrix
            offsets = np.zeros(y_matrix.shape[1], dtype=np.float64)

        donor_idx = np.array([i for i in range(y_matrix.shape[1]) if i != treated_idx])
        y1_pre = y_fit[pre_mask, treated_idx]
        y0_pre = y_fit[np.ix_(pre_mask, donor_idx)]

        # Period-demean before any CV or final solve. See _period_demean_pre.
        # This is the change required for parity with R augsynth's ridge step;
        # _loo_cv_lambda receives the period-demeaned matrices so each fold's
        # held-out prediction lives in the same demeaned space (matches R's
        # internal cv_lambda behavior).
        y0_pre_pdem, y1_pre_pdem = _period_demean_pre(y0_pre, y1_pre)

        # Dispatch on (lambda_, lambda_grid) per D7.
        cv_path: NDArray[np.float64] | None
        if self._lambda_init is not None:
            if self._lambda_grid_init is not None:
                warnings.warn(
                    f"lambda_grid was supplied alongside an explicit "
                    f"lambda_={float(self._lambda_init):g}; the grid is ignored. "
                    f"Pass lambda_=None to use lambda_grid for CV.",
                    UserWarning,
                    stacklevel=2,
                )
            effective_lambda = float(self._lambda_init)
            cv_path = None
        else:
            if self._lambda_grid_init is None:
                grid = _build_auto_lambda_grid(y1_pre)
            else:
                grid = np.asarray(self._lambda_grid_init, dtype=np.float64)
            effective_lambda, cv_path = _loo_cv_lambda(y0_pre_pdem, y1_pre_pdem, grid)
            best_idx = int(cv_path[:, 1].argmin())
            _warn_if_lambda_at_boundary(best_idx, grid)

        # Final fit at the chosen lambda. We reuse the matrices we already
        # prepared rather than re-running _fit_at_lambda's panel prep.
        y0_full = y_fit[:, donor_idx]
        omega = Synth._solve_simplex_qp(y1_pre_pdem, y0_pre_pdem)
        gamma, _ = _ridge_augment(omega, y0_pre_pdem, y1_pre_pdem, lambda_=effective_lambda)

        full_correction = y0_full @ gamma
        treated_offset = offsets[treated_idx]
        synthetic_scm = y0_full @ omega + treated_offset
        synthetic = synthetic_scm + full_correction
        actual = y_matrix[:, treated_idx]
        gap = actual - synthetic

        donor_names = [units[i] for i in donor_idx]
        scm_weights = {name: float(w) for name, w in zip(donor_names, omega, strict=True)}
        augmentation_weights = {name: float(w) for name, w in zip(donor_names, gamma, strict=True)}
        effective = omega + gamma
        weights = {name: float(w) for name, w in zip(donor_names, effective, strict=True)}

        pre_actual_mean = float(actual[pre_mask].mean())
        pre_actual_scale = abs(pre_actual_mean) if pre_actual_mean != 0 else 1.0
        rmspe_pre = float(np.sqrt(np.mean(gap[pre_mask] ** 2)) / pre_actual_scale)
        att = float(gap[~pre_mask].mean())
        att_pct = att / pre_actual_scale * float(np.sign(pre_actual_mean or 1.0))

        self.units_: list[str] = units
        self.periods_: NDArray[Any] = periods
        self.actual_: NDArray[np.float64] = actual
        self.synthetic_: NDArray[np.float64] = synthetic
        self.synthetic_scm_: NDArray[np.float64] = synthetic_scm
        self.ridge_correction_: NDArray[np.float64] = full_correction
        self.gap_: NDArray[np.float64] = gap
        self.pre_mask_: NDArray[np.bool_] = pre_mask
        self.weights_: dict[str, float] = weights
        self.scm_weights_: dict[str, float] = scm_weights
        self.augmentation_weights_: dict[str, float] = augmentation_weights
        self.lambda_: float = float(effective_lambda)
        self.lambda_cv_path_: NDArray[np.float64] | None = cv_path
        self.att_: float = att
        self.att_pct_: float = att_pct
        self.rmspe_pre_: float = rmspe_pre
        self.treated_: Any = treated
        self.treatment_time_: Any = treatment_time

        return self
