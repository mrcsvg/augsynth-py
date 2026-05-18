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
        demeaning before invocation.
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

    omega = Synth._solve_simplex_qp(y1_pre, y0_pre)
    gamma, _ = _ridge_augment(omega, y0_pre, y1_pre, lambda_=lambda_)

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
