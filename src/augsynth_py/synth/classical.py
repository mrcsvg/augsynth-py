"""Classical synthetic control estimator.

This module implements the simplex-constrained, outcome-only formulation of
synthetic control as used by the R ``augsynth`` package with
``progfunc = "None"``. It is the form analyzed in Doudchenko & Imbens (2016)
and is the building block on which Ben-Michael, Feller & Rothstein (2021)
construct the augmented variant.

The earlier Abadie-Diamond-Hainmueller (2010) formulation with predictor
optimization and a nested ``V`` matrix is a separate estimator (deferred);
this module does not implement it.

References
----------
Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods
    for Comparative Case Studies. *JASA*, 105(490), 493-505.
Doudchenko, N., & Imbens, G. W. (2016). Balancing, Regression,
    Difference-in-Differences and Synthetic Control Methods: A Synthesis.
    NBER Working Paper No. 22791.
"""

from __future__ import annotations

import logging
from typing import Any

import cvxpy as cp
import numpy as np
import polars as pl
from numpy.typing import NDArray

from augsynth_py.synth._panel import (
    apply_unit_fixedeff,
    imbalance,
    long_to_wide,
    pre_period_mask,
)

logger = logging.getLogger(__name__)


class Synth:
    r"""Classical synthetic control via a simplex-constrained QP on outcomes.

    Fits donor weights :math:`w \in \Delta^{J}` (the simplex: ``w >= 0``,
    ``sum(w) == 1``) that minimize the pre-treatment squared error between the
    treated unit's outcome trajectory and the weighted donor combination. The
    counterfactual post-treatment path is then ``Y_donors @ w``, optionally
    re-centered to the treated unit's pre-period level when ``fixedeff=True``.

    Parameters
    ----------
    fixedeff
        If True (default), subtract each unit's pre-period mean from its outcome
        trajectory before fitting weights, then add back the treated unit's
        pre-period mean when reporting the counterfactual. Matches the
        ``fixedeff = TRUE`` branch of ``augsynth::augsynth``.

    Attributes
    ----------
    weights_ : dict[str, float]
        Donor unit name -> weight, all non-negative, summing to 1.
    units_ : list[str]
        All units in the order used internally; ``units_[0]`` is the treated
        unit. For a multi-treated fit (``treated`` passed as an iterable),
        ``units_[0]`` is the comma-joined group label (e.g. ``"u0,u1"``)
        rather than a single panel unit name.
    periods_ : numpy.ndarray
        Time index of length ``T`` in row order of the internal outcome matrix.
    actual_ : numpy.ndarray
        Length-``T`` observed outcome trajectory of the treated unit.
    synthetic_ : numpy.ndarray
        Length-``T`` counterfactual trajectory on the original outcome scale.
    gap_ : numpy.ndarray
        ``actual_ - synthetic_``.
    att_ : float
        Mean post-treatment gap (in original outcome units). For a
        multi-treated fit this is the effect on the treated-group *mean*;
        for flow outcomes (where summing across units and periods is
        meaningful, e.g. sales or conversions), the total incremental effect
        is ``att_ * n_treated * n_post_periods`` (downstream consumers such
        as geolift-py do this multiplication — do not double-count).
    att_pct_ : float
        ``att_`` divided by ``mean(actual_[pre])``. Same treated-group-mean
        semantics as ``att_`` for multi-treated fits.
    rmspe_pre_ : float
        Pre-period root-mean-squared prediction error, normalized by
        ``|mean(actual_[pre])|``.
    l2_imbalance_ : float
        Pre-period L2 imbalance ``||y1_pre - Y0_pre @ w||_2``, computed in the
        same space the weights were fit in (i.e. the demeaned pre-period fit
        space when ``fixedeff=True``). Matches the R ``augsynth``
        ``l2_imbalance`` definition.
    scaled_l2_imbalance_ : float
        ``l2_imbalance_`` divided by the L2 imbalance of uniform ``1/J`` donor
        weights, matching R ``augsynth``'s ``scaled_l2_imbalance``. Values
        below 1 mean the fitted weights balance better than the uniform
        baseline; as a ratio it is invariant to any constant normalization of
        the norm.
    """

    def __init__(self, *, fixedeff: bool = True) -> None:
        self.fixedeff = fixedeff

    def fit(
        self,
        panel: pl.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        treated: Any,
        treatment_time: Any,
    ) -> Synth:
        """Fit donor weights and compute the counterfactual path.

        Parameters
        ----------
        panel
            Long-format panel; must be balanced across ``(unit, time)``.
        unit, time, outcome
            Column names in ``panel``.
        treated
            Value in the ``unit`` column identifying the treated unit, or an
            iterable of such values. An iterable designates a treated *group*;
            the units are collapsed to their elementwise mean and excluded
            from the donor pool (see
            :func:`augsynth_py.synth._panel.long_to_wide`). ``actual_``,
            ``synthetic_``, ``att_``, and ``att_pct_`` then refer to the
            treated-group mean. Typed ``Any`` because polars unit values are
            heterogeneous (str, int, date, ...).
        treatment_time
            First period considered post-treatment. Rows with ``time <
            treatment_time`` form the pre-period used for fitting.

        Returns
        -------
        self
        """
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

        weights = self._solve_simplex_qp(y1_pre, y0_pre)
        l2_imb, scaled_l2_imb = imbalance(y1_pre, y0_pre, weights)

        synthetic_fit = y_fit[:, donor_idx] @ weights
        # Re-add the treated unit's pre-period offset so the counterfactual is on
        # the same scale as the observed series. When fixedeff=False, offsets is
        # all zeros and this is a no-op.
        synthetic = synthetic_fit + offsets[treated_idx]
        actual = y_matrix[:, treated_idx]
        gap = actual - synthetic

        pre_actual = actual[pre_mask]
        pre_actual_mean = float(pre_actual.mean())
        pre_actual_scale = abs(pre_actual_mean) if pre_actual_mean != 0 else 1.0

        rmspe_pre = float(np.sqrt(np.mean(gap[pre_mask] ** 2)) / pre_actual_scale)
        att = float(gap[~pre_mask].mean())
        att_pct = att / pre_actual_scale * np.sign(pre_actual_mean or 1.0)

        donor_names = [units[i] for i in donor_idx]
        self.weights_: dict[str, float] = {
            name: float(w) for name, w in zip(donor_names, weights, strict=True)
        }
        self.units_: list[str] = units
        self.periods_: NDArray[Any] = periods
        self.actual_: NDArray[np.float64] = actual
        self.synthetic_: NDArray[np.float64] = synthetic
        self.gap_: NDArray[np.float64] = gap
        self.att_: float = att
        self.att_pct_: float = float(att_pct)
        self.rmspe_pre_: float = rmspe_pre
        self.l2_imbalance_: float = l2_imb
        self.scaled_l2_imbalance_: float = scaled_l2_imb
        self.pre_mask_: NDArray[np.bool_] = pre_mask
        self.treated_: Any = treated
        self.treatment_time_: Any = treatment_time

        return self

    @staticmethod
    def _solve_simplex_qp(
        y1_pre: NDArray[np.float64],
        y0_pre: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve ``min ||y1 - Y0 w||^2`` subject to ``w >= 0``, ``sum(w) == 1``.

        Uses CVXPY with the Clarabel solver. Reports infeasibility / numerical
        failure as ``RuntimeError`` rather than silently returning garbage.
        """
        # Normalize by the absolute max so the QP is well-conditioned regardless
        # of the outcome's natural scale. The argmin is invariant to a positive
        # scalar on the residual (it scales the objective uniformly), so this
        # affects numerics only, not the optimum.
        scale = float(max(np.abs(y1_pre).max(), np.abs(y0_pre).max(), 1.0))
        y1_n = y1_pre / scale
        y0_n = y0_pre / scale

        n_donors = y0_n.shape[1]
        w = cp.Variable(n_donors, nonneg=True)
        objective = cp.Minimize(cp.sum_squares(y1_n - y0_n @ w))
        constraints = [cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(
                f"Simplex QP solver failed with status {problem.status!r}. "
                f"This usually means a degenerate donor pool."
            )
        if problem.status == "optimal_inaccurate":
            logger.info(
                "Clarabel reported 'optimal_inaccurate'; weights may be "
                "slightly off the simplex boundary."
            )

        weights: NDArray[np.float64] = np.asarray(w.value, dtype=np.float64)
        # Clip away small negatives from finite-precision arithmetic and
        # renormalize, so the returned vector exactly satisfies the constraints.
        weights = np.clip(weights, a_min=0.0, a_max=None)
        total = float(weights.sum())
        if total <= 0:
            raise RuntimeError("Solver returned all-zero weights; check the donor pool.")
        normalized: NDArray[np.float64] = weights / total
        return normalized
