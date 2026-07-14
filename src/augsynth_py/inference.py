"""Conformal inference for synthetic controls (Chernozhukov, Wüthrich & Zhu 2021).

Implements the permutation-test core of the exact and robust conformal inference
method of Chernozhukov, Wüthrich & Zhu (2021), "An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls", JASA (hereafter
CWZ 2021).

The public conformal p-value and confidence-interval surface is built on top of
the two private helpers defined here in later tasks. This module currently
implements only the pure permutation-test core: a residual test statistic and a
permutation p-value with two permutation schemes (moving block and iid).

References
----------
Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Side = Literal["two-sided", "left", "right"]
PermutationType = Literal["block", "iid"]


class _FittedSC(Protocol):
    """Structural type for a fitted synthetic-control estimator.

    Both :class:`augsynth_py.synth.classical.Synth` and
    :class:`augsynth_py.synth.augmented.AugSynth` satisfy this Protocol after
    ``.fit(...)``. Only the three attributes used by :func:`conformal_pvalue`
    are declared.

    Attributes
    ----------
    gap_ : NDArray[np.float64]
        Actual minus synthetic outcome over *all* periods, ordered by period.
    pre_mask_ : NDArray[np.bool_]
        Boolean mask, same length as ``gap_``, True for pre-treatment periods
        (post-treatment periods are ``~pre_mask_``).
    att_ : float
        Mean post-treatment gap (the point estimate of the ATT).
    """

    gap_: NDArray[np.float64]
    pre_mask_: NDArray[np.bool_]
    att_: float


def _post_statistic(
    residuals: NDArray[np.float64],
    post_mask: NDArray[np.bool_],
    side: Side,
) -> float:
    r"""Compute the CWZ test statistic on the post-window residuals.

    Implements the (unnormalized) test statistic of CWZ 2021, §3, Eq. defining
    :math:`S_q` (the mean-absolute statistic, :math:`q = 1`). The :math:`S_q`
    normalization constant :math:`1/\sqrt{T_1}` of the paper is identical across
    permutations and cancels in the p-value ranking, so it is omitted here. The
    ``"right"`` and ``"left"`` variants use the signed sum for one-sided tests.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Residual vector over the full window (pre- and post-treatment).
    post_mask : NDArray[np.bool_]
        Boolean mask, same length as ``residuals``, selecting the post-treatment
        positions.
    side : str
        One of ``"two-sided"`` (``sum(abs(post))``), ``"right"``
        (``sum(post)``), or ``"left"`` (``-sum(post)``).

    Returns
    -------
    float
        The value of the test statistic on the selected post-window residuals.

    Raises
    ------
    ValueError
        If ``side`` is not one of the recognized values.
    """
    post = residuals[post_mask]
    if side == "two-sided":
        return float(np.sum(np.abs(post)))
    if side == "right":
        return float(np.sum(post))
    if side == "left":
        return float(-np.sum(post))
    raise ValueError(f"Unknown side {side!r}; expected 'two-sided', 'left', or 'right'.")


def _permutation_pvalue(
    residuals: NDArray[np.float64],
    post_mask: NDArray[np.bool_],
    side: Side,
    permutation_type: PermutationType,
    ns: int,
    rng: np.random.Generator | None,
) -> float:
    r"""Permutation p-value for the CWZ conformal test.

    The full residual vector is permuted and the test statistic is recomputed on
    the *fixed* post positions (``post_mask``), then compared to the observed
    statistic (the statistic on the unpermuted residuals). See CWZ 2021, §3
    (Eq. defining :math:`S_q` and the permutation p-value).

    Two permutation schemes are supported:

    - ``"block"``: the moving-block (cyclic-shift) scheme, deterministic. The
      reference set is the :math:`T` cyclic shifts ``np.roll(residuals, j)`` for
      ``j = 0, ..., T-1``; ``j = 0`` reproduces the observed statistic. Since the
      observed statistic is always a member of the reference set,
      :math:`p = \#\{j : S_j \ge S_{\mathrm{obs}}\} / T`, which is always at
      least :math:`1/T`. ``ns`` and ``rng`` are ignored.
    - ``"iid"``: ``ns`` random permutations drawn from ``rng``, using the
      finite-sample-valid convention
      :math:`p = (1 + \#\{S_{\mathrm{perm}} \ge S_{\mathrm{obs}}\}) / (1 + ns)`.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Residual vector over the full window.
    post_mask : NDArray[np.bool_]
        Boolean mask selecting the post-treatment positions.
    side : str
        Test direction; see :func:`_post_statistic`.
    permutation_type : str
        Either ``"block"`` or ``"iid"``.
    ns : int
        Number of random permutations (``"iid"`` only; ignored for ``"block"``).
    rng : numpy.random.Generator or None
        Random generator, required for ``"iid"``.

    Returns
    -------
    float
        The permutation p-value in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``permutation_type`` is unknown, ``side`` is unknown, or
        ``permutation_type == "iid"`` and ``rng`` is None.
    """
    s_obs = _post_statistic(residuals, post_mask, side)

    if permutation_type == "block":
        t = int(residuals.shape[0])
        count = 0
        for j in range(t):
            shifted = np.roll(residuals, j)
            if _post_statistic(shifted, post_mask, side) >= s_obs:
                count += 1
        return count / t

    if permutation_type == "iid":
        if rng is None:
            raise ValueError("permutation_type='iid' requires a non-None rng.")
        count = 0
        for _ in range(ns):
            permuted = rng.permutation(residuals)
            if _post_statistic(permuted, post_mask, side) >= s_obs:
                count += 1
        return (1 + count) / (1 + ns)

    raise ValueError(f"Unknown permutation_type {permutation_type!r}; expected 'block' or 'iid'.")


def conformal_pvalue(
    fit: _FittedSC,
    h0: float = 0.0,
    *,
    permutation_type: PermutationType = "block",
    side: Side = "two-sided",
    ns: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Conformal p-value for a constant post-period effect (CWZ 2021).

    Tests the null hypothesis that the treatment effect equals a constant ``h0``
    over the whole post-treatment window, for an already-fitted
    synthetic-control estimator, via the exact permutation test of CWZ 2021, §3.

    No refit under the null
    -----------------------
    In this package the synthetic-control weights are fit on the *pre-period
    only* (both ``Synth`` and ``AugSynth`` match on ``pre_mask_``). Subtracting a
    constant post-period effect ``h0`` from the treated post-treatment outcomes
    leaves every pre-period value untouched, so the counterfactual refit under
    :math:`H_0` recovers the *same* weights. The residual under :math:`H_0` is
    therefore available in closed form from the fitted gaps, with no refit:

    .. math::

        u_t = \mathrm{gap}_t              \quad t \in \text{pre-period}
        u_t = \mathrm{gap}_t - h_0        \quad t \in \text{post-period}

    The residual vector is then handed to the permutation core
    (:func:`_permutation_pvalue`), which recomputes the CWZ test statistic on the
    fixed post positions across the reference permutations.

    Parameters
    ----------
    fit : _FittedSC
        A fitted estimator exposing ``gap_``, ``pre_mask_`` and ``att_`` (any
        :class:`~augsynth_py.synth.classical.Synth` or
        :class:`~augsynth_py.synth.augmented.AugSynth` after ``.fit``).
    h0 : float, optional
        The hypothesized constant post-period effect. Defaults to ``0.0`` (the
        sharp null of no effect).
    permutation_type : {"block", "iid"}, optional
        Permutation scheme. ``"block"`` (default) is the deterministic
        moving-block (cyclic-shift) scheme and needs no ``rng``. ``"iid"`` draws
        ``ns`` random permutations and requires ``rng``.
    side : {"two-sided", "left", "right"}, optional
        Direction of the test; see :func:`_post_statistic`.
    ns : int, optional
        Number of random permutations for ``permutation_type="iid"``; ignored
        for ``"block"``. Defaults to ``1000``.
    rng : numpy.random.Generator or None, optional
        Random generator; required for ``permutation_type="iid"``, ignored for
        ``"block"``.

    Returns
    -------
    float
        The conformal p-value in ``[0, 1]``.

    Notes
    -----
    The sibling ``geolift-py`` package maps its public arguments onto this
    function: ``conformal_type`` -> ``permutation_type`` and ``side_of_test`` ->
    ``side``.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    residuals = np.asarray(fit.gap_, dtype=np.float64).copy()
    pre_mask = np.asarray(fit.pre_mask_, dtype=np.bool_)
    post_mask = ~pre_mask
    residuals[post_mask] -= h0
    return _permutation_pvalue(residuals, post_mask, side, permutation_type, ns, rng)


def conformal_interval(
    fit: _FittedSC,
    *,
    alpha: float = 0.05,
    grid_size: int = 100,
    permutation_type: PermutationType = "block",
    side: Side = "two-sided",
    ns: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    r"""Confidence interval for the constant post-period effect by test inversion.

    Builds a :math:`(1 - \alpha)` confidence interval for the constant
    post-period treatment effect by inverting the conformal test of CWZ 2021,
    §3: the interval is the *acceptance region* of the test,

    .. math::

        \mathrm{CI}_{1-\alpha} = \{h_0 : p(h_0) \ge \alpha\},

    where :math:`p(h_0)` is :func:`conformal_pvalue` evaluated at the null
    constant effect :math:`h_0`. Because the p-value is only available
    pointwise, the acceptance region is approximated on a finite grid and the
    interval is reported as ``(min, max)`` of the accepted grid points.

    Grid construction
    -----------------
    The grid is centred on ``center = mean(post gap) == fit.att_`` and spans
    ``center ± 6 * sd(post gap)`` (sample standard deviation, ``ddof=1``), with
    ``grid_size`` equally spaced points. The value ``0.0`` is always unioned in
    so the sharp null of no effect is evaluated. When the post window has a
    single period, or the post gaps are degenerate with zero variance,
    ``sd == 0`` and the span falls back to ``abs(center)`` (or ``1.0`` if
    ``center`` is also zero), so the grid is never a single point.

    If no grid point is accepted (an empty acceptance region, e.g. an extreme
    effect with a very fine but narrow grid), ``(nan, nan)`` is returned.

    Parameters
    ----------
    fit : _FittedSC
        A fitted estimator exposing ``gap_``, ``pre_mask_`` and ``att_`` (any
        :class:`~augsynth_py.synth.classical.Synth` or
        :class:`~augsynth_py.synth.augmented.AugSynth` after ``.fit``).
    alpha : float, optional
        Significance level; the interval has nominal coverage :math:`1 - \alpha`.
        Defaults to ``0.05``.
    grid_size : int, optional
        Number of equally spaced grid points before ``0.0`` is unioned in.
        Defaults to ``100``.
    permutation_type : {"block", "iid"}, optional
        Permutation scheme threaded to :func:`conformal_pvalue`. Defaults to
        ``"block"``.
    side : {"two-sided", "left", "right"}, optional
        Direction of the underlying test threaded to :func:`conformal_pvalue`. A
        proper two-sided confidence interval uses ``"two-sided"`` (the default).
    ns : int, optional
        Number of random permutations for ``permutation_type="iid"``; ignored
        for ``"block"``. Defaults to ``1000``.
    rng : numpy.random.Generator or None, optional
        Random generator threaded to :func:`conformal_pvalue`; required for
        ``permutation_type="iid"``. On the iid path the same generator is
        consumed sequentially across grid points, which is intentional.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds of the confidence interval, or
        ``(nan, nan)`` if no grid point is accepted.

    Notes
    -----
    The sibling ``geolift-py`` package consumes this function to populate
    ``ConfIntervals(method="conformal")``.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    gap = np.asarray(fit.gap_, dtype=np.float64)
    pre_mask = np.asarray(fit.pre_mask_, dtype=np.bool_)
    post_gap = gap[~pre_mask]

    center = float(np.mean(post_gap))
    sd = float(np.std(post_gap, ddof=1)) if post_gap.size > 1 else 0.0
    spread = 6.0 * sd
    if spread == 0.0:
        spread = abs(center) if center != 0.0 else 1.0

    grid = np.linspace(center - spread, center + spread, grid_size)
    grid = np.union1d(grid, [0.0])

    accepted = [
        float(h0)
        for h0 in grid
        if conformal_pvalue(
            fit,
            float(h0),
            permutation_type=permutation_type,
            side=side,
            ns=ns,
            rng=rng,
        )
        >= alpha
    ]

    if not accepted:
        return (float("nan"), float("nan"))
    return (float(min(accepted)), float(max(accepted)))
