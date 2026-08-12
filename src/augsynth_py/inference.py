"""Conformal inference for synthetic controls (Chernozhukov, Wüthrich & Zhu 2021).

Implements the permutation-test core of the exact and robust conformal inference
method of Chernozhukov, Wüthrich & Zhu (2021), "An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls", JASA (hereafter
CWZ 2021).

The public conformal p-value and confidence-interval surface is built on top of
the pure permutation-test core (a residual test statistic and a permutation
p-value with two schemes, moving block and iid) and the estimator's
full-window refit-under-null hook (``_conformal_null_residuals``), which
supplies the exchangeable residuals the permutation test operates on.

References
----------
Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

import warnings
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Side = Literal["two-sided", "left", "right"]
PermutationType = Literal["block", "iid"]


class _FittedSC(Protocol):
    """Structural type for a fitted synthetic-control estimator.

    Both :class:`augsynth_py.synth.classical.Synth` and
    :class:`augsynth_py.synth.augmented.AugSynth` satisfy this Protocol after
    ``.fit(...)``.

    Attributes
    ----------
    gap_ : NDArray[np.float64]
        Actual minus synthetic outcome over *all* periods, ordered by period.
        Used only to *place* the confidence-interval grid
        (:func:`conformal_interval`); the p-value itself does not read it.
    pre_mask_ : NDArray[np.bool_]
        Boolean mask, same length as ``gap_``, True for pre-treatment periods
        (post-treatment periods are ``~pre_mask_``).
    att_ : float
        Mean post-treatment gap (the point estimate of the ATT); the CI grid
        centre.

    Methods
    -------
    _conformal_null_residuals(h0)
        Full-window residuals under the constant-effect null ``h0`` (CWZ 2021),
        obtained by refitting the synthetic control over the entire window with
        the treated post-period adjusted by ``h0``. See the estimator method
        for the exact construction.
    """

    gap_: NDArray[np.float64]
    pre_mask_: NDArray[np.bool_]
    att_: float

    def _conformal_null_residuals(self, h0: float) -> NDArray[np.float64]: ...


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

    Refit under the null
    --------------------
    CWZ 2021's exactness rests on the full-window residuals being *exchangeable*
    under :math:`H_0`. To obtain them, the treated unit's post-period outcomes
    are adjusted by ``h0`` and the synthetic control is **refit over the entire
    window** (all ``T`` periods), not the pre-period only. This is delegated to
    the estimator's ``_conformal_null_residuals(h0)`` method (implemented on both
    :class:`~augsynth_py.synth.classical.Synth` and
    :class:`~augsynth_py.synth.augmented.AugSynth`); the resulting length-``T``
    residual vector is handed to the permutation core
    (:func:`_permutation_pvalue`), which recomputes the CWZ test statistic on the
    fixed post positions across the reference permutations.

    This refit is intentional and distinct from the point-estimate path (whose
    weights are pre-period-anchored): the ATT is a pre-period counterfactual,
    whereas the conformal residual is CWZ's full-window exchangeability
    construct. It matches R ``augsynth``'s conformal inference exactly on the
    ``GeoLift_PreTest`` fixture (see
    ``tests/validation_against_r/test_conformal.py``).

    Parameters
    ----------
    fit : _FittedSC
        A fitted estimator exposing ``pre_mask_``, ``att_`` and the
        ``_conformal_null_residuals`` method (any
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
    The planned sibling ``geoexp`` package will map its public arguments onto
    this function: ``conformal_type`` -> ``permutation_type`` and
    ``side_of_test`` -> ``side``.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    residuals = np.asarray(fit._conformal_null_residuals(h0), dtype=np.float64)
    post_mask = ~np.asarray(fit.pre_mask_, dtype=np.bool_)
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

    Only the two-sided test can be inverted into a confidence interval: the
    acceptance region of a one-sided statistic is a half-line, not a bounded
    interval, so a finite ``(min, max)`` over the grid would be a meaningless
    grid-edge artifact. ``side`` must therefore be ``"two-sided"`` (the
    default); use :func:`conformal_pvalue` directly for one-sided testing.

    Grid construction
    -----------------
    The grid is centred on ``center = mean(post gap) == fit.att_`` and spans
    ``center ± 6 * sd(post gap)`` (sample standard deviation, ``ddof=1``), with
    ``grid_size`` equally spaced points. The value ``0.0`` is always unioned in
    so the sharp null of no effect is evaluated. When the post window has a
    single period, or the post gaps are degenerate with zero variance,
    ``sd == 0`` and the span falls back to ``abs(center)`` (or ``1.0`` if
    ``center`` is also zero), so the grid is never a single point.

    Parameters
    ----------
    fit : _FittedSC
        A fitted estimator exposing ``gap_``, ``pre_mask_``, ``att_`` and the
        ``_conformal_null_residuals`` method (any
        :class:`~augsynth_py.synth.classical.Synth` or
        :class:`~augsynth_py.synth.augmented.AugSynth` after ``.fit``). ``gap_``
        and ``att_`` place the grid; each grid point is scored by refitting via
        :func:`conformal_pvalue`.
    alpha : float, optional
        Significance level; the interval has nominal coverage :math:`1 - \alpha`.
        Defaults to ``0.05``.
    grid_size : int, optional
        Number of equally spaced grid points before ``0.0`` is unioned in.
        Defaults to ``100``.
    permutation_type : {"block", "iid"}, optional
        Permutation scheme threaded to :func:`conformal_pvalue`. Defaults to
        ``"block"``.
    side : {"two-sided"}, optional
        Kept in the signature to make the two-sided requirement explicit. Only
        ``"two-sided"`` (the default) is accepted; any other value raises
        ``ValueError`` because a one-sided acceptance region is unbounded.
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
        ``(nan, nan)`` if *no* grid point is accepted (an empty acceptance
        region). Refining or widening the grid does *not* recover a non-empty
        interval in that case.

        Empty ``(nan, nan)`` requires the *peak* of the ``p(h0)`` curve to fall
        below ``alpha``. Under the block scheme ``p(h0)`` peaks near a
        well-specified ``h0`` (large — the ``j = 0`` identity shift makes the
        residuals nearly tie, so ``p`` approaches 1) and decays to a **floor of
        ``1/T``** (``T`` = total periods; the identity shift always ties itself,
        so the count is at least 1) at extreme ``h0``. An empty region therefore
        arises only from **residual non-exchangeability** — a poor / trending /
        heteroskedastic fit that depresses even the peak below ``alpha``. It is
        *only possible when* ``1/T < alpha`` (i.e. ``T > 1/alpha``, e.g.
        ``T > 20`` for a 95% CI); otherwise the ``1/T`` floor keeps every ``h0``
        accepted.

        The complementary regime, ``T <= 1/alpha`` (e.g. ``T = 14`` at
        ``alpha = 0.05``, floor ``1/14 ~ 0.071 >= 0.05``), is *not* empty: the
        floor holds ``p(h0) >= 1/T >= alpha`` at **every** ``h0``, so the
        acceptance region is **unbounded**. That hits the truncation guard
        (widening + ``UserWarning`` + finite truncated bounds; see *Notes*),
        never ``(nan, nan)``.

    Notes
    -----
    The reported bounds are the innermost accepted grid points, so each bound is
    accurate only to about one grid spacing, ``2 * spread / (grid_size - 1)``;
    increasing ``grid_size`` tightens this resolution.

    **Truncation guard.** The grid span starts at ``6 * sd(post gap)``, derived
    from the *point-estimate* ``gap_`` dispersion, but each grid point is scored
    by the *full-window refit* acceptance region — a different, often wider
    quantity. When the donors fit tightly (small ``sd``) yet the conformal test
    has little power (a broad range of ``h0`` is accepted), the initial span is
    far narrower than the region and ``min`` / ``max`` would clip to the grid
    edges, silently understating (anti-conservatively) the interval. To avoid
    this, if an accepted endpoint reaches a grid boundary the span is doubled
    (centre, ``grid_size`` and the unioned ``0.0`` preserved) and the grid is
    re-scored, up to 8 doublings, stopping once both endpoints are strictly
    interior. If truncation persists at the cap, a :class:`UserWarning` is
    emitted and the widest computed bounds are returned as a lower bound on the
    true interval. This is distinct from the empty-region ``(nan, nan)`` case in
    *Returns*: there *no* ``h0`` is accepted (peak p-value below ``alpha``),
    which widening cannot fix; here a boundary-reaching region is accepted and
    widening extends it.

    **Non-contiguity warning.** The acceptance region of the refit-under-null
    test is not guaranteed to be an interval: because the residuals depend on
    ``h0`` through the refit, ``p(h0)`` can dip below ``alpha`` on a stretch
    strictly inside the accepted envelope (demonstrated on the Basque panel,
    where the rejected gap contains ``att_`` itself — identically in R; see
    ``docs/methodology.md`` §5.5). When the accepted grid points are not
    consecutive, a :class:`UserWarning` is emitted and the returned ``(min,
    max)`` envelope *overstates* (never understates) the acceptance region —
    conservative in direction, but not a connected interval. Under
    ``permutation_type="iid"`` the same warning can also fire from Monte-Carlo
    noise at grid points whose p-value sits near ``alpha``.

    The planned sibling ``geoexp`` package will consume this function to
    populate ``ConfIntervals(method="conformal")``.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    if side != "two-sided":
        raise ValueError(
            "conformal_interval inverts the two-sided conformal test; one-sided "
            "intervals are not supported. Use conformal_pvalue(..., side=...) for "
            "one-sided testing."
        )

    gap = np.asarray(fit.gap_, dtype=np.float64)
    pre_mask = np.asarray(fit.pre_mask_, dtype=np.bool_)
    post_gap = gap[~pre_mask]

    # The grid centre is the point estimate att_, which equals mean(post_gap) by
    # construction on both Synth and AugSynth (att_ = mean(gap[~pre_mask])); this
    # is why att_ is part of the _FittedSC protocol. post_gap still supplies the
    # dispersion for the span.
    center = float(fit.att_)
    sd = float(np.std(post_gap, ddof=1)) if post_gap.size > 1 else 0.0
    # Degenerate / near-perfect pre-fit: the point-estimate post gaps carry no
    # usable dispersion (a single post period, or sd at the solver-noise floor
    # ~1e-14). A 6*sd span then collapses to a point that both mis-locates the
    # interval and cannot be widened by doubling, so fall back to a center-scaled
    # span. The threshold is relative to the outcome scale so it fires on solver
    # noise but never on a genuine (however small) residual spread.
    scale = abs(center) if center != 0.0 else 1.0
    spread = 6.0 * sd
    if spread < 1e-6 * scale:
        spread = scale

    def _accepted(lin: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Score the sorted candidate grid (``0.0`` always evaluated).

        Returns the candidate array and the boolean acceptance mask over it,
        so the caller can recover both the accepted values and their grid
        positions (the positions drive the non-contiguity diagnostic).
        """
        candidates = np.union1d(lin, [0.0])
        mask = np.fromiter(
            (
                conformal_pvalue(
                    fit,
                    float(h0),
                    permutation_type=permutation_type,
                    side=side,
                    ns=ns,
                    rng=rng,
                )
                >= alpha
                for h0 in candidates
            ),
            dtype=bool,
            count=candidates.size,
        )
        return candidates, mask

    # Test inversion on a finite grid. The grid span is derived from the
    # point-estimate gap_ dispersion, but each point is scored by the
    # full-window refit acceptance region — a *different*, often wider quantity
    # (tight donors -> small sd, yet a low-power conformal test accepts a broad
    # range of h0). If the accepted set reaches a linspace boundary the interval
    # is truncated and would silently understate coverage, so widen the span
    # (double it, keeping the centre, grid_size, and the unioned 0.0) and
    # re-score until both endpoints are strictly interior or the cap is hit.
    max_doublings = 8
    truncated = False
    accepted_idx = np.empty(0, dtype=np.intp)
    candidates = np.empty(0, dtype=np.float64)
    for attempt in range(max_doublings + 1):
        lin = np.linspace(center - spread, center + spread, grid_size)
        candidates, accept_mask = _accepted(lin)
        if not accept_mask.any():
            # Empty acceptance region: widening cannot recover it (the peak
            # p-value over h0 is below alpha — structural, see Returns).
            return (float("nan"), float("nan"))
        accepted_idx = np.flatnonzero(accept_mask)
        lower = float(candidates[accepted_idx[0]])
        upper = float(candidates[accepted_idx[-1]])
        truncated = (lower <= float(lin[0])) or (upper >= float(lin[-1]))
        if not truncated or attempt == max_doublings:
            break
        spread *= 2.0

    if truncated:
        warnings.warn(
            "conformal_interval may be truncated: the acceptance region still "
            f"reaches a grid boundary after {max_doublings} span doublings "
            f"(final half-width {spread:g} about center {center:g}). The "
            "returned bounds understate the interval; the estimator likely has "
            "little conformal power (the test accepts h0 across a very wide "
            "range). Treat the bounds as a lower bound on the true interval.",
            UserWarning,
            stacklevel=2,
        )
    # min/max recovers the CI only when the two-sided acceptance region is
    # contiguous — which is empirical, not guaranteed by convexity, since under
    # refit-under-null the residuals themselves depend on h0 through the refit
    # (so the statistic is not a fixed convex function of h0). Non-contiguity
    # occurs on real panels: on the Basque panel a rejected 1/T-floor gap sits
    # strictly inside the envelope and contains att_ itself, identically in R
    # (methodology.md §5.5). When that happens the returned bounds are the
    # min/max ENVELOPE, wider than the true acceptance region (conservative
    # direction), and the warning below stops them being misread as a
    # connected region. On the iid path Monte-Carlo noise across grid points
    # can additionally open spurious single-point gaps.
    interior_gaps = int(np.count_nonzero(np.diff(accepted_idx) > 1))
    if interior_gaps > 0:
        warnings.warn(
            f"conformal_interval: the acceptance region is non-contiguous at "
            f"alpha={alpha:g} — {interior_gaps} rejected gap(s) lie strictly "
            "inside the returned bounds. The bounds are the min/max envelope "
            "of the accepted grid points and overstate (never understate) the "
            "true acceptance region; do not read them as a connected interval. "
            "This is a genuine property of CWZ test inversion on some panels "
            "(see docs/methodology.md §5.5); under permutation_type='iid' it "
            "can also be Monte-Carlo noise at grid points with p near alpha.",
            UserWarning,
            stacklevel=2,
        )
    return (float(candidates[accepted_idx[0]]), float(candidates[accepted_idx[-1]]))
