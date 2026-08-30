"""Conformal inference for synthetic controls (Chernozhukov, Wüthrich & Zhu 2021).

Implements the permutation-test core of the exact and robust conformal inference
method of Chernozhukov, Wüthrich & Zhu (2021), "An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls", JASA (hereafter
CWZ 2021).

The public surface — :func:`conformal_pvalue`, :func:`conformal_test` (the
p-value plus the full permutation distribution of the test statistic), and
:func:`conformal_interval` — is built on top of the pure permutation-test core
(a residual test statistic and a permutation reference distribution with three
schemes: deterministic moving-block cyclic shifts, random contiguous-block
shuffles of a caller-chosen ``block_size``, and iid random permutations) and
the estimator's full-window refit-under-null hook
(``conformal_null_residuals``), which supplies the exchangeable residuals the
permutation test operates on. :func:`adjust_pvalues` provides standard
multiple-testing adjustments for collections of conformal p-values.

References
----------
Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Side = Literal["two-sided", "left", "right"]
PermutationType = Literal["block", "iid"]
AdjustMethod = Literal["bonferroni", "holm", "bh"]


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
    conformal_null_residuals(h0)
        Full-window residuals under the constant-effect null ``h0`` (CWZ 2021),
        obtained by refitting the synthetic control over the entire window with
        the treated post-period adjusted by ``h0``. See the estimator method
        for the exact construction.
    """

    gap_: NDArray[np.float64]
    pre_mask_: NDArray[np.bool_]
    att_: float

    def conformal_null_residuals(self, h0: float) -> NDArray[np.float64]: ...


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


def _permutation_distribution(
    residuals: NDArray[np.float64],
    post_mask: NDArray[np.bool_],
    side: Side,
    permutation_type: PermutationType,
    block_size: int | None,
    ns: int,
    rng: np.random.Generator | None,
) -> tuple[float, NDArray[np.float64], bool]:
    r"""Observed statistic and permutation reference distribution (CWZ 2021, §3).

    The full residual vector is permuted and the test statistic is recomputed on
    the *fixed* post positions (``post_mask``) for every permutation in the
    reference set. See CWZ 2021, §3 (Eq. defining :math:`S_q` and the
    permutation p-value). Three schemes are supported:

    - ``"block"`` with ``block_size=None``: the moving-block (cyclic-shift)
      scheme of CWZ 2021 §3, deterministic. The reference set is the :math:`T`
      cyclic shifts ``np.roll(residuals, j)`` for ``j = 0, ..., T-1``; ``j = 0``
      reproduces the observed statistic, so the observed statistic is a member
      of the returned distribution. ``ns`` and ``rng`` are ignored.
    - ``"block"`` with an integer ``block_size``: ``ns`` random shuffles of the
      contiguous length-``block_size`` blocks that partition the residual
      vector (the last block is shorter when :math:`T` is not a multiple).
      Within-block order is preserved, so serial dependence up to lag
      ``block_size - 1`` survives the permutation — the caller-tunable analogue
      of the moving-block scheme's robustness to weak dependence. ``block_size
      = 1`` degenerates to iid random permutations. Requires ``rng``.
    - ``"iid"``: ``ns`` random permutations of the individual residuals drawn
      from ``rng`` (all permutations equally likely), the scheme for iid
      errors. ``block_size`` must be ``None``.

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
    block_size : int or None
        Length of the contiguous blocks for the random-block scheme; ``None``
        selects the deterministic cyclic-shift scheme. Only valid with
        ``permutation_type="block"``; must satisfy ``1 <= block_size < T``.
    ns : int
        Number of random permutations (random schemes only; ignored for the
        deterministic cyclic-shift scheme).
    rng : numpy.random.Generator or None
        Random generator, required for the random schemes.

    Returns
    -------
    tuple[float, NDArray[np.float64], bool]
        ``(statistic, null_distribution, observed_included)``: the observed
        statistic, the statistics of the reference permutations, and whether
        the identity permutation (hence the observed statistic) is a member of
        the reference set — which decides the p-value convention in
        :func:`_pvalue_from_distribution`.

    Raises
    ------
    ValueError
        If ``permutation_type`` or ``side`` is unknown, a random scheme is
        requested without ``rng``, ``block_size`` is passed with ``"iid"``, or
        ``block_size`` is outside ``[1, T)``.
    """
    s_obs = _post_statistic(residuals, post_mask, side)
    t = int(residuals.shape[0])

    if permutation_type == "block":
        if block_size is None:
            stats = np.fromiter(
                (_post_statistic(np.roll(residuals, j), post_mask, side) for j in range(t)),
                dtype=np.float64,
                count=t,
            )
            return s_obs, stats, True
        if not 1 <= block_size < t:
            raise ValueError(
                f"block_size must be in [1, T) with T={t} total periods, got {block_size}: "
                "at block_size >= T there is a single block and every shuffle is the "
                "identity, so the test is degenerate."
            )
        if rng is None:
            raise ValueError("permutation_type='block' with a block_size requires a non-None rng.")
        blocks = [np.arange(s, min(s + block_size, t)) for s in range(0, t, block_size)]
        stats = np.empty(ns, dtype=np.float64)
        for i in range(ns):
            order = rng.permutation(len(blocks))
            permuted = residuals[np.concatenate([blocks[k] for k in order])]
            stats[i] = _post_statistic(permuted, post_mask, side)
        return s_obs, stats, False

    if permutation_type == "iid":
        if block_size is not None:
            raise ValueError(
                "block_size applies to permutation_type='block' only; use "
                "permutation_type='block' with block_size for random block shuffles "
                "(block_size=1 is equivalent to 'iid')."
            )
        if rng is None:
            raise ValueError("permutation_type='iid' requires a non-None rng.")
        stats = np.empty(ns, dtype=np.float64)
        for i in range(ns):
            stats[i] = _post_statistic(rng.permutation(residuals), post_mask, side)
        return s_obs, stats, False

    raise ValueError(f"Unknown permutation_type {permutation_type!r}; expected 'block' or 'iid'.")


def _pvalue_from_distribution(
    statistic: float,
    null_distribution: NDArray[np.float64],
    observed_included: bool,
) -> float:
    r"""Turn a permutation reference distribution into a p-value.

    When the reference set contains the identity permutation (the deterministic
    cyclic-shift scheme), :math:`p = \#\{S_j \ge S_{\mathrm{obs}}\} / n` — the
    identity always ties itself, so :math:`p \ge 1/n`. For random schemes the
    observed statistic is not a member and the finite-sample-valid add-one
    convention applies:
    :math:`p = (1 + \#\{S_{\mathrm{perm}} \ge S_{\mathrm{obs}}\}) / (1 + n)`
    (Phipson & Smyth 2010, "Permutation p-values should never be zero").
    """
    count = int(np.count_nonzero(null_distribution >= statistic))
    n = int(null_distribution.shape[0])
    if observed_included:
        return count / n
    return (1 + count) / (1 + n)


def _permutation_pvalue(
    residuals: NDArray[np.float64],
    post_mask: NDArray[np.bool_],
    side: Side,
    permutation_type: PermutationType,
    ns: int,
    rng: np.random.Generator | None,
    *,
    block_size: int | None = None,
) -> float:
    """Permutation p-value for the CWZ conformal test.

    Composition of :func:`_permutation_distribution` (which documents the
    schemes and their parameters) and :func:`_pvalue_from_distribution` (which
    documents the two p-value conventions).
    """
    s_obs, null_distribution, observed_included = _permutation_distribution(
        residuals, post_mask, side, permutation_type, block_size, ns, rng
    )
    return _pvalue_from_distribution(s_obs, null_distribution, observed_included)


# eq=False: the generated __eq__ would compare the ndarray fields elementwise,
# making `result_a == result_b` raise on truth-value ambiguity; identity
# semantics are the useful behaviour for a result object holding arrays.
@dataclass(frozen=True, eq=False)
class ConformalTestResult:
    r"""Full output of one conformal permutation test (CWZ 2021, §3).

    Returned by :func:`conformal_test`. Exposes the intermediate quantities
    between the refit-under-null residuals and the scalar p-value — in
    particular the permutation distribution of the test statistic, so the
    observed statistic can be placed against the reference distribution (e.g.
    a histogram of ``null_distribution`` with a vertical line at
    ``statistic``) instead of collapsing straight to ``pvalue``.

    Attributes
    ----------
    pvalue : float
        The conformal p-value, identical to what :func:`conformal_pvalue`
        returns for the same arguments.
    statistic : float
        The observed test statistic :math:`S_{\mathrm{obs}}` — the CWZ
        statistic evaluated on the post positions of the unpermuted residuals
        (see :func:`_post_statistic`; the constant :math:`1/\sqrt{T_1}`
        normalization is omitted, as it cancels in the p-value rank).
    null_distribution : NDArray[np.float64]
        The test statistic of every reference permutation. Under the
        deterministic cyclic-shift scheme this has length :math:`T` and its
        first entry (the ``j = 0`` identity shift) equals ``statistic``; under
        the random schemes it has length ``ns`` and ``statistic`` is not a
        member.
    residuals : NDArray[np.float64]
        The full-window refit-under-null residual vector the permutations were
        applied to (``conformal_null_residuals(h0)``).
    post_mask : NDArray[np.bool_]
        Boolean mask over ``residuals`` selecting the post-treatment
        positions the statistic is read on.
    observed_included : bool
        Whether the reference set contains the identity permutation, hence the
        p-value convention: ``True`` (cyclic shifts) means
        ``pvalue = mean(null_distribution >= statistic)``; ``False`` (random
        schemes) means the add-one convention
        ``pvalue = (1 + count) / (1 + len(null_distribution))``.
    h0 : float
        The hypothesized constant post-period effect that was tested.
    side : str
        Test direction the statistic was computed with.
    permutation_type : str
        Permutation scheme family, ``"block"`` or ``"iid"``.
    block_size : int or None
        Block length of the random-block scheme, or ``None`` for the
        deterministic cyclic-shift scheme / iid permutations.
    """

    pvalue: float
    statistic: float
    null_distribution: NDArray[np.float64]
    residuals: NDArray[np.float64]
    post_mask: NDArray[np.bool_]
    observed_included: bool
    h0: float
    side: Side
    permutation_type: PermutationType
    block_size: int | None


def conformal_test(
    fit: _FittedSC,
    h0: float = 0.0,
    *,
    permutation_type: PermutationType = "block",
    block_size: int | None = None,
    side: Side = "two-sided",
    ns: int = 1000,
    rng: np.random.Generator | None = None,
) -> ConformalTestResult:
    r"""Conformal permutation test with its full reference distribution.

    Identical to :func:`conformal_pvalue` — same null, same refit-under-null
    residual construction, same permutation schemes, bit-identical p-value for
    the same arguments (and the same ``rng`` state on the random schemes) —
    but returning the whole :class:`ConformalTestResult` instead of the scalar
    p-value: the observed statistic, the statistic of every reference
    permutation, and the residual vector they were computed from. Use it when
    the p-value alone is not enough — typically to show *how far* the observed
    post-period behaviour sits from the permutation reference distribution,
    not merely that :math:`p` is small.

    Parameters
    ----------
    fit : _FittedSC
        A fitted estimator exposing ``pre_mask_``, ``att_`` and the
        ``conformal_null_residuals`` method (any
        :class:`~augsynth_py.synth.classical.Synth` or
        :class:`~augsynth_py.synth.augmented.AugSynth` after ``.fit``).
    h0 : float, optional
        The hypothesized constant post-period effect. Defaults to ``0.0`` (the
        sharp null of no effect).
    permutation_type : {"block", "iid"}, optional
        Permutation scheme family; see :func:`conformal_pvalue`.
    block_size : int or None, optional
        ``None`` (default) with ``"block"`` selects the deterministic
        cyclic-shift scheme; an integer in ``[1, T)`` selects ``ns`` random
        shuffles of contiguous length-``block_size`` blocks (within-block
        order preserved), which requires ``rng``. See :func:`conformal_pvalue`.
    side : {"two-sided", "left", "right"}, optional
        Direction of the test; see :func:`_post_statistic`.
    ns : int, optional
        Number of random permutations for the random schemes; ignored by the
        deterministic cyclic-shift scheme. Defaults to ``1000``.
    rng : numpy.random.Generator or None, optional
        Random generator; required for the random schemes.

    Returns
    -------
    ConformalTestResult
        The p-value together with the observed statistic, the permutation
        reference distribution, the null residuals, and the test metadata.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    residuals = np.asarray(fit.conformal_null_residuals(h0), dtype=np.float64)
    post_mask = ~np.asarray(fit.pre_mask_, dtype=np.bool_)
    statistic, null_distribution, observed_included = _permutation_distribution(
        residuals, post_mask, side, permutation_type, block_size, ns, rng
    )
    return ConformalTestResult(
        pvalue=_pvalue_from_distribution(statistic, null_distribution, observed_included),
        statistic=statistic,
        null_distribution=null_distribution,
        residuals=residuals,
        post_mask=post_mask,
        observed_included=observed_included,
        h0=h0,
        side=side,
        permutation_type=permutation_type,
        block_size=block_size,
    )


def conformal_pvalue(
    fit: _FittedSC,
    h0: float = 0.0,
    *,
    permutation_type: PermutationType = "block",
    block_size: int | None = None,
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
    the estimator's ``conformal_null_residuals(h0)`` method (implemented on both
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
        ``conformal_null_residuals`` method (any
        :class:`~augsynth_py.synth.classical.Synth` or
        :class:`~augsynth_py.synth.augmented.AugSynth` after ``.fit``).
    h0 : float, optional
        The hypothesized constant post-period effect. Defaults to ``0.0`` (the
        sharp null of no effect).
    permutation_type : {"block", "iid"}, optional
        Permutation scheme family. ``"block"`` (default) with
        ``block_size=None`` is the deterministic moving-block (cyclic-shift)
        scheme of CWZ 2021 §3 and needs no ``rng``; with an integer
        ``block_size`` it becomes the random block-shuffle scheme below.
        ``"iid"`` draws ``ns`` random permutations of the individual residuals
        and requires ``rng``.
    block_size : int or None, optional
        ``None`` (default) keeps the deterministic cyclic-shift scheme. An
        integer ``b`` in ``[1, T)`` (``T`` = total periods) instead draws
        ``ns`` random shuffles of the contiguous length-``b`` blocks that
        partition the residual vector (the last block is shorter when ``T`` is
        not a multiple of ``b``). Within-block order is preserved, so serial
        dependence up to lag ``b - 1`` survives each permutation — a
        caller-tunable middle ground between ``"iid"`` (``b = 1`` is
        equivalent to it) and the cyclic-shift scheme, useful when the
        residuals are serially dependent and the ``1/T`` granularity of the
        cyclic scheme is too coarse. Requires ``rng``; the add-one p-value
        convention below applies. Only valid with ``permutation_type="block"``.
    side : {"two-sided", "left", "right"}, optional
        Direction of the test; see :func:`_post_statistic`.
    ns : int, optional
        Number of random permutations for the random schemes (``"iid"``, or
        ``"block"`` with a ``block_size``); ignored by the deterministic
        cyclic-shift scheme. Defaults to ``1000``.
    rng : numpy.random.Generator or None, optional
        Random generator; required for the random schemes, ignored by the
        deterministic cyclic-shift scheme.

    Returns
    -------
    float
        The conformal p-value in ``[0, 1]``. Deterministic cyclic-shift
        scheme: a multiple of ``1/T`` with a ``1/T`` floor. Random schemes:
        the add-one convention ``(1 + count) / (1 + ns)``, with a
        ``1/(1 + ns)`` floor.

    Notes
    -----
    Use :func:`conformal_test` to additionally obtain the observed statistic
    and the full permutation distribution it is ranked against, and
    :func:`adjust_pvalues` when several conformal p-values are reported
    together.

    The planned sibling ``geoexp`` package will map its public arguments onto
    this function: ``conformal_type`` -> ``permutation_type`` and
    ``side_of_test`` -> ``side``.

    References
    ----------
    Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
    Conformal Inference Method for Counterfactual and Synthetic Controls. JASA,
    116(536), 1849-1864.
    """
    residuals = np.asarray(fit.conformal_null_residuals(h0), dtype=np.float64)
    post_mask = ~np.asarray(fit.pre_mask_, dtype=np.bool_)
    return _permutation_pvalue(
        residuals, post_mask, side, permutation_type, ns, rng, block_size=block_size
    )


def conformal_interval(
    fit: _FittedSC,
    *,
    alpha: float = 0.05,
    grid_size: int = 100,
    permutation_type: PermutationType = "block",
    block_size: int | None = None,
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
        ``conformal_null_residuals`` method (any
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
    block_size : int or None, optional
        Block length for the random block-shuffle scheme, threaded to
        :func:`conformal_pvalue` (see there for semantics and constraints).
        ``None`` (default) keeps the deterministic cyclic-shift scheme.
    side : {"two-sided"}, optional
        Kept in the signature to make the two-sided requirement explicit. Only
        ``"two-sided"`` (the default) is accepted; any other value raises
        ``ValueError`` because a one-sided acceptance region is unbounded.
    ns : int, optional
        Number of random permutations for the random schemes (``"iid"``, or
        ``"block"`` with a ``block_size``); ignored by the deterministic
        cyclic-shift scheme. Defaults to ``1000``.
    rng : numpy.random.Generator or None, optional
        Random generator threaded to :func:`conformal_pvalue`; required for
        the random schemes. On the random paths the same generator is
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
    conservative in direction, but not a connected interval. Under the random
    permutation schemes (``"iid"``, or ``"block"`` with a ``block_size``) the
    same warning can also fire from Monte-Carlo noise at grid points whose
    p-value sits near ``alpha``.

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
                    block_size=block_size,
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
            "(see docs/methodology.md §5.5); under the random permutation "
            "schemes ('iid', or 'block' with a block_size) it can also be "
            "Monte-Carlo noise at grid points with p near alpha.",
            UserWarning,
            stacklevel=2,
        )
    return (float(candidates[accepted_idx[0]]), float(candidates[accepted_idx[-1]]))


def adjust_pvalues(
    pvalues: Sequence[float] | NDArray[np.float64],
    method: AdjustMethod = "holm",
) -> NDArray[np.float64]:
    r"""Adjust a collection of p-values for multiple testing.

    Conformal p-values are marginal: each one is valid for its own test. When
    several are reported together — one per placebo window, per candidate
    treated set, per tested ``h0``, or per outcome — the family-wise error
    rate (or false discovery rate) is no longer the nominal level, and the
    p-values should be adjusted before thresholding. This helper implements
    the three standard adjustments; each adjusted p-value can be compared
    directly to the desired level :math:`\alpha`.

    Parameters
    ----------
    pvalues : Sequence[float] or NDArray[np.float64]
        The raw p-values, each in ``[0, 1]``. One-dimensional and non-empty.
    method : {"holm", "bonferroni", "bh"}, optional
        - ``"holm"`` (default): Holm's step-down procedure (Holm 1979).
          Controls the family-wise error rate under arbitrary dependence, and
          is uniformly at least as powerful as Bonferroni — there is no
          assumption under which plain Bonferroni is preferable.
        - ``"bonferroni"``: :math:`\min(1, m \, p_i)`. Included for
          comparability with other software.
        - ``"bh"``: the Benjamini-Hochberg step-up procedure (Benjamini &
          Hochberg 1995), controlling the false discovery rate under
          independence or positive regression dependence. Appropriate for
          large exploratory families (e.g. a scan over many candidate
          designs) where FWER control is needlessly strict.

    Returns
    -------
    NDArray[np.float64]
        The adjusted p-values, in the same order as the input, each in
        ``[0, 1]``. Matches R's ``p.adjust`` with methods ``"holm"``,
        ``"bonferroni"`` and ``"BH"``.

    Raises
    ------
    ValueError
        If ``pvalues`` is empty or not one-dimensional, contains values
        outside ``[0, 1]`` or non-finite values, or ``method`` is unknown.

    References
    ----------
    Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery
    Rate: A Practical and Powerful Approach to Multiple Testing. Journal of
    the Royal Statistical Society, Series B, 57(1), 289-300.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be one-dimensional, got shape {p.shape}.")
    if p.size == 0:
        raise ValueError("pvalues must be non-empty.")
    if not np.all(np.isfinite(p)) or bool(np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("pvalues must all be finite and in [0, 1].")

    m = p.size
    if method == "bonferroni":
        return np.minimum(1.0, m * p)

    if method not in ("holm", "bh"):
        raise ValueError(f"Unknown method {method!r}; expected 'holm', 'bonferroni', or 'bh'.")

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        # Step-down: (m - j) * p_(j) for ascending rank j = 0..m-1, made
        # monotone non-decreasing by a running maximum.
        adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ranked))
    else:
        # Step-up: m/j * p_(j) for one-based rank j, made monotone by a
        # running minimum taken from the largest p downwards.
        raw = ranked * (m / np.arange(1, m + 1))
        adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted_sorted
    return out
