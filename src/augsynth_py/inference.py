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

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Side = Literal["two-sided", "left", "right"]
PermutationType = Literal["block", "iid"]


def _post_statistic(
    residuals: NDArray[np.float64],
    post_mask: NDArray[np.bool_],
    side: str,
) -> float:
    r"""Compute the CWZ test statistic on the post-window residuals.

    Implements the (unnormalized) test statistic of CWZ 2021, Section 3. The
    :math:`S_q` normalization constant :math:`1/\sqrt{T_1}` of the paper is
    identical across permutations and cancels in the p-value ranking, so it is
    omitted here.

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
    side: str,
    permutation_type: str,
    ns: int,
    rng: np.random.Generator | None,
) -> float:
    r"""Permutation p-value for the CWZ conformal test.

    The full residual vector is permuted and the test statistic is recomputed on
    the *fixed* post positions (``post_mask``), then compared to the observed
    statistic (the statistic on the unpermuted residuals). See CWZ 2021,
    Section 3.

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
