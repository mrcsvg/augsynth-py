"""Internal panel reshaping and pre-period demeaning helpers.

These are not part of the public API. They are shared by every estimator that
needs to lift a long panel into a ``(T, N)`` outcome matrix with stable column
ordering — currently :class:`augsynth_py.synth.classical.Synth`, soon also the
augmented variant.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from augsynth_py.exceptions import (
    EmptyDonorPoolError,
    SinglePeriodError,
    TreatedUnitNotFoundError,
)


def long_to_wide(
    panel: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treated: Any,
) -> tuple[NDArray[np.float64], list[str], NDArray[Any], int]:
    """Reshape a long panel into a ``(T, N)`` outcome matrix.

    ``treated`` may be a single unit value or a sequence of unit values. When a
    sequence is given, the treated units are collapsed into a single column at
    index 0 holding their elementwise mean (the treated *group* trajectory),
    and they are removed from the donor pool. Per R ``augsynth`` semantics for a
    block treatment with a single intervention time, this pooled-mean series is
    the fit target. Collapsing here (before any fixed-effect demeaning) is
    identical to demeaning each treated unit and then averaging: pre-period
    demeaning is a linear map applied identically to each column, so it
    distributes over the cross-column average.

    Parameters
    ----------
    panel
        Long-format panel with at least the columns ``unit``, ``time``, ``outcome``.
    unit, time, outcome
        Column names.
    treated
        Value in the ``unit`` column identifying the treated unit, or a
        sequence of such values identifying the treated group.

    Returns
    -------
    Y
        ``(T, N)`` outcome matrix with rows sorted ascending by ``time`` and the
        treated column placed at index ``treated_idx`` (always 0). For a
        sequence ``treated``, column 0 is the elementwise mean of the treated
        units.
    units
        Length-``N`` list of unit names in the column order of ``Y``. The
        treated column is always at position ``treated_idx``; for a sequence
        ``treated``, ``units[0]`` is the comma-joined label of the treated
        group, sorted by string representation (so ``[2, 10]`` labels as
        ``"10,2"``).
    periods
        Length-``T`` 1-D array of the ``time`` values in row order.
    treated_idx
        Column index of the treated column in ``Y`` (always 0).

    Raises
    ------
    ValueError
        If a required column is missing, ``treated`` is an empty sequence, or
        any ``(unit, time)`` pair is missing / duplicated (unbalanced panel).
    TreatedUnitNotFoundError
        If any treated value does not appear in the ``unit`` column.
    EmptyDonorPoolError
        If no donor units remain after removing the treated unit(s).
    """
    missing = {unit, time, outcome} - set(panel.columns)
    if missing:
        raise ValueError(
            f"Panel is missing required column(s): {sorted(missing)}. Have columns: {panel.columns}"
        )

    # Normalize `treated` to a list of one or more unit values. A bare str/bytes
    # is a scalar, not a sequence of characters; any other iterable (list, tuple,
    # set, np.ndarray, pl.Series, ...) is a sequence of unit values.
    if isinstance(treated, (str, bytes)):
        treated_list = [treated]
    elif isinstance(treated, Iterable):
        treated_list = sorted(set(treated), key=str)
    else:
        treated_list = [treated]
    if not treated_list:
        raise ValueError("`treated` must name at least one unit.")

    all_units = panel[unit].unique().sort().to_list()
    unknown = [t for t in treated_list if t not in all_units]
    if unknown:
        raise TreatedUnitNotFoundError(
            f"Treated unit(s) {unknown!r} not found in column {unit!r}. "
            f"Sample of values present: {all_units[:5]}"
        )

    treated_set = set(treated_list)
    donors = [u for u in all_units if u not in treated_set]
    if not donors:
        raise EmptyDonorPoolError(
            f"Panel has only treated unit(s) {treated_list!r}; no donors to fit against."
        )

    wide = panel.pivot(
        values=outcome,
        index=time,
        on=unit,
        aggregate_function=None,
    ).sort(time)

    if wide.select(pl.exclude(time)).null_count().to_numpy().sum() > 0:
        raise ValueError(
            "Panel is unbalanced: at least one (unit, time) pair is missing. "
            "Synthetic control requires a balanced panel."
        )

    donor_names = [str(d) for d in donors]
    treated_names = [str(t) for t in treated_list]
    donor_matrix = wide.select(donor_names).to_numpy().astype(np.float64)
    treated_matrix = wide.select(treated_names).to_numpy().astype(np.float64)
    treated_col = treated_matrix.mean(axis=1, keepdims=True)  # (T, 1) group mean

    ordered_units = [",".join(treated_names), *donor_names]
    y_matrix = np.concatenate([treated_col, donor_matrix], axis=1)
    periods = wide[time].to_numpy()
    treated_idx = 0

    return y_matrix, ordered_units, periods, treated_idx


def imbalance(
    y1_pre: NDArray[np.float64],
    y0_pre: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    r"""Pre-period L2 imbalance and its uniform-weight-scaled version.

    Matches the R ``augsynth`` definitions:

    .. math::

        \text{l2} = \lVert y_{1,\text{pre}} - Y_{0,\text{pre}} w \rVert_2, \quad
        \text{scaled} = \frac{\text{l2}}{\lVert y_{1,\text{pre}}
        - Y_{0,\text{pre}} w_{\text{unif}} \rVert_2},

    with :math:`w_{\text{unif}} = \mathbf{1}/J`. The scaled version is a ratio
    and is therefore invariant to any constant normalization of the norm.

    Parameters
    ----------
    y1_pre
        Treated pre-period vector, shape ``(T0,)``. Caller supplies the same
        (demeaned) space the weights were fit in.
    y0_pre
        Donor pre-period matrix, shape ``(T0, J)``.
    weights
        Length-``J`` fitted weight vector.

    Returns
    -------
    l2_imbalance, scaled_l2_imbalance
    """
    n_donors = y0_pre.shape[1]
    resid = y1_pre - y0_pre @ weights
    l2 = float(np.linalg.norm(resid))
    resid_unif = y1_pre - y0_pre @ np.full(n_donors, 1.0 / n_donors)
    denom = float(np.linalg.norm(resid_unif))
    scaled = l2 / denom if denom > 0 else 0.0
    return l2, scaled


def pre_period_mask(
    periods: NDArray[Any],
    treatment_time: Any,
) -> NDArray[np.bool_]:
    """Return the boolean mask ``periods < treatment_time``.

    Raises
    ------
    ValueError
        If ``treatment_time`` falls outside ``periods`` or leaves zero pre or
        post periods.
    """
    mask: NDArray[np.bool_] = (periods < treatment_time).astype(bool)
    n_pre = int(mask.sum())
    n_post = int((~mask).sum())
    if n_pre == 0:
        raise ValueError(
            f"treatment_time={treatment_time!r} leaves zero pre-treatment periods. "
            f"It must be strictly greater than the earliest period."
        )
    if n_post == 0:
        raise ValueError(
            f"treatment_time={treatment_time!r} leaves zero post-treatment periods. "
            f"It must be no greater than the latest period."
        )
    if n_pre < 2:
        raise SinglePeriodError(f"Need at least two pre-treatment periods to fit; got {n_pre}.")
    return mask


def apply_unit_fixedeff(
    y_matrix: NDArray[np.float64],
    pre_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract each unit's pre-period mean from its column.

    Parameters
    ----------
    y_matrix
        ``(T, N)`` outcome matrix.
    pre_mask
        Length-``T`` boolean mask selecting pre-treatment rows.

    Returns
    -------
    y_demeaned
        ``y_matrix`` minus the per-column pre-period mean.
    offsets
        Length-``N`` vector of pre-period means (the offsets subtracted).
    """
    offsets = y_matrix[pre_mask].mean(axis=0)
    return y_matrix - offsets[None, :], offsets
