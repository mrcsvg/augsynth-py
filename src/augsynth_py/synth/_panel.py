"""Internal panel reshaping and pre-period demeaning helpers.

These are not part of the public API. They are shared by every estimator that
needs to lift a long panel into a ``(T, N)`` outcome matrix with stable column
ordering — currently :class:`augsynth_py.synth.classical.Synth`, soon also the
augmented variant.
"""

from __future__ import annotations

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

    Parameters
    ----------
    panel
        Long-format panel with at least the columns ``unit``, ``time``, ``outcome``.
    unit, time, outcome
        Column names.
    treated
        Value in the ``unit`` column identifying the treated unit.

    Returns
    -------
    Y
        ``(T, N)`` outcome matrix with rows sorted ascending by ``time`` and the
        treated unit's column placed at index ``treated_idx``.
    units
        Length-``N`` list of unit names in the column order of ``Y``. The treated
        unit is always at position ``treated_idx`` (returned separately).
    periods
        Length-``T`` 1-D array of the ``time`` values in row order.
    treated_idx
        Column index of the treated unit in ``Y``.

    Raises
    ------
    ValueError
        If a required column is missing, or any ``(unit, time)`` pair is
        missing / duplicated (unbalanced panel).
    TreatedUnitNotFoundError
        If ``treated`` does not appear in the ``unit`` column.
    EmptyDonorPoolError
        If the panel contains only the treated unit.
    """
    missing = {unit, time, outcome} - set(panel.columns)
    if missing:
        raise ValueError(
            f"Panel is missing required column(s): {sorted(missing)}. Have columns: {panel.columns}"
        )

    all_units = panel[unit].unique().sort().to_list()
    if treated not in all_units:
        raise TreatedUnitNotFoundError(
            f"Treated unit {treated!r} not found in column {unit!r}. "
            f"Sample of values present: {all_units[:5]}"
        )

    donors = [u for u in all_units if u != treated]
    if not donors:
        raise EmptyDonorPoolError(
            f"Panel has only the treated unit {treated!r}; no donors to fit against."
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

    ordered_units = [str(treated)] + [str(d) for d in donors]
    treated_idx = 0
    y_matrix = wide.select(ordered_units).to_numpy().astype(np.float64)
    periods = wide[time].to_numpy()

    return y_matrix, ordered_units, periods, treated_idx


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
