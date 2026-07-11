"""Unit tests for the internal panel helpers (no R required)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from augsynth_py.exceptions import EmptyDonorPoolError, TreatedUnitNotFoundError
from augsynth_py.synth._panel import long_to_wide


def _toy_panel() -> pl.DataFrame:
    # 4 units (a,b,c,d), 3 periods (1,2,3). Values chosen so column means differ.
    rows = []
    values = {
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [100.0, 200.0, 300.0],
        "d": [4.0, 5.0, 6.0],
    }
    for u, vs in values.items():
        for t, v in zip([1, 2, 3], vs, strict=True):
            rows.append({"unit": u, "time": t, "Y": v})
    return pl.DataFrame(rows)


def test_long_to_wide_scalar_treated_unchanged() -> None:
    """A scalar `treated` keeps the existing single-treated contract."""
    panel = _toy_panel()
    y, units, _periods, treated_idx = long_to_wide(
        panel, unit="unit", time="time", outcome="Y", treated="a"
    )
    assert treated_idx == 0
    assert units[0] == "a"
    assert units[1:] == ["b", "c", "d"]
    np.testing.assert_allclose(y[:, 0], [1.0, 2.0, 3.0])


def test_long_to_wide_list_treated_collapses_to_mean() -> None:
    """A list `treated` places the mean of the treated units at column 0 and
    removes them from the donor pool."""
    panel = _toy_panel()
    y, units, _periods, treated_idx = long_to_wide(
        panel, unit="unit", time="time", outcome="Y", treated=["a", "d"]
    )
    assert treated_idx == 0
    # Column 0 is the elementwise mean of a and d.
    np.testing.assert_allclose(y[:, 0], [(1 + 4) / 2, (2 + 5) / 2, (3 + 6) / 2])
    # Donors are the remaining units, sorted; treated units are gone.
    assert units[1:] == ["b", "c"]
    assert y.shape == (3, 3)  # 1 collapsed treated + 2 donors


def test_long_to_wide_list_treated_label_is_deterministic() -> None:
    """The collapsed treated column gets a stable, readable label."""
    panel = _toy_panel()
    _, units, _, _ = long_to_wide(panel, unit="unit", time="time", outcome="Y", treated=["d", "a"])
    # Sorted + comma-joined, independent of input order.
    assert units[0] == "a,d"


def test_long_to_wide_single_element_list_equals_scalar() -> None:
    """`treated=["a"]` is identical to `treated="a"` in the matrix."""
    panel = _toy_panel()
    y_list, _, _, _ = long_to_wide(panel, unit="unit", time="time", outcome="Y", treated=["a"])
    y_scalar, _, _, _ = long_to_wide(panel, unit="unit", time="time", outcome="Y", treated="a")
    np.testing.assert_allclose(y_list, y_scalar)


def test_long_to_wide_duplicate_treated_entries_deduplicated() -> None:
    """Duplicates in the treated list must not double-weight a unit's series."""
    panel = _toy_panel()
    y, units, _, _ = long_to_wide(
        panel, unit="unit", time="time", outcome="Y", treated=["a", "d", "a"]
    )
    assert units[0] == "a,d"
    np.testing.assert_allclose(y[:, 0], [(1 + 4) / 2, (2 + 5) / 2, (3 + 6) / 2])


def test_long_to_wide_ndarray_treated_collapses_like_list() -> None:
    """Non-list iterables (e.g. np.ndarray) are treated as sequences too."""
    panel = _toy_panel()
    y, units, _, treated_idx = long_to_wide(
        panel, unit="unit", time="time", outcome="Y", treated=np.array(["a", "d"])
    )
    assert treated_idx == 0
    assert units[0] == "a,d"
    np.testing.assert_allclose(y[:, 0], [(1 + 4) / 2, (2 + 5) / 2, (3 + 6) / 2])


def test_long_to_wide_empty_treated_list_raises() -> None:
    panel = _toy_panel()
    with pytest.raises(ValueError, match="at least one unit"):
        long_to_wide(panel, unit="unit", time="time", outcome="Y", treated=[])


def test_long_to_wide_unknown_treated_in_list_raises() -> None:
    panel = _toy_panel()
    with pytest.raises(TreatedUnitNotFoundError):
        long_to_wide(panel, unit="unit", time="time", outcome="Y", treated=["a", "zzz"])


def test_long_to_wide_all_units_treated_raises_empty_donor() -> None:
    panel = _toy_panel()
    with pytest.raises(EmptyDonorPoolError):
        long_to_wide(panel, unit="unit", time="time", outcome="Y", treated=["a", "b", "c", "d"])
