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


# ---------------------------------------------------------------------------
# wide_to_long (inverse of long_to_wide)
# ---------------------------------------------------------------------------


def _round_trip_panel() -> pl.DataFrame:
    from datetime import date, timedelta

    rows = []
    for u in ("a", "b", "c", "d"):
        for s in range(6):
            rows.append(
                {
                    "unit": u,
                    "date": date(2021, 1, 1) + timedelta(days=s),
                    "y": float(ord(u) * 10 + s),
                }
            )
    return pl.DataFrame(rows)


def test_wide_to_long_round_trips_long_to_wide_with_dates() -> None:
    from augsynth_py.synth._panel import long_to_wide, wide_to_long

    panel = _round_trip_panel()
    y, units, periods, t_idx = long_to_wide(
        panel, unit="unit", time="date", outcome="y", treated="b"
    )
    long = wide_to_long(y, units, periods)

    assert long.columns == ["unit", "time", "outcome"]
    assert long.height == y.size
    assert long["time"].dtype == pl.Date
    y2, units2, periods2, t_idx2 = long_to_wide(
        long, unit="unit", time="time", outcome="outcome", treated=units[t_idx]
    )
    np.testing.assert_array_equal(y2, y)
    assert units2 == units
    np.testing.assert_array_equal(periods2, periods)
    assert t_idx2 == t_idx


def test_wide_to_long_keeps_group_label_as_a_unit() -> None:
    from augsynth_py.synth._panel import long_to_wide, wide_to_long

    panel = _round_trip_panel()
    y, units, periods, _ = long_to_wide(
        panel, unit="unit", time="date", outcome="y", treated=["a", "c"]
    )
    long = wide_to_long(y, units, periods)
    assert set(long["unit"].unique().to_list()) == {"a,c", "b", "d"}
    group = long.filter(pl.col("unit") == "a,c").sort("time")["outcome"].to_numpy()
    np.testing.assert_allclose(group, y[:, 0])


def test_wide_to_long_custom_column_names_and_int_periods() -> None:
    from augsynth_py.synth._panel import wide_to_long

    y = np.arange(6.0).reshape(3, 2)  # 3 periods x 2 units
    long = wide_to_long(y, ["u0", "u1"], np.array([10, 11, 12]), unit="g", time="t", outcome="v")
    assert long.columns == ["g", "t", "v"]
    assert long.filter(pl.col("g") == "u1")["v"].to_list() == [1.0, 3.0, 5.0]
    assert long["t"].to_list() == [10, 11, 12, 10, 11, 12]


def test_wide_to_long_rejects_shape_mismatch() -> None:
    from augsynth_py.synth._panel import wide_to_long

    y = np.zeros((3, 2))
    with pytest.raises(ValueError, match="units"):
        wide_to_long(y, ["only_one"], np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="periods"):
        wide_to_long(y, ["a", "b"], np.array([0, 1]))
    with pytest.raises(ValueError, match="2-D"):
        wide_to_long(np.zeros(3), ["a"], np.array([0, 1, 2]))
