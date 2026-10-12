"""The R -> Polars bridge itself: dtypes, NA, and the classes it stringifies."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from ._r_frames import from_r_data_frame


@pytest.fixture(scope="module")
def mixed_frame(r_session: Any) -> pl.DataFrame:
    return from_r_data_frame(
        r_session(
            "data.frame("
            's = c("a", NA), '
            "i = c(1L, NA), "
            "d = c(1.5, NA), "
            "b = c(TRUE, NA), "
            'f = factor(c("y", "x")), '
            'dt = as.Date(c("2021-01-01", "2021-01-02")), '
            "stringsAsFactors = FALSE)"
        )
    )


def test_schema(mixed_frame: pl.DataFrame) -> None:
    assert mixed_frame.schema == pl.Schema(
        {
            "s": pl.String,
            "i": pl.Int64,
            "d": pl.Float64,
            "b": pl.Boolean,
            "f": pl.String,
            "dt": pl.String,
        }
    )


def test_values_and_na(mixed_frame: pl.DataFrame) -> None:
    assert mixed_frame.get_column("s").to_list() == ["a", None]
    assert mixed_frame.get_column("i").to_list() == [1, None]
    assert mixed_frame.get_column("b").to_list() == [True, None]
    d = mixed_frame.get_column("d").to_list()
    assert d[0] == 1.5 and d[1] != d[1]  # R numeric NA arrives as nan


def test_factor_and_date_are_stringified(mixed_frame: pl.DataFrame) -> None:
    # Factor levels, not the integer codes; ISO dates, not days since epoch.
    assert mixed_frame.get_column("f").to_list() == ["y", "x"]
    assert mixed_frame.get_column("dt").to_list() == ["2021-01-01", "2021-01-02"]


def test_geolift_pretest_schema_unchanged(r_geolift_pretest: pl.DataFrame) -> None:
    # The schema every parity test was written against: `date` stays a
    # string (R stores it as character), parsed by each test as needed.
    assert r_geolift_pretest.schema == pl.Schema(
        {"location": pl.String, "Y": pl.Int64, "date": pl.String}
    )
    assert r_geolift_pretest.shape == (3600, 3)
