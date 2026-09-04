"""Small Polars conversion helpers for R parity tests."""

from typing import Any

import polars as pl


def from_r_data_frame(frame: Any) -> pl.DataFrame:
    """Convert an R data.frame without routing through pandas."""
    import rpy2.robjects as ro

    columns: dict[str, list[object]] = {}
    for name in frame.names:
        column = frame.rx2(name)
        if set(column.rclass) & {"factor", "Date", "POSIXct"}:
            column = ro.r["as.character"](column)
        columns[str(name)] = list(column)
    return pl.DataFrame(columns)


def to_r_data_frame(frame: pl.DataFrame) -> Any:
    """Convert the scalar dtypes used by parity fixtures to an R data.frame."""
    import rpy2.robjects as ro

    columns = {}
    for name, dtype in frame.schema.items():
        values = frame.get_column(name).to_list()
        if dtype == pl.String:
            columns[name] = ro.StrVector(values)
        elif dtype == pl.Boolean:
            columns[name] = ro.BoolVector(values)
        elif dtype.is_integer():
            columns[name] = ro.IntVector(values)
        elif dtype.is_numeric():
            columns[name] = ro.FloatVector(values)
        else:
            raise TypeError(f"Unsupported Polars dtype for R conversion: {name}={dtype}")
    return ro.DataFrame(columns)
