"""R ``data.frame`` -> Polars conversion for the parity suite.

Adapted from the helper in mrcsvg/augsynth-py#22 (author: GuiMarthe). Keeping
the bridge Polars-native means the ``validation`` extra needs neither pandas
nor pyarrow: rpy2 alone is enough. The library itself never touches pandas.

Dtype mapping: R ``character`` -> ``pl.String``, ``integer`` -> ``pl.Int64``,
``numeric`` -> ``pl.Float64``, ``logical`` -> ``pl.Boolean``. ``factor``,
``Date`` and ``POSIXct`` columns are converted to character on the R side
first (a factor's levels, an ISO date / timestamp string); tests parse them as
they need, exactly as they did with the string columns the pandas route
produced. R ``NA`` becomes ``null``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

_AS_CHARACTER = frozenset({"factor", "Date", "POSIXct", "POSIXlt"})


def from_r_data_frame(frame: Any) -> pl.DataFrame:
    """Convert an R ``data.frame`` to Polars without routing through pandas."""
    import rpy2.rinterface as ri
    import rpy2.robjects as ro

    # rpy2 surfaces integer / character / logical NA as singleton objects
    # (subclasses of int / str / bool), not as None. Numeric NA arrives as nan
    # and is left alone: it is already a valid float.
    na_singletons = {id(na) for na in (ri.NA_Integer, ri.NA_Character, ri.NA_Logical)}

    columns: dict[str, list[object]] = {}
    for name in frame.names:
        column = frame.rx2(name)
        if _AS_CHARACTER & set(column.rclass):
            column = ro.r["as.character"](column)
        columns[str(name)] = [None if id(v) in na_singletons else v for v in column]
    return pl.DataFrame(columns)
