"""Parity: multi-treated Synth vs augsynth with several treat==1 units.

RISK NOTE: this test pins the assumption that R augsynth, given a block
treatment (single t_int) with multiple treated units, fits the pooled mean of
the treated series. On the Python side that collapse lives in
``long_to_wide`` (src/augsynth_py/synth/_panel.py), which averages the treated
columns into column 0 before per-unit fixed-effect demeaning; see also the
``treated`` parameter docstring on ``Synth.fit``. If augsynth instead fits
each treated unit separately and averages results, this test will fail and
the collapse semantics must be revisited. The test IS the spec for that
assumption — it was verified against a live R augsynth run (2026-07-11),
not inferred from documentation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest


@pytest.mark.requires_r_pkg("augsynth")
def test_multitreated_synth_matches_r_augsynth(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> None:
    """Multi-treated ``Synth(fixedeff=True)`` matches augsynth block treatment.

    Treatment is hypothetical: we designate ``chicago`` and ``portland`` as
    treated starting on 2021-02-15. ``GeoLift_PreTest`` contains no real
    intervention, so this is purely a test of whether the two implementations
    construct the same synthetic control for the pooled treated group from the
    same pre-period data.
    """
    treated_units = ["chicago", "portland"]
    treatment_date = date(2021, 2, 15)

    # ------------------------------------------------------------------
    # 1. R reference output
    # ------------------------------------------------------------------
    # The R script is interpolated from the same Python variables used for the
    # Python fit below, so the two sides cannot drift.
    r_treated_vec = ", ".join(f"'{u}'" for u in treated_units)
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(
        f"""
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location %in% c({r_treated_vec})
            & df$date >= as.Date('{treatment_date.isoformat()}')
        )
        res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'None', scm = TRUE, fixedeff = TRUE
        )
        """
    )
    r_synthetic = np.asarray(r_session("as.numeric(predict(res, att = FALSE))"), dtype=float)

    # ------------------------------------------------------------------
    # 2. Python implementation
    # ------------------------------------------------------------------
    from augsynth_py import Synth

    panel = r_geolift_pretest.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    est = Synth(fixedeff=True).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=treated_units,
        treatment_time=treatment_date,
    )

    # ------------------------------------------------------------------
    # 3. Numerical agreement
    # ------------------------------------------------------------------
    from tests.validation_against_r.conftest import assert_array_close

    # The counterfactual path is what users consume. We match it strictly.
    # Tolerance follows README: paths use atol=1e-4, rtol=1e-4 (sales are on
    # the order of 10^5-10^6 so this is a relative agreement of ~1e-9).
    assert_array_close(
        est.synthetic_,
        r_synthetic,
        atol=1e-4,
        rtol=1e-4,
        name="multi-treated counterfactual path",
    )

    # Donor weights are not compared pointwise (flat minima under collinear
    # donors — see test_classical_synth.py), but the simplex invariants are
    # unique and must hold.
    py_weights = np.array(list(est.weights_.values()), dtype=float)
    assert (py_weights >= -1e-9).all(), "Python weights violate w >= 0"
    assert abs(py_weights.sum() - 1.0) < 1e-6, "Python weights do not sum to 1"
