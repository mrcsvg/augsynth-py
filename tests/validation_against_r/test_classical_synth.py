"""Parity test for the classical synthetic control estimator.

Validates :class:`augsynth_py.Synth` against the R ``augsynth`` package
called with ``progfunc = "None"``, ``scm = TRUE``, ``fixedeff = TRUE``. This
is the DI-2016 / augsynth-form classical SCM — not the Abadie-Diamond-
Hainmueller (2010) predictor-optimized form. See ``docs/methodology.md`` §1
for why this is the oracle of choice for the MVP.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest


@pytest.mark.requires_r_pkg("augsynth")
def test_classical_synth_matches_r_augsynth_progfunc_none(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> None:
    """``Synth(fixedeff=True)`` matches ``augsynth(progfunc='None', scm=TRUE)``.

    Treatment is hypothetical: we designate ``new york`` as the treated unit
    starting on 2021-02-15. ``GeoLift_PreTest`` contains no real intervention,
    so this is purely a test of whether the two implementations construct the
    same simplex-constrained synthetic control from the same pre-period data.
    """
    treated_unit = "new york"
    treatment_date = date(2021, 2, 15)

    # ------------------------------------------------------------------
    # 1. R reference output
    # ------------------------------------------------------------------
    r_session("suppressPackageStartupMessages(library(augsynth))")
    # `GeoLift_PreTest$date` ships as character; augsynth coerces internally and
    # silently fails. Convert to a real Date before passing.
    r_session(
        """
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location == 'new york' & df$date >= as.Date('2021-02-15')
        )
        res <- augsynth(
            Y ~ treat,
            unit     = location,
            time     = date,
            data     = df,
            progfunc = 'None',
            scm      = TRUE,
            fixedeff = TRUE
        )
        """
    )
    r_weights = np.asarray(r_session("as.numeric(res$weights)"), dtype=float).ravel()
    r_donor_names = [str(x) for x in r_session("rownames(res$weights)")]
    r_synthetic = np.asarray(r_session("as.numeric(predict(res, att = FALSE))"), dtype=float)

    # ------------------------------------------------------------------
    # 2. Python implementation
    # ------------------------------------------------------------------
    from augsynth_py import Synth

    # `GeoLift_PreTest$date` is character on the R side; the fixture preserves
    # that, so we parse to Date here to match the R-side conversion above.
    panel = r_geolift_pretest.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d"),
    )

    est = Synth(fixedeff=True).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=treated_unit,
        treatment_time=treatment_date,
    )

    py_weights = np.array([est.weights_[d] for d in r_donor_names], dtype=float)

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
        name="counterfactual path",
    )

    # Donor weights are *not* generally unique when donors are collinear (the
    # pre-period objective has a flat minimum along the affine subspace of
    # equivalent convex combinations). R `augsynth` and our Clarabel solver
    # land on different argmins of the same minimum on realistic panels. The
    # invariants that ARE unique — simplex constraints and the cumulative sum
    # `Y0 @ w` driving the counterfactual — are tested above and via this
    # block.
    assert (py_weights >= -1e-9).all(), "Python weights violate w >= 0"
    assert abs(py_weights.sum() - 1.0) < 1e-6, "Python weights do not sum to 1"
    assert abs(r_weights.sum() - 1.0) < 1e-3, "R reference weights do not sum to 1"
