"""Parity: multi-treated Synth vs augsynth with several treat==1 units.

RISK NOTE: this test pins the assumption that R augsynth, given a block
treatment (single t_int) with multiple treated units, fits the pooled mean of
the treated series. If augsynth instead fits each treated unit separately and
averages results, this test will fail and the collapse semantics must be
revisited. The test IS the spec for that assumption.
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
    treated_units = ["chicago", "portland"]
    treatment_date = date(2021, 2, 15)

    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(
        """
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location %in% c('chicago', 'portland') & df$date >= as.Date('2021-02-15')
        )
        res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'None', scm = TRUE, fixedeff = TRUE
        )
        """
    )
    r_synthetic = np.asarray(r_session("as.numeric(predict(res, att = FALSE))"), dtype=float)

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

    from tests.validation_against_r.conftest import assert_array_close

    assert_array_close(
        est.synthetic_,
        r_synthetic,
        atol=1e-4,
        rtol=1e-4,
        name="multi-treated counterfactual path",
    )
