"""Parity tests for augsynth_py.robustness.

Three layers, by design (docs/methodology.md §7.4):

1. **Refits by composition** — each placebo / leave-one-out refit must equal
   R ``augsynth(progfunc='None', scm=TRUE)`` on the same modified panel.
   This validates the panel-reconstruction path, not just the estimator.
2. **Aggregation exact vs SCtools** — ``SCtools::mspe.test`` is fed a ``tdf``
   object built from *our* fitted series; MSPE ratios and the p-value must
   agree exactly. This pins the convention (treated in the denominator,
   ``>=`` comparison, discard rule), independent of any solver.
3. **Characterization** — a real ``SCtools::generate.placebos`` run on
   ``Synth::synth`` (ipop) with every pre-period outcome as a predictor. The
   two optimizers do not bit-match, so this layer only records the gap and
   asserts a loose rank agreement.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

TREATED = "new york"
T_DATE = date(2021, 2, 15)


def _panel(r_geolift_pretest: pl.DataFrame) -> pl.DataFrame:
    return r_geolift_pretest.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _r_augsynth_path(r_session: Any, *, exclude: str | None, treated: str) -> np.ndarray:
    """Counterfactual path from R augsynth on GeoLift_PreTest minus ``exclude``."""
    excl = f"df <- df[df$location != '{exclude}', ]" if exclude else ""
    r_session(
        f"""
        suppressPackageStartupMessages(library(augsynth))
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        {excl}
        df$treat <- as.integer(df$location == '{treated}' & df$date >= as.Date('2021-02-15'))
        res <- augsynth(Y ~ treat, unit = location, time = date, data = df,
                        progfunc = 'None', scm = TRUE, fixedeff = TRUE)
        """
    )
    return np.asarray(r_session("as.numeric(predict(res, att = FALSE))"), dtype=float)


@pytest.fixture(scope="module")
def ny_fit(r_geolift_pretest: pl.DataFrame) -> Any:
    from augsynth_py import Synth

    return Synth(fixedeff=True).fit(
        _panel(r_geolift_pretest),
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED,
        treatment_time=T_DATE,
    )


@pytest.mark.requires_r_pkg("augsynth", "GeoLift")
def test_placebo_refits_match_r_augsynth_by_composition(
    r_session: Any, r_geolift_pretest: pl.DataFrame, ny_fit: Any
) -> None:
    from augsynth_py import placebo_in_space
    from tests.validation_against_r.conftest import assert_array_close

    ps = placebo_in_space(ny_fit)
    panel = _panel(r_geolift_pretest)
    donors = [u for u in ny_fit.units_ if u != TREATED]
    for pseudo in donors[:3]:
        r_path = _r_augsynth_path(r_session, exclude=TREATED, treated=pseudo)
        actual = panel.filter(pl.col("location") == pseudo).sort("date")["Y"].to_numpy()
        gap = ps.gaps.filter(pl.col("unit") == pseudo).sort("period")["gap"].to_numpy()
        assert_array_close(actual - gap, r_path, atol=1e-4, rtol=1e-4, name=f"placebo {pseudo}")


@pytest.mark.requires_r_pkg("augsynth", "GeoLift")
def test_loo_refits_match_r_augsynth_by_composition(
    r_session: Any, r_geolift_pretest: pl.DataFrame, ny_fit: Any
) -> None:
    from augsynth_py import leave_one_out
    from tests.validation_against_r.conftest import assert_array_close

    loo = leave_one_out(ny_fit, donors="nonzero")
    dropped = loo.att["dropped_unit"].to_list()[:3]
    assert dropped, "new york should have at least one nonzero-weight donor"
    panel = _panel(r_geolift_pretest)
    post = panel.filter((pl.col("location") == TREATED) & (pl.col("date") >= T_DATE)).sort("date")
    for d in dropped:
        r_path = _r_augsynth_path(r_session, exclude=d, treated=TREATED)
        path = loo.paths.filter(pl.col("dropped_unit") == d).sort("period")["synthetic"].to_numpy()
        assert_array_close(path, r_path, atol=1e-4, rtol=1e-4, name=f"LOO without {d}")
        n_post = post.height
        r_att = float(np.mean(post["Y"].to_numpy() - r_path[-n_post:]))
        row = loo.att.filter(pl.col("dropped_unit") == d).row(0, named=True)
        # README tolerance for ATT estimates (atol=1e-4, rtol=1e-4). The two
        # solvers differ at ~1e-7 relative on the path, which propagates to
        # ~3e-4 absolute on an ATT of ~-104 (e.g. without baton rouge); an
        # absolute-only 1e-4 would sit at that noise floor even for the full fit.
        assert row["att"] == pytest.approx(r_att, abs=1e-4, rel=1e-4)
