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


def _r_augsynth_path(r_session: Any, *, exclude: str, treated: str) -> np.ndarray:
    """Counterfactual path from R augsynth on GeoLift_PreTest minus ``exclude``."""
    r_session(
        f"""
        suppressPackageStartupMessages(library(augsynth))
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df <- df[df$location != '{exclude}', ]
        df$treat <- as.integer(df$location == '{treated}'
                               & df$date >= as.Date('{T_DATE.isoformat()}'))
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
    n_post = post.height
    for d in dropped:
        r_path = _r_augsynth_path(r_session, exclude=d, treated=TREATED)
        path = loo.paths.filter(pl.col("dropped_unit") == d).sort("period")["synthetic"].to_numpy()
        assert_array_close(path, r_path, atol=1e-4, rtol=1e-4, name=f"LOO without {d}")
        r_att = float(np.mean(post["Y"].to_numpy() - r_path[-n_post:]))
        row = loo.att.filter(pl.col("dropped_unit") == d).row(0, named=True)
        # README tolerance for ATT estimates (atol=1e-4, rtol=1e-4). The two
        # solvers differ at about 1e-6 relative on the path (3.3e-3 absolute on
        # values in the thousands), which propagates to ~3e-4 absolute on an
        # ATT of ~-104 (e.g. without baton rouge); an absolute-only 1e-4 would
        # sit at that noise floor even for the full fit (plan decision #12).
        assert row["att"] == pytest.approx(r_att, abs=1e-4, rel=1e-4)


def _build_tdf_in_r(r_session: Any, fit: Any, ps: Any) -> None:
    """Assemble an SCtools ``tdf`` object in R from our fitted series.

    Period index 0..T-1 plays the role of ``year``; ``t1`` is the first post
    index. ``mspe.placs`` and ``loss.v`` are our own pre-period MSPEs, so the
    only thing SCtools computes is the post/pre ratio and the rank p-value.
    ``SCtools:::is_tdf`` is ``inherits(x, "tdf")`` (0.3.3.1), so a classed
    list with the fields ``mspe.test`` reads is enough.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector, StrVector

    panel_actual = {u: fit._y_matrix[:, i] for i, u in enumerate(fit.units_)}
    successful = ps.ratios.filter(~pl.col("is_treated") & pl.col("error").is_null())
    placebos = successful["unit"].to_list()
    n = len(placebos)
    pre_mask = np.asarray(fit.pre_mask_, dtype=bool)
    n_pre = int(pre_mask.sum())
    n_periods = len(fit.periods_)
    # `year >= t1` selects the post rows, which assumes the pre-period is a prefix.
    assert pre_mask[:n_pre].all() and not pre_mask[n_pre:].any()

    # Column order is what mspe.test indexes by: n placebo synthetic series,
    # n placebo actual series, Y1, synthetic.Y1, year.
    cols: dict[str, np.ndarray] = {}
    for u in placebos:
        gap = ps.gaps.filter(pl.col("unit") == u).sort("period")["gap"].to_numpy()
        cols[f"synthetic.{u}"] = panel_actual[u] - gap
    for u in placebos:
        cols[u] = panel_actual[u]
    cols["Y1"] = fit.actual_
    cols["synthetic.Y1"] = fit.synthetic_
    cols["year"] = np.arange(n_periods, dtype=float)

    ro.globalenv["tdf_cols"] = ro.ListVector(
        {k: FloatVector(np.asarray(v, dtype=float)) for k, v in cols.items()}
    )
    ro.globalenv["tdf_mspe_placs"] = FloatVector(successful["rmspe_pre"].to_numpy() ** 2)
    ro.globalenv["tdf_loss_v"] = FloatVector([float(ps.ratios["rmspe_pre"][0]) ** 2])
    ro.globalenv["tdf_units"] = StrVector(placebos)
    r_session(
        f"""
        suppressPackageStartupMessages(library(SCtools))
        tdf <- list(
          df = as.data.frame(tdf_cols, check.names = FALSE),
          mspe.placs = data.frame(mspe = tdf_mspe_placs),
          t0 = 0, t1 = {n_pre}, tr = 1,
          names.and.numbers = data.frame(unit.numbers = seq_len({n}), unit.names = tdf_units),
          n = {n}, treated.name = '{TREATED}', loss.v = tdf_loss_v
        )
        class(tdf) <- 'tdf'
        """
    )
    assert [str(c) for c in r_session("names(tdf$df)")] == list(cols)


@pytest.mark.requires_r_pkg("SCtools", "GeoLift")
@pytest.mark.parametrize("mspe_limit", [None, 5.0, 2.0])
def test_placebo_aggregation_matches_sctools_mspe_test_exactly(
    r_session: Any, ny_fit: Any, mspe_limit: float | None
) -> None:
    from augsynth_py import placebo_in_space

    ps = placebo_in_space(ny_fit, mspe_limit=mspe_limit)
    # The builder keeps every successful placebo regardless of `discarded`, so
    # the tdf always holds all of them; SCtools applies the discard rule itself.
    _build_tdf_in_r(r_session, ny_fit, ps)
    if mspe_limit is None:
        r_session("res <- mspe.test(tdf)")
    else:
        # When no placebo clears the limit SCtools warns ("No placebos have a
        # pre-MSPE above mspe.limit") and falls back to no discard; our result
        # also discards nothing then, so the assertions still hold. rpy2
        # re-emits R warnings as RRuntimeWarning, which pyproject's
        # `filterwarnings = error` would turn into a failure, hence the wrap.
        # None of the limits above trigger it on GeoLift_PreTest (5.0 drops 6
        # placebos, 2.0 drops 9); the wrap is defensive.
        r_session(
            f"res <- suppressWarnings(mspe.test(tdf, discard.extreme = TRUE, mspe.limit = {mspe_limit}))"
        )
    r_ratios = np.asarray(r_session("as.numeric(res$test$MSPE.ratios)"), dtype=float)
    r_units = [str(u) for u in r_session("as.character(res$test$unit)")]
    r_p = float(r_session("res$p.val")[0])

    kept = ps.ratios.filter(~pl.col("discarded") & pl.col("error").is_null())
    ours = {
        row["unit"]: row["ratio"] ** 2 for row in kept.iter_rows(named=True)
    }  # RMSPE² = MSPE ratio
    assert r_units == [u for u in r_units if u in ours]  # same kept set, treated last in R
    assert set(r_units) == set(ours)
    np.testing.assert_allclose([ours[u] for u in r_units], r_ratios, rtol=1e-12, atol=0)
    assert ps.pvalue == r_p
    assert set(ps.discarded) == set(ps.ratios["unit"]) - set(ours) - {TREATED}
