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
        # values in the thousands), which propagates to 2.6e-4 absolute on an
        # ATT of ~-104 (e.g. without baton rouge); an absolute-only 1e-4 would
        # sit at that noise floor even for the full fit (plan decision #12).
        assert row["att"] == pytest.approx(r_att, abs=1e-4, rel=1e-4)


def _build_tdf_in_r(r_session: Any, fit: Any, ps: Any) -> list[str]:
    """Assemble an SCtools ``tdf`` object in R from our fitted series.

    Period index 0..T-1 plays the role of ``year``; ``t1`` is the first post
    index. ``mspe.placs`` and ``loss.v`` are our own pre-period MSPEs, so the
    only thing SCtools computes is the post/pre ratio and the rank p-value;
    R nevertheless recomputes both denominators from the pre rows and checks
    them against ours (the ``stopifnot`` block), so they are oracle-verified
    rather than merely supplied.
    ``SCtools:::is_tdf`` is ``inherits(x, "tdf")`` (0.3.3.1), so a classed
    list with the fields ``mspe.test`` reads is enough. Returns the placebo
    units in ``tdf`` column order.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector, StrVector

    panel_actual = {u: fit._y_matrix[:, i] for i, u in enumerate(fit.units_)}
    successful = ps.ratios.filter(~pl.col("is_treated") & pl.col("error").is_null())
    placebos: list[str] = [str(u) for u in successful["unit"].to_list()]
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
        # R recomputes the pre-MSPE denominators from the series: if our
        # pre-window definition drifted, supplying them ourselves would move
        # both sides of the comparison together and hide it.
        pre <- subset(tdf$df, year < tdf$t1 & year >= tdf$t0)
        stopifnot(isTRUE(all.equal(
          sapply(seq_len(tdf$n), function(i) cvTools::mspe(pre[, i], pre[, i + tdf$n])),
          tdf$mspe.placs$mspe, tolerance = 1e-12)))
        stopifnot(isTRUE(all.equal(cvTools::mspe(pre$Y1, pre$synthetic.Y1), tdf$loss.v,
                                   tolerance = 1e-12)))
        """
    )
    assert [str(c) for c in r_session("names(tdf$df)")] == list(cols)
    return placebos


@pytest.mark.requires_r_pkg("SCtools", "GeoLift")
@pytest.mark.parametrize("mspe_limit", [None, 5.0, 2.0])
def test_placebo_aggregation_matches_sctools_mspe_test_exactly(
    r_session: Any, ny_fit: Any, mspe_limit: float | None
) -> None:
    from augsynth_py import placebo_in_space

    ps = placebo_in_space(ny_fit, mspe_limit=mspe_limit)
    # The builder keeps every successful placebo regardless of `discarded`, so
    # the tdf always holds all of them; SCtools applies the discard rule itself.
    placebos = _build_tdf_in_r(r_session, ny_fit, ps)
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
    assert r_units[-1] == TREATED  # SCtools appends the treated row last
    assert set(r_units) == set(ours)  # same kept set
    np.testing.assert_allclose([ours[u] for u in r_units], r_ratios, rtol=1e-12, atol=0)
    assert ps.pvalue == r_p
    assert set(ps.discarded) == set(placebos) - set(r_units)


@pytest.mark.slow
@pytest.mark.requires_r_pkg("SCtools", "Synth", "GeoLift")
def test_characterize_against_real_sctools_placebos(
    r_session: Any, r_geolift_pretest: pl.DataFrame, ny_fit: Any
) -> None:
    """Layer 3: not a parity assertion. R Synth optimises V and W with ipop
    over 45 single-period 'special predictors'; we solve the simplex QP with
    Clarabel. Ratios differ at the solver level; ranks mostly agree. The
    numbers this prints are recorded in docs/methodology.md §7.4.

    The variant characterized is ``Synth(fixedeff=False)``, fitted here for
    this test only: R ``Synth`` has no unit demeaning, and against the
    ``fixedeff=True`` fixture the ranks agree only at rho ~ 0.76 (printed as
    well, for the record). Two harness details, both explained inline in
    the R code: ``controls.identifier`` must be sorted or SCtools labels
    every placebo with another unit's name, and ``Sigf.ipop = 4`` because
    ipop at Synth's default 5 diverges on one placebo. Expect 80-120 s of
    wall time: 39 placebo fits plus the treated fit, each with nested V
    optimisation.
    """
    from scipy.stats import spearmanr

    from augsynth_py import Synth, placebo_in_space

    r_session(
        """
        suppressPackageStartupMessages({library(Synth); library(SCtools)})
        df <- as.data.frame(GeoLift_PreTest)
        df$date <- as.Date(df$date)
        df$t <- as.integer(df$date - min(df$date)) + 1L     # 1..90
        df$id <- as.integer(factor(df$location))
        tr <- unique(df$id[df$location == 'new york'])
        # Sorted on purpose: dataprep returns X0 / Z0 / Y0plot in sorted-id
        # order whatever order it was given, but names.and.numbers (and
        # colnames(X0)) keep the given order, and generate.placebos indexes
        # the matrices by position while labelling placebo i from
        # names.and.numbers[i]. With an unsorted vector every placebo carries
        # another unit's name (the ratios themselves are right). Sorting makes
        # the two orders coincide.
        ctrl <- sort(setdiff(unique(df$id), tr))
        pre_idx <- 1:45
        sp <- lapply(pre_idx, function(k) list('Y', k, 'mean'))
        dp <- dataprep(foo = df, predictors = NULL, special.predictors = sp,
                       dependent = 'Y', unit.variable = 'id', time.variable = 't',
                       treatment.identifier = tr, controls.identifier = ctrl,
                       time.predictors.prior = pre_idx, time.optimize.ssr = pre_idx,
                       unit.names.variable = 'location', time.plot = 1:90)
        # Sigf.ipop = 4, not Synth's default 5: with 45 single-period
        # predictors on 38 donors the inner QP's H = X0' V X0 has rcond ~5e-8
        # and kernlab::ipop at 5 significant digits diverges on the nashville
        # placebo ("infinite or missing values in 'x'" from its iterate-norm
        # svd); 3 and 4 converge, 5-7 do not. The same precision is used for
        # the treated fit so the two sides of the ratio are comparable.
        so <- synth(dp, Sigf.ipop = 4)
        # suppressMessages: tibble name-repair chatter (messages, not warnings).
        tdf_real <- suppressMessages(generate.placebos(dp, so, Sigf.ipop = 4))
        res_real <- mspe.test(tdf_real)
        """
    )
    r_test = {
        str(u): float(v)
        for u, v in zip(
            r_session("as.character(res_real$test$unit)"),
            r_session("as.numeric(res_real$test$MSPE.ratios)"),
            strict=True,
        )
    }
    r_p = float(r_session("res_real$p.val")[0])
    r_session("rm(df, dp, so, tdf_real, res_real, sp, ctrl, tr, pre_idx)")

    def _ratios(fit: Any) -> tuple[dict[str, float], Any]:
        ps = placebo_in_space(fit)
        return {row["unit"]: row["ratio"] ** 2 for row in ps.ratios.iter_rows(named=True)}, ps

    def _rho(ours: dict[str, float]) -> float:
        common = sorted(set(ours) & set(r_test))
        assert len(common) == len(ours), "SCtools and we must rank the same units"
        return float(spearmanr([ours[u] for u in common], [r_test[u] for u in common]).statistic)

    nofe_fit = Synth(fixedeff=False).fit(
        _panel(r_geolift_pretest),
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED,
        treatment_time=T_DATE,
    )
    ours, ps = _ratios(nofe_fit)
    rho = _rho(ours)
    rel_gap = float(np.median([abs(ours[u] - r_test[u]) / r_test[u] for u in ours]))
    rho_fe = _rho(_ratios(ny_fit)[0])
    print(
        f"\n[layer 3] spearman rho={rho:.4f}  median |rel gap| in MSPE ratio={rel_gap:.4g}  "
        f"p(ours)={ps.pvalue:.4f}  p(SCtools/ipop)={r_p:.4f}  "
        f"[rho with fixedeff=True: {rho_fe:.4f}]"
    )
    assert rho > 0.9  # loose by design; see docstring
