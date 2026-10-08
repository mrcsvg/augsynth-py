"""Unit tests for augsynth_py.robustness (no R required).

Glass-box equivalence against manual filter + fit loops, result conventions
(RMSPE ratio, p-value with the treated unit in the denominator, donor
selection), error handling, parallel-vs-sequential agreement, and warning
aggregation. Numerical parity against R lives in
tests/validation_against_r/test_robustness.py.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from augsynth_py.synth.augmented import AugSynth
from augsynth_py.synth.classical import Synth

N_PERIODS = 24
T0 = 16  # first post period
UNIT, TIME, OUT = "city", "t", "y"


def make_panel(
    seed: int = 11, *, shock_unit: str | None = None, shock: float = 0.0
) -> pl.DataFrame:
    """Treated u0 = 0.7*u1 + 0.3*u2 (+ noise); donors u1..u6 at levels 10..60.

    Levels are distinct so that, with ``fixedeff=False``, the simplex weights
    are essentially unique (0.7 on u1, 0.3 on u2, ~0 elsewhere). ``shock``
    is added to ``shock_unit`` in the post-period only, planting a
    contaminated donor (the Brumadinho pattern from the NR-35 notebook).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_PERIODS)
    trend = 50.0 + 5.0 * np.sin(t / 3.0)
    series: dict[str, np.ndarray] = {}
    for u in range(1, 7):
        series[f"u{u}"] = trend + 10.0 * u + rng.normal(0.0, 0.3, N_PERIODS)
    series["u0"] = 0.7 * series["u1"] + 0.3 * series["u2"] + rng.normal(0.0, 0.3, N_PERIODS)
    if shock_unit is not None:
        series[shock_unit] = series[shock_unit] + np.where(t >= T0, shock, 0.0)
    rows = [
        {UNIT: u, TIME: int(s), OUT: float(y[s])}
        for u, y in sorted(series.items())
        for s in range(N_PERIODS)
    ]
    return pl.DataFrame(rows)


def fit_synth(panel: pl.DataFrame, treated: Any = "u0", **kw: Any) -> Synth:
    return Synth(**kw).fit(
        panel, unit=UNIT, time=TIME, outcome=OUT, treated=treated, treatment_time=T0
    )


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def shocked_panel() -> pl.DataFrame:
    return make_panel(shock_unit="u1", shock=20.0)


# ---------------------------------------------------------------------------
# leave_one_out
# ---------------------------------------------------------------------------


def test_loo_matches_manual_refits(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out

    fit = fit_synth(panel, fixedeff=False)
    res = leave_one_out(fit, donors="all")

    assert res.att.columns == [
        "dropped_unit",
        "weight",
        "att",
        "delta_att",
        "rmspe_pre",
        "lambda_",
        "error",
    ]
    assert res.att["dropped_unit"].to_list() == [f"u{i}" for i in range(1, 7)]
    assert res.att_full == fit.att_
    for row in res.att.iter_rows(named=True):
        manual = fit_synth(panel.filter(pl.col(UNIT) != row["dropped_unit"]), fixedeff=False)
        assert row["att"] == pytest.approx(manual.att_, abs=1e-9)
        assert row["delta_att"] == pytest.approx(manual.att_ - fit.att_, abs=1e-9)
        assert row["rmspe_pre"] == pytest.approx(manual.rmspe_pre_, abs=1e-12)
        assert row["weight"] == fit.weights_[row["dropped_unit"]]
        assert row["error"] is None
        assert row["lambda_"] is None  # Synth has no ridge penalty
        path = res.paths.filter(pl.col("dropped_unit") == row["dropped_unit"]).sort("period")
        np.testing.assert_allclose(path["synthetic"].to_numpy(), manual.synthetic_, atol=1e-9)
    assert res.paths.columns == ["dropped_unit", "period", "synthetic"]
    assert res.paths.height == 6 * N_PERIODS


def test_loo_names_the_contaminated_donor(shocked_panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out

    fit = fit_synth(shocked_panel, fixedeff=False)
    res = leave_one_out(fit)  # donors="nonzero" -> u1, u2

    assert set(res.att["dropped_unit"].to_list()) == {"u1", "u2"}
    assert res.most_influential == "u1"
    lo, hi = res.att_range
    assert lo <= hi  # range is ordered
    assert lo == res.att["att"].min()
    assert hi == res.att["att"].max()
    # Every refit stays negative, as does the full fit: sign is stable even
    # though the magnitude swings with the contaminated donor in or out.
    assert fit.att_ < 0
    assert res.sign_stable is True


def test_loo_donor_selection_modes(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out

    fit = fit_synth(panel, fixedeff=False)
    assert set(leave_one_out(fit).att["dropped_unit"]) == {"u1", "u2"}
    assert leave_one_out(fit, donors="all").att.height == 6
    assert leave_one_out(fit, donors=["u3", "u5"]).att["dropped_unit"].to_list() == ["u3", "u5"]
    assert leave_one_out(fit, donors=["u3", "u3"]).att.height == 1
    with pytest.raises(ValueError, match="not donors"):
        leave_one_out(fit, donors=["u3", "nope"])
    with pytest.raises(ValueError, match=r"pass a list, e\.g\. donors=\['some'\]"):
        leave_one_out(fit, donors="some")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        leave_one_out(fit, donors=[])
    with pytest.raises(ValueError, match="tol"):
        leave_one_out(fit, tol=-1.0)


def test_loo_paths_schema_when_all_refits_fail(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out

    # Two-unit panel: dropping the only donor leaves an empty pool, so the
    # single refit fails and every success-derived field must degrade cleanly.
    fit = fit_synth(panel.filter(pl.col(UNIT).is_in(["u0", "u1"])), fixedeff=False)
    res = leave_one_out(fit, donors="all", on_error="record")

    assert res.att.height == 1
    assert res.att["error"][0].startswith("EmptyDonorPoolError")
    assert res.paths.height == 0
    assert res.paths.schema["dropped_unit"] == pl.String
    assert res.paths.schema["synthetic"] == pl.Float64
    assert res.most_influential is None
    assert res.sign_stable is False
    assert all(np.isnan(v) for v in res.att_range)


def test_loo_requires_a_fitted_estimator() -> None:
    from augsynth_py.robustness import leave_one_out

    with pytest.raises(TypeError, match="fitted"):
        leave_one_out(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fitted"):
        leave_one_out(Synth())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# placebo_in_space
# ---------------------------------------------------------------------------


def _raw_rmspe(est: Synth, post: bool) -> float:
    mask = ~est.pre_mask_ if post else est.pre_mask_
    return float(np.sqrt(np.mean((est.actual_[mask] - est.synthetic_[mask]) ** 2)))


def test_placebo_matches_manual_refits_and_pvalue_convention(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = fit_synth(panel)
    res = placebo_in_space(fit)

    assert res.ratios.columns == [
        "unit",
        "is_treated",
        "rmspe_pre",
        "rmspe_post",
        "ratio",
        "rank",
        "discarded",
        "lambda_",
        "error",
    ]
    # Treated row first, then donors in fit.units_ order.
    assert res.ratios["unit"].to_list() == ["u0", "u1", "u2", "u3", "u4", "u5", "u6"]
    assert res.ratios["is_treated"].to_list() == [True] + [False] * 6
    assert res.statistic == pytest.approx(_raw_rmspe(fit, True) / _raw_rmspe(fit, False))
    assert res.ratios["ratio"][0] == res.statistic

    pool = panel.filter(pl.col(UNIT) != "u0")  # the treated unit leaves the pool
    for row in res.ratios.filter(~pl.col("is_treated")).iter_rows(named=True):
        manual = fit_synth(pool, treated=row["unit"])
        assert row["rmspe_pre"] == pytest.approx(_raw_rmspe(manual, False), abs=1e-9)
        assert row["rmspe_post"] == pytest.approx(_raw_rmspe(manual, True), abs=1e-9)
        assert row["ratio"] == pytest.approx(row["rmspe_post"] / row["rmspe_pre"])
        gap = res.gaps.filter(pl.col("unit") == row["unit"]).sort("period")["gap"].to_numpy()
        np.testing.assert_allclose(gap, manual.actual_ - manual.synthetic_, atol=1e-9)
        assert row["error"] is None and row["discarded"] is False

    ratios = res.ratios["ratio"].to_numpy()
    expected_p = np.sum(ratios >= res.statistic) / len(ratios)  # treated included
    assert res.pvalue == expected_p
    assert res.pvalue >= 1 / 7
    ranks = res.ratios["rank"].to_list()
    assert sorted(ranks) == list(range(1, 8))
    assert ranks[0] == 1 + int(np.sum(ratios > res.statistic))
    assert res.discarded == ()
    assert res.mspe_limit is None
    assert res.gaps.columns == ["unit", "period", "gap"]
    assert res.gaps.height == 7 * N_PERIODS


def test_placebo_statistic_is_extreme_for_a_real_effect(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    big = panel.with_columns(
        pl.when((pl.col(UNIT) == "u0") & (pl.col(TIME) >= T0))
        .then(pl.col(OUT) * 1.5)
        .otherwise(pl.col(OUT))
        .alias(OUT)
    )
    res = placebo_in_space(fit_synth(big))
    assert res.ratios["rank"][0] == 1
    assert res.pvalue == pytest.approx(1 / 7)


def test_placebo_mspe_limit_discards_and_recomputes(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = fit_synth(panel)
    base = placebo_in_space(fit)
    t_pre = base.ratios["rmspe_pre"][0]
    pre_mspe_ratio = (base.ratios["rmspe_pre"].to_numpy() / t_pre) ** 2
    donors_only = pre_mspe_ratio[1:]
    limit = float(np.sort(donors_only)[3])  # discards the top 3 by >= (SCtools convention)
    expected_discard = {
        u for u, r in zip(base.ratios["unit"][1:], donors_only, strict=True) if r >= limit
    }
    assert len(expected_discard) == 3

    res = placebo_in_space(fit, mspe_limit=limit)
    assert set(res.discarded) == expected_discard
    assert res.mspe_limit == limit
    disc = res.ratios.filter(pl.col("discarded"))
    assert set(disc["unit"]) == expected_discard
    assert disc["rank"].null_count() == disc.height
    kept = res.ratios.filter(~pl.col("discarded"))
    assert kept.height == 4
    kept_ratios = kept["ratio"].to_numpy()
    assert res.pvalue == np.sum(kept_ratios >= res.statistic) / 4
    assert sorted(kept["rank"].to_list()) == [1, 2, 3, 4]
    # The treated row is never discarded.
    assert res.ratios.filter(pl.col("is_treated"))["discarded"][0] is False

    with pytest.raises(ValueError, match="mspe_limit"):
        placebo_in_space(fit, mspe_limit=0.0)


def test_placebo_needs_two_donors() -> None:
    from augsynth_py.robustness import placebo_in_space

    two = make_panel().filter(pl.col(UNIT).is_in(["u0", "u1"]))
    fit = fit_synth(two)
    with pytest.raises(ValueError, match="at least two donors"):
        placebo_in_space(fit)


def _patch_rmspe(monkeypatch: pytest.MonkeyPatch, fit: Synth, overrides: dict[str, str]) -> None:
    """Make ``_rmspe`` return degenerate values for chosen units.

    ``placebo_in_space`` calls ``_rmspe(actual, synthetic, mask)`` twice per
    unit (pre mask, then post mask) with ``actual`` being that unit's own
    outcome series. The series are the columns of ``fit._y_matrix`` and the
    refit round-trips them exactly, so matching ``actual`` against the column
    of a unit identifies the unit deterministically. ``mask[0]`` tells pre
    (True) from post (False) because the first period is pre-treatment.
    ``overrides`` maps unit -> "nan" (0 pre, 0 post -> 0/0) or "inf" (0 pre,
    real post -> x/0).
    """
    import augsynth_py.robustness as rb

    real = rb._rmspe
    y = np.asarray(fit._y_matrix)
    cols = {u: y[:, i] for i, u in enumerate(fit.units_)}

    def fake(actual: Any, synthetic: Any, mask: Any) -> float:
        for u, mode in overrides.items():
            if np.allclose(actual, cols[u], atol=1e-12) and (mode == "nan" or bool(mask[0])):
                return 0.0
        return float(real(actual, synthetic, mask))

    monkeypatch.setattr(rb, "_rmspe", fake)


def test_placebo_nan_and_inf_ratios(panel: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = fit_synth(panel)
    clean = placebo_in_space(fit)
    _patch_rmspe(monkeypatch, fit, {"u3": "nan", "u5": "nan", "u4": "inf"})
    res = placebo_in_space(fit)

    by_unit = {r["unit"]: r for r in res.ratios.iter_rows(named=True)}
    for u in ("u3", "u5"):
        assert np.isnan(by_unit[u]["ratio"])
        assert by_unit[u]["rank"] is None
        assert by_unit[u]["discarded"] is False and by_unit[u]["error"] is None
    assert by_unit["u4"]["ratio"] == float("inf")
    assert by_unit["u4"]["rank"] == 1
    # Treated row and statistic untouched.
    assert res.statistic == clean.statistic
    assert by_unit["u0"]["ratio"] == clean.statistic
    # Undefined ratios leave the denominator; inf stays in it.
    kept = res.ratios.filter(pl.col("rank").is_not_null())["ratio"].to_numpy()
    assert len(kept) == 5  # u0, u1, u2, u4(inf), u6
    assert res.pvalue == np.sum(kept >= res.statistic) / len(kept)
    assert by_unit["u0"]["rank"] == 1 + int(np.sum(kept > res.statistic))


def test_placebo_nan_statistic_warns(panel: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = fit_synth(panel)
    _patch_rmspe(monkeypatch, fit, {"u0": "nan"})
    with pytest.warns(UserWarning, match="undefined"):
        res = placebo_in_space(fit)

    assert np.isnan(res.statistic)
    assert np.isnan(res.pvalue)
    assert res.ratios["rank"][0] is None
    assert res.ratios["is_treated"][0] is True
    # Placebos are still ranked among themselves.
    assert sorted(res.ratios["rank"].drop_nulls().to_list()) == list(range(1, 7))


def test_placebo_all_discarded_warns(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = fit_synth(panel)
    with pytest.warns(UserWarning, match="no placebo"):
        res = placebo_in_space(fit, mspe_limit=1e-12)

    assert len(res.discarded) == 6
    assert res.pvalue == 1.0
    assert res.ratios["rank"].to_list() == [1] + [None] * 6


def test_placebo_rank_ties_share_the_smaller_rank(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    dup = panel.filter(pl.col(UNIT) == "u1").with_columns(pl.lit("u1_dup").alias(UNIT))
    res = placebo_in_space(fit_synth(pl.concat([panel, dup])))

    by_unit = {r["unit"]: r for r in res.ratios.iter_rows(named=True)}
    a, b = by_unit["u1"], by_unit["u1_dup"]
    assert a["ratio"] == b["ratio"]
    assert a["rank"] == b["rank"]
    ranks = res.ratios["rank"].to_list()
    k = a["rank"]
    # Competition ranking: the tied pair shares k and k + 1 is skipped.
    assert sorted(ranks) == sorted([r for r in range(1, 9) if r != k + 1] + [k])


# ---------------------------------------------------------------------------
# Shared behaviour: errors, parallelism, treated groups, lambda policy, warnings
# ---------------------------------------------------------------------------


class _Flaky(Synth):
    """Synth that refuses to fit when the treated unit is ``u2``."""

    def fit(self, panel: pl.DataFrame, **kw: Any) -> Synth:  # type: ignore[override]
        if kw["treated"] == "u2":
            raise RuntimeError("boom")
        return super().fit(panel, **kw)


def test_on_error_raise_propagates_and_record_keeps_going(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    fit = _Flaky().fit(panel, unit=UNIT, time=TIME, outcome=OUT, treated="u0", treatment_time=T0)
    with pytest.raises(RuntimeError, match="boom"):
        placebo_in_space(fit)

    res = placebo_in_space(fit, on_error="record")
    bad = res.ratios.filter(pl.col("unit") == "u2").row(0, named=True)
    assert bad["error"] == "RuntimeError: boom"
    assert bad["ratio"] is None and bad["rank"] is None and bad["discarded"] is False
    assert res.ratios.filter(pl.col("error").is_null()).height == 6
    # Errored placebos leave the denominator.
    kept = res.ratios.filter(pl.col("error").is_null())["ratio"].to_numpy()
    assert res.pvalue == np.sum(kept >= res.statistic) / 6
    assert "u2" not in set(res.gaps["unit"])
    # The error row must come back from a loky worker exactly as it does inline.
    par = placebo_in_space(fit, on_error="record", n_jobs=2)
    assert_frame_equal(res.ratios, par.ratios, check_exact=True)

    with pytest.raises(ValueError, match="on_error"):
        placebo_in_space(fit, on_error="ignore")  # type: ignore[arg-type]


def test_loo_on_error_raise_on_empty_pool(panel: pl.DataFrame) -> None:
    from augsynth_py.exceptions import EmptyDonorPoolError
    from augsynth_py.robustness import leave_one_out

    # LOO always refits the real treated unit, so a refit can only fail on the
    # pool: with a single donor, dropping it empties the pool. The "record"
    # side of this case is covered by test_loo_paths_schema_when_all_refits_fail.
    two = panel.filter(pl.col(UNIT).is_in(["u0", "u1"]))
    fit = fit_synth(two)
    with pytest.raises(EmptyDonorPoolError):
        leave_one_out(fit, donors="all")


def test_parallel_matches_sequential(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out, placebo_in_space

    fit = fit_synth(panel)
    seq, par = placebo_in_space(fit), placebo_in_space(fit, n_jobs=2)
    assert_frame_equal(seq.ratios, par.ratios, check_exact=True)
    assert_frame_equal(seq.gaps, par.gaps, check_exact=True)
    assert seq.pvalue == par.pvalue
    lseq, lpar = leave_one_out(fit, donors="all"), leave_one_out(fit, donors="all", n_jobs=2)
    assert_frame_equal(lseq.att, lpar.att, check_exact=True)
    assert_frame_equal(lseq.paths, lpar.paths, check_exact=True)


def test_prototype_is_never_refit_in_place(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out, placebo_in_space

    fit = fit_synth(panel)
    before = (fit.att_, dict(fit.weights_), fit.synthetic_.copy())
    placebo_in_space(fit)
    leave_one_out(fit, donors="all")
    assert fit.att_ == before[0]
    assert fit.weights_ == before[1]
    np.testing.assert_array_equal(fit.synthetic_, before[2])

    # A CV prototype re-runs the CV in every refit; none of it may leak back.
    # The default grid lands on its boundary for this panel, so the fit and the
    # refits warn; those warnings are not what this test is about.
    with pytest.warns(UserWarning):
        cv = AugSynth().fit(
            panel, unit=UNIT, time=TIME, outcome=OUT, treated="u0", treatment_time=T0
        )
    assert cv.lambda_cv_path_ is not None
    snapshot = (cv.att_, dict(cv.weights_), cv.lambda_, cv.lambda_cv_path_.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        placebo_in_space(cv)
        leave_one_out(cv, donors="all")
    assert cv.att_ == snapshot[0]
    assert cv.weights_ == snapshot[1]
    assert cv.lambda_ == snapshot[2]
    np.testing.assert_array_equal(cv.lambda_cv_path_, snapshot[3])


def test_treated_group_is_collapsed_and_excluded(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out, placebo_in_space

    fit = fit_synth(panel, treated=["u0", "u1"])
    ps = placebo_in_space(fit)
    assert ps.ratios["unit"].to_list() == ["u0,u1", "u2", "u3", "u4", "u5", "u6"]
    assert ps.ratios.height == 6
    loo = leave_one_out(fit, donors="all")
    assert loo.att["dropped_unit"].to_list() == ["u2", "u3", "u4", "u5", "u6"]
    # Each LOO refit is the group fit on the panel without that donor.
    manual = fit_synth(panel.filter(pl.col(UNIT) != "u3"), treated=["u0", "u1"])
    assert loo.att.filter(pl.col("dropped_unit") == "u3")["att"][0] == pytest.approx(
        manual.att_, abs=1e-9
    )


def test_augsynth_lambda_policy_follows_the_prototype(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import leave_one_out, placebo_in_space

    frozen = AugSynth(lambda_=0.5).fit(
        panel, unit=UNIT, time=TIME, outcome=OUT, treated="u0", treatment_time=T0
    )
    loo = leave_one_out(frozen, donors="all")
    assert loo.att["lambda_"].to_list() == [0.5] * 6
    ps = placebo_in_space(frozen)
    assert ps.ratios["lambda_"].to_list() == [0.5] * 7
    assert loo.lambda_at_boundary == () and ps.lambda_at_boundary == ()

    grid = np.array([1e-3, 1e3])
    with pytest.warns(UserWarning):
        cv = AugSynth(lambda_grid=grid).fit(
            panel, unit=UNIT, time=TIME, outcome=OUT, treated="u0", treatment_time=T0
        )
    # A two-point grid makes every CV choice a boundary choice.
    with pytest.warns(UserWarning) as rec:
        loo_cv = leave_one_out(cv, donors="all")
    summary = [w for w in rec if "refits selected a lambda at the boundary" in str(w.message)]
    assert len(summary) == 1
    assert not any(str(w.message).startswith("CV-selected lambda") for w in rec)
    assert set(loo_cv.lambda_at_boundary) == {f"u{i}" for i in range(1, 7)}
    # The boundary warning from every refit is the evidence that CV re-ran in
    # each one: on this grid the prototype's lambda equals the refits' lambda,
    # so the values alone could not tell a re-run from a copy. Only check that
    # every refit reported one.
    assert loo_cv.att["lambda_"].null_count() == 0

    # Warnings raised inside loky workers must reach the caller the same way.
    with pytest.warns(UserWarning) as rec_par:
        loo_par = leave_one_out(cv, donors="all", n_jobs=2)
    summary_par = [
        w for w in rec_par if "refits selected a lambda at the boundary" in str(w.message)
    ]
    assert len(summary_par) == 1
    assert set(loo_par.lambda_at_boundary) == set(loo_cv.lambda_at_boundary)


def test_result_types_are_frozen(panel: pl.DataFrame) -> None:
    from dataclasses import FrozenInstanceError

    from augsynth_py.robustness import leave_one_out, placebo_in_space

    fit = fit_synth(panel)
    ps = placebo_in_space(fit)
    loo = leave_one_out(fit)
    with pytest.raises(FrozenInstanceError):
        ps.pvalue = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loo.att_full = 0.0  # type: ignore[misc]


class _Noisy(Synth):
    """Synth that emits a custom diagnostic warning on every fit."""

    def fit(self, panel: pl.DataFrame, **kw: Any) -> Synth:  # type: ignore[override]
        warnings.warn("custom diagnostic", UserWarning, stacklevel=2)
        return super().fit(panel, **kw)


def test_non_boundary_warnings_are_reemitted_once_with_labels(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    with pytest.warns(UserWarning, match="custom diagnostic"):
        fit = _Noisy().fit(
            panel, unit=UNIT, time=TIME, outcome=OUT, treated="u0", treatment_time=T0
        )
    with pytest.warns(UserWarning) as rec:
        placebo_in_space(fit)
    # Six refits each warned once; the caller sees a single aggregated warning.
    assert len(rec) == 1
    message = str(rec[0].message)
    assert "custom diagnostic" in message
    assert "raised in 6 refit(s)" in message
    assert all(f"u{i}" in message for i in range(1, 7))


def test_treated_group_gaps_use_the_collapsed_series(panel: pl.DataFrame) -> None:
    from augsynth_py.robustness import placebo_in_space

    group = ["u0", "u1"]
    fit = fit_synth(panel, treated=group)
    res = placebo_in_space(fit)

    treated_gap = res.gaps.filter(pl.col("unit") == "u0,u1").sort("period")["gap"].to_numpy()
    np.testing.assert_array_equal(treated_gap, fit.actual_ - fit.synthetic_)
    # Placebos refit on the pool with the whole treated group removed.
    pool = panel.filter(~pl.col(UNIT).is_in(group))
    for d in ["u2", "u3", "u4", "u5", "u6"]:
        manual = fit_synth(pool, treated=d)
        gap = res.gaps.filter(pl.col("unit") == d).sort("period")["gap"].to_numpy()
        np.testing.assert_allclose(gap, manual.actual_ - manual.synthetic_, atol=1e-9)


def test_robustness_names_are_package_exports() -> None:
    import augsynth_py

    for name in ("placebo_in_space", "leave_one_out", "PlaceboInSpaceResult", "LeaveOneOutResult"):
        assert name in augsynth_py.__all__
        assert hasattr(augsynth_py, name)


def test_result_types_compare_by_identity(panel: pl.DataFrame) -> None:
    """Frozen dataclasses holding DataFrames: ``eq=False`` like ConformalTestResult.

    With the default ``eq=True`` the generated ``__eq__`` compares the frame
    fields, which raises TypeError (polars refuses an ambiguous truth value),
    and the frozen+eq combination also makes instances unhashable in practice.
    """
    from augsynth_py.robustness import leave_one_out, placebo_in_space

    fit = fit_synth(panel, fixedeff=False)
    ps_a, ps_b = placebo_in_space(fit), placebo_in_space(fit)
    loo_a, loo_b = leave_one_out(fit), leave_one_out(fit)

    for a, b in ((ps_a, ps_b), (loo_a, loo_b)):
        assert a == a
        assert a != b  # same inputs, different objects: identity, not value
        assert hash(a) != hash(b) or a is b
        assert len({a, b}) == 2
