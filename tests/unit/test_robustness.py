"""Unit tests for augsynth_py.robustness (no R required).

Glass-box equivalence against manual filter + fit loops, result conventions
(RMSPE ratio, p-value with the treated unit in the denominator, donor
selection), error handling, parallel-vs-sequential agreement, and warning
aggregation. Numerical parity against R lives in
tests/validation_against_r/test_robustness.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

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
    with pytest.raises(ValueError, match="donors"):
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
