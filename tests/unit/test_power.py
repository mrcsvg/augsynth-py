"""Unit tests for augsynth_py.power (no R required).

Covers the window arithmetic, effect injection, the glass-box equivalence
between simulate_power and a direct fit + conformal_pvalue at effect 0, the
aggregation logic (power_curve / mde), error handling, reproducibility, and
parallel-vs-sequential agreement. Numerical parity against R's GeoLiftPower
lives in tests/validation_against_r/test_power.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import conformal_pvalue
from augsynth_py.power import (
    PowerResults,
    _inject_effect,
    _windows,
    simulate_power,
)
from augsynth_py.synth.augmented import AugSynth
from augsynth_py.synth.classical import Synth

N_UNITS = 6
N_PERIODS = 24


def make_panel(seed: int = 3) -> pl.DataFrame:
    """Balanced panel with a shared trend, unit levels, and small noise.

    The treated unit (u0) is well approximated by the donors, so a large
    injected lift is cleanly detectable and the no-effect p-value is large.
    """
    rng = np.random.default_rng(seed)
    trend = 50.0 + 5.0 * np.sin(np.arange(N_PERIODS) / 3.0)
    rows = []
    for u in range(N_UNITS):
        level = 10.0 * u
        y = trend + level + rng.normal(0.0, 0.3, N_PERIODS)
        rows.extend({"city": f"u{u}", "t": int(s), "y": float(y[s])} for s in range(N_PERIODS))
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return make_panel()


# ---------------------------------------------------------------------------
# Window arithmetic and truncation
# ---------------------------------------------------------------------------


def test_windows_slide_back_from_the_end() -> None:
    periods = list(range(1, 11))  # 1..10
    windows = _windows(periods, duration=2, lookback=3)
    assert [(w.idx, w.start, w.end) for w in windows] == [(0, 9, 10), (1, 8, 9), (2, 7, 8)]
    assert [len(w.kept_periods) for w in windows] == [10, 9, 8]
    assert [w.n_pre for w in windows] == [8, 7, 6]
    assert windows[2].window_periods == [7, 8]
    assert windows[2].kept_periods == list(range(1, 9))


def test_grid_shape_and_schema(panel: pl.DataFrame) -> None:
    res = simulate_power(
        panel,
        estimator=Synth(),
        unit="city",
        time="t",
        outcome="y",
        treated="u0",
        durations=[2, 3],
        effect_sizes=[0.0, 0.4],
        lookback_window=2,
    )
    sims = res.simulations
    assert sims.height == 2 * 2 * 2
    assert sims.columns == [
        "duration",
        "effect_size",
        "window",
        "treatment_start",
        "treatment_end",
        "n_pre_periods",
        "pvalue",
        "att",
        "att_pct",
        "rmspe_pre",
        "error",
    ]
    # Duration 3, window 1 on a 0..23 axis: treated periods 20..22, 20 pre.
    row = sims.filter((pl.col("duration") == 3) & (pl.col("window") == 1)).row(0, named=True)
    assert (row["treatment_start"], row["treatment_end"], row["n_pre_periods"]) == (20, 22, 20)
    assert sims.get_column("error").null_count() == sims.height


# ---------------------------------------------------------------------------
# Effect injection
# ---------------------------------------------------------------------------


def test_inject_multiplicative_touches_only_treated_window(panel: pl.DataFrame) -> None:
    out = _inject_effect(
        panel,
        unit="city",
        time="t",
        outcome="y",
        treated=["u0", "u1"],
        window_periods=[22, 23],
        effect=0.25,
        effect_type="multiplicative",
    )
    joined = panel.join(out, on=["city", "t"], suffix="_new")
    in_cell = pl.col("city").is_in(["u0", "u1"]) & pl.col("t").is_in([22, 23])
    changed = joined.filter(in_cell)
    untouched = joined.filter(~in_cell)
    assert np.allclose(changed["y_new"].to_numpy(), changed["y"].to_numpy() * 1.25)
    assert (untouched["y_new"] == untouched["y"]).all()


def test_inject_additive(panel: pl.DataFrame) -> None:
    out = _inject_effect(
        panel,
        unit="city",
        time="t",
        outcome="y",
        treated=["u0"],
        window_periods=[23],
        effect=-2.0,
        effect_type="additive",
    )
    before = panel.filter((pl.col("city") == "u0") & (pl.col("t") == 23))["y"][0]
    after = out.filter((pl.col("city") == "u0") & (pl.col("t") == 23))["y"][0]
    assert after == pytest.approx(before - 2.0)


# ---------------------------------------------------------------------------
# Glass-box equivalence with a direct fit at effect 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimator", "treated"),
    [
        (Synth(), "u0"),
        (Synth(), ["u0", "u1"]),
        (AugSynth(lambda_=1.0), "u0"),  # fixed lambda: no CV, keeps the test fast
    ],
)
def test_zero_effect_matches_direct_conformal(
    panel: pl.DataFrame, estimator: Any, treated: Any
) -> None:
    duration = 4
    res = simulate_power(
        panel,
        estimator=estimator,
        unit="city",
        time="t",
        outcome="y",
        treated=treated,
        durations=duration,
        effect_sizes=[0.0],
        lookback_window=1,
    )
    row = res.simulations.row(0, named=True)

    # Same fit and test, assembled by hand: lookback 1 keeps the full panel
    # and treats the last `duration` periods.
    direct = type(estimator)(**_init_kwargs(estimator)).fit(
        panel,
        unit="city",
        time="t",
        outcome="y",
        treated=treated,
        treatment_time=N_PERIODS - duration,
    )
    assert row["pvalue"] == conformal_pvalue(direct)
    assert row["att"] == pytest.approx(direct.att_, abs=1e-12)
    assert row["att_pct"] == pytest.approx(direct.att_pct_, abs=1e-12)
    assert row["rmspe_pre"] == pytest.approx(direct.rmspe_pre_, abs=1e-12)


def _init_kwargs(estimator: Any) -> dict[str, Any]:
    if isinstance(estimator, AugSynth):
        return {
            "fixedeff": estimator.fixedeff,
            "lambda_": estimator._lambda_init,
            "lambda_grid": estimator._lambda_grid_init,
        }
    return {"fixedeff": estimator.fixedeff}


# ---------------------------------------------------------------------------
# Detection behaviour
# ---------------------------------------------------------------------------


def test_large_effect_detected_null_effect_not(panel: pl.DataFrame) -> None:
    res = simulate_power(
        panel,
        estimator=Synth(),
        unit="city",
        time="t",
        outcome="y",
        treated="u0",
        durations=3,
        effect_sizes=[0.0, 0.5],
        lookback_window=2,
    )
    curve = res.power_curve()
    by_effect = {row["effect_size"]: row["power"] for row in curve.iter_rows(named=True)}
    # Deterministic given the fixed panel seed and the block scheme.
    assert by_effect[0.5] == 1.0
    assert by_effect[0.0] < 1.0
    p0 = res.simulations.filter(pl.col("effect_size") == 0.0)["pvalue"].min()
    p_big = res.simulations.filter(pl.col("effect_size") == 0.5)["pvalue"].max()
    assert p_big < p0


def test_prototype_is_never_fitted_in_place(panel: pl.DataFrame) -> None:
    proto = Synth()
    simulate_power(
        panel,
        estimator=proto,
        unit="city",
        time="t",
        outcome="y",
        treated="u0",
        durations=3,
        effect_sizes=[0.2],
    )
    assert not hasattr(proto, "att_")


# ---------------------------------------------------------------------------
# Aggregation: power_curve and mde
# ---------------------------------------------------------------------------


def _results_from_rows(rows: list[dict[str, Any]], alpha: float = 0.1) -> PowerResults:
    sims = pl.DataFrame(
        rows,
        schema_overrides={
            "duration": pl.Int64,
            "effect_size": pl.Float64,
            "pvalue": pl.Float64,
            "error": pl.String,
        },
    )
    return PowerResults(simulations=sims, alpha=alpha, effect_type="multiplicative")


def _row(duration: int, effect: float, pvalue: float | None) -> dict[str, Any]:
    return {
        "duration": duration,
        "effect_size": effect,
        "pvalue": pvalue,
        "error": None if pvalue is not None else "RuntimeError: boom",
    }


def test_power_curve_counts_and_failed_rows() -> None:
    res = _results_from_rows(
        [
            _row(3, 0.1, 0.05),
            _row(3, 0.1, 0.5),
            _row(3, 0.1, None),  # failed simulation: excluded from denominator
            _row(3, 0.0, 0.8),
        ]
    )
    curve = res.power_curve()
    r = curve.filter(pl.col("effect_size") == 0.1).row(0, named=True)
    assert (r["n_simulations"], r["n_failed"], r["n_detected"]) == (3, 1, 1)
    assert r["power"] == pytest.approx(0.5)
    # Boundary: detection is pvalue <= alpha, not strict <.
    assert res.power_curve(alpha=0.5).filter(pl.col("effect_size") == 0.1).row(0, named=True)[
        "power"
    ] == pytest.approx(1.0)


def test_power_curve_all_failed_cell_has_null_power() -> None:
    res = _results_from_rows([_row(3, 0.1, None)])
    assert res.power_curve().row(0, named=True)["power"] is None


def test_mde_smallest_magnitude_meeting_target() -> None:
    res = _results_from_rows(
        [
            _row(3, 0.05, 0.5),  # power 0
            _row(3, 0.10, 0.05),  # power 1
            _row(3, 0.15, 0.05),  # power 1
            _row(3, 0.0, 0.9),
        ]
    )
    assert res.mde() == pytest.approx(0.10)
    assert res.mde(target_power=1.0) == pytest.approx(0.10)


def test_mde_none_when_no_effect_reaches_target() -> None:
    res = _results_from_rows([_row(3, 0.05, 0.5), _row(3, 0.0, 0.9)])
    assert res.mde() is None


def test_mde_negative_grid_returns_negative_effect() -> None:
    res = _results_from_rows([_row(3, -0.05, 0.5), _row(3, -0.10, 0.05)])
    assert res.mde() == pytest.approx(-0.10)


def test_mde_mixed_signs_raises() -> None:
    res = _results_from_rows([_row(3, -0.10, 0.05), _row(3, 0.10, 0.05)])
    with pytest.raises(ValueError, match="mixes positive and negative"):
        res.mde()


def test_mde_requires_duration_when_ambiguous() -> None:
    res = _results_from_rows([_row(3, 0.10, 0.05), _row(5, 0.10, 0.05)])
    with pytest.raises(ValueError, match="several durations"):
        res.mde()
    assert res.mde(duration=5) == pytest.approx(0.10)
    with pytest.raises(ValueError, match="not in results"):
        res.mde(duration=4)


def test_mde_target_power_validated() -> None:
    res = _results_from_rows([_row(3, 0.10, 0.05)])
    with pytest.raises(ValueError, match="target_power"):
        res.mde(target_power=0.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class _AlwaysFails:
    """Estimator stub whose fit always raises."""

    def fit(self, panel: pl.DataFrame, **kwargs: Any) -> _AlwaysFails:
        raise RuntimeError("solver exploded")


def test_on_error_raise_propagates(panel: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="solver exploded"):
        simulate_power(
            panel,
            estimator=_AlwaysFails(),  # type: ignore[arg-type]
            unit="city",
            time="t",
            outcome="y",
            treated="u0",
            durations=3,
            effect_sizes=[0.1],
        )


def test_on_error_record_keeps_going(panel: pl.DataFrame) -> None:
    res = simulate_power(
        panel,
        estimator=_AlwaysFails(),  # type: ignore[arg-type]
        unit="city",
        time="t",
        outcome="y",
        treated="u0",
        durations=3,
        effect_sizes=[0.1, 0.2],
        on_error="record",
    )
    assert res.simulations.height == 2
    assert res.simulations.get_column("pvalue").null_count() == 2
    assert all(
        e == "RuntimeError: solver exploded" for e in res.simulations.get_column("error").to_list()
    )


# ---------------------------------------------------------------------------
# Reproducibility and parallelism
# ---------------------------------------------------------------------------


def test_iid_reproducible_and_parallel_invariant(panel: pl.DataFrame) -> None:
    kwargs: dict[str, Any] = {
        "estimator": Synth(),
        "unit": "city",
        "time": "t",
        "outcome": "y",
        "treated": "u0",
        "durations": 3,
        "effect_sizes": [0.0, 0.3],
        "lookback_window": 2,
        "permutation_type": "iid",
        "ns": 200,
    }
    a = simulate_power(panel, rng=np.random.default_rng(42), **kwargs)
    b = simulate_power(panel, rng=np.random.default_rng(42), **kwargs)
    c = simulate_power(panel, rng=np.random.default_rng(42), n_jobs=2, **kwargs)
    assert (
        a.simulations.get_column("pvalue").to_list() == b.simulations.get_column("pvalue").to_list()
    )
    assert (
        a.simulations.get_column("pvalue").to_list() == c.simulations.get_column("pvalue").to_list()
    )


def test_block_parallel_matches_sequential(panel: pl.DataFrame) -> None:
    kwargs: dict[str, Any] = {
        "estimator": Synth(),
        "unit": "city",
        "time": "t",
        "outcome": "y",
        "treated": "u0",
        "durations": 3,
        "effect_sizes": [0.0, 0.3],
    }
    seq = simulate_power(panel, **kwargs)
    par = simulate_power(panel, n_jobs=2, **kwargs)
    assert (
        seq.simulations.get_column("pvalue").to_list()
        == par.simulations.get_column("pvalue").to_list()
    )


def test_random_block_scheme_reproducible_and_detects(panel: pl.DataFrame) -> None:
    # permutation_type="block" with a block_size is a random scheme: it must be
    # seeded per task like "iid" (reproducible for a given rng, n_jobs
    # invariant) and still detect a large injected lift.
    kwargs: dict[str, Any] = {
        "estimator": Synth(),
        "unit": "city",
        "time": "t",
        "outcome": "y",
        "treated": "u0",
        "durations": 3,
        "effect_sizes": [0.0, 0.5],
        "permutation_type": "block",
        "block_size": 2,
        "ns": 200,
    }
    a = simulate_power(panel, rng=np.random.default_rng(7), **kwargs)
    b = simulate_power(panel, rng=np.random.default_rng(7), **kwargs)
    c = simulate_power(panel, rng=np.random.default_rng(7), n_jobs=2, **kwargs)
    assert (
        a.simulations.get_column("pvalue").to_list() == b.simulations.get_column("pvalue").to_list()
    )
    assert (
        a.simulations.get_column("pvalue").to_list() == c.simulations.get_column("pvalue").to_list()
    )
    curve = a.power_curve(alpha=0.1)
    big = curve.filter(pl.col("effect_size") == 0.5).get_column("power")[0]
    assert big == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validation_errors(panel: pl.DataFrame) -> None:
    common: dict[str, Any] = {
        "estimator": Synth(),
        "unit": "city",
        "time": "t",
        "outcome": "y",
        "treated": "u0",
    }
    with pytest.raises(ValueError, match="not found in panel"):
        simulate_power(panel, **{**common, "outcome": "nope"}, durations=3)
    with pytest.raises(ValueError, match="not present in panel"):
        simulate_power(panel, **{**common, "treated": "zz"}, durations=3)
    with pytest.raises(ValueError, match="keeps a pre-period"):
        simulate_power(panel, **common, durations=N_PERIODS - 1, lookback_window=2)
    with pytest.raises(ValueError, match="duplicates"):
        simulate_power(panel, **common, durations=[3, 3])
    with pytest.raises(ValueError, match="duplicates"):
        simulate_power(panel, **common, durations=3, effect_sizes=[0.1, 0.1])
    with pytest.raises(ValueError, match="must be > -1"):
        simulate_power(panel, **common, durations=3, effect_sizes=[-1.0])
    with pytest.raises(ValueError, match="finite"):
        simulate_power(panel, **common, durations=3, effect_sizes=[float("nan")])
    with pytest.raises(ValueError, match="effect_type"):
        simulate_power(panel, **common, durations=3, effect_type="log")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lookback_window"):
        simulate_power(panel, **common, durations=3, lookback_window=0)
    with pytest.raises(ValueError, match="alpha"):
        simulate_power(panel, **common, durations=3, alpha=1.0)
    with pytest.raises(ValueError, match="on_error"):
        simulate_power(panel, **common, durations=3, on_error="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        simulate_power(panel, **common, durations=[])
    with pytest.raises(ValueError, match="block_size applies to permutation_type='block'"):
        simulate_power(panel, **common, durations=3, permutation_type="iid", block_size=2)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        simulate_power(panel, **common, durations=3, block_size=0)
    # Random schemes refuse to run without a caller-supplied generator,
    # matching conformal_pvalue's stance (contract review R-4).
    with pytest.raises(ValueError, match="requires a non-None rng"):
        simulate_power(panel, **common, durations=3, permutation_type="iid")
    with pytest.raises(ValueError, match="requires a non-None rng"):
        simulate_power(panel, **common, durations=3, block_size=2)


# ---------------------------------------------------------------------------
# Public contract: exports, protocol method, and run metadata (review R-1/R-2)
# ---------------------------------------------------------------------------


def test_power_contract_names_are_package_exports() -> None:
    import augsynth_py

    for name in ("PowerEstimator", "PowerParams", "EffectType", "DEFAULT_EFFECT_SIZES"):
        assert name in augsynth_py.__all__
        assert getattr(augsynth_py, name) is not None
    assert augsynth_py.DEFAULT_EFFECT_SIZES == (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def test_conformal_null_residuals_hook_is_public() -> None:
    # The estimator hook required by PowerEstimator (and conformal_pvalue) is
    # a public name; the pre-0.5 private spelling is gone.
    for cls in (Synth, AugSynth):
        assert hasattr(cls, "conformal_null_residuals")
        assert not hasattr(cls, "_conformal_null_residuals")


def test_params_recorded_on_results(panel: pl.DataFrame) -> None:
    res = simulate_power(
        panel,
        estimator=Synth(),
        unit="city",
        time="t",
        outcome="y",
        treated={"u0", "u1"},
        durations=3,
        effect_sizes=[0.0, 0.2],
        lookback_window=2,
        ns=500,
    )
    p = res.params
    assert p is not None
    assert set(p.treated) == {"u0", "u1"}
    assert p.durations == (3,)
    assert p.effect_sizes == (0.0, 0.2)
    assert p.lookback_window == 2
    assert p.permutation_type == "block"
    assert p.block_size is None
    assert p.side == "two-sided"
    assert p.ns == 500
    assert "Synth" in p.estimator
    # Direct construction (e.g. rebuilding from a filtered frame) needs no
    # params; the field defaults to None.
    rebuilt = PowerResults(
        simulations=res.simulations, alpha=res.alpha, effect_type=res.effect_type
    )
    assert rebuilt.params is None
