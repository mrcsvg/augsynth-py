"""Multi-treated-unit fits collapse to the treated-group mean (no R)."""

from __future__ import annotations

import numpy as np
import polars as pl

from augsynth_py import AugSynth, Synth


def _panel(n_units: int = 6, n_periods: int = 12, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"u{i}" for i in range(n_units)]
    base = rng.normal(size=(n_periods, n_units)) * 5 + np.arange(n_units) * 10
    rows = []
    for j, name in enumerate(names):
        for t in range(n_periods):
            rows.append({"unit": name, "time": t, "Y": float(base[t, j])})
    return pl.DataFrame(rows)


def test_synth_list_of_one_equals_scalar() -> None:
    panel = _panel()
    a = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=8
    )
    b = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated=["u0"], treatment_time=8
    )
    np.testing.assert_allclose(a.synthetic_, b.synthetic_)
    assert a.weights_ == b.weights_


def test_synth_multitreated_actual_is_group_mean() -> None:
    """`actual_` for treated=[u0,u1] is the elementwise mean of u0 and u1."""
    panel = _panel()
    wide = panel.pivot(values="Y", index="time", on="unit").sort("time")
    expected_actual = (wide["u0"].to_numpy() + wide["u1"].to_numpy()) / 2

    est = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u0", "u1"],
        treatment_time=8,
    )
    np.testing.assert_allclose(est.actual_, expected_actual)
    # u0 and u1 must not appear as donors.
    assert "u0" not in est.weights_ and "u1" not in est.weights_


def test_augsynth_multitreated_runs_and_excludes_treated_from_donors() -> None:
    panel = _panel()
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u0", "u1"],
        treatment_time=8,
    )
    assert "u0" not in est.weights_ and "u1" not in est.weights_
    assert est.actual_.shape == est.synthetic_.shape
