"""Multi-treated-unit fits collapse to the treated-group mean (no R)."""

from __future__ import annotations

import numpy as np
import polars as pl

from augsynth_py import AugSynth, Synth


def _make_panel(n_units: int = 6, n_periods: int = 12, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"u{i}" for i in range(n_units)]
    base = rng.normal(size=(n_periods, n_units)) * 5 + np.arange(n_units) * 10
    rows = []
    for j, name in enumerate(names):
        for t in range(n_periods):
            rows.append({"unit": name, "time": t, "Y": float(base[t, j])})
    return pl.DataFrame(rows)


def _group_mean(panel: pl.DataFrame, members: list[str]) -> np.ndarray:
    """Independent pivot-based expectation of the treated-group mean path."""
    wide = panel.pivot(values="Y", index="time", on="unit").sort("time")
    stacked = np.column_stack([wide[m].to_numpy() for m in members])
    mean: np.ndarray = stacked.mean(axis=1)
    return mean


def test_synth_list_of_one_equals_scalar() -> None:
    panel = _make_panel()
    a = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=8
    )
    b = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated=["u0"], treatment_time=8
    )
    np.testing.assert_allclose(a.synthetic_, b.synthetic_)
    assert a.weights_.keys() == b.weights_.keys()
    for name in a.weights_:
        np.testing.assert_allclose(a.weights_[name], b.weights_[name])


def test_synth_multitreated_actual_is_group_mean() -> None:
    """`actual_` for treated=[u0,u1] is the elementwise mean of u0 and u1."""
    panel = _make_panel()
    expected_actual = _group_mean(panel, ["u0", "u1"])

    est = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u0", "u1"],
        treatment_time=8,
    )
    np.testing.assert_allclose(est.actual_, expected_actual)
    # units_[0] is the comma-joined group label, not a panel unit name.
    assert est.units_[0] == "u0,u1"
    # u0 and u1 must not appear as donors.
    assert "u0" not in est.weights_ and "u1" not in est.weights_


def test_synth_multitreated_order_invariant() -> None:
    """treated=[u1,u0] gives the same label and fit as treated=[u0,u1]."""
    panel = _make_panel()
    a = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u0", "u1"],
        treatment_time=8,
    )
    b = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u1", "u0"],
        treatment_time=8,
    )
    assert a.units_[0] == b.units_[0] == "u0,u1"
    np.testing.assert_allclose(a.synthetic_, b.synthetic_)


def test_augsynth_multitreated_runs_and_excludes_treated_from_donors() -> None:
    panel = _make_panel()
    expected_actual = _group_mean(panel, ["u0", "u1"])
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel,
        unit="unit",
        time="time",
        outcome="Y",
        treated=["u0", "u1"],
        treatment_time=8,
    )
    assert "u0" not in est.weights_ and "u1" not in est.weights_
    np.testing.assert_allclose(est.actual_, expected_actual)
    assert est.actual_.shape == est.synthetic_.shape
