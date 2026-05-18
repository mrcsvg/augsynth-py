"""Unit tests for the classical synthetic control estimator.

These tests do not depend on R. Numerical parity against ``augsynth`` lives in
``tests/validation_against_r/test_classical_synth.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from augsynth_py import Synth
from augsynth_py.exceptions import (
    EmptyDonorPoolError,
    TreatedUnitNotFoundError,
)


def _long_panel(y_matrix: np.ndarray, units: list[str]) -> pl.DataFrame:
    """Build a long polars panel from a ``(T, N)`` outcome matrix."""
    n_periods, n_units = y_matrix.shape
    assert n_units == len(units)
    rows = []
    for t in range(n_periods):
        for j, name in enumerate(units):
            rows.append({"unit": name, "day": t, "y": float(y_matrix[t, j])})
    return pl.DataFrame(rows)


def test_perfect_donor_gets_full_weight(rng: np.random.Generator) -> None:
    """One donor matches the treated's pre-period exactly → it should get weight 1."""
    n_periods = 30
    treatment_time = 20
    treated_trajectory = np.linspace(100, 130, n_periods) + rng.normal(0, 0.5, n_periods)

    # Donor B copies treated's pre-period exactly; its post-period is arbitrary.
    donor_b = treated_trajectory.copy()
    donor_b[treatment_time:] += rng.normal(0, 5, n_periods - treatment_time)

    # Three uncorrelated donors.
    donor_c = np.linspace(50, 40, n_periods) + rng.normal(0, 1, n_periods)
    donor_d = np.full(n_periods, 80.0) + rng.normal(0, 1, n_periods)
    donor_e = np.linspace(200, 210, n_periods) + rng.normal(0, 1, n_periods)

    y_matrix = np.column_stack([treated_trajectory, donor_b, donor_c, donor_d, donor_e])
    panel = _long_panel(y_matrix, units=["A_treated", "B_match", "C", "D", "E"])

    est = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="A_treated",
        treatment_time=treatment_time,
    )

    assert est.weights_["B_match"] > 0.95
    for other in ("C", "D", "E"):
        assert est.weights_[other] < 0.05

    # RMSPE should be tiny because B reproduces A's pre-period up to its own noise.
    assert est.rmspe_pre_ < 0.02


def test_weights_lie_on_simplex(rng: np.random.Generator) -> None:
    """For any non-degenerate panel, weights are non-negative and sum to 1."""
    n_periods, n_donors = 40, 6
    treatment_time = 25
    y_matrix = rng.normal(100, 10, (n_periods, n_donors + 1))
    units = ["treated"] + [f"d{i}" for i in range(n_donors)]
    panel = _long_panel(y_matrix, units=units)

    est = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="treated",
        treatment_time=treatment_time,
    )

    weights = np.array(list(est.weights_.values()))
    assert (weights >= -1e-9).all()
    assert abs(weights.sum() - 1.0) < 1e-6


def test_fit_is_deterministic(rng: np.random.Generator) -> None:
    """Two consecutive fits on the same panel must produce identical weights."""
    y_matrix = rng.normal(100, 10, (40, 7))
    panel = _long_panel(y_matrix, units=["t", *(f"d{i}" for i in range(6))])

    w1 = (
        Synth()
        .fit(panel, unit="unit", time="day", outcome="y", treated="t", treatment_time=25)
        .weights_
    )
    w2 = (
        Synth()
        .fit(panel, unit="unit", time="day", outcome="y", treated="t", treatment_time=25)
        .weights_
    )

    for k in w1:
        assert abs(w1[k] - w2[k]) < 1e-9


def test_treated_unit_must_exist(rng: np.random.Generator) -> None:
    panel = _long_panel(rng.normal(100, 10, (10, 3)), units=["a", "b", "c"])
    with pytest.raises(TreatedUnitNotFoundError):
        Synth().fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="missing",
            treatment_time=5,
        )


def test_donor_pool_must_be_nonempty(rng: np.random.Generator) -> None:
    panel = _long_panel(rng.normal(100, 10, (10, 1)), units=["solo"])
    with pytest.raises(EmptyDonorPoolError):
        Synth().fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="solo",
            treatment_time=5,
        )


def test_treatment_time_boundaries(rng: np.random.Generator) -> None:
    panel = _long_panel(rng.normal(100, 10, (10, 3)), units=["a", "b", "c"])

    # No pre-treatment periods.
    with pytest.raises(ValueError, match="pre-treatment"):
        Synth().fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="a",
            treatment_time=0,
        )

    # No post-treatment periods.
    with pytest.raises(ValueError, match="post-treatment"):
        Synth().fit(
            panel,
            unit="unit",
            time="day",
            outcome="y",
            treated="a",
            treatment_time=10,
        )


def test_unbalanced_panel_is_rejected(rng: np.random.Generator) -> None:
    panel = _long_panel(rng.normal(100, 10, (10, 3)), units=["a", "b", "c"])
    # Drop one (unit, day) cell.
    truncated = panel.filter(~((pl.col("unit") == "b") & (pl.col("day") == 4)))
    with pytest.raises(ValueError, match="unbalanced"):
        Synth().fit(
            truncated,
            unit="unit",
            time="day",
            outcome="y",
            treated="a",
            treatment_time=5,
        )


def test_scm_beats_naive_donor_mean(rng: np.random.Generator) -> None:
    """RMSPE-pre under SCM must be no worse than under equal-weight donor mean.

    The minimizer over the simplex contains the equal-weight vector as a
    feasible point, so the optimal objective is by construction <= equal-weight
    objective. The strict inequality is overwhelmingly likely on random data.
    """
    n_periods = 40
    treatment_time = 25
    # Heterogeneous donors so equal-weight average is a poor reconstruction.
    treated = np.linspace(100, 130, n_periods)
    donors = np.column_stack(
        [
            np.linspace(80, 90, n_periods),
            np.linspace(120, 140, n_periods),
            np.linspace(110, 125, n_periods),
            np.linspace(60, 100, n_periods),
        ]
    )
    treated_noisy = treated + rng.normal(0, 0.3, n_periods)
    donors_noisy = donors + rng.normal(0, 0.3, donors.shape)

    y_matrix = np.column_stack([treated_noisy, donors_noisy])
    panel = _long_panel(y_matrix, units=["t", "d0", "d1", "d2", "d3"])

    est = Synth(fixedeff=True).fit(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=treatment_time,
    )

    # Equal-weight donor mean as baseline (with same fixed-effect offsetting).
    pre = np.arange(n_periods) < treatment_time
    treated_offset = treated_noisy[pre].mean()
    donor_offsets = donors_noisy[pre].mean(axis=0)
    naive_synth_pre = (donors_noisy[pre] - donor_offsets).mean(axis=1) + treated_offset
    naive_rmspe = np.sqrt(np.mean((treated_noisy[pre] - naive_synth_pre) ** 2)) / abs(
        treated_offset
    )

    assert est.rmspe_pre_ <= naive_rmspe + 1e-9


def test_fixedeff_false_keeps_levels(rng: np.random.Generator) -> None:
    """With ``fixedeff=False`` the synthetic must lie inside the donor envelope.

    Convex combinations cannot exceed ``max(donors)`` or fall below ``min(donors)``
    pointwise, so this is a direct consequence of the simplex constraint.
    """
    n_periods = 30
    y_matrix = rng.normal(100, 5, (n_periods, 5))
    # Make the treated systematically higher than every donor.
    y_matrix[:, 0] += 50
    panel = _long_panel(y_matrix, units=["t", "d0", "d1", "d2", "d3"])

    est = Synth(fixedeff=False).fit(
        panel,
        unit="unit",
        time="day",
        outcome="y",
        treated="t",
        treatment_time=20,
    )

    donor_cols = y_matrix[:, 1:]
    assert (est.synthetic_ <= donor_cols.max(axis=1) + 1e-6).all()
    assert (est.synthetic_ >= donor_cols.min(axis=1) - 1e-6).all()
