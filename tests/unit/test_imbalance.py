"""Unit tests for L2 imbalance (no R)."""

from __future__ import annotations

import numpy as np
import polars as pl

from augsynth_py import AugSynth, Synth
from augsynth_py.synth._panel import imbalance


def test_imbalance_matches_manual_norm() -> None:
    rng = np.random.default_rng(1)
    y1 = rng.normal(size=10)
    y0 = rng.normal(size=(10, 4))
    # Non-uniform weights so the scaled assertion distinguishes the fitted
    # numerator from the uniform-baseline denominator.
    w = np.array([0.7, 0.1, 0.1, 0.1])
    l2, scaled = imbalance(y1, y0, w)

    resid = y1 - y0 @ w
    w_unif = np.full(4, 0.25)
    resid_unif = y1 - y0 @ w_unif
    np.testing.assert_allclose(l2, np.linalg.norm(resid))
    np.testing.assert_allclose(scaled, np.linalg.norm(resid) / np.linalg.norm(resid_unif))


def test_scaled_imbalance_is_one_for_uniform_weights() -> None:
    rng = np.random.default_rng(2)
    y1 = rng.normal(size=8)
    y0 = rng.normal(size=(8, 3))
    _, scaled = imbalance(y1, y0, np.full(3, 1 / 3))
    np.testing.assert_allclose(scaled, 1.0)


def test_zero_denominator_follows_ieee_convention() -> None:
    # Identical donor columns equal to y1: the uniform 1/J baseline fits
    # exactly, so the denominator is 0. Convention matches unguarded R
    # division: x/0 = Inf, 0/0 = NaN.
    y1 = np.array([1.0, 2.0, 3.0])
    y0 = np.column_stack([y1, y1])

    # Weights sum to 2 -> fitted residual is -y1, so l2 > 0: scaled is inf.
    l2, scaled = imbalance(y1, y0, np.array([1.0, 1.0]))
    assert l2 > 0.0
    assert scaled == float("inf")

    # Uniform weights also fit exactly -> 0/0: scaled is nan.
    l2, scaled = imbalance(y1, y0, np.array([0.5, 0.5]))
    np.testing.assert_allclose(l2, 0.0, atol=1e-15)
    assert np.isnan(scaled)


def test_synth_exposes_imbalance_attributes() -> None:
    rng = np.random.default_rng(3)
    names = [f"u{i}" for i in range(5)]
    rows = []
    for j, name in enumerate(names):
        for t in range(10):
            rows.append({"unit": name, "time": t, "Y": float(rng.normal() + j)})
    panel = pl.DataFrame(rows)
    est = Synth(fixedeff=True).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=7
    )
    assert isinstance(est.l2_imbalance_, float)
    assert isinstance(est.scaled_l2_imbalance_, float)
    assert est.l2_imbalance_ >= 0.0
    # A fitted SCM should not be worse than uniform weights.
    assert est.scaled_l2_imbalance_ <= 1.0 + 1e-9


def test_augsynth_exposes_imbalance_attributes() -> None:
    rng = np.random.default_rng(4)
    names = [f"u{i}" for i in range(5)]
    rows = []
    for j, name in enumerate(names):
        for t in range(12):
            rows.append({"unit": name, "time": t, "Y": float(rng.normal() + j)})
    panel = pl.DataFrame(rows)
    est = AugSynth(fixedeff=True, lambda_=1.0).fit(
        panel, unit="unit", time="time", outcome="Y", treated="u0", treatment_time=8
    )
    assert isinstance(est.l2_imbalance_, float)
    assert isinstance(est.scaled_l2_imbalance_, float)
    assert est.l2_imbalance_ >= 0.0
