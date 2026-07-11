"""Unit tests for L2 imbalance (no R)."""

from __future__ import annotations

import numpy as np
import polars as pl

from augsynth_py import Synth
from augsynth_py.synth._panel import imbalance


def test_imbalance_matches_manual_norm() -> None:
    rng = np.random.default_rng(1)
    y1 = rng.normal(size=10)
    y0 = rng.normal(size=(10, 4))
    w = np.array([0.25, 0.25, 0.25, 0.25])
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
