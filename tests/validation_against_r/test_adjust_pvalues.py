"""Parity test for :func:`augsynth_py.inference.adjust_pvalues` against R's
``stats::p.adjust``.

``adjust_pvalues`` computes in exact rational arithmetic and rounds once, so a
mathematically tied adjusted p-value stays ``== alpha`` (issue #29, scenario
S4b). R computes in plain floating point; the two must still agree to within
the default parity tolerance, and in practice to a few ulps. The tied cases
below are exactly where they differ in the last bit: R's ``"BH"`` gives
``0.05000000000000001`` where we give ``0.05``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from augsynth_py.inference import adjust_pvalues

from .conftest import assert_array_close

_R_METHOD = {"bonferroni": "bonferroni", "holm": "holm", "bh": "BH"}


def _family_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(29)
    t = 140
    return [
        ("textbook", np.array([0.01, 0.02, 0.03, 0.04])),
        ("mixed", np.array([0.005, 0.009, 0.05, 0.3, 0.9])),
        ("bonf-tie-7-140", np.array([1 / 140] + [1.0] * 6)),
        ("holm-tie-7-70", np.array([1 / 70] + [1.0] * 6)),
        ("bh-tie-83-83-j20", np.array([1 / 83] * 20 + [1.0] * 63)),
        ("block-grid", rng.integers(1, t + 1, 40) / t),
        ("uniform", rng.uniform(0.0, 1.0, 60)),
    ]


@pytest.mark.parametrize("method", ["bonferroni", "holm", "bh"])
@pytest.mark.parametrize(
    ("label", "p"), _family_cases(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_adjust_pvalues_matches_r_p_adjust(
    r_session: Any, method: str, label: str, p: np.ndarray
) -> None:
    from rpy2.robjects import FloatVector

    r_adjusted = np.asarray(
        r_session["p.adjust"](FloatVector(p.tolist()), method=_R_METHOD[method]),
        dtype=float,
    )
    ours = adjust_pvalues(p, method=method)

    assert_array_close(ours, r_adjusted, name=f"adjust_pvalues[{method}, {label}]")
    # Stronger than the default tolerance: the only difference from R is
    # floating-point rounding of the same formula.
    np.testing.assert_allclose(ours, r_adjusted, rtol=1e-14, atol=0)
