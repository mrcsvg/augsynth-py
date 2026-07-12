"""Parity: L2 imbalance metrics vs R augsynth (Synth + AugSynth).

Pins two facts about the Task 4-5 imbalance implementation
(:func:`augsynth_py.synth._panel.imbalance` and its estimator call sites):

1. **Normalization.** R augsynth's ``l2_imbalance`` is the plain (unnormalized)
   L2 norm of the pre-period residual in the demeaned fit space — no division
   by ``sqrt(T0)`` or ``T0``. The raw-l2 test would fail with a constant
   R/Python ratio if R normalized; it does not (verified against a live R
   augsynth run, 2026-07-12). The scaled ratio is invariant to any constant
   norm normalization, so it pins the residual itself.
2. **Weights choice for AugSynth.** R augsynth's ``l2_imbalance`` under
   ``progfunc='Ridge'`` is computed with the EFFECTIVE weights (omega + gamma),
   not the SCM-only omega — matching the call site in
   ``src/augsynth_py/synth/augmented.py``. With period demeaning and
   ``lambda > 0``, gamma sums to exactly 0 and omega sums to 1, so the value is
   identical in period-demeaned vs raw unit-demeaned space; the weights choice
   was the only genuinely open question, and this file pins it.

Same panel/date-parsing conventions as :file:`test_augsynth.py` and
:file:`test_multitreated.py` (treated ``'new york'``, treatment 2021-02-15).
One shared R+Python fit per estimator via module-scoped fixtures.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

TREATED_UNIT = "new york"
TREATMENT_DATE = date(2021, 2, 15)


def _panel_with_date_column(panel: pl.DataFrame) -> pl.DataFrame:
    """Return ``panel`` with ``date`` parsed to a real ``Date``.

    The GeoLift fixture ships ``date`` as a character column; same convention
    as :file:`test_augsynth.py`.
    """
    return panel.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _r_scalar(r_session: Any, expr: str) -> float:
    """Evaluate ``expr`` in R and return it as a Python float."""
    return float(np.asarray(r_session(expr), dtype=float).ravel()[0])


# ---------------------------------------------------------------------------
# Shared fits (one R fit + one Python fit per estimator for the whole module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_imbalance_pair(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> tuple[float, float, Any]:
    """Fit SCM-only augsynth (R) and ``Synth`` (Python) once.

    Returns ``(r_l2_imbalance, r_scaled_l2_imbalance, python_estimator)``.
    R variable names are prefixed ``imb_`` so the session-scoped R interpreter
    is not clobbered by (or does not clobber) other parity modules.
    """
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(
        f"""
        imb_df <- GeoLift_PreTest
        imb_df$date <- as.Date(imb_df$date)
        imb_df$treat <- as.integer(
            imb_df$location == '{TREATED_UNIT}'
            & imb_df$date >= as.Date('{TREATMENT_DATE.isoformat()}')
        )
        imb_none_res <- augsynth(
            Y ~ treat, unit = location, time = date, data = imb_df,
            progfunc = 'None', scm = TRUE, fixedeff = TRUE
        )
        """
    )
    r_l2 = _r_scalar(r_session, "imb_none_res$l2_imbalance")
    r_scaled = _r_scalar(r_session, "imb_none_res$scaled_l2_imbalance")

    from augsynth_py import Synth

    panel = _panel_with_date_column(r_geolift_pretest)
    est = Synth(fixedeff=True).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED_UNIT,
        treatment_time=TREATMENT_DATE,
    )
    return r_l2, r_scaled, est


@pytest.fixture(scope="module")
def augsynth_imbalance_pair(
    r_session: Any,
    r_geolift_pretest: pl.DataFrame,
) -> tuple[float, float, Any]:
    """Fit Ridge augsynth (R) and ``AugSynth`` (Python) at one pinned lambda.

    Returns ``(r_l2_imbalance, r_scaled_l2_imbalance, python_estimator)``.

    Lambda convention follows :file:`test_augsynth.py` Test A exactly: the
    pinned value is ``multiplier * var(y_pre_treated, ddof=0)`` and is passed
    verbatim to both sides (no rescaling between R and Python). We use the
    mid regime (multiplier 1.0) so the ridge correction is well-conditioned
    and gamma is meaningfully nonzero — a fixed unscaled lambda like 10.0
    would be ~5 orders of magnitude below the outcome variance and would
    barely regularize.
    """
    panel = _panel_with_date_column(r_geolift_pretest)
    pre = (
        panel.filter((pl.col("location") == TREATED_UNIT) & (pl.col("date") < TREATMENT_DATE))
        .get_column("Y")
        .to_numpy()
    )
    pinned_lambda = 1.0 * float(np.var(pre, ddof=0))

    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(f"imb_pinned_lambda <- {pinned_lambda!r}")
    r_session(
        f"""
        imb_df <- GeoLift_PreTest
        imb_df$date <- as.Date(imb_df$date)
        imb_df$treat <- as.integer(
            imb_df$location == '{TREATED_UNIT}'
            & imb_df$date >= as.Date('{TREATMENT_DATE.isoformat()}')
        )
        imb_ridge_res <- augsynth(
            Y ~ treat, unit = location, time = date, data = imb_df,
            progfunc = 'Ridge', scm = TRUE, fixedeff = TRUE,
            lambda = imb_pinned_lambda
        )
        """
    )
    r_l2 = _r_scalar(r_session, "imb_ridge_res$l2_imbalance")
    r_scaled = _r_scalar(r_session, "imb_ridge_res$scaled_l2_imbalance")

    from augsynth_py import AugSynth

    est = AugSynth(fixedeff=True, lambda_=pinned_lambda).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED_UNIT,
        treatment_time=TREATMENT_DATE,
    )
    return r_l2, r_scaled, est


# ---------------------------------------------------------------------------
# Synth (progfunc = 'None')
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_synth_scaled_imbalance_matches_r(
    synth_imbalance_pair: tuple[float, float, Any],
) -> None:
    """``Synth.scaled_l2_imbalance_`` matches R ``res$scaled_l2_imbalance``.

    The scaled ratio is invariant to any constant normalization of the norm,
    so this must match tightly regardless of R's normalization convention.
    Tolerance 1e-4 (abs): the ratio is O(0.4) here and each side carries a
    solver floor of ~1e-7 on this fixture, so 1e-4 leaves a wide margin
    without being loose enough to hide a wrong-space or wrong-weights bug
    (those move the ratio at the 1e-2 level or more).
    """
    _, r_scaled, est = synth_imbalance_pair
    py_scaled = est.scaled_l2_imbalance_
    diff = abs(py_scaled - r_scaled)
    assert diff < 1e-4, (
        f"Synth scaled_l2_imbalance: R={r_scaled!r}, Python={py_scaled!r}, "
        f"abs diff={diff:.3e} exceeds 1e-4"
    )


@pytest.mark.requires_r_pkg("augsynth")
def test_synth_raw_l2_imbalance_matches_r(
    synth_imbalance_pair: tuple[float, float, Any],
) -> None:
    """``Synth.l2_imbalance_`` matches R ``res$l2_imbalance`` (unnormalized).

    Pins the absolute scale: R augsynth stores the plain L2 norm of the
    demeaned pre-period residual, with no ``sqrt(T0)`` / ``T0`` division.
    Tolerance 1e-3 (abs): the value is O(1400) on this fixture, so this is a
    relative agreement of ~1e-6 — the observed solver-floor difference is
    ~5e-5. A constant normalization mismatch would show up as an R/Python
    ratio of e.g. ~6.7 (sqrt(T0=45)), which the failure message surfaces.
    """
    r_l2, _, est = synth_imbalance_pair
    py_l2 = est.l2_imbalance_
    diff = abs(py_l2 - r_l2)
    ratio = r_l2 / py_l2 if py_l2 != 0.0 else float("inf")
    assert diff < 1e-3, (
        f"Synth l2_imbalance: R={r_l2!r}, Python={py_l2!r}, abs diff={diff:.3e} "
        f"exceeds 1e-3; ratio R/Python={ratio!r} (a constant ratio such as "
        f"sqrt(T0) or T0 indicates a normalization-convention mismatch to fix "
        f"at the estimator call site, never inside the imbalance helper)"
    )


# ---------------------------------------------------------------------------
# AugSynth (progfunc = 'Ridge', pinned lambda)
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_augsynth_imbalance_matches_r(
    augsynth_imbalance_pair: tuple[float, float, Any],
) -> None:
    """``AugSynth`` imbalance attributes match R Ridge augsynth's.

    This is the test that pins the weights choice: R computes ``l2_imbalance``
    for ``progfunc='Ridge'`` with the effective weights (omega + gamma), same
    as the call site in ``augmented.py``. Had R used the SCM-only omega, both
    assertions here would fail by far more than the tolerance (the ridge
    correction shrinks the imbalance by ~40% on this fixture).

    Tolerances match the Synth tests above: scaled 1e-4 abs (ratio is O(0.23),
    scale-invariant), raw 1e-3 abs (value is O(840), i.e. relative ~1e-6).
    """
    r_l2, r_scaled, est = augsynth_imbalance_pair

    py_scaled = est.scaled_l2_imbalance_
    scaled_diff = abs(py_scaled - r_scaled)
    assert scaled_diff < 1e-4, (
        f"AugSynth scaled_l2_imbalance: R={r_scaled!r}, Python={py_scaled!r}, "
        f"abs diff={scaled_diff:.3e} exceeds 1e-4"
    )

    py_l2 = est.l2_imbalance_
    l2_diff = abs(py_l2 - r_l2)
    ratio = r_l2 / py_l2 if py_l2 != 0.0 else float("inf")
    assert l2_diff < 1e-3, (
        f"AugSynth l2_imbalance: R={r_l2!r}, Python={py_l2!r}, "
        f"abs diff={l2_diff:.3e} exceeds 1e-3; ratio R/Python={ratio!r}"
    )
