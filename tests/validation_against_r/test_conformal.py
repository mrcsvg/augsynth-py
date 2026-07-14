"""Parity tests for CWZ 2021 conformal inference against R ``augsynth``.

Validates :func:`augsynth_py.inference.conformal_pvalue` and
:func:`augsynth_py.inference.conformal_interval` against R ``augsynth``'s
conformal inference (Chernozhukov, Wüthrich & Zhu 2021). Both sides use the
**full-window refit-under-null** construction: under a constant post-period
null ``h0``, the treated post-period outcomes are adjusted by ``h0`` and the
synthetic control is refit over the entire window, then the moving-block /
iid permutation test runs on the resulting exchangeable residuals.

R conformal API (discovered, used as an oracle — not translated)
----------------------------------------------------------------
``augsynth``'s ``summary()`` / ``conformal_inf()`` computes the *aggregate*
(average post-effect) conformal p-value by appending all post outcomes into the
balancing matrix ``X`` (a dummy single post column ``y``) and calling
``compute_permute_pval(agg_wide, ascm, h0, post_length, type, q, ns, stat_func)``.
That internal function subtracts ``h0`` from the treated post-columns, refits via
``fit_augsynth_internal`` over the full window, and runs the permutation test.
We call it directly with our chosen ``h0`` to obtain R's conformal p-value at an
arbitrary constant null — exactly the quantity our ``conformal_pvalue`` computes.

Defaults pinned for the comparison: ``type='block'`` (deterministic moving
block; ``ns`` ignored), ``q=1`` (mean-absolute statistic), ``stat_func=NULL``.

Convention pinned (decision log)
--------------------------------
- Moving block = the ``T`` cyclic shifts of the full residual vector, statistic
  recomputed on the fixed post positions, ``p = #{S_shift >= S_obs} / T``.
- Residual under ``H_0`` = full-window refit with fixed effects demeaned over
  the **entire** (h0-adjusted) window and simplex/ridge matching over all ``T``
  periods (verified: our ``_conformal_null_residuals(0)`` reproduces R's
  ``compute_permute_test_stats`` residual vector to solver tolerance, ~1e-4).

Aggregate CI note
-----------------
R does **not** expose an aggregate/average-effect conformal CI: in
``conformal_inf`` the average row's ``lb``/``ub`` are ``NA`` (only per-period
CIs, built from a different per-period null, are populated). There is therefore
no R oracle for our aggregate ``conformal_interval``. It is validated
transitively: the interval is the test-inversion acceptance region of the same
``conformal_pvalue`` that is asserted exact against R here at several ``h0``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

from augsynth_py.inference import conformal_interval, conformal_pvalue
from augsynth_py.synth.augmented import AugSynth
from augsynth_py.synth.classical import Synth

TREATED_UNIT = "new york"
# 2021-01-01 .. 2021-03-31 (90 days); 2021-02-15 splits into 45 pre / 45 post,
# matching the sibling parity tests (test_classical_synth.py, test_augsynth.py).
TREATMENT_DATE = date(2021, 2, 15)
TREATMENT_DATE_R = "2021-02-15"


def _panel_with_date(panel: pl.DataFrame) -> pl.DataFrame:
    """Parse the GeoLift ``date`` character column to a real Date."""
    return panel.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _treated_pre_variance(panel: pl.DataFrame) -> float:
    """``var(y_pre_treated, ddof=0)`` — the AugSynth lambda scale (see test_augsynth.py)."""
    pre = (
        panel.filter((pl.col("location") == TREATED_UNIT) & (pl.col("date") < TREATMENT_DATE))
        .get_column("Y")
        .to_numpy()
    )
    return float(np.var(pre, ddof=0))


@pytest.fixture(scope="module")
def r_conformal_env(r_session: Any, r_geolift_pretest: pl.DataFrame) -> Any:
    """Fit the R oracles once and expose the aggregate-conformal wide_data.

    Side-effects in the (session-scoped) R global env, reused across tests:

    - ``res`` / ``agg`` / ``pl_``   : classical SCM (progfunc='None') aggregate setup.
    - ``res2`` / ``agg2`` / ``pln2``: Ridge SCM (pinned lambda) aggregate setup.

    ``agg*`` is the ``cbind(X, y)`` + dummy-``y`` reshape ``conformal_inf`` uses
    for the average post-effect; ``compute_permute_pval(agg*, res*, h0, pl*, ...)``
    then returns R's conformal p-value at the constant null ``h0``.
    """
    panel = _panel_with_date(r_geolift_pretest)
    sigma_sq = _treated_pre_variance(panel)
    pinned_lambda = 1.0 * sigma_sq

    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(f"pinned_lambda <- {pinned_lambda!r}")
    r_session(
        f"""
        df <- GeoLift_PreTest
        df$date <- as.Date(df$date)
        df$treat <- as.integer(
            df$location == '{TREATED_UNIT}' & df$date >= as.Date('{TREATMENT_DATE_R}')
        )

        res <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'None', scm = TRUE, fixedeff = TRUE
        )
        wide_data <- res$data
        agg <- wide_data
        agg$X <- cbind(wide_data$X, wide_data$y)
        agg$y <- matrix(1, nrow = nrow(wide_data$X), ncol = 1)
        pl_ <- ncol(wide_data$y)

        res2 <- augsynth(
            Y ~ treat, unit = location, time = date, data = df,
            progfunc = 'Ridge', scm = TRUE, fixedeff = TRUE, lambda = pinned_lambda
        )
        wide2 <- res2$data
        agg2 <- wide2
        agg2$X <- cbind(wide2$X, wide2$y)
        agg2$y <- matrix(1, nrow = nrow(wide2$X), ncol = 1)
        pln2 <- ncol(wide2$y)
        """
    )
    return r_session


def _r_conformal_pval(
    r_session: Any,
    *,
    agg: str,
    res: str,
    post_len: str,
    h0: float,
    perm_type: str,
    ns: int,
) -> float:
    """R's conformal p-value at constant null ``h0`` via ``compute_permute_pval``."""
    expr = (
        f"augsynth:::compute_permute_pval({agg}, {res}, {h0!r}, {post_len}, "
        f"'{perm_type}', 1, {ns}, NULL)"
    )
    return float(np.asarray(r_session(expr), dtype=float).ravel()[0])


@pytest.fixture(scope="module")
def synth_fit(r_geolift_pretest: pl.DataFrame) -> Synth:
    """Python classical SCM matching the R ``progfunc='None'`` oracle."""
    panel = _panel_with_date(r_geolift_pretest)
    return Synth(fixedeff=True).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED_UNIT,
        treatment_time=TREATMENT_DATE,
    )


@pytest.fixture(scope="module")
def augsynth_fit(r_geolift_pretest: pl.DataFrame) -> AugSynth:
    """Python Ridge ASCM at the pinned lambda matching the R ``progfunc='Ridge'`` oracle."""
    panel = _panel_with_date(r_geolift_pretest)
    pinned_lambda = 1.0 * _treated_pre_variance(panel)
    return AugSynth(fixedeff=True, lambda_=pinned_lambda).fit(
        panel,
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED_UNIT,
        treatment_time=TREATMENT_DATE,
    )


# ---------------------------------------------------------------------------
# 1. Block p-value at the sharp null h0 = 0 -- deterministic, EXACT.
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_block_pvalue_h0_zero_matches_r_exact(
    r_conformal_env: Any,
    synth_fit: Synth,
) -> None:
    """``conformal_pvalue(fit, 0, block)`` equals R's aggregate conformal p exactly.

    The block scheme is deterministic on both sides (the p-value is a multiple
    of ``1/T``), so agreement is exact up to float noise, not a statistical
    tolerance. On this fixture both sides return ``74/90``.
    """
    r_p = _r_conformal_pval(
        r_conformal_env, agg="agg", res="res", post_len="pl_", h0=0.0, perm_type="block", ns=1000
    )
    py_p = conformal_pvalue(synth_fit, 0.0, permutation_type="block", side="two-sided")
    assert py_p == pytest.approx(r_p, abs=1e-8)


# ---------------------------------------------------------------------------
# 2. Block p-value at nonzero h0 -- deterministic, EXACT.
#    Transitively validates the test-inverted conformal_interval.
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
@pytest.mark.parametrize("h0", [-20.0, 10.0, 50.0])
def test_block_pvalue_nonzero_h0_matches_r_exact(
    r_conformal_env: Any,
    synth_fit: Synth,
    h0: float,
) -> None:
    """``conformal_pvalue(fit, h0, block)`` matches R exactly for nonzero ``h0``.

    R's ``compute_permute_pval(..., h0, ...)`` subtracts ``h0`` from the treated
    post-columns and refits over the full window before permuting — the same
    construction as our ``_conformal_null_residuals(h0)``. Exercising several
    ``h0`` validates the whole p(h0) curve, and hence the acceptance region that
    ``conformal_interval`` inverts (for which R exposes no aggregate oracle).
    """
    r_p = _r_conformal_pval(
        r_conformal_env, agg="agg", res="res", post_len="pl_", h0=h0, perm_type="block", ns=1000
    )
    py_p = conformal_pvalue(synth_fit, h0, permutation_type="block", side="two-sided")
    assert py_p == pytest.approx(r_p, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. iid p-value -- stochastic, statistical tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_iid_pvalue_matches_r_within_tolerance(
    r_conformal_env: Any,
    synth_fit: Synth,
) -> None:
    """iid conformal p agrees with R within a Monte-Carlo tolerance.

    Both sides draw ``ns=5000`` random permutations; RNG state does not transfer
    between R and Python, so exact agreement is impossible. The ``0.03``
    tolerance guards against gross scheme errors (wrong statistic, wrong
    residual construction) while absorbing the sampling noise of two
    independent 5000-draw estimates.
    """
    r_p = _r_conformal_pval(
        r_conformal_env, agg="agg", res="res", post_len="pl_", h0=0.0, perm_type="iid", ns=5000
    )
    py_p = conformal_pvalue(
        synth_fit, 0.0, permutation_type="iid", ns=5000, rng=np.random.default_rng(0)
    )
    assert abs(py_p - r_p) <= 0.03


# ---------------------------------------------------------------------------
# 4. Conformal interval -- no aggregate R oracle; validate structurally.
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
def test_conformal_interval_is_finite_and_brackets_att(
    synth_fit: Synth,
) -> None:
    """The test-inverted CI is finite and brackets ``att_``.

    R exposes no aggregate/average-effect conformal CI (``conformal_inf``'s
    average row has ``NA`` bounds), so there is no direct numeric oracle. The
    interval is validated transitively by the exact p(h0) parity above (it is
    the acceptance region of that same p-value); here we assert its structural
    invariants on the real fixture.
    """
    lo, hi = conformal_interval(
        synth_fit, alpha=0.05, grid_size=400, permutation_type="block", side="two-sided"
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert lo <= synth_fit.att_ <= hi


# ---------------------------------------------------------------------------
# 5. AugSynth (Ridge, pinned lambda) block p-value -- deterministic, EXACT.
# ---------------------------------------------------------------------------


@pytest.mark.requires_r_pkg("augsynth")
@pytest.mark.parametrize("h0", [0.0, 30.0])
def test_augsynth_block_pvalue_matches_r_exact(
    r_conformal_env: Any,
    augsynth_fit: AugSynth,
    h0: float,
) -> None:
    """AugSynth conformal block p matches R ``progfunc='Ridge'`` exactly.

    The ridge augmentation is refit over the full window under the null at the
    same pinned ``lambda = var(y_pre_treated, ddof=0)`` on both sides (see
    test_augsynth.py for the lambda convention). On this fixture both sides
    return ``65/90`` at ``h0 = 0``.
    """
    r_p = _r_conformal_pval(
        r_conformal_env, agg="agg2", res="res2", post_len="pln2", h0=h0, perm_type="block", ns=1000
    )
    py_p = conformal_pvalue(augsynth_fit, h0, permutation_type="block", side="two-sided")
    assert py_p == pytest.approx(r_p, abs=1e-8)
