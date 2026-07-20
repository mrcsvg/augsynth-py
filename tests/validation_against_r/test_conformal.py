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
from pathlib import Path
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
    # abs=1e-8 is legitimate for a solver-based rank statistic: the p-value is an
    # integer count of cyclic shifts over T, so R and Python agree exactly iff
    # they agree on that integer. The smallest gap between distinct shift
    # statistics on this fixture (~0.1+) dwarfs the combined R+Python Clarabel
    # residual noise (~1e-4), so no rank can flip — the counts are identical and
    # the two p-values coincide to float precision.
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
def test_conformal_interval_brackets_accepted_nulls(
    synth_fit: Synth,
) -> None:
    """The test-inverted CI contains every null the exact p(h0) parity accepts.

    R exposes no aggregate/average-effect conformal CI (``conformal_inf``'s
    average row has ``NA`` bounds), so there is no direct numeric oracle. The
    interval is validated transitively: it is the acceptance region of the same
    ``conformal_pvalue`` asserted exact against R above, so it must contain every
    ``h0`` with ``p >= alpha``. Both ``h0 = 0`` (p ~ 0.82) and ``h0 = 50``
    (p ~ 0.84) are accepted, so both — and ``att_`` — must lie inside.

    On this fixture the post gap is highly volatile, so the aggregate conformal
    test has little power: the acceptance region is enormous and exceeds the
    truncation guard's doubling cap, which deterministically emits a
    ``UserWarning``. That is the correct, non-anti-conservative behavior (the
    old silent-clip would have understated the interval).
    """
    with pytest.warns(UserWarning, match="truncated"):
        lo, hi = conformal_interval(
            synth_fit, alpha=0.05, grid_size=400, permutation_type="block", side="two-sided"
        )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert lo <= synth_fit.att_ <= hi
    assert lo <= 0.0 <= hi
    assert lo <= 50.0 <= hi


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


# ---------------------------------------------------------------------------
# 6. Basque panel (Abadie & Gardeazabal 2003): p(h0) curve -- deterministic,
#    EXACT. Second fixture, decoupled from GeoLift_PreTest; pins the strongly
#    asymmetric acceptance region investigated as known-issue I-1.
# ---------------------------------------------------------------------------

BASQUE_CSV = (
    Path(__file__).resolve().parents[2] / "notebooks" / "_data" / "basque_ag2003.csv"
).as_posix()
BASQUE_TREATED = "Basque Country (Pais Vasco)"
BASQUE_TREATMENT_YEAR = 1975

# h0 probes spanning the full structure of the Basque p(h0) curve: both 1/T
# floors (-5, +10), the sharp left cliff (-3.75 -> -3.0), the REJECTED gap at
# the point estimate (-0.5, p = 1/43 < alpha: the acceptance region is
# non-contiguous and contains att_ ~ -0.69 only via min/max bridging) with its
# accepted flank (-0.75, p = 3/43), the sharp null (0), the broad right
# plateau (+3.0), its peak (+4.5, p = 36/43 -- far from att_), and the right
# decay (+7.5). Exact agreement across these pins the whole acceptance region
# {h0 : p(h0) >= alpha}, i.e. the markedly asymmetric conformal interval of
# I-1, as a property of the method shared with R rather than an artifact.
BASQUE_H0_PROBES = [-5.0, -3.75, -3.0, -0.75, -0.5, 0.0, 3.0, 4.5, 7.5, 10.0]


@pytest.fixture(scope="module")
def basque_panel() -> pl.DataFrame:
    """AG 2003 Basque panel (17 regions x 43 years), Spain aggregate dropped."""
    return pl.read_csv(BASQUE_CSV).filter(pl.col("regionname") != "Spain (Espana)")


@pytest.fixture(scope="module")
def basque_synth_fit(basque_panel: pl.DataFrame) -> Synth:
    """Python classical SCM on the Basque panel, no fixed effects (as in I-1)."""
    return Synth(fixedeff=False).fit(
        basque_panel,
        unit="regionname",
        time="year",
        outcome="gdpcap",
        treated=BASQUE_TREATED,
        treatment_time=BASQUE_TREATMENT_YEAR,
    )


@pytest.fixture(scope="module")
def r_basque_conformal_env(r_session: Any) -> Any:
    """R oracle on the Basque panel: aggregate-conformal setup, fixedeff=FALSE.

    Same ``cbind(X, y)`` + dummy-``y`` reshape as ``r_conformal_env``, but built
    from the in-repo Basque CSV so it needs only the ``augsynth`` R package (not
    ``GeoLift``). Exposes ``res_bq`` / ``agg_bq`` / ``plb`` in the R global env.
    """
    r_session("suppressPackageStartupMessages(library(augsynth))")
    r_session(
        f"""
        bq <- read.csv('{BASQUE_CSV}')
        bq <- bq[bq$regionname != 'Spain (Espana)', ]
        bq$treat <- as.integer(
            bq$regionname == '{BASQUE_TREATED}' & bq$year >= {BASQUE_TREATMENT_YEAR}
        )

        res_bq <- augsynth(
            gdpcap ~ treat, unit = regionname, time = year, data = bq,
            progfunc = 'None', scm = TRUE, fixedeff = FALSE
        )
        wide_bq <- res_bq$data
        agg_bq <- wide_bq
        agg_bq$X <- cbind(wide_bq$X, wide_bq$y)
        agg_bq$y <- matrix(1, nrow = nrow(wide_bq$X), ncol = 1)
        plb <- ncol(wide_bq$y)
        """
    )
    return r_session


@pytest.mark.requires_r_pkg("augsynth")
@pytest.mark.parametrize("h0", BASQUE_H0_PROBES)
def test_basque_pvalue_curve_matches_r_exact(
    r_basque_conformal_env: Any,
    basque_synth_fit: Synth,
    h0: float,
) -> None:
    """The Basque p(h0) curve equals R's pointwise, including its asymmetry.

    Resolves known-issue I-1: the 95% interval (-3.22, 7.69) around
    ``att_ = -0.69`` is skewed strongly positive because p(h0) itself is
    asymmetric — it peaks near h0 = +4.5, not at ``att_``, and collapses to the
    1/T floor already at h0 = -3.75 on the left. The acceptance region is even
    non-contiguous: p(h0) = 1/43 < 0.05 for h0 in roughly [-0.70, -0.35] — a
    rejected gap containing ``att_`` itself, identical in R, which the
    interval's min/max extraction bridges (audit D-6). A full 89-point sweep
    over h0 in [-12, 16] matched R with max |p_py - p_r| = 4.4e-16 (one ulp);
    this test pins the probes that define that shape. The mechanism is the
    full-window refit against a donor pool whose convex hull cannot reach the
    treated series from below: raising the treated post-period (negative h0)
    concentrates residual mass in the post window (share 0.79-0.96 of total),
    while lowering it (positive h0) lets the refit spread misfit across the
    whole window (share ~ 0.48), so post-window block sums rank as
    unexceptional among the cyclic shifts and p stays high. Near att_ the
    refit zeroes the pre-window residuals while the post window keeps genuine
    dispersion, so the post block ranks first among all shifts — the local
    exchangeability failure that produces the rejected gap at the truth.
    """
    r_p = _r_conformal_pval(
        r_basque_conformal_env,
        agg="agg_bq",
        res="res_bq",
        post_len="plb",
        h0=h0,
        perm_type="block",
        ns=1000,
    )
    py_p = conformal_pvalue(basque_synth_fit, h0, permutation_type="block", side="two-sided")
    # Same rationale as test_block_pvalue_h0_zero_matches_r_exact: the block
    # p-value is an integer count over T = 43 cyclic shifts; the smallest gap
    # between distinct shift statistics on this fixture dwarfs solver noise, so
    # the counts are identical and agreement is float-exact.
    assert py_p == pytest.approx(r_p, abs=1e-8)
