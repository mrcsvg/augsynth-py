"""R oracle wrappers for the showcase notebook.

This module isolates all `rpy2` calls so the showcase notebook
(`01_showcase_sp_campaign.ipynb`) can stay readable. It will later be reused by
`tests/validation_against_r/` once the native Python estimators land.

Layout:
    - `make_panel()`    : DGP for the synthetic geo-experiment.
    - `naive_did()`     : pure-Python 2x2 difference-in-differences.
    - `fit_scm_r()`     : classical SCM via `augsynth(progfunc='None')`.
    - `fit_augsynth_r()`: ridge-augmented SCM via `augsynth(progfunc='Ridge')`.
    - `placebo_test()`  : permutation inference re-fitting each donor as treated.

All R wrappers return a `FitResult` dataclass — notebook code never touches rpy2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Lazy R bridge
# ---------------------------------------------------------------------------

# Caching the bridge object means importing this module is cheap; rpy2 only
# loads the first time an R wrapper is actually called.


def _r() -> Any:
    if not hasattr(_r, "_cached"):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        _r._cached = type(  # type: ignore[attr-defined]
            "RBridge",
            (),
            dict(
                ro=ro,
                pandas2ri=pandas2ri,
                base=importr("base"),
                augsynth=importr("augsynth"),
            ),
        )()
    return _r._cached  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

# 27 BR capitals (26 states + DF). Baselines and regions are roughly realistic
# but not factual. Order matters only for reproducibility of the seeded RNG.
CITIES_BR: list[tuple[str, str, float]] = [
    ("São Paulo", "Sudeste", 1_000_000),
    ("Rio de Janeiro", "Sudeste", 700_000),
    ("Belo Horizonte", "Sudeste", 500_000),
    ("Vitória", "Sudeste", 250_000),
    ("Curitiba", "Sul", 450_000),
    ("Porto Alegre", "Sul", 400_000),
    ("Florianópolis", "Sul", 300_000),
    ("Salvador", "Nordeste", 380_000),
    ("Recife", "Nordeste", 350_000),
    ("Fortaleza", "Nordeste", 360_000),
    ("Natal", "Nordeste", 200_000),
    ("João Pessoa", "Nordeste", 180_000),
    ("Maceió", "Nordeste", 170_000),
    ("Aracaju", "Nordeste", 160_000),
    ("São Luís", "Nordeste", 220_000),
    ("Teresina", "Nordeste", 210_000),
    ("Manaus", "Norte", 290_000),
    ("Belém", "Norte", 270_000),
    ("Porto Velho", "Norte", 150_000),
    ("Rio Branco", "Norte", 120_000),
    ("Boa Vista", "Norte", 110_000),
    ("Macapá", "Norte", 130_000),
    ("Palmas", "Norte", 100_000),
    ("Brasília", "Centro-Oeste", 600_000),
    ("Goiânia", "Centro-Oeste", 380_000),
    ("Campo Grande", "Centro-Oeste", 280_000),
    ("Cuiabá", "Centro-Oeste", 250_000),
]


# (lat, lon) for each BR capital, used by the showcase notebook map plots.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "São Paulo": (-23.55, -46.63),
    "Rio de Janeiro": (-22.91, -43.17),
    "Belo Horizonte": (-19.92, -43.94),
    "Vitória": (-20.32, -40.34),
    "Curitiba": (-25.43, -49.27),
    "Porto Alegre": (-30.03, -51.23),
    "Florianópolis": (-27.59, -48.55),
    "Salvador": (-12.97, -38.51),
    "Recife": (-8.05, -34.88),
    "Fortaleza": (-3.73, -38.52),
    "Natal": (-5.79, -35.20),
    "João Pessoa": (-7.12, -34.86),
    "Maceió": (-9.65, -35.71),
    "Aracaju": (-10.91, -37.07),
    "São Luís": (-2.53, -44.30),
    "Teresina": (-5.09, -42.81),
    "Manaus": (-3.12, -60.02),
    "Belém": (-1.46, -48.50),
    "Porto Velho": (-8.76, -63.90),
    "Rio Branco": (-9.97, -67.81),
    "Boa Vista": (2.82, -60.67),
    "Macapá": (0.04, -51.07),
    "Palmas": (-10.18, -48.33),
    "Brasília": (-15.79, -47.88),
    "Goiânia": (-16.67, -49.27),
    "Campo Grande": (-20.45, -54.62),
    "Cuiabá": (-15.60, -56.10),
}


def make_panel(
    *,
    seed: int = 42,
    n_days: int = 90,
    treatment_day: int = 60,
    treated_city: str = "São Paulo",
    true_att_pct: float = 0.08,
    pre_fit_difficulty: float = 1.0,
    treated_trend_boost: float = 0.0,
) -> pl.DataFrame:
    """Generate a synthetic geo-experiment panel.

    Returns a long polars DataFrame with columns:
        day, city, region, sales, is_treated_unit, post, treat

    The DGP per city `i` per day `t` is:
        sales[i,t] = baseline[i] * (1 + trend[i]*t + seasonal[t]
                                    + region_shock[g(i),t] + idio[i,t])
                     + treated_trend_boost*baseline[treated]*t * 1{i=treated}
                     + ATT * 1{i=treated, t>=treatment_day}

    `pre_fit_difficulty` > 1 inflates idio noise → harder pre-period fit.

    `treated_trend_boost` > 0 adds an extra trend slope to the treated unit only,
    pushing it OUTSIDE the convex hull of donor trajectories. With donor trends
    drawn from U(0.0005, 0.0015), a boost of 0.0025 reliably places the treated
    unit beyond any convex combination of donors — exactly the failure mode of
    classical SCM that AugSynth (Ben-Michael et al. 2021) is designed to fix.
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)

    seasonal_pct = 0.04 * np.sin(2 * np.pi * days / 7)

    regions = sorted({r for _, r, _ in CITIES_BR})
    region_shocks: dict[str, np.ndarray] = {}
    for region in regions:
        s = np.zeros(n_days)
        for t in range(1, n_days):
            s[t] = 0.7 * s[t - 1] + rng.normal(0, 0.01)
        region_shocks[region] = s

    treated_baseline = next(b for c, _, b in CITIES_BR if c == treated_city)
    rows: list[dict[str, Any]] = []
    for city, region, baseline in CITIES_BR:
        trend = rng.uniform(0.0005, 0.0015)
        idio = rng.normal(0, 0.025 * pre_fit_difficulty, size=n_days)
        sales_pct = 1.0 + trend * days + seasonal_pct + region_shocks[region] + idio
        sales = baseline * sales_pct

        if city == treated_city:
            if treated_trend_boost != 0.0:
                sales = sales + treated_trend_boost * treated_baseline * days
            te = np.zeros(n_days)
            te[treatment_day:] = true_att_pct * treated_baseline
            sales = sales + te

        for t in range(n_days):
            rows.append(
                dict(
                    day=int(t),
                    city=city,
                    region=region,
                    sales=float(sales[t]),
                    is_treated_unit=(city == treated_city),
                    post=(t >= treatment_day),
                )
            )

    return pl.DataFrame(rows).with_columns(
        treat=(pl.col("is_treated_unit") & pl.col("post")).cast(pl.Int32),
    )


# ---------------------------------------------------------------------------
# Naive DiD — pure Python
# ---------------------------------------------------------------------------


def naive_did(
    panel: pl.DataFrame,
    treated_city: str = "São Paulo",
    treatment_day: int = 60,
) -> dict[str, float]:
    """Two-by-two DiD: (treated_post − treated_pre) − (donors_post − donors_pre)."""
    pre = pl.col("day") < treatment_day
    post = pl.col("day") >= treatment_day
    is_treated = pl.col("city") == treated_city

    sp_pre = panel.filter(is_treated & pre)["sales"].mean()
    sp_post = panel.filter(is_treated & post)["sales"].mean()
    don_pre = panel.filter(~is_treated & pre)["sales"].mean()
    don_post = panel.filter(~is_treated & post)["sales"].mean()

    att_abs = (sp_post - sp_pre) - (don_post - don_pre)
    return dict(
        att_abs=float(att_abs),
        att_pct=float(att_abs / sp_pre),
        sp_pre=float(sp_pre),
        sp_post=float(sp_post),
        don_pre=float(don_pre),
        don_post=float(don_post),
    )


# ---------------------------------------------------------------------------
# R wrappers
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Plain-Python container for an SCM/AugSynth fit. No R objects inside."""

    method: str  # "SCM" or "AugSynth"
    treated_city: str
    treatment_day: int
    weights: dict[str, float]  # donor city -> weight
    synthetic: np.ndarray  # length n_days
    actual: np.ndarray  # length n_days
    gap: np.ndarray  # actual − synthetic
    att_avg: float  # mean gap in post period (absolute units)
    att_avg_pct: float  # att_avg / mean(actual_pre)
    rmspe_pre: float  # pre-period RMSPE / mean(actual_pre)
    att_by_period: np.ndarray | None = None  # ATT trajectory (post period only) if available
    att_periods: np.ndarray | None = None  # corresponding period indices for att_by_period
    ci_lower: np.ndarray | None = None  # lower CI bound (post period) if available
    ci_upper: np.ndarray | None = None  # upper CI bound (post period) if available
    chosen_lambda: float | None = None  # ridge λ chosen by CV (AugSynth only)
    lambda_path: np.ndarray | None = None  # shape (n_grid, 2): [lambda, cv_loss]


def _fit_via_augsynth(
    panel: pl.DataFrame,
    treated_city: str,
    treatment_day: int,
    *,
    progfunc: str,
    method_label: str,
    fixedeff: bool = True,
    inference: str | None = None,
    unit: str = "city",
    time: str = "day",
    outcome: str = "sales",
) -> FitResult:
    r = _r()
    pdf = panel.select(time, unit, outcome, "treat").to_pandas()

    with (r.ro.default_converter + r.pandas2ri.converter).context():
        rdf = r.ro.conversion.get_conversion().py2rpy(pdf)

    r.ro.globalenv["panel_r"] = rdf
    extra = f", inference='{inference}'" if inference else ""
    fe_arg = "TRUE" if fixedeff else "FALSE"
    r.ro.r(
        f"res <- augsynth::augsynth({outcome} ~ treat, unit = {unit}, time = {time}, "
        f"data = panel_r, progfunc = '{progfunc}', scm = TRUE, fixedeff = {fe_arg}{extra})"
    )

    # weights: named single-column matrix, donors as rownames
    w_arr = np.asarray(r.ro.r("res$weights"), dtype=float).ravel()
    w_names = list(r.ro.r("rownames(res$weights)"))
    weights = {name: float(w) for name, w in zip(w_names, w_arr, strict=True)}

    synthetic = np.asarray(r.ro.r("predict(res, att = FALSE)"), dtype=float)

    actual = panel.filter(pl.col(unit) == treated_city).sort(time)[outcome].to_numpy().astype(float)
    gap = actual - synthetic

    pre_actual_mean = float(np.mean(actual[:treatment_day]))
    rmspe_pre = float(np.sqrt(np.mean(gap[:treatment_day] ** 2)) / pre_actual_mean)
    att_avg = float(np.mean(gap[treatment_day:]))
    att_avg_pct = att_avg / pre_actual_mean

    att_by_period: np.ndarray | None = None
    att_periods: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    try:
        att_df_r = r.ro.r("as.data.frame(summary(res)$att)")
        with (r.ro.default_converter + r.pandas2ri.converter).context():
            att_df = r.ro.conversion.get_conversion().rpy2py(att_df_r)
        if "Estimate" in att_df.columns:
            att_by_period = att_df["Estimate"].to_numpy(dtype=float)
        if "Time" in att_df.columns:
            att_periods = att_df["Time"].to_numpy(dtype=float)
        if "lower_bound" in att_df.columns:
            ci_lower = att_df["lower_bound"].to_numpy(dtype=float)
        if "upper_bound" in att_df.columns:
            ci_upper = att_df["upper_bound"].to_numpy(dtype=float)
    except Exception:
        pass

    chosen_lambda: float | None = None
    lambda_path: np.ndarray | None = None
    if progfunc.lower() == "ridge":
        try:
            chosen_lambda = float(np.asarray(r.ro.r("res$lambda"), dtype=float).ravel()[0])
        except Exception:
            chosen_lambda = None
        try:
            lambdas = np.asarray(r.ro.r("res$lambdas"), dtype=float).ravel()
            errors = np.asarray(r.ro.r("res$lambda_errors"), dtype=float).ravel()
            if lambdas.size and errors.size == lambdas.size:
                lambda_path = np.column_stack([lambdas, errors])
        except Exception:
            lambda_path = None

    return FitResult(
        method=method_label,
        treated_city=treated_city,
        treatment_day=treatment_day,
        weights=weights,
        synthetic=synthetic,
        actual=actual,
        gap=gap,
        att_avg=att_avg,
        att_avg_pct=att_avg_pct,
        rmspe_pre=rmspe_pre,
        att_by_period=att_by_period,
        att_periods=att_periods,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        chosen_lambda=chosen_lambda,
        lambda_path=lambda_path,
    )


def fit_scm_r(
    panel: pl.DataFrame,
    treated_city: str = "São Paulo",
    treatment_day: int = 60,
    *,
    fixedeff: bool = True,
    unit: str = "city",
    time: str = "day",
    outcome: str = "sales",
) -> FitResult:
    """Classical synthetic control via `augsynth(progfunc='None', scm=TRUE)`.

    `fixedeff=True` adds unit fixed effects (de-means each unit's outcome before
    weighting). This is the recommended default with heterogeneous baselines —
    set to `False` to see the raw-level SCM failure mode.
    """
    return _fit_via_augsynth(
        panel,
        treated_city,
        treatment_day,
        progfunc="None",
        method_label="SCM",
        fixedeff=fixedeff,
        unit=unit,
        time=time,
        outcome=outcome,
    )


def fit_augsynth_r(
    panel: pl.DataFrame,
    treated_city: str = "São Paulo",
    treatment_day: int = 60,
    *,
    fixedeff: bool = True,
    conformal: bool = False,
    unit: str = "city",
    time: str = "day",
    outcome: str = "sales",
) -> FitResult:
    """Ridge-augmented SCM via `augsynth(progfunc='Ridge', scm=TRUE)`.

    Set `conformal=True` to attach period-wise conformal CIs (Chernozhukov,
    Wuthrich & Zhu 2021), available natively in `augsynth` via
    `inference='conformal'`.
    """
    return _fit_via_augsynth(
        panel,
        treated_city,
        treatment_day,
        progfunc="Ridge",
        method_label="AugSynth",
        fixedeff=fixedeff,
        inference="conformal" if conformal else None,
        unit=unit,
        time=time,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Placebo permutation
# ---------------------------------------------------------------------------


def placebo_test(
    panel: pl.DataFrame,
    treated_city: str = "São Paulo",
    treatment_day: int = 60,
    fitter: Callable[..., FitResult] = fit_scm_r,
    rmspe_threshold_mult: float = 5.0,
) -> dict[str, Any]:
    """Permutation inference: re-fit treating each donor as if treated.

    Following Abadie's convention, placebos with `rmspe_pre` greater than
    `rmspe_threshold_mult` × the treated unit's `rmspe_pre` are filtered out
    (a placebo with bad pre-fit can't speak to the post-period gap).

    Returns dict with: treated, placebos, p_value, threshold.
    """
    treated_fit = fitter(panel, treated_city=treated_city, treatment_day=treatment_day)
    threshold = treated_fit.rmspe_pre * rmspe_threshold_mult

    placebos: dict[str, FitResult] = {}
    donor_cities = [c for c, _, _ in CITIES_BR if c != treated_city]
    for donor in donor_cities:
        re_panel = panel.with_columns(
            is_treated_unit=(pl.col("city") == donor),
            treat=((pl.col("city") == donor) & pl.col("post")).cast(pl.Int32),
        )
        try:
            placebos[donor] = fitter(re_panel, treated_city=donor, treatment_day=treatment_day)
        except Exception:
            continue

    kept_placebos = {k: v for k, v in placebos.items() if v.rmspe_pre <= threshold}
    treated_abs = abs(treated_fit.att_avg)
    n_extreme = sum(1 for v in kept_placebos.values() if abs(v.att_avg) >= treated_abs)
    p_value = (n_extreme + 1) / (len(kept_placebos) + 1)

    return dict(
        treated=treated_fit,
        placebos=kept_placebos,
        all_placebos=placebos,
        threshold=threshold,
        p_value=p_value,
    )
