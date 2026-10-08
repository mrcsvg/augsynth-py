"""Robustness checks for a fitted synthetic control.

Two refit-based diagnostics from the synthetic-control literature, computed on
an already-fitted :class:`~augsynth_py.Synth` or :class:`~augsynth_py.AugSynth`:

* :func:`placebo_in_space` — Abadie, Diamond & Hainmueller (2010, §III.B):
  reassign treatment to each donor in turn, refit, and place the treated
  unit's post/pre RMSPE ratio in the placebo distribution (Abadie 2021, §5).
* :func:`leave_one_out` — Abadie, Diamond & Hainmueller (2015, §4.4) and
  Abadie (2021, §6): drop each contributing donor in turn and refit, reporting
  the range of the ATT and the donor whose removal moves it most.

Both rebuild the panel from the fitted object's stored outcome matrix and
refit a ``copy.deepcopy`` of the estimator, so its configuration (``fixedeff``,
the ridge-penalty policy) carries over unchanged: an ``AugSynth`` fitted with a
cross-validated penalty re-runs the CV in every refit; one fitted with a fixed
``lambda_`` keeps it. Nothing here touches the solver or the CV.

Conventions
-----------
* The placebo statistic is the ratio of post- to pre-treatment **RMSPE** on
  the outcome scale, as in ADH 2010 Figure 7. R ``SCtools::mspe.test`` reports
  the ratio of **MSPE**; one is the square of the other, so rank and p-value
  coincide.
* The p-value counts the treated unit in the denominator —
  ``#{ratio >= treated ratio} / (J_kept + 1)``, the treated unit's own ratio
  included (Abadie 2021 §5; SCtools ``mspe.test``). It is a rank statistic
  with floor ``1 / (J_kept + 1)``.
* ``mspe_limit`` reproduces the ADH 2010 practice of discarding placebos
  whose pre-treatment MSPE is at least ``mspe_limit`` times the treated
  unit's (ADH use 20, 5 and 2; SCtools uses ``>=`` with default 20). Off by
  default: Abadie (2021) argues the ratio statistic already penalises a poor
  pre-period fit.
* Refit diagnostics report donors by the string labels in ``fit.units_``
  (``long_to_wide`` stringifies unit values), so an integer unit ``7`` appears
  as ``"7"`` and a treated group as its comma-joined label.

References
----------
Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods
for Comparative Case Studies. *JASA*, 105(490), 493-505.

Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative Politics and
the Synthetic Control Method. *American Journal of Political Science*, 59(2),
495-510.

Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements,
and Methodological Aspects. *Journal of Economic Literature*, 59(2), 391-425.

R ``SCtools`` (``generate.placebos``, ``mspe.test``) is the parity oracle for
the aggregation arithmetic only; the refits themselves are validated against
R ``augsynth`` (``tests/validation_against_r/test_robustness.py``).
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray

from augsynth_py.synth._panel import periods_to_series, wide_to_long

logger = logging.getLogger(__name__)

OnError = Literal["raise", "record"]
DonorSelection = Literal["nonzero", "all"]

# Column names of the panel rebuilt for every refit.
_UNIT, _TIME, _OUTCOME = "unit", "time", "outcome"
# Substring identifying AugSynth's CV-at-grid-edge warning (see
# synth/augmented.py::_warn_if_lambda_at_boundary). Those are aggregated into
# one warning per call instead of one per refit.
_LAMBDA_BOUNDARY_MARKER = "boundary of the grid"


class _RefittableSC(Protocol):
    """Fitted estimator exposing enough state to be refit on a modified panel.

    Both :class:`~augsynth_py.Synth` and :class:`~augsynth_py.AugSynth`
    satisfy this after ``.fit(...)``. The two underscore attributes
    (``_y_matrix``, ``_treated_idx``) are the scaffolding those classes keep
    for the CWZ refit-under-null; this module documents them as the
    refit-diagnostics extension point in the same way
    ``conformal_null_residuals`` is the conformal one. A third-party estimator
    that wants these diagnostics exposes the same attributes and ``fit``
    signature.
    """

    units_: list[str]
    periods_: NDArray[Any]
    actual_: NDArray[np.float64]
    synthetic_: NDArray[np.float64]
    pre_mask_: NDArray[np.bool_]
    weights_: dict[str, float]
    att_: float
    rmspe_pre_: float
    treatment_time_: Any
    _y_matrix: NDArray[np.float64]
    _treated_idx: int

    def fit(
        self,
        panel: pl.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        treated: Any,
        treatment_time: Any,
    ) -> Any: ...


_REQUIRED_ATTRS: tuple[str, ...] = (
    "units_",
    "periods_",
    "actual_",
    "synthetic_",
    "pre_mask_",
    "weights_",
    "att_",
    "rmspe_pre_",
    "treatment_time_",
    "_y_matrix",
    "_treated_idx",
    "fit",
)


@dataclass(frozen=True)
class LeaveOneOutResult:
    """Output of :func:`leave_one_out`.

    Attributes
    ----------
    att_full : float
        ATT of the fit that was passed in (all donors).
    att : pl.DataFrame
        One row per dropped donor, in the order they were dropped. Columns:

        - ``dropped_unit`` (str): donor removed from the pool.
        - ``weight`` (f64): its weight in the full fit (``fit.weights_``).
        - ``att`` (f64): ATT of the refit without it; null on error.
        - ``delta_att`` (f64): ``att - att_full``; null on error.
        - ``rmspe_pre`` (f64): normalized pre-period RMSPE of the refit, same
          definition as the estimator's ``rmspe_pre_``; null on error.
        - ``lambda_`` (f64): ridge penalty the refit used (AugSynth); null
          for estimators without one, or on error.
        - ``error`` (str): ``"ExcType: message"`` under
          ``on_error="record"``; null otherwise.
    att_range : tuple[float, float]
        ``(min, max)`` of ``att`` over successful refits with a finite ATT —
        the leave-one-out envelope of the estimate. ``(nan, nan)`` if no
        refit qualifies.
    most_influential : str or None
        Dropped donor with the largest ``|delta_att|`` among successful refits
        with a finite ATT; ties go to the donor dropped first. None if no
        refit qualifies.
    sign_stable : bool
        True when ``att_full`` is nonzero and every successful refit with a
        finite ATT has the same sign; False when no refit succeeded.
    paths : pl.DataFrame
        Long frame ``dropped_unit, period, synthetic`` with the counterfactual
        path of every successful refit, for plotting against ``fit.actual_``.
    lambda_at_boundary : tuple[str, ...]
        Dropped donors whose refit selected a CV penalty at the edge of the
        grid (AugSynth diagnostic); empty for other estimators.
    """

    att_full: float
    att: pl.DataFrame
    att_range: tuple[float, float]
    most_influential: str | None
    sign_stable: bool
    paths: pl.DataFrame
    lambda_at_boundary: tuple[str, ...]


@dataclass(frozen=True)
class PlaceboInSpaceResult:
    """Output of :func:`placebo_in_space`.

    Attributes
    ----------
    statistic : float
        The treated unit's post/pre RMSPE ratio (ADH 2010, Fig. 7).
    pvalue : float
        Share of kept units (placebos plus the treated unit itself) whose
        ratio is at least ``statistic``: ``#{ratio >= statistic} /
        (J_kept + 1)``. Its floor is ``1 / (J_kept + 1)``. Placebos with an
        undefined ratio (0/0) are excluded from the rank and the p-value;
        ``nan`` when the treated unit's own ratio is undefined.
    ratios : pl.DataFrame
        One row per unit — the treated unit first, then donors in
        ``fit.units_`` order. Columns:

        - ``unit`` (str), ``is_treated`` (bool).
        - ``rmspe_pre`` / ``rmspe_post`` (f64): raw RMSPE on the outcome
          scale over the pre / post periods; null on error.
        - ``ratio`` (f64): ``rmspe_post / rmspe_pre`` (``inf`` for a perfect
          pre-fit with post error, ``nan`` for 0/0); null on error.
        - ``rank`` (i64): 1 for the largest ratio among kept units, ties share
          the smaller rank; null for discarded or errored rows and for
          placebos with an undefined ratio (0/0), which are excluded from the
          rank and the p-value. An ``inf`` ratio is kept and ranks first.
        - ``discarded`` (bool): dropped by ``mspe_limit``.
        - ``lambda_`` (f64): ridge penalty the refit used (AugSynth), or the
          original fit's penalty on the treated row; null otherwise.
        - ``error`` (str): ``"ExcType: message"`` under
          ``on_error="record"``; null otherwise.
    gaps : pl.DataFrame
        Long frame ``unit, period, gap`` (actual minus synthetic) for the
        treated unit and every successful placebo — the ADH "spaghetti" plot
        data.
    discarded : tuple[str, ...]
        Donors excluded by ``mspe_limit``.
    mspe_limit : float or None
        The limit that was applied.
    lambda_at_boundary : tuple[str, ...]
        Placebo units whose refit selected a CV penalty at the grid edge.
    """

    statistic: float
    pvalue: float
    ratios: pl.DataFrame
    gaps: pl.DataFrame
    discarded: tuple[str, ...]
    mspe_limit: float | None
    lambda_at_boundary: tuple[str, ...]


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


def _check_refittable(fit: Any) -> None:
    missing = [a for a in _REQUIRED_ATTRS if not hasattr(fit, a)]
    if missing:
        raise TypeError(
            "Robustness checks need a fitted Synth/AugSynth (or an estimator implementing "
            f"the refit contract documented in augsynth_py.robustness); missing: {missing}. "
            "Did you call .fit()?"
        )


def _check_on_error(on_error: str) -> bool:
    if on_error not in ("raise", "record"):
        raise ValueError(f"on_error must be 'raise' or 'record'; got {on_error!r}.")
    return on_error == "record"


def _split_units(fit: _RefittableSC) -> tuple[list[str], int, str, list[int], list[str]]:
    """Return (units, treated_idx, treated_label, donor_col_idx, donor_labels)."""
    units = [str(u) for u in fit.units_]
    t_idx = int(fit._treated_idx)
    donor_idx = [i for i in range(len(units)) if i != t_idx]
    return units, t_idx, units[t_idx], donor_idx, [units[i] for i in donor_idx]


def _rmspe(
    actual: NDArray[np.float64], synthetic: NDArray[np.float64], mask: NDArray[np.bool_]
) -> float:
    """Raw (unnormalized) RMSPE over the masked periods."""
    diff = (
        np.asarray(actual, dtype=np.float64)[mask] - np.asarray(synthetic, dtype=np.float64)[mask]
    )
    return float(np.sqrt(np.mean(diff**2)))


def _ratio(pre: float, post: float) -> float:
    """Post / pre with the IEEE conventions for a zero pre-period error."""
    if pre > 0.0:
        return post / pre
    return float("inf") if post > 0.0 else float("nan")


def _lambda_of(est: Any) -> float | None:
    """Ridge penalty of a fitted estimator as a float; None if it has none (Synth)."""
    lam = getattr(est, "lambda_", None)
    return None if lam is None else float(lam)


def _refit(
    prototype: _RefittableSC,
    panel: pl.DataFrame,
    treated: str,
    treatment_time: Any,
    record_errors: bool,
) -> tuple[Any | None, list[str], str | None]:
    """Fit a deep copy of ``prototype`` on ``panel``; capture its warnings.

    Module-level so joblib can pickle it. Returns ``(fitted, warning_messages,
    error)``: on failure under ``record_errors`` ``fitted`` is None and
    ``error`` is ``"ExcType: message"``; otherwise the exception propagates.
    Warnings are captured rather than emitted because worker processes cannot
    surface them to the caller; :func:`_surface_warnings` re-emits them once.
    """
    est = copy.deepcopy(prototype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted = est.fit(
                panel,
                unit=_UNIT,
                time=_TIME,
                outcome=_OUTCOME,
                treated=treated,
                treatment_time=treatment_time,
            )
        # Broad on purpose: any solver / CV / validation failure in this refit
        # either propagates (on_error="raise") or becomes an error row.
        except Exception as exc:
            if not record_errors:
                raise
            return None, [str(w.message) for w in caught], f"{type(exc).__name__}: {exc}"
    return fitted, [str(w.message) for w in caught], None


def _run_all(
    n_jobs: int,
    prototype: _RefittableSC,
    jobs: list[tuple[pl.DataFrame, str]],
    treatment_time: Any,
    record_errors: bool,
) -> list[tuple[Any | None, list[str], str | None]]:
    logger.info("robustness: %d refits, n_jobs=%d", len(jobs), n_jobs)
    if n_jobs == 1:
        return [_refit(prototype, p, t, treatment_time, record_errors) for p, t in jobs]
    out: list[tuple[Any | None, list[str], str | None]] = list(
        Parallel(n_jobs=n_jobs)(
            delayed(_refit)(prototype, p, t, treatment_time, record_errors) for p, t in jobs
        )
    )
    return out


def _surface_warnings(labels: list[str], messages: list[list[str]]) -> tuple[str, ...]:
    """Re-emit captured refit warnings once per call; return λ-boundary labels.

    AugSynth's "CV-selected lambda is at the boundary" warning is collapsed
    into one summary naming the affected refits. Any other warning text is
    re-emitted once, suffixed with the refits that raised it.
    """
    at_boundary: list[str] = []
    other: dict[str, list[str]] = {}
    for label, msgs in zip(labels, messages, strict=True):
        for m in msgs:
            if _LAMBDA_BOUNDARY_MARKER in m:
                if label not in at_boundary:
                    at_boundary.append(label)
            else:
                other.setdefault(m, []).append(label)
    if at_boundary:
        warnings.warn(
            f"{len(at_boundary)} of {len(labels)} refits selected a lambda at the boundary of "
            f"the CV grid (refits: {', '.join(at_boundary)}). The result's "
            "`lambda_at_boundary` lists them; widen `lambda_grid` on the estimator if it matters.",
            UserWarning,
            stacklevel=3,
        )
    for m, labs in other.items():
        warnings.warn(
            f"{m} (raised in {len(labs)} refit(s): {', '.join(labs)})",
            UserWarning,
            stacklevel=3,
        )
    return tuple(at_boundary)


def _tiled_periods(fit: _RefittableSC, n_copies: int) -> pl.Series:
    """``fit.periods_`` repeated ``n_copies`` times, dtype preserved (see periods_to_series)."""
    n_periods = len(fit.periods_)
    return periods_to_series(fit.periods_, "period").gather(np.tile(np.arange(n_periods), n_copies))


# ---------------------------------------------------------------------------
# leave_one_out
# ---------------------------------------------------------------------------


def leave_one_out(
    fit: _RefittableSC,
    *,
    donors: DonorSelection | Sequence[Any] = "nonzero",
    tol: float = 1e-6,
    n_jobs: int = 1,
    on_error: OnError = "raise",
) -> LeaveOneOutResult:
    """Refit without each donor in turn and report how the ATT moves.

    The leave-one-out robustness check of Abadie, Diamond & Hainmueller
    (2015, §4.4), recommended in Abadie (2021, §6): if the estimate hinges on
    a single donor, dropping that donor exposes it. The result names the donor
    whose removal moves the ATT most, so a contaminated donor (a shock in the
    pool that is not in the treated unit) is identified rather than averaged
    into a "range".

    Parameters
    ----------
    fit
        A fitted :class:`~augsynth_py.Synth` or :class:`~augsynth_py.AugSynth`.
        Never modified; every refit works on a deep copy, so the estimator's
        configuration — including whether the ridge penalty is cross-validated
        or fixed — carries over unchanged.
    donors : {"nonzero", "all"} or sequence, optional
        Which donors to drop. ``"nonzero"`` (default) drops each donor whose
        weight in ``fit`` satisfies ``|w| > tol`` — for ``AugSynth`` that
        includes negative weights; ``"all"`` drops every donor; a sequence
        names the donors explicitly (compared as strings against
        ``fit.units_``).
    tol : float, optional
        Weight magnitude below which a donor counts as unused for
        ``donors="nonzero"``. Clarabel at its default tolerances leaves
        ~1e-7 on weights that should be zero, hence the default; 1e-6 is
        negligible against weights that sum to 1.
    n_jobs : int, optional
        Parallel workers for the refits (:class:`joblib.Parallel`; ``-1`` uses
        all cores). Defaults to 1.
    on_error : {"raise", "record"}, optional
        ``"raise"`` (default) propagates the first refit failure.
        ``"record"`` keeps going and stores ``"ExcType: message"`` in the
        ``error`` column of the failed row, with null estimates.

    Returns
    -------
    LeaveOneOutResult

    Raises
    ------
    TypeError
        If ``fit`` does not expose the refit contract (e.g. it is unfitted).
    ValueError
        For an unknown ``donors`` mode, names that are not donors of ``fit``,
        an empty selection, a negative ``tol`` or a bad ``on_error``.
    EmptyDonorPoolError
        Propagated from the refit when dropping a donor leaves no pool (a
        single-donor fit); recorded per row under ``on_error="record"``.

    Notes
    -----
    Refits are compared on the ATT. With ``fixedeff=True`` and donors that
    share a trend, several weight vectors fit the pre-period equally well and
    the solver's choice among them is not meaningful; the ATT, which is what
    this diagnostic reads, is unique up to the pre-period fit.

    **Parallelism and warnings.** Refits run in :class:`joblib.Parallel`
    worker processes, each capturing its own warnings, which this function
    re-emits once. The capture uses the process-global warnings filter, so
    ``n_jobs > 1`` under a thread-based joblib backend, or concurrent calls
    from several threads, can misattribute or drop warnings; attribution is
    guaranteed only under the default process backend. joblib does not nest
    workers: a call made from inside a worker runs its refits sequentially,
    so parallelize at the caller level or here, not both.
    """
    _check_refittable(fit)
    record = _check_on_error(on_error)
    if tol < 0:
        raise ValueError(f"tol must be >= 0; got {tol}.")
    units, _t_idx, treated_label, _donor_idx, all_donors = _split_units(fit)

    if isinstance(donors, str):
        if donors == "nonzero":
            selected = [d for d in all_donors if abs(float(fit.weights_[d])) > tol]
            if not selected:
                raise ValueError(
                    f"No donor has |weight| > tol={tol:g}; nothing to leave out. "
                    "Use donors='all' or a lower tol."
                )
        elif donors == "all":
            selected = list(all_donors)
        else:
            raise ValueError(
                f"donors must be 'nonzero', 'all' or a sequence of donor labels; got {donors!r}."
            )
    else:
        selected = list(dict.fromkeys(str(d) for d in donors))  # dedupe, keep order
        if not selected:
            raise ValueError("donors must name at least one donor.")
        unknown = [d for d in selected if d not in all_donors]
        if unknown:
            raise ValueError(f"{unknown!r} are not donors of this fit. Donors: {all_donors}.")

    y = np.asarray(fit._y_matrix, dtype=np.float64)
    jobs: list[tuple[pl.DataFrame, str]] = []
    for d in selected:
        keep = [i for i, u in enumerate(units) if u != d]
        jobs.append(
            (wide_to_long(y[:, keep], [units[i] for i in keep], fit.periods_), treated_label)
        )
    runs = _run_all(n_jobs, fit, jobs, fit.treatment_time_, record)

    att_full = float(fit.att_)
    rows: list[dict[str, Any]] = []
    path_units: list[str] = []
    path_values: list[NDArray[np.float64]] = []
    for d, (fitted, _msgs, err) in zip(selected, runs, strict=True):
        w = float(fit.weights_[d])
        if fitted is None:
            rows.append(
                {
                    "dropped_unit": d,
                    "weight": w,
                    "att": None,
                    "delta_att": None,
                    "rmspe_pre": None,
                    "lambda_": None,
                    "error": err,
                }
            )
            continue
        att = float(fitted.att_)
        rows.append(
            {
                "dropped_unit": d,
                "weight": w,
                "att": att,
                "delta_att": att - att_full,
                "rmspe_pre": float(fitted.rmspe_pre_),
                "lambda_": _lambda_of(fitted),
                "error": None,
            }
        )
        path_units.append(d)
        path_values.append(np.asarray(fitted.synthetic_, dtype=np.float64))

    att_frame = pl.DataFrame(
        rows,
        schema={
            "dropped_unit": pl.String,
            "weight": pl.Float64,
            "att": pl.Float64,
            "delta_att": pl.Float64,
            "rmspe_pre": pl.Float64,
            "lambda_": pl.Float64,
            "error": pl.String,
        },
    )
    ok = att_frame.filter(pl.col("error").is_null() & pl.col("att").is_finite())
    if ok.height > 0:
        atts = ok["att"].to_numpy()
        att_range = (float(atts.min()), float(atts.max()))
        most = str(
            ok.sort(pl.col("delta_att").abs(), descending=True, maintain_order=True)[
                "dropped_unit"
            ][0]
        )
        sign_stable = bool(att_full != 0.0 and np.all(np.sign(atts) == np.sign(att_full)))
    else:
        att_range = (float("nan"), float("nan"))
        most = None
        sign_stable = False

    n_periods = len(fit.periods_)
    paths = pl.DataFrame(
        {
            "dropped_unit": [d for d in path_units for _ in range(n_periods)],
            "period": _tiled_periods(fit, len(path_units)),
            "synthetic": np.concatenate(path_values) if path_values else np.zeros(0),
        },
        schema_overrides={"dropped_unit": pl.String, "synthetic": pl.Float64},
    )
    at_boundary = _surface_warnings(selected, [m for _f, m, _e in runs])
    return LeaveOneOutResult(
        att_full=att_full,
        att=att_frame,
        att_range=att_range,
        most_influential=most,
        sign_stable=sign_stable,
        paths=paths,
        lambda_at_boundary=at_boundary,
    )


# ---------------------------------------------------------------------------
# placebo_in_space
# ---------------------------------------------------------------------------


def placebo_in_space(
    fit: _RefittableSC,
    *,
    mspe_limit: float | None = None,
    n_jobs: int = 1,
    on_error: OnError = "raise",
) -> PlaceboInSpaceResult:
    """In-space placebo test of whether the treated unit's gap is unusual among donors.

    Reassigns treatment to each donor in turn (with the real treated unit
    removed from the pool), refits, and compares the treated unit's post/pre
    RMSPE ratio with the placebo ratios (Abadie, Diamond & Hainmueller 2010,
    §III.B; Abadie 2021, §5). The ratio statistic rewards a large
    post-treatment gap only relative to the pre-treatment fit, so a placebo
    with a poor pre-fit does not look like a large effect.

    Parameters
    ----------
    fit
        A fitted :class:`~augsynth_py.Synth` or :class:`~augsynth_py.AugSynth`.
        Never modified; every refit works on a deep copy (see
        :func:`leave_one_out` for the configuration carry-over).
    mspe_limit : float, optional
        If given, placebos whose pre-period MSPE is at least ``mspe_limit``
        times the treated unit's are discarded before ranking (ADH 2010 use
        20, 5, 2; R ``SCtools`` uses ``>=`` with default 20). None (default)
        keeps every placebo, following Abadie (2021).
    n_jobs : int, optional
        Parallel workers for the refits (:class:`joblib.Parallel`).
    on_error : {"raise", "record"}, optional
        See :func:`leave_one_out`. Errored placebos are excluded from the
        rank and the p-value.

    Returns
    -------
    PlaceboInSpaceResult

    Raises
    ------
    TypeError
        If ``fit`` does not expose the refit contract.
    ValueError
        If the fit has fewer than two donors (no pool is left for a placebo),
        ``mspe_limit`` is not positive, or ``on_error`` is invalid.

    Notes
    -----
    The p-value is a rank statistic with a floor of ``1 / (J_kept + 1)``:
    with 9 donors the smallest attainable value is 0.1. It is an exact
    permutation-style quantity under the hypothesis that treatment was
    assigned at random among the units, and is reported as such — see
    ``docs/methodology.md`` §7.
    """
    _check_refittable(fit)
    record = _check_on_error(on_error)
    if mspe_limit is not None and not mspe_limit > 0:
        raise ValueError(f"mspe_limit must be positive; got {mspe_limit!r}.")
    _units, _t_idx, treated_label, donor_idx, donors = _split_units(fit)
    if len(donors) < 2:
        raise ValueError(
            "placebo_in_space needs at least two donors: with a single donor there is no pool "
            "left to build a placebo from once that donor is the pseudo-treated unit."
        )

    y = np.asarray(fit._y_matrix, dtype=np.float64)
    pool = wide_to_long(y[:, donor_idx], donors, fit.periods_)  # real treated unit removed
    runs = _run_all(n_jobs, fit, [(pool, d) for d in donors], fit.treatment_time_, record)

    pre = np.asarray(fit.pre_mask_, dtype=bool)
    t_pre = _rmspe(fit.actual_, fit.synthetic_, pre)
    t_post = _rmspe(fit.actual_, fit.synthetic_, ~pre)
    statistic = _ratio(t_pre, t_post)

    rows: list[dict[str, Any]] = [
        {
            "unit": treated_label,
            "is_treated": True,
            "rmspe_pre": t_pre,
            "rmspe_post": t_post,
            "ratio": statistic,
            "lambda_": _lambda_of(fit),
            "error": None,
        }
    ]
    gap_units: list[str] = [treated_label]
    gap_values: list[NDArray[np.float64]] = [
        np.asarray(fit.actual_, dtype=np.float64) - np.asarray(fit.synthetic_, dtype=np.float64)
    ]
    for d, (fitted, _msgs, err) in zip(donors, runs, strict=True):
        if fitted is None:
            rows.append(
                {
                    "unit": d,
                    "is_treated": False,
                    "rmspe_pre": None,
                    "rmspe_post": None,
                    "ratio": None,
                    "lambda_": None,
                    "error": err,
                }
            )
            continue
        p_mask = np.asarray(fitted.pre_mask_, dtype=bool)
        p_pre = _rmspe(fitted.actual_, fitted.synthetic_, p_mask)
        p_post = _rmspe(fitted.actual_, fitted.synthetic_, ~p_mask)
        rows.append(
            {
                "unit": d,
                "is_treated": False,
                "rmspe_pre": p_pre,
                "rmspe_post": p_post,
                "ratio": _ratio(p_pre, p_post),
                "lambda_": _lambda_of(fitted),
                "error": None,
            }
        )
        gap_units.append(d)
        gap_values.append(
            np.asarray(fitted.actual_, dtype=np.float64)
            - np.asarray(fitted.synthetic_, dtype=np.float64)
        )

    # Discard rule (ADH 2010 / SCtools): (pre_j / pre_treated)^2 >= mspe_limit.
    # Never applies to the treated row or to errored rows. _ratio's conventions
    # carry over: a zero treated pre-RMSPE (inf) discards, 0/0 (nan) does not.
    for r in rows:
        r["discarded"] = bool(
            not r["is_treated"]
            and r["error"] is None
            and mspe_limit is not None
            and _ratio(t_pre, float(r["rmspe_pre"])) ** 2 >= mspe_limit
        )

    # The comparison set: successful, kept rows with a defined ratio. A 0/0
    # ratio (nan) compares False against everything, so it would rank first and
    # pad the denominator; it is left out instead. inf is the most extreme
    # ratio and stays in (inf >= inf is True).
    def _kept(r: dict[str, Any]) -> bool:
        return bool(r["error"] is None and not r["discarded"] and not np.isnan(r["ratio"]))

    kept_ratios = np.array([float(r["ratio"]) for r in rows if _kept(r)], dtype=np.float64)
    for r in rows:
        r["rank"] = 1 + int(np.sum(kept_ratios > float(r["ratio"]))) if _kept(r) else None

    if np.isnan(statistic):
        pvalue = float("nan")
        warnings.warn(
            f"The treated unit {treated_label!r} has zero pre- and post-period RMSPE, so the "
            "post/pre RMSPE ratio statistic is undefined (0/0); pvalue is nan and the treated "
            "rank is null.",
            UserWarning,
            stacklevel=2,
        )
    else:
        pvalue = float(np.sum(kept_ratios >= statistic) / len(kept_ratios))
    if not any(_kept(r) for r in rows if not r["is_treated"]):
        warnings.warn(
            "placebo_in_space: no placebo survived mspe_limit / refit errors / undefined "
            "ratios; the p-value is uninformative (the treated unit alone is in the "
            "comparison set).",
            UserWarning,
            stacklevel=2,
        )

    ratios = pl.DataFrame(
        rows,
        schema={
            "unit": pl.String,
            "is_treated": pl.Boolean,
            "rmspe_pre": pl.Float64,
            "rmspe_post": pl.Float64,
            "ratio": pl.Float64,
            "rank": pl.Int64,
            "discarded": pl.Boolean,
            "lambda_": pl.Float64,
            "error": pl.String,
        },
    )

    n_periods = len(fit.periods_)
    gaps = pl.DataFrame(
        {
            "unit": [u for u in gap_units for _ in range(n_periods)],
            "period": _tiled_periods(fit, len(gap_units)),
            "gap": np.concatenate(gap_values),
        },
        schema_overrides={"unit": pl.String, "gap": pl.Float64},
    )
    discarded = tuple(str(r["unit"]) for r in rows if r["discarded"])
    at_boundary = _surface_warnings(donors, [m for _f, m, _e in runs])
    return PlaceboInSpaceResult(
        statistic=statistic,
        pvalue=pvalue,
        ratios=ratios,
        gaps=gaps,
        discarded=discarded,
        mspe_limit=mspe_limit,
        lambda_at_boundary=at_boundary,
    )
