"""Simulation-based power analysis for synthetic-control designs.

Implements GeoLift-style power simulation for a *given* treated set: inject a
known effect into placebo-in-time treatment windows at the end of the observed
panel, refit the synthetic control on each simulated dataset, and record
whether conformal inference (CWZ 2021, see :mod:`augsynth_py.inference`)
detects the injected effect. The detection rate across simulations is the
empirical power; the minimum detectable effect (MDE) is the smallest injected
effect whose power reaches a chosen threshold.

Scope
-----
This module answers *what can this design detect?* for a treated set the
caller already fixed. Choosing a design — market selection over candidate
treated sets, budget/ROI, multi-cell — belongs to the planned sibling
``geoexp`` package (see the package-boundary section of ``CLAUDE.md``).

Method
------
For each requested duration ``d`` and window index ``i`` (``i = 0`` is the
most recent), the pseudo-experiment occupies the ``d`` consecutive periods
ending ``i`` periods before the end of the panel. Periods after the window are
dropped — an experiment that ends at ``t`` has no data past ``t`` — so earlier
windows see a shorter pre-period. This is the in-time placebo ("backdating")
design of Abadie (2021, §5), applied at the end of the pre-intervention data
as in GeoLift's power analysis (``lookback_window`` sliding windows).

The injected effect is multiplicative by default — treated outcomes in the
window are scaled by ``1 + effect_size``, matching GeoLift's fractional-lift
convention — with an additive alternative in raw outcome units. Detection
tests the sharp null of no effect via :func:`~augsynth_py.inference.conformal_pvalue`
(``h0 = 0``) and flags ``pvalue <= alpha``. With the deterministic ``"block"``
permutation scheme the p-values are multiples of ``1/T``, so an ``alpha``
sitting exactly on that grid makes the ``<=`` convention observable; see
``docs/methodology.md`` §6.

References
----------
Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements,
and Methodological Aspects. *Journal of Economic Literature*, 59(2), 391-425.

Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
Conformal Inference Method for Counterfactual and Synthetic Controls. *JASA*,
116(536), 1849-1864.

GeoLift documentation (Meta): ``GeoLiftPower``. Used as a parity oracle only
(``tests/validation_against_r/test_power.py``), never as a source to
translate from.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy.typing import NDArray

from augsynth_py.inference import PermutationType, Side, conformal_pvalue

logger = logging.getLogger(__name__)

EffectType = Literal["multiplicative", "additive"]

#: GeoLift's default effect-size grid (``seq(0, 0.25, 0.05)``): fractional
#: lifts from 0% to 25% in 5-point steps. The zero entry measures the
#: empirical size (false-positive rate) of the test.
DEFAULT_EFFECT_SIZES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


class PowerEstimator(Protocol):
    """Structural type for estimators usable in :func:`simulate_power`.

    Both :class:`~augsynth_py.synth.classical.Synth` and
    :class:`~augsynth_py.synth.augmented.AugSynth` satisfy this Protocol. It
    is the union of the ``fit`` entry point, the fitted attributes the power
    loop reports, and the surface :func:`~augsynth_py.inference.conformal_pvalue`
    requires (its ``_FittedSC`` protocol).

    Attributes
    ----------
    gap_ : NDArray[np.float64]
        Actual minus synthetic outcome over all periods.
    pre_mask_ : NDArray[np.bool_]
        True for pre-treatment periods.
    att_ : float
        Mean post-treatment gap.
    att_pct_ : float
        ``att_`` relative to the treated pre-period mean.
    rmspe_pre_ : float
        Normalized pre-period RMSPE (fit-quality diagnostic).
    """

    gap_: NDArray[np.float64]
    pre_mask_: NDArray[np.bool_]
    att_: float
    att_pct_: float
    rmspe_pre_: float

    def fit(
        self,
        panel: pl.DataFrame,
        *,
        unit: str,
        time: str,
        outcome: str,
        treated: Any,
        treatment_time: Any,
    ) -> PowerEstimator:
        """Fit on a long-format panel; see :meth:`Synth.fit`."""
        ...

    def _conformal_null_residuals(self, h0: float) -> NDArray[np.float64]: ...


class _Window(NamedTuple):
    """One placebo-in-time pseudo-experiment window."""

    idx: int  # 0 = most recent possible window
    start: Any  # first treated period (the fit's treatment_time)
    end: Any  # last treated period; data after this is dropped
    n_pre: int  # periods strictly before start
    kept_periods: list[Any]  # all periods up to and including end
    window_periods: list[Any]  # the duration periods receiving the effect


class _Task(NamedTuple):
    """One (duration, effect, window) simulation cell."""

    duration: int
    effect: float
    window: _Window
    seed: int | None  # per-task rng seed; None under the deterministic scheme


@dataclass(frozen=True)
class PowerResults:
    """Results of a :func:`simulate_power` run.

    Attributes
    ----------
    simulations : pl.DataFrame
        One row per (duration, effect size, window) simulation. Columns:

        - ``duration`` (i64): treatment length in periods.
        - ``effect_size`` (f64): injected effect.
        - ``window`` (i64): window index; 0 is the most recent.
        - ``treatment_start`` / ``treatment_end``: first/last treated period,
          in the panel's time dtype.
        - ``n_pre_periods`` (i64): pre-period length for this window.
        - ``pvalue`` (f64): conformal p-value of the no-effect null; null if
          the simulation errored under ``on_error="record"``.
        - ``att`` / ``att_pct`` (f64): estimated ATT on the simulated data,
          absolute and relative to the treated pre-period mean.
        - ``rmspe_pre`` (f64): normalized pre-period RMSPE of the fit.
        - ``error`` (str): ``"ExcType: message"`` for failed simulations
          under ``on_error="record"``; null otherwise.
    alpha : float
        Significance level used as the default detection threshold.
    effect_type : str
        ``"multiplicative"`` or ``"additive"`` — how ``effect_size`` was
        injected, hence how the MDE must be read (fractional lift vs raw
        outcome units).
    """

    simulations: pl.DataFrame
    alpha: float
    effect_type: str

    def power_curve(self, *, alpha: float | None = None) -> pl.DataFrame:
        """Aggregate detection rates by (duration, effect size).

        Parameters
        ----------
        alpha : float, optional
            Detection threshold; defaults to the ``alpha`` the simulation ran
            with. A simulation counts as detected when ``pvalue <= alpha``.

        Returns
        -------
        pl.DataFrame
            Columns ``duration``, ``effect_size``, ``n_simulations``,
            ``n_failed``, ``n_detected``, ``power``, sorted by duration then
            effect size. ``power = n_detected / (n_simulations - n_failed)``
            (failed simulations are excluded from the denominator); it is null
            when every simulation in the cell failed. The ``effect_size == 0``
            row is the empirical size (false-positive rate) of the test.
        """
        a = self.alpha if alpha is None else alpha
        n_valid = pl.col("n_simulations") - pl.col("n_failed")
        return (
            self.simulations.group_by(["duration", "effect_size"])
            .agg(
                n_simulations=pl.len(),
                n_failed=pl.col("pvalue").is_null().sum(),
                n_detected=(pl.col("pvalue") <= a).sum(),
            )
            .with_columns(
                power=pl.when(n_valid > 0).then(pl.col("n_detected") / n_valid).otherwise(None)
            )
            .sort(["duration", "effect_size"])
        )

    def mde(
        self,
        *,
        target_power: float = 0.8,
        alpha: float | None = None,
        duration: int | None = None,
    ) -> float | None:
        """Minimum detectable effect at a target power.

        The MDE is the smallest-magnitude nonzero effect size in the simulated
        grid whose power reaches ``target_power``. No interpolation is done:
        the answer is a grid point, so its resolution is the grid's.

        Parameters
        ----------
        target_power : float, optional
            Power threshold, in ``(0, 1]``. Defaults to the conventional 0.8.
        alpha : float, optional
            Detection threshold passed to :meth:`power_curve`.
        duration : int, optional
            Which duration to read the MDE for. May be omitted when the
            simulation ran a single duration.

        Returns
        -------
        float or None
            The MDE (interpreted per ``effect_type``: fractional lift when
            multiplicative, raw outcome units when additive), or None when no
            simulated effect reaches ``target_power`` — enlarge the effect
            grid or the design.

        Raises
        ------
        ValueError
            If ``target_power`` is outside ``(0, 1]``, if ``duration`` is
            omitted while several durations are present (or names an absent
            one), or if the nonzero effect grid mixes signs — a single MDE is
            ill-defined across directions; filter ``simulations`` by sign and
            rebuild first.
        """
        if not 0.0 < target_power <= 1.0:
            raise ValueError(f"target_power must be in (0, 1], got {target_power}.")
        curve = self.power_curve(alpha=alpha)
        available = [int(d) for d in curve.get_column("duration").unique().sort().to_list()]
        if duration is None:
            if len(available) > 1:
                raise ValueError(f"results contain several durations {available}; pass duration=.")
            duration = available[0]
        elif duration not in available:
            raise ValueError(f"duration {duration} not in results; available: {available}.")
        nonzero = curve.filter((pl.col("duration") == duration) & (pl.col("effect_size") != 0.0))
        signs = {e > 0 for e in nonzero.get_column("effect_size").to_list()}
        if len(signs) > 1:
            raise ValueError(
                "effect grid mixes positive and negative sizes; a single MDE is "
                "ill-defined across directions. Filter `simulations` by sign and "
                "rebuild PowerResults, or simulate one direction at a time."
            )
        hit = nonzero.filter(pl.col("power") >= target_power)
        if hit.is_empty():
            return None
        effects = hit.get_column("effect_size")
        pos = effects.abs().arg_min()
        assert pos is not None  # hit is non-empty by the guard above
        return float(effects[pos])


def simulate_power(
    panel: pl.DataFrame,
    *,
    estimator: PowerEstimator,
    unit: str,
    time: str,
    outcome: str,
    treated: Any | Iterable[Any],
    durations: int | Sequence[int],
    effect_sizes: Sequence[float] = DEFAULT_EFFECT_SIZES,
    effect_type: EffectType = "multiplicative",
    lookback_window: int = 1,
    alpha: float = 0.1,
    permutation_type: PermutationType = "block",
    block_size: int | None = None,
    side: Side = "two-sided",
    ns: int = 1000,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
    on_error: Literal["raise", "record"] = "raise",
) -> PowerResults:
    """Estimate detection power for a given treated set by simulation.

    Runs the full grid ``durations x effect_sizes x lookback_window``: for
    each cell, truncate the panel at the window end, inject the effect into
    the treated units over the window, fit a fresh copy of ``estimator``, and
    compute the conformal p-value of the no-effect null (CWZ 2021). See the
    module docstring for the method and its GeoLift correspondence.

    Parameters
    ----------
    panel : pl.DataFrame
        Long-format panel; must be balanced across ``(unit, time)``. It is
        never modified — injections happen on per-simulation copies.
    estimator : PowerEstimator
        Unfitted estimator *prototype* — e.g. ``Synth()`` or ``AugSynth()``.
        Each simulation fits a ``copy.deepcopy`` of it; the object passed in
        is never fitted in place. Note the CV caveat below.
    unit, time, outcome : str
        Column names in ``panel``.
    treated : Any or Iterable[Any]
        The treated unit, or an iterable of units forming a treated group,
        with the same semantics as :meth:`Synth.fit` (``str``/``bytes`` read
        as a single unit value). The effect is injected into every listed
        unit; a group is then collapsed to its mean by the estimator.
    durations : int or Sequence[int]
        Treatment length(s) in periods (GeoLift's ``treatment_periods``).
    effect_sizes : Sequence[float], optional
        Effects to inject (GeoLift's ``effect_size``). Defaults to GeoLift's
        grid ``(0, 0.05, ..., 0.25)``. Include 0 to measure the empirical
        size; use negative values for cost-reduction (negative-lift) designs.
        Multiplicative effects must be greater than -1.
    effect_type : {"multiplicative", "additive"}, optional
        ``"multiplicative"`` (default, GeoLift's convention) scales treated
        window outcomes by ``1 + effect_size``; ``"additive"`` adds
        ``effect_size`` in raw outcome units per period.
    lookback_window : int, optional
        Number of placebo-in-time windows to simulate per (duration, effect)
        cell, sliding back one period at a time from the end of the panel.
        Defaults to 1 (GeoLift's default): only the most recent window.
    alpha : float, optional
        Significance level for detection (``pvalue <= alpha``). Defaults to
        0.1, GeoLift's default for geo-experiments. Stored on the result;
        aggregation can override it without re-simulating.
    permutation_type : {"block", "iid"}, optional
        Conformal permutation scheme (GeoLift's ``conformal_type``).
        ``"block"`` (default) with ``block_size=None`` is deterministic and
        needs no ``rng``.
    block_size : int or None, optional
        Block length for the random block-shuffle scheme of
        :func:`~augsynth_py.inference.conformal_pvalue` (see there for
        semantics); ``None`` (default) keeps the deterministic cyclic-shift
        scheme. Only valid with ``permutation_type="block"``. Each window has
        its own total length ``T``, so ``block_size`` must be smaller than the
        *shortest* simulated window (``min(n_pre_periods) + duration``);
        violations surface per simulation, per ``on_error``.
    side : {"two-sided", "left", "right"}, optional
        Test direction (GeoLift's ``side_of_test``; its ``"one_sided"`` with
        positive lifts corresponds to ``"right"``).
    ns : int, optional
        Number of permutations for the random schemes (``"iid"``, or
        ``"block"`` with a ``block_size``); ignored by the deterministic
        cyclic-shift scheme.
    rng : numpy.random.Generator, optional
        Source of per-simulation seeds for the random schemes. Each
        simulation draws p-values from an independent generator seeded up
        front in task order, so results are reproducible for a given ``rng``
        regardless of ``n_jobs``. Ignored by the deterministic cyclic-shift
        scheme; a fresh unseeded generator is used when omitted.
    n_jobs : int, optional
        Parallel workers for the simulation grid, forwarded to
        :class:`joblib.Parallel` (``-1`` uses all cores). Defaults to 1.
    on_error : {"raise", "record"}, optional
        ``"raise"`` (default) propagates the first simulation failure.
        ``"record"`` stores ``"ExcType: message"`` in the ``error`` column
        with a null ``pvalue`` and continues; :meth:`PowerResults.power_curve`
        excludes failed simulations from the power denominator.

    Returns
    -------
    PowerResults
        Per-simulation rows plus :meth:`~PowerResults.power_curve` /
        :meth:`~PowerResults.mde` aggregation.

    Raises
    ------
    ValueError
        On unknown columns, treated units absent from the panel, non-finite
        or duplicated effect sizes, a multiplicative effect at or below -1,
        non-positive durations or duplicates, ``lookback_window < 1``,
        ``alpha`` outside ``(0, 1)``, an unknown ``effect_type`` or
        ``on_error``, a ``block_size`` below 1 or combined with
        ``permutation_type="iid"``, or a panel too short for the requested grid
        (``T >= max(durations) + lookback_window`` is required so every
        window keeps at least one pre-period).

    Notes
    -----
    **Cross-validated penalties.** ``AugSynth()`` with no explicit ``lambda_``
    re-runs its LOO-CV inside every simulation — the R-faithful behaviour
    (GeoLift refits ``augsynth`` per simulation), at roughly one CV per fit.
    Freeze the penalty across simulations with ``AugSynth(lambda_=...)`` when
    that cost matters and the design justifies it.

    **GeoLift argument correspondence** (`GeoLiftPower`): ``locations`` ->
    ``treated``, ``treatment_periods`` -> ``durations``, ``effect_size`` ->
    ``effect_sizes``, ``lookback_window`` -> ``lookback_window``,
    ``model``/``fixed_effects`` -> ``estimator`` (``"none"`` ~ ``Synth()``,
    ``"ridge"`` ~ ``AugSynth()``), ``conformal_type`` -> ``permutation_type``,
    ``side_of_test`` -> ``side``, ``alpha`` -> ``alpha``. GeoLift's ``cpic``
    and investment outputs are design-costing concerns and live in the
    planned sibling ``geoexp`` package, not here.
    """
    for name, col in (("unit", unit), ("time", time), ("outcome", outcome)):
        if col not in panel.columns:
            raise ValueError(f"{name} column {col!r} not found in panel.")

    duration_list = [int(d) for d in ([durations] if isinstance(durations, int) else durations)]
    if not duration_list:
        raise ValueError("durations must be non-empty.")
    if any(d < 1 for d in duration_list):
        raise ValueError(f"durations must be >= 1, got {duration_list}.")
    if len(set(duration_list)) != len(duration_list):
        raise ValueError(f"durations contains duplicates: {duration_list}.")

    effect_list = [float(e) for e in effect_sizes]
    if not effect_list:
        raise ValueError("effect_sizes must be non-empty.")
    if any(not math.isfinite(e) for e in effect_list):
        raise ValueError(f"effect_sizes must be finite, got {effect_list}.")
    if len(set(effect_list)) != len(effect_list):
        raise ValueError(f"effect_sizes contains duplicates: {effect_list}.")
    if effect_type not in ("multiplicative", "additive"):
        raise ValueError(
            f"Unknown effect_type {effect_type!r}; expected 'multiplicative' or 'additive'."
        )
    if effect_type == "multiplicative" and any(e <= -1.0 for e in effect_list):
        raise ValueError(
            "multiplicative effect sizes must be > -1 (a lift of -1 zeroes the outcome)."
        )

    if block_size is not None:
        if permutation_type != "block":
            raise ValueError(
                "block_size applies to permutation_type='block' only; use "
                "permutation_type='block' with block_size for random block shuffles."
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}.")

    if lookback_window < 1:
        raise ValueError(f"lookback_window must be >= 1, got {lookback_window}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if on_error not in ("raise", "record"):
        raise ValueError(f"Unknown on_error {on_error!r}; expected 'raise' or 'record'.")

    periods = panel.get_column(time).unique().sort().to_list()
    t_len = len(periods)
    max_duration = max(duration_list)
    if t_len < max_duration + lookback_window:
        raise ValueError(
            f"panel has {t_len} periods but the grid needs at least "
            f"max(durations) + lookback_window = {max_duration + lookback_window} "
            f"so every window keeps a pre-period."
        )

    treated_list = _treated_values(treated)
    known_units = set(panel.get_column(unit).unique().to_list())
    missing = [u for u in treated_list if u not in known_units]
    if missing:
        raise ValueError(f"treated unit(s) not present in panel: {missing}.")
    # Preserve the caller's scalar-vs-group distinction for the estimator,
    # but hand workers a stable list (a generator would be consumed once).
    treated_for_fit: Any = treated_list if _is_group(treated) else treated_list[0]

    tasks: list[_Task] = []
    for d in duration_list:
        windows = _windows(periods, d, lookback_window)
        for e in effect_list:
            tasks.extend(_Task(d, e, w, None) for w in windows)

    # Random permutation schemes ("iid", or "block" with a block_size) need a
    # per-task generator; the deterministic cyclic-shift scheme does not.
    if permutation_type == "iid" or block_size is not None:
        master = rng if rng is not None else np.random.default_rng()
        seeds = master.integers(0, 2**63 - 1, size=len(tasks))
        tasks = [task._replace(seed=int(s)) for task, s in zip(tasks, seeds, strict=True)]

    logger.info(
        "power grid: %d fits (%d durations x %d effects x %d windows), n_jobs=%d",
        len(tasks),
        len(duration_list),
        len(effect_list),
        lookback_window,
        n_jobs,
    )

    record = on_error == "record"
    if n_jobs == 1:
        rows = [
            _run_one(
                estimator,
                panel,
                unit,
                time,
                outcome,
                treated_for_fit,
                treated_list,
                effect_type,
                permutation_type,
                block_size,
                side,
                ns,
                record,
                task,
            )
            for task in tasks
        ]
    else:
        rows = list(
            Parallel(n_jobs=n_jobs)(
                delayed(_run_one)(
                    estimator,
                    panel,
                    unit,
                    time,
                    outcome,
                    treated_for_fit,
                    treated_list,
                    effect_type,
                    permutation_type,
                    block_size,
                    side,
                    ns,
                    record,
                    task,
                )
                for task in tasks
            )
        )

    simulations = pl.DataFrame(
        rows,
        schema_overrides={
            "duration": pl.Int64,
            "effect_size": pl.Float64,
            "window": pl.Int64,
            "n_pre_periods": pl.Int64,
            "pvalue": pl.Float64,
            "att": pl.Float64,
            "att_pct": pl.Float64,
            "rmspe_pre": pl.Float64,
            "error": pl.String,
        },
    )
    return PowerResults(simulations=simulations, alpha=alpha, effect_type=effect_type)


def _is_group(treated: Any) -> bool:
    """Report whether ``treated`` designates a group (estimator ``fit`` semantics)."""
    return isinstance(treated, Iterable) and not isinstance(treated, str | bytes)


def _treated_values(treated: Any) -> list[Any]:
    """Normalize ``treated`` to a concrete list of unit values."""
    return list(treated) if _is_group(treated) else [treated]


def _windows(periods: list[Any], duration: int, lookback: int) -> list[_Window]:
    """Enumerate the ``lookback`` most recent placebo windows of ``duration``.

    Window ``i`` occupies the ``duration`` consecutive periods ending ``i``
    periods before the end of ``periods`` (sorted ascending). The caller
    guarantees ``len(periods) >= duration + lookback``, so every window keeps
    at least one pre-period.
    """
    t_len = len(periods)
    out: list[_Window] = []
    for i in range(lookback):
        end_pos = t_len - 1 - i
        start_pos = end_pos - duration + 1
        out.append(
            _Window(
                idx=i,
                start=periods[start_pos],
                end=periods[end_pos],
                n_pre=start_pos,
                kept_periods=periods[: end_pos + 1],
                window_periods=periods[start_pos : end_pos + 1],
            )
        )
    return out


def _inject_effect(
    panel: pl.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treated: list[Any],
    window_periods: list[Any],
    effect: float,
    effect_type: EffectType,
) -> pl.DataFrame:
    """Apply the effect to treated rows inside the window; all others untouched."""
    in_cell = pl.col(unit).is_in(treated) & pl.col(time).is_in(window_periods)
    if effect_type == "multiplicative":
        adjusted = pl.col(outcome) * (1.0 + effect)
    else:
        adjusted = pl.col(outcome) + effect
    return panel.with_columns(
        pl.when(in_cell).then(adjusted).otherwise(pl.col(outcome)).alias(outcome)
    )


def _run_one(
    prototype: PowerEstimator,
    panel: pl.DataFrame,
    unit: str,
    time: str,
    outcome: str,
    treated_for_fit: Any,
    treated_list: list[Any],
    effect_type: EffectType,
    permutation_type: PermutationType,
    block_size: int | None,
    side: Side,
    ns: int,
    record_errors: bool,
    task: _Task,
) -> dict[str, Any]:
    """Run one simulation cell: truncate, inject, fit, test.

    Module-level (not a closure) so joblib can pickle it to worker processes.
    The prototype is deep-copied here, worker-side, so the caller's estimator
    is never fitted in place and parallel workers never share fitted state.
    """
    w = task.window
    base: dict[str, Any] = {
        "duration": task.duration,
        "effect_size": task.effect,
        "window": w.idx,
        "treatment_start": w.start,
        "treatment_end": w.end,
        "n_pre_periods": w.n_pre,
    }
    try:
        est = copy.deepcopy(prototype)
        sub = panel.filter(pl.col(time).is_in(w.kept_periods))
        if task.effect != 0.0:
            sub = _inject_effect(
                sub,
                unit=unit,
                time=time,
                outcome=outcome,
                treated=treated_list,
                window_periods=w.window_periods,
                effect=task.effect,
                effect_type=effect_type,
            )
        fitted = est.fit(
            sub,
            unit=unit,
            time=time,
            outcome=outcome,
            treated=treated_for_fit,
            treatment_time=w.start,
        )
        task_rng = np.random.default_rng(task.seed) if task.seed is not None else None
        pvalue = conformal_pvalue(
            fitted,
            0.0,
            permutation_type=permutation_type,
            block_size=block_size,
            side=side,
            ns=ns,
            rng=task_rng,
        )
    # Broad on purpose: any solver/CV/validation failure in this cell either
    # propagates (on_error="raise") or becomes an `error` row ("record").
    except Exception as exc:
        if not record_errors:
            raise
        return {
            **base,
            "pvalue": None,
            "att": None,
            "att_pct": None,
            "rmspe_pre": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "pvalue": float(pvalue),
        "att": float(fitted.att_),
        "att_pct": float(fitted.att_pct_),
        "rmspe_pre": float(fitted.rmspe_pre_),
        "error": None,
    }
