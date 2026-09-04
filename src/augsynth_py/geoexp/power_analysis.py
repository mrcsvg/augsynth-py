"""Evaluate candidate assignments with GeoLift-compatible power analysis."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from augsynth_py.power import (
    DEFAULT_EFFECT_SIZES,
    PowerEstimator,
    PowerResults,
    simulate_power,
)

TreatmentPod = Literal["pod_A", "pod_B"]
Ranking = Literal["geolift"] | Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class _Assignment:
    treatment_markets: tuple[Any, ...]
    donor_markets: tuple[Any, ...]
    treatment_pod: TreatmentPod


@dataclass(frozen=True, eq=False)
class GeoLiftPowerAnalysisResults:
    """Power-analysis results for a collection of candidate assignments."""

    _power_results: Mapping[str, PowerResults]
    _assignments: Mapping[str, _Assignment]
    _simulations: pl.DataFrame
    _effect_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_power_results", MappingProxyType(dict(self._power_results)))
        object.__setattr__(self, "_assignments", MappingProxyType(dict(self._assignments)))
        object.__setattr__(self, "_simulations", self._simulations.clone())

    def get_power_results(self, sample: str) -> PowerResults:
        """Return the native single-assignment result for ``sample``."""
        try:
            return self._power_results[sample]
        except KeyError as exc:
            raise KeyError(f"Unknown sample {sample!r}") from exc

    def as_df(
        self,
        *,
        sample: str | None = None,
        duration: int | None = None,
        effect_size: float | None = None,
        window: int | None = None,
    ) -> pl.DataFrame:
        """Return enriched simulation rows with optional scalar filters."""
        return _filter_frame(
            self._simulations,
            sample=sample,
            duration=duration,
            effect_size=effect_size,
            window=window,
        )

    def power_curve(
        self,
        *,
        sample: str | None = None,
        duration: int | None = None,
        effect_size: float | None = None,
    ) -> pl.DataFrame:
        """Return native power curves as one Polars DataFrame."""
        frames = [
            result.power_curve()
            .with_columns(sample=pl.lit(sample_name))
            .select("sample", pl.exclude("sample"))
            for sample_name, result in self._selected_results(sample).items()
        ]
        combined = pl.concat(frames) if frames else _empty_power_curve()
        return _filter_frame(combined, duration=duration, effect_size=effect_size)

    def mde(
        self,
        *,
        target_power: float = 0.8,
        sample: str | None = None,
        duration: int | None = None,
    ) -> pl.DataFrame:
        """Return one MDE grid point per selected sample and duration."""
        records = []
        for sample_name, result in self._selected_results(sample).items():
            assert result.params is not None
            for evaluated_duration in result.params.durations:
                if duration is not None and evaluated_duration != duration:
                    continue
                records.append(
                    {
                        "sample": sample_name,
                        "duration": evaluated_duration,
                        "mde": result.mde(
                            target_power=target_power,
                            duration=evaluated_duration,
                        ),
                    }
                )
        return pl.DataFrame(
            records,
            schema={"sample": pl.String, "duration": pl.Int64, "mde": pl.Float64},
        )

    def evaluated_designs(
        self,
        *,
        target_power: float = 0.8,
        sample: str | None = None,
        duration: int | None = None,
    ) -> pl.DataFrame:
        """Return one analytical row per candidate assignment and duration."""
        curves = self.power_curve()
        mdes = self.mde(target_power=target_power)
        records = []

        for mde_row in mdes.iter_rows(named=True):
            sample_name = cast(str, mde_row["sample"])
            evaluated_duration = cast(int, mde_row["duration"])
            mde = cast(float | None, mde_row["mde"])
            assignment = self._assignments[sample_name]
            simulations = self.as_df(sample=sample_name, duration=evaluated_duration)
            at_mde = (
                simulations.clear()
                if mde is None
                else simulations.filter((pl.col("effect_size") == mde) & ~pl.col("failed"))
            )
            null_rows = simulations.filter(
                (pl.col("effect_size") == 0) & pl.col("pvalue").is_not_null()
            )
            curve_at_mde = (
                curves.clear()
                if mde is None
                else curves.filter(
                    (pl.col("sample") == sample_name)
                    & (pl.col("duration") == evaluated_duration)
                    & (pl.col("effect_size") == mde)
                )
            )
            average_att = cast(float | None, at_mde.get_column("att").mean())
            average_detected_lift = cast(
                float | None,
                at_mde.get_column("detected_lift").mean(),
            )
            recovered_effect = (
                average_detected_lift if self._effect_type == "multiplicative" else average_att
            )
            h1_calibration_error = (
                round(abs(recovered_effect - mde), 3)
                if recovered_effect is not None and mde is not None
                else None
            )
            native_failed = simulations.get_column("pvalue").is_null()
            enrichment_failed = simulations.get_column("enrichment_error").is_not_null()
            failed = native_failed | enrichment_failed

            records.append(
                {
                    "sample": sample_name,
                    "duration": evaluated_duration,
                    "treatment_markets": list(assignment.treatment_markets),
                    "donor_markets": list(assignment.donor_markets),
                    "treatment_pod": assignment.treatment_pod,
                    "mde": mde,
                    "power_at_mde": _first_or_none(curve_at_mde, "power"),
                    "average_att": average_att,
                    "average_detected_lift": average_detected_lift,
                    "h1_calibration_error": h1_calibration_error,
                    "null_bias": null_rows.get_column("detected_lift").mean(),
                    "average_rmspe_pre": at_mde.get_column("rmspe_pre").mean(),
                    "n_simulations": simulations.height,
                    "n_valid": int((~failed).sum()),
                    "n_native_failed": int(native_failed.sum()),
                    "n_enrichment_failed": int(enrichment_failed.sum()),
                    "n_failed": int(failed.sum()),
                }
            )

        designs = pl.DataFrame(
            records,
            schema_overrides={column: pl.Float64 for column in _EVALUATED_DESIGN_FLOAT_COLUMNS},
        ).select(_EVALUATED_DESIGN_COLUMNS)
        return _filter_frame(designs, sample=sample, duration=duration)

    def rank_designs(
        self,
        ranking: Ranking = "geolift",
        *,
        sample: str | None = None,
        duration: int | None = None,
        target_power: float = 0.8,
        **ranking_kwargs: Any,
    ) -> pl.DataFrame:
        """Rank evaluated designs with GeoLift or a Polars callable."""
        designs = self.evaluated_designs(
            target_power=target_power,
            sample=sample,
            duration=duration,
        )
        if callable(ranking):
            ranked = ranking(designs.clone(), **ranking_kwargs)
            if not isinstance(ranked, pl.DataFrame):
                raise TypeError("A custom ranking must return a Polars DataFrame")
            missing = [column for column in ("sample", "duration") if column not in ranked.columns]
            if missing:
                raise ValueError(f"Custom ranking removed required columns: {missing}")
            return ranked.clone()
        if ranking != "geolift":
            raise ValueError("ranking must be 'geolift' or a callable")
        if ranking_kwargs:
            names = ", ".join(sorted(ranking_kwargs))
            raise TypeError(f"GeoLift ranking does not accept custom arguments: {names}")
        if self._effect_type != "multiplicative":
            raise ValueError("GeoLift ranking is only defined for multiplicative effects")
        return _rank_geolift(designs)

    def get_best_design(
        self,
        ranking: Ranking = "geolift",
        *,
        sample: str | None = None,
        duration: int | None = None,
        target_power: float = 0.8,
        **ranking_kwargs: Any,
    ) -> pl.DataFrame:
        """Return the first ranked design as a one-row DataFrame."""
        ranked = self.rank_designs(
            ranking,
            sample=sample,
            duration=duration,
            target_power=target_power,
            **ranking_kwargs,
        )
        if ranked.is_empty():
            raise ValueError("Ranking returned no eligible designs")
        return ranked.head(1)

    def _selected_results(self, sample: str | None) -> dict[str, PowerResults]:
        if sample is None:
            return dict(self._power_results)
        if sample not in self._power_results:
            return {}
        return {sample: self._power_results[sample]}


@dataclass(frozen=True)
class GeoLiftPowerAnalysis:
    """Configure reusable GeoLift power analysis for candidate assignments.

    Native power aggregation treats ``pvalue == alpha`` as detected. R GeoLift
    uses a strict boundary, so parity comparisons should avoid an alpha on the
    conformal p-value grid. H1 calibration uses detected lift for multiplicative
    effects and ATT for additive effects. Built-in GeoLift ranking is available
    only for multiplicative effects.
    """

    estimator: InitVar[PowerEstimator]
    durations: int | Sequence[int]
    treatment_pod: TreatmentPod
    rng: InitVar[np.random.Generator | None] = None
    unit_col: str = "location"
    time_col: str = "time"
    outcome_col: str = "Y"
    effect_sizes: Sequence[float] = DEFAULT_EFFECT_SIZES
    effect_type: Literal["multiplicative", "additive"] = "multiplicative"
    lookback_window: int = 1
    alpha: float = 0.1
    permutation_type: Literal["iid", "block"] = "block"
    block_size: int | None = None
    side: Literal["two-sided", "left", "right"] = "two-sided"
    ns: int = 1000
    n_jobs: int = 1
    on_error: Literal["raise", "record"] = "raise"
    _estimator: PowerEstimator = field(init=False, repr=False)
    _random_seed: int | None = field(init=False, repr=False)

    def __post_init__(
        self,
        estimator: PowerEstimator,
        rng: np.random.Generator | None,
    ) -> None:
        durations = _normalize_durations(self.durations)
        effect_sizes = _normalize_effect_sizes(self.effect_sizes, self.effect_type)
        _validate_configuration(self, rng)

        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "effect_sizes", effect_sizes)
        object.__setattr__(self, "_estimator", copy.deepcopy(estimator))
        seed = _seed_from_rng(rng) if self._uses_randomness else None
        object.__setattr__(self, "_random_seed", seed)

    def evaluate(
        self,
        panel: pl.DataFrame,
        samples: Mapping[str, Mapping[str, Sequence[Any]]],
    ) -> GeoLiftPowerAnalysisResults:
        """Evaluate named candidate assignments against a Polars panel."""
        durations = _normalize_durations(self.durations)
        validated_panel = _validate_panel(
            panel,
            unit_col=self.unit_col,
            time_col=self.time_col,
            outcome_col=self.outcome_col,
            minimum_periods=max(durations) + self.lookback_window + 1,
        )
        assignments = _validate_assignments(
            samples,
            known_markets=validated_panel.get_column(self.unit_col)
            .unique(maintain_order=True)
            .to_list(),
            treatment_pod=self.treatment_pod,
        )

        power_results: dict[str, PowerResults] = {}
        enriched = []
        for sample, assignment in assignments.items():
            markets = (*assignment.treatment_markets, *assignment.donor_markets)
            sample_panel = validated_panel.filter(pl.col(self.unit_col).is_in(markets))
            native = simulate_power(
                sample_panel,
                estimator=self._estimator,
                unit=self.unit_col,
                time=self.time_col,
                outcome=self.outcome_col,
                treated=assignment.treatment_markets,
                durations=durations,
                effect_sizes=self.effect_sizes,
                effect_type=self.effect_type,
                lookback_window=self.lookback_window,
                alpha=self.alpha,
                permutation_type=self.permutation_type,
                block_size=self.block_size,
                side=self.side,
                ns=self.ns,
                rng=_sample_rng(self._random_seed, sample) if self._uses_randomness else None,
                n_jobs=self.n_jobs,
                on_error=self.on_error,
            )
            power_results[sample] = native
            enriched.append(
                _enrich_simulations(
                    native,
                    sample=sample,
                    assignment=assignment,
                    panel=sample_panel,
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                    outcome_col=self.outcome_col,
                    effect_type=self.effect_type,
                )
            )

        return GeoLiftPowerAnalysisResults(
            power_results,
            assignments,
            pl.concat(enriched),
            self.effect_type,
        )

    @property
    def _uses_randomness(self) -> bool:
        return self.permutation_type == "iid" or self.block_size is not None


def _normalize_durations(durations: int | Sequence[int]) -> tuple[int, ...]:
    values = (
        (durations,)
        if isinstance(durations, Integral) and not isinstance(durations, bool)
        else tuple(cast(Sequence[int], durations))
    )
    if not values or any(
        not isinstance(value, Integral) or isinstance(value, bool) or value < 1 for value in values
    ):
        raise ValueError("durations must contain positive integers")
    normalized = tuple(int(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("durations must not contain duplicates")
    return normalized


def _normalize_effect_sizes(effect_sizes: Sequence[float], effect_type: str) -> tuple[float, ...]:
    values = tuple(effect_sizes)
    if not values or any(
        not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("effect_sizes must contain finite numbers")
    normalized = tuple(float(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("effect_sizes must not contain duplicates")
    if effect_type == "multiplicative" and any(value <= -1 for value in normalized):
        raise ValueError("multiplicative effect_sizes must be greater than -1")
    return normalized


def _validate_configuration(
    config: GeoLiftPowerAnalysis,
    rng: np.random.Generator | None,
) -> None:
    columns = (config.unit_col, config.time_col, config.outcome_col)
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Column names must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("unit_col, time_col, and outcome_col must be distinct")
    if config.treatment_pod not in ("pod_A", "pod_B"):
        raise ValueError("treatment_pod must be 'pod_A' or 'pod_B'")
    if config.effect_type not in ("multiplicative", "additive"):
        raise ValueError("effect_type must be 'multiplicative' or 'additive'")
    if config.permutation_type not in ("iid", "block"):
        raise ValueError("permutation_type must be 'iid' or 'block'")
    if config.block_size is not None and (
        not isinstance(config.block_size, Integral)
        or isinstance(config.block_size, bool)
        or config.block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if config.block_size is not None and config.permutation_type != "block":
        raise ValueError("block_size requires permutation_type='block'")
    uses_randomness = config.permutation_type == "iid" or config.block_size is not None
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if uses_randomness and rng is None:
        raise ValueError("Random permutation modes require rng")
    if config.side not in ("two-sided", "left", "right"):
        raise ValueError("side must be 'two-sided', 'left', or 'right'")
    if not isinstance(config.lookback_window, Integral) or config.lookback_window < 1:
        raise ValueError("lookback_window must be a positive integer")
    if not isinstance(config.alpha, Real) or not 0 < config.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(config.ns, Integral) or isinstance(config.ns, bool) or config.ns < 1:
        raise ValueError("ns must be a positive integer")
    if (
        not isinstance(config.n_jobs, Integral)
        or isinstance(config.n_jobs, bool)
        or config.n_jobs == 0
    ):
        raise ValueError("n_jobs must be a nonzero integer")
    if config.on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")


def _validate_panel(
    panel: pl.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    minimum_periods: int,
) -> pl.DataFrame:
    if not isinstance(panel, pl.DataFrame):
        raise TypeError("panel must be a Polars DataFrame")
    required = [unit_col, time_col, outcome_col]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    validated = panel.select(required)
    if any(validated.null_count().row(0)):
        raise ValueError("Power analysis requires a complete panel without missing values")
    if validated.select(pl.struct(unit_col, time_col).is_duplicated().any()).item():
        raise ValueError("Panel contains duplicate market-time observations")
    outcome = validated.get_column(outcome_col)
    if not outcome.dtype.is_numeric():
        raise ValueError(f"Outcome column {outcome_col!r} must be numeric")
    if not outcome.is_finite().all():
        raise ValueError(f"Outcome column {outcome_col!r} must contain only finite values")

    market_count = validated.get_column(unit_col).n_unique()
    period_count = validated.get_column(time_col).n_unique()
    if market_count < 2:
        raise ValueError("Power analysis requires at least two markets")
    if validated.height != market_count * period_count:
        raise ValueError("Power analysis requires one observation per market and time")
    if period_count < minimum_periods:
        raise ValueError(
            f"Panel has {period_count} periods but at least {minimum_periods} are required"
        )
    try:
        return validated.sort([time_col, unit_col], maintain_order=True)
    except pl.exceptions.InvalidOperationError as exc:
        raise ValueError("Panel time and market identifiers must be sortable") from exc


def _validate_assignments(
    samples: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    known_markets: Sequence[Any],
    treatment_pod: TreatmentPod,
) -> dict[str, _Assignment]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a mapping of named candidate assignments")
    if not samples:
        raise ValueError("samples must contain at least one candidate assignment")

    known = set(known_markets)
    assignments = {}
    for sample, pods in samples.items():
        if not isinstance(sample, str) or not sample:
            raise ValueError("Sample identifiers must be non-empty strings")
        if not isinstance(pods, Mapping):
            raise TypeError(f"Assignment {sample!r} must be a mapping")
        if set(pods) != {"pod_A", "pod_B"}:
            raise ValueError(f"Assignment {sample!r} must contain exactly pod_A and pod_B")
        pod_a = _normalize_pod(pods["pod_A"], sample=sample, pod="pod_A")
        pod_b = _normalize_pod(pods["pod_B"], sample=sample, pod="pod_B")
        overlap = set(pod_a) & set(pod_b)
        if overlap:
            raise ValueError(f"Assignment {sample!r} has markets in both pods: {sorted(overlap)}")
        unknown = [market for market in (*pod_a, *pod_b) if market not in known]
        if unknown:
            raise ValueError(
                f"Assignment {sample!r} contains markets not found in the panel: {unknown}"
            )
        treated = pod_a if treatment_pod == "pod_A" else pod_b
        donors = pod_b if treatment_pod == "pod_A" else pod_a
        assignments[sample] = _Assignment(treated, donors, treatment_pod)
    return assignments


def _normalize_pod(markets: Sequence[Any], *, sample: str, pod: str) -> tuple[Any, ...]:
    if isinstance(markets, str | bytes) or not isinstance(markets, Sequence):
        raise TypeError(f"{sample!r} {pod} must be a market sequence, not a scalar")
    values = tuple(markets)
    if not values:
        raise ValueError(f"{sample!r} {pod} must not be empty")
    try:
        unique_count = len(set(values))
    except TypeError as exc:
        raise ValueError(f"{sample!r} {pod} contains an unhashable market identifier") from exc
    if unique_count != len(values):
        raise ValueError(f"{sample!r} {pod} contains duplicate markets")
    return values


def _seed_from_rng(rng: np.random.Generator | None) -> int | None:
    if rng is None:
        return None
    snapshot = copy.deepcopy(rng)
    return int(snapshot.integers(0, 2**63 - 1))


def _sample_rng(seed: int | None, sample: str) -> np.random.Generator:
    assert seed is not None
    sample_words = np.frombuffer(sample.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
    return np.random.default_rng(np.random.SeedSequence([int(seed), *sample_words.tolist()]))


def _enrich_simulations(
    result: PowerResults,
    *,
    sample: str,
    assignment: _Assignment,
    panel: pl.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    effect_type: str,
) -> pl.DataFrame:
    periods = panel.get_column(time_col).unique().sort().to_list()
    period_positions = {period: position for position, period in enumerate(periods)}
    records = []

    for row in result.simulations.iter_rows(named=True):
        start = period_positions[row["treatment_start"]]
        end = period_positions[row["treatment_end"]]
        window_periods = periods[start : end + 1]
        base_post_total = (
            panel.filter(
                pl.col(unit_col).is_in(assignment.treatment_markets)
                & pl.col(time_col).is_in(window_periods)
            )
            .group_by(time_col)
            .agg(pl.col(outcome_col).mean())
            .get_column(outcome_col)
            .sum()
        )
        injected_post_total = (
            base_post_total * (1 + row["effect_size"])
            if effect_type == "multiplicative"
            else base_post_total + row["effect_size"] * row["duration"]
        )
        estimated_incrementality = None
        synthetic_post_total = None
        detected_lift = None
        enrichment_error = None

        if row["error"] is None:
            att = row["att"]
            if att is None or not math.isfinite(att):
                enrichment_error = "ATT is missing or non-finite"
            else:
                estimated_incrementality = att * row["duration"]
                synthetic_post_total = injected_post_total - estimated_incrementality
                if not math.isfinite(synthetic_post_total) or synthetic_post_total == 0:
                    enrichment_error = "Synthetic post-period total is zero or non-finite"
                else:
                    detected_lift = estimated_incrementality / synthetic_post_total
                    if not math.isfinite(detected_lift):
                        detected_lift = None
                        enrichment_error = "Detected lift is non-finite"

        records.append(
            {
                "sample": sample,
                "treatment_markets": list(assignment.treatment_markets),
                "donor_markets": list(assignment.donor_markets),
                "treatment_pod": assignment.treatment_pod,
                **row,
                "injected_treated_post_total": injected_post_total,
                "estimated_incrementality": estimated_incrementality,
                "synthetic_post_total": synthetic_post_total,
                "detected_lift": detected_lift,
                "enrichment_error": enrichment_error,
                "failed": row["error"] is not None or enrichment_error is not None,
            }
        )
    return pl.DataFrame(
        records,
        schema_overrides={
            "effect_size": pl.Float64,
            "pvalue": pl.Float64,
            "att": pl.Float64,
            "att_pct": pl.Float64,
            "rmspe_pre": pl.Float64,
            "error": pl.String,
            "injected_treated_post_total": pl.Float64,
            "estimated_incrementality": pl.Float64,
            "synthetic_post_total": pl.Float64,
            "detected_lift": pl.Float64,
            "enrichment_error": pl.String,
        },
    )


def _filter_frame(frame: pl.DataFrame, **filters: object) -> pl.DataFrame:
    predicates = [pl.col(column) == value for column, value in filters.items() if value is not None]
    return (frame.filter(*predicates) if predicates else frame).clone()


def _first_or_none(frame: pl.DataFrame, column: str) -> float | None:
    return None if frame.is_empty() else cast(float, frame[0, column])


def _empty_power_curve() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "sample": pl.String,
            "duration": pl.Int64,
            "effect_size": pl.Float64,
            "n_simulations": pl.UInt32,
            "n_failed": pl.UInt32,
            "n_detected": pl.UInt32,
            "power": pl.Float64,
        }
    )


def _rank_geolift(designs: pl.DataFrame) -> pl.DataFrame:
    if designs.is_empty():
        return designs.with_columns(pl.lit(None, dtype=pl.Float64).alias("rank"))

    incomplete = (
        pl.col("mde").is_null()
        | pl.col("mde").is_nan()
        | pl.col("power_at_mde").is_null()
        | pl.col("power_at_mde").is_nan()
        | pl.col("h1_calibration_error").is_null()
        | pl.col("h1_calibration_error").is_nan()
        | (pl.col("n_failed") > 0)
    )
    eligible = designs.filter(~incomplete)
    ineligible = designs.filter(incomplete)

    if not eligible.is_empty():
        eligible = (
            eligible.with_columns(
                _rank_mde=pl.col("mde").abs().rank("dense"),
                _rank_power=pl.col("power_at_mde").rank("dense"),
                _rank_calibration=pl.col("h1_calibration_error").round(3).rank("dense"),
                _treatment_key=pl.col("treatment_markets")
                .list.eval(pl.element().cast(pl.String).str.to_lowercase())
                .list.sort()
                .list.join("\0"),
            )
            .with_columns(
                pl.mean_horizontal("_rank_mde", "_rank_power", "_rank_calibration").alias(
                    "_composite_rank"
                )
            )
            .with_columns(rank=pl.col("_composite_rank").rank("min"))
            .sort(["rank", "_treatment_key"], maintain_order=True)
        )

    rank_offset = (
        int(cast(float, eligible.get_column("rank").max())) if not eligible.is_empty() else 0
    )
    ineligible = ineligible.sort(["sample", "duration"], maintain_order=True).with_columns(
        rank=pl.int_range(
            rank_offset + 1,
            rank_offset + ineligible.height + 1,
            eager=True,
        ).cast(pl.Float64)
    )
    return pl.concat([eligible, ineligible], how="diagonal_relaxed").drop(
        "_rank_mde",
        "_rank_power",
        "_rank_calibration",
        "_composite_rank",
        "_treatment_key",
        strict=False,
    )


_EVALUATED_DESIGN_FLOAT_COLUMNS = [
    "mde",
    "power_at_mde",
    "average_att",
    "average_detected_lift",
    "h1_calibration_error",
    "null_bias",
    "average_rmspe_pre",
]

_EVALUATED_DESIGN_COLUMNS = [
    "sample",
    "duration",
    "treatment_markets",
    "donor_markets",
    "treatment_pod",
    "mde",
    "power_at_mde",
    "average_att",
    "average_detected_lift",
    "h1_calibration_error",
    "null_bias",
    "average_rmspe_pre",
    "n_simulations",
    "n_valid",
    "n_native_failed",
    "n_enrichment_failed",
    "n_failed",
]
