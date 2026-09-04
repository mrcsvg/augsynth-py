"""Generate GeoLift-compatible candidate treatment groups."""

import copy
from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field
from numbers import Integral
from typing import cast

import numpy as np
import polars as pl


@dataclass(frozen=True, eq=False)
class MarketSelectionResults:
    """Candidate assignments produced by market selection."""

    _candidates: pl.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "_candidates", self._candidates.clone())

    def as_df(self) -> pl.DataFrame:
        """Return the candidate table as a defensive copy."""
        return self._candidates.clone()

    def as_samples(self) -> dict[str, dict[str, list[object]]]:
        """Return assignments with treatment in ``pod_A`` and donors in ``pod_B``."""
        rows = self._candidates.select("treatment_markets", "donor_markets").iter_rows()
        return {
            f"sample{sample_number}": {
                "pod_A": list(treatment_markets),
                "pod_B": list(donor_markets),
            }
            for sample_number, (treatment_markets, donor_markets) in enumerate(rows, start=1)
        }


@dataclass(frozen=True)
class MarketSelector:
    """Configure and generate candidate treatment groups.

    The selector stores one market-selection policy and can apply it to equivalent
    canonical panels. It does not retain input data or results between calls.

    Parameters
    ----------
    market_counts
        Numbers of markets to include in each candidate treatment group.
    location_col
        Panel column identifying markets.
    time_col
        Panel column identifying time periods.
    outcome_col
        Numeric outcome used to compare market histories.
    include_markets
        Markets that every retained candidate must contain.
    exclude_markets
        Markets that cannot receive treatment but remain available as donors.
    run_stochastic_process
        Whether to sample one market from each adjacent pair of similarity ranks
        instead of taking the top ``market_count`` markets.
    rng
        Random generator for stochastic selection. Required when
        ``run_stochastic_process=True``. The selector snapshots its state, so
        repeated calls return the same candidates.
    """

    market_counts: Sequence[int]
    location_col: str = "location"
    time_col: str = "time"
    outcome_col: str = "Y"
    include_markets: Iterable[object] = ()
    exclude_markets: Iterable[object] = ()
    run_stochastic_process: bool = False
    rng: InitVar[np.random.Generator | None] = None
    _rng: np.random.Generator | None = field(init=False, repr=False, compare=False)

    def __post_init__(self, rng: np.random.Generator | None) -> None:
        """Normalize and validate configuration that does not depend on panel data."""
        counts = _validate_market_counts(self.market_counts)
        included = _normalize_market_list(self.include_markets, name="include_markets")
        excluded = _normalize_market_list(self.exclude_markets, name="exclude_markets")
        _validate_column_names(self.location_col, self.time_col, self.outcome_col)
        if not isinstance(self.run_stochastic_process, bool):
            raise ValueError("run_stochastic_process must be a boolean")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if self.run_stochastic_process and rng is None:
            raise ValueError("Stochastic market selection requires rng")

        overlap = set(included) & set(excluded)
        if overlap:
            raise ValueError(f"include_markets and exclude_markets overlap: {sorted(overlap)}")

        object.__setattr__(self, "market_counts", counts)
        object.__setattr__(self, "include_markets", included)
        object.__setattr__(self, "exclude_markets", excluded)
        object.__setattr__(self, "_rng", copy.deepcopy(rng))

    def select(self, data: pl.DataFrame) -> MarketSelectionResults:
        """Return candidate treatment groups and their corresponding donor pools.

        ``data`` must be a complete long panel with one row per market and time.
        """
        candidates = _select_markets(
            data,
            market_counts=self.market_counts,
            location_col=self.location_col,
            time_col=self.time_col,
            outcome_col=self.outcome_col,
            include_markets=cast(tuple[str, ...], self.include_markets),
            exclude_markets=cast(tuple[str, ...], self.exclude_markets),
            run_stochastic_process=self.run_stochastic_process,
            rng=copy.deepcopy(self._rng),
        )
        return MarketSelectionResults(candidates)


def _select_markets(
    data: pl.DataFrame,
    *,
    market_counts: Sequence[int],
    location_col: str,
    time_col: str,
    outcome_col: str,
    include_markets: Sequence[str],
    exclude_markets: Sequence[str],
    run_stochastic_process: bool,
    rng: np.random.Generator | None,
) -> pl.DataFrame:
    panel = _normalize_locations(data, location_col=location_col)
    original_markets = data.get_column(location_col).unique(maintain_order=True).to_list()
    normalized_markets = panel.get_column(location_col).unique(maintain_order=True).to_list()
    original_identifier = dict(zip(normalized_markets, original_markets, strict=True))
    panel = _validate_panel(
        panel,
        location_col=location_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )
    all_markets = panel.get_column(location_col).unique(maintain_order=True).to_list()

    _validate_known_markets(include_markets, all_markets, name="include_markets")
    _validate_known_markets(exclude_markets, all_markets, name="exclude_markets")

    eligible_markets = [market for market in all_markets if market not in exclude_markets]
    market_counts = _limit_stochastic_market_counts(
        market_counts,
        eligible_market_count=len(eligible_markets),
        run_stochastic_process=run_stochastic_process,
    )
    _validate_group_sizes(
        market_counts,
        included=include_markets,
        eligible_markets=eligible_markets,
        all_markets=all_markets,
    )
    ranked_markets = _rank_eligible_markets(
        panel,
        eligible_markets=eligible_markets,
        location_col=location_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )

    records: list[dict[str, object]] = []
    for market_count in market_counts:
        groups = _candidate_groups(
            ranked_markets,
            market_count,
            run_stochastic_process=run_stochastic_process,
            rng=rng,
        )
        groups = [group for group in groups if set(include_markets).issubset(group)]
        if not groups:
            raise ValueError(
                "No candidate treatment groups satisfy include_markets "
                f"for market_count={market_count}"
            )

        records.extend(
            {
                "market_count": market_count,
                "candidate_id": candidate_id,
                "treatment_markets": [original_identifier[market] for market in group],
                "donor_markets": [
                    original_identifier[market] for market in all_markets if market not in group
                ],
            }
            for candidate_id, group in enumerate(groups, start=1)
        )

    return pl.DataFrame(records)


def _rank_eligible_markets(
    panel: pl.DataFrame,
    *,
    eligible_markets: Sequence[str],
    location_col: str,
    time_col: str,
    outcome_col: str,
) -> dict[str, list[str]]:
    if len(eligible_markets) == 1:
        market = eligible_markets[0]
        return {market: [market]}

    outcomes = (
        panel.filter(pl.col(location_col).is_in(eligible_markets))
        .pivot(on=location_col, index=time_col, values=outcome_col)
        .sort(time_col)
        .select(eligible_markets)
    )
    constant_markets = [
        market for market in eligible_markets if outcomes.get_column(market).n_unique() == 1
    ]
    if constant_markets:
        raise ValueError(f"Cannot correlate constant markets: {constant_markets}")

    correlations = np.corrcoef(outcomes.to_numpy(), rowvar=False)
    return {
        anchor: [
            anchor,
            *[
                eligible_markets[index]
                for index in sorted(
                    (index for index in range(len(eligible_markets)) if index != anchor_index),
                    key=lambda index: correlations[anchor_index, index],
                    reverse=True,
                )
            ],
        ]
        for anchor_index, anchor in enumerate(eligible_markets)
    }


def _validate_panel(
    data: pl.DataFrame,
    *,
    location_col: str,
    time_col: str,
    outcome_col: str,
) -> pl.DataFrame:
    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a Polars DataFrame")

    required = [location_col, time_col, outcome_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    panel = data.select(required)
    if any(panel.null_count().row(0)):
        raise ValueError("Market selection requires a complete panel without missing values")
    if panel.select(pl.struct(location_col, time_col).is_duplicated().any()).item():
        raise ValueError("Panel contains duplicate market-time observations")
    outcome = panel.get_column(outcome_col)
    if not outcome.dtype.is_numeric():
        raise ValueError(f"Outcome column '{outcome_col}' must be numeric")
    if not outcome.is_finite().all():
        raise ValueError(f"Outcome column '{outcome_col}' must contain only finite values")

    market_count = panel.get_column(location_col).n_unique()
    time_count = panel.get_column(time_col).n_unique()
    if panel.height != market_count * time_count:
        raise ValueError("Market selection requires one observation per market and time")
    if market_count < 2:
        raise ValueError("Market selection requires at least two markets")

    return panel


def _normalize_locations(data: pl.DataFrame, *, location_col: str) -> pl.DataFrame:
    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a Polars DataFrame")
    if location_col not in data.columns:
        raise ValueError(f"Data is missing required columns: ['{location_col}']")
    if data.get_column(location_col).null_count():
        raise ValueError("Market selection requires market identifiers without missing values")

    panel = data.with_columns(pl.col(location_col).cast(pl.String).str.to_lowercase())
    if panel.get_column(location_col).n_unique() != data.get_column(location_col).n_unique():
        raise ValueError("Market identifiers must remain unique after case normalization")
    return panel


def _normalize_market_list(markets: Iterable[object], *, name: str) -> tuple[str, ...]:
    values = (markets,) if isinstance(markets, str) else markets
    normalized = tuple(str(market).lower() for market in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate markets")
    return normalized


def _limit_stochastic_market_counts(
    market_counts: Sequence[int],
    *,
    eligible_market_count: int,
    run_stochastic_process: bool,
) -> tuple[int, ...]:
    counts = tuple(market_counts)
    if not run_stochastic_process:
        return counts
    if eligible_market_count < 2:
        raise ValueError(
            "Stochastic market selection requires at least two treatment-eligible markets"
        )

    maximum = eligible_market_count // 2
    limited = tuple(count for count in counts if count <= maximum)
    return limited or (maximum,)


def _validate_column_names(location_col: str, time_col: str, outcome_col: str) -> None:
    columns = (location_col, time_col, outcome_col)
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Column names must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("location_col, time_col, and outcome_col must be distinct")


def _validate_known_markets(
    markets: Iterable[str], all_markets: Sequence[str], *, name: str
) -> None:
    unknown = sorted(set(markets) - set(all_markets))
    if unknown:
        raise ValueError(f"Markets in {name} not found in the data: {unknown}")


def _validate_market_counts(market_counts: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(market_counts)
    if not counts:
        raise ValueError("market_counts must contain at least one treatment-group size")
    invalid = (
        not isinstance(count, Integral) or isinstance(count, bool) or count <= 0 for count in counts
    )
    if any(invalid):
        raise ValueError("market_counts must contain positive integers")
    if len(counts) != len(set(counts)):
        raise ValueError("market_counts must not contain duplicates")
    return counts


def _validate_group_sizes(
    counts: Sequence[int],
    *,
    included: Sequence[str],
    eligible_markets: Sequence[str],
    all_markets: Sequence[str],
) -> None:
    if any(count < len(included) for count in counts):
        raise ValueError("market_counts cannot be smaller than the number of included markets")
    if any(count > len(eligible_markets) for count in counts):
        raise ValueError("market_counts cannot exceed the number of treatment-eligible markets")
    if any(count >= len(all_markets) for count in counts):
        raise ValueError("Each candidate treatment group must leave at least one donor market")


def _candidate_groups(
    ranked_markets: dict[str, list[str]],
    market_count: int,
    *,
    run_stochastic_process: bool = False,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, ...]]:
    if not run_stochastic_process:
        groups = [tuple(sorted(markets[:market_count])) for markets in ranked_markets.values()]
    else:
        if rng is None:  # pragma: no cover
            raise ValueError("A random generator is required for stochastic selection")
        selected_positions = [2 * pair + int(rng.integers(0, 2)) for pair in range(market_count)]
        groups = [
            tuple(sorted(markets[position] for position in selected_positions))
            for markets in ranked_markets.values()
        ]
    return list(dict.fromkeys(groups))
