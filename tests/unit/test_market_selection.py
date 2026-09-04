import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from augsynth_py.geoexp import MarketSelectionResults, MarketSelector


@pytest.fixture
def panel() -> pl.DataFrame:
    values = {
        "A": [1, 2, 4, 8, 7, 3],
        "B": [2, 3, 5, 7, 6, 5],
        "C": [4, 1, 3, 9, 2, 8],
        "D": [8, 2, 6, 1, 4, 0],
    }
    return pl.DataFrame(
        [
            {"time": time, "location": location, "Y": outcome}
            for time in range(1, 7)
            for location, outcomes in values.items()
            for outcome in [outcomes[time - 1]]
        ]
    )


def test_market_selector_returns_unique_sorted_treatment_groups(panel: pl.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1, 2]).select(panel)

    expected = pl.DataFrame(
        {
            "market_count": [1, 1, 1, 1, 2, 2, 2],
            "candidate_id": [1, 2, 3, 4, 1, 2, 3],
            "treatment_markets": [
                ("A",),
                ("B",),
                ("C",),
                ("D",),
                ("A", "B"),
                ("B", "C"),
                ("A", "D"),
            ],
            "donor_markets": [
                ("B", "C", "D"),
                ("A", "C", "D"),
                ("A", "B", "D"),
                ("A", "B", "C"),
                ("C", "D"),
                ("A", "D"),
                ("B", "C"),
            ],
        }
    )

    assert isinstance(selection, MarketSelectionResults)
    assert_frame_equal(selection.as_df(), expected)


def test_samples_follow_candidate_row_order_across_market_counts(panel: pl.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1, 2]).select(panel)

    candidates = selection.as_df()
    samples = selection.as_samples()

    assert list(samples) == [f"sample{number}" for number in range(1, 8)]
    for sample, candidate in zip(samples.values(), candidates.iter_rows(named=True), strict=True):
        assert sample == {
            "pod_A": candidate["treatment_markets"],
            "pod_B": candidate["donor_markets"],
        }


def test_stochastic_selector_samples_ranked_pairs_reproducibly(panel: pl.DataFrame) -> None:
    selector = MarketSelector(
        market_counts=[2],
        run_stochastic_process=True,
        rng=np.random.default_rng(0),
    )

    first = selector.select(panel).as_df()
    second = selector.select(panel).as_df()

    expected = pl.DataFrame(
        {
            "market_count": [2, 2, 2],
            "candidate_id": [1, 2, 3],
            "treatment_markets": [("B", "D"), ("A", "D"), ("A", "B")],
            "donor_markets": [
                ("A", "C"),
                ("B", "C"),
                ("C", "D"),
            ],
        }
    )

    assert_frame_equal(first, expected)
    assert_frame_equal(second, expected)


def test_excluded_markets_are_not_selected_but_remain_donors(panel: pl.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1], exclude_markets=["d"]).select(panel)

    result = selection.as_df()
    assert result.get_column("treatment_markets").to_list() == [["A"], ["B"], ["C"]]
    assert result[0, "donor_markets"].to_list() == ["B", "C", "D"]
    assert all("D" not in sample["pod_A"] for sample in selection.as_samples().values())
    assert all("D" in sample["pod_B"] for sample in selection.as_samples().values())


def test_samples_partition_every_market(panel: pl.DataFrame) -> None:
    samples = MarketSelector(market_counts=[1, 2]).select(panel).as_samples()

    for sample in samples.values():
        donors = set(sample["pod_A"])
        treatment = set(sample["pod_B"])
        assert donors.isdisjoint(treatment)
        assert donors | treatment == {"A", "B", "C", "D"}


def test_include_markets_filters_complete_candidates(panel: pl.DataFrame) -> None:
    result = MarketSelector(market_counts=[2], include_markets="a").select(panel).as_df()

    assert result.get_column("treatment_markets").to_list() == [["A", "B"], ["A", "D"]]


def test_numeric_market_identifiers_are_preserved(panel: pl.DataFrame) -> None:
    numeric_panel = panel.with_columns(
        pl.col("location").replace_strict({"A": 1, "B": 2, "C": 3, "D": 4})
    )

    selection = MarketSelector(market_counts=[1]).select(numeric_panel)

    candidates = selection.as_df()
    assert candidates.get_column("treatment_markets").to_list() == [[1], [2], [3], [4]]
    assert candidates[0, "donor_markets"].to_list() == [2, 3, 4]
    assert selection.as_samples()["sample1"] == {"pod_A": [1], "pod_B": [2, 3, 4]}
    assert all(type(market) is int for group in candidates["treatment_markets"] for market in group)
    assert all(
        type(market) is int
        for sample in selection.as_samples().values()
        for pod in sample.values()
        for market in pod
    )


def test_result_conversions_return_defensive_copies(panel: pl.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1]).select(panel)
    expected_candidates = selection.as_df()
    expected_samples = selection.as_samples()

    candidates = selection.as_df()
    samples = selection.as_samples()
    samples["sample1"]["pod_A"].append("changed")
    samples["sample1"]["pod_B"].clear()
    samples.pop("sample2")

    assert candidates is not selection.as_df()
    assert_frame_equal(selection.as_df(), expected_candidates)
    assert selection.as_samples() == expected_samples


def test_selector_snapshots_configuration_and_does_not_mutate_data(panel: pl.DataFrame) -> None:
    market_counts = [1]
    include_markets = ["A"]
    selector = MarketSelector(
        market_counts=market_counts,
        include_markets=include_markets,
    )
    original = panel.clone()

    market_counts.append(2)
    include_markets.append("B")
    first = selector.select(panel).as_df()
    second = selector.select(panel).as_df()

    assert selector.market_counts == (1,)
    assert selector.include_markets == ("a",)
    assert_frame_equal(first, second)
    assert_frame_equal(panel, original)


def test_selector_validates_configuration_on_construction() -> None:
    with pytest.raises(ValueError, match="market_counts"):
        MarketSelector(market_counts=[])

    with pytest.raises(ValueError, match="positive integers"):
        MarketSelector(market_counts=[0])

    with pytest.raises(ValueError, match="overlap"):
        MarketSelector(
            market_counts=[1],
            include_markets=["A"],
            exclude_markets=["a"],
        )

    with pytest.raises(ValueError, match="distinct"):
        MarketSelector(market_counts=[1], location_col="market", time_col="market")

    with pytest.raises(ValueError, match="requires rng"):
        MarketSelector(market_counts=[1], run_stochastic_process=True)

    with pytest.raises(TypeError, match="rng"):
        MarketSelector(market_counts=[1], rng=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="run_stochastic_process"):
        MarketSelector(market_counts=[1], run_stochastic_process=1)


def test_stochastic_selection_limits_treatment_size_to_half_eligible_markets(
    panel: pl.DataFrame,
) -> None:
    result = (
        MarketSelector(
            market_counts=[3],
            run_stochastic_process=True,
            rng=np.random.default_rng(0),
        )
        .select(panel)
        .as_df()
    )

    assert result.get_column("market_count").to_list() == [2, 2, 2]


def test_selector_supports_custom_panel_columns(panel: pl.DataFrame) -> None:
    custom_panel = panel.rename({"location": "market", "time": "date", "Y": "revenue"})
    selector = MarketSelector(
        market_counts=[1],
        location_col="market",
        time_col="date",
        outcome_col="revenue",
    )

    result = selector.select(custom_panel).as_df()

    assert result.get_column("treatment_markets").to_list() == [["A"], ["B"], ["C"], ["D"]]


def test_invalid_market_selection_inputs_raise_clear_errors(panel: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="Polars DataFrame"):
        MarketSelector(market_counts=[1]).select([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not found"):
        MarketSelector(market_counts=[1], include_markets=["missing"]).select(panel)

    with pytest.raises(ValueError, match="duplicate"):
        MarketSelector(market_counts=[1]).select(
            panel.with_row_index().with_columns(
                pl.when(pl.col("index") == 0).then(2).otherwise(pl.col("time")).alias("time")
            )
        )

    with pytest.raises(ValueError, match="identifiers without missing values"):
        MarketSelector(market_counts=[1]).select(
            panel.with_row_index().with_columns(
                pl.when(pl.col("index") == 0)
                .then(None)
                .otherwise(pl.col("location"))
                .alias("location")
            )
        )

    with pytest.raises(ValueError, match="unique after case normalization"):
        MarketSelector(market_counts=[1]).select(
            panel.with_columns(pl.col("location").replace("B", "a"))
        )


@pytest.mark.parametrize("column", ["location", "time", "Y"])
def test_required_columns_are_validated(panel: pl.DataFrame, column: str) -> None:
    missing = panel.drop(column)

    with pytest.raises(ValueError, match="required columns"):
        MarketSelector(market_counts=[1]).select(missing)


def test_constant_market_is_rejected(panel: pl.DataFrame) -> None:
    panel = panel.with_columns(
        pl.when(pl.col("location") == "D").then(1).otherwise(pl.col("Y")).alias("Y")
    )

    with pytest.raises(ValueError, match="constant"):
        MarketSelector(market_counts=[1]).select(panel)


def test_perfect_correlation_does_not_remove_anchor_candidates(panel: pl.DataFrame) -> None:
    a_outcomes = dict(panel.filter(pl.col("location") == "A").select("time", "Y").iter_rows())
    panel = panel.with_columns(
        pl.when(pl.col("location") == "B")
        .then(pl.col("time").replace_strict(a_outcomes))
        .otherwise(pl.col("Y"))
        .alias("Y")
    )

    result = MarketSelector(market_counts=[1]).select(panel).as_df()

    assert result.get_column("treatment_markets").to_list() == [["A"], ["B"], ["C"], ["D"]]


def test_one_eligible_market_can_use_excluded_markets_as_donors(panel: pl.DataFrame) -> None:
    result = (
        MarketSelector(
            market_counts=[1],
            exclude_markets=["B", "C", "D"],
        )
        .select(panel)
        .as_df()
    )

    assert result[0, "treatment_markets"].to_list() == ["A"]
    assert result[0, "donor_markets"].to_list() == ["B", "C", "D"]


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_non_finite_outcomes_are_rejected(panel: pl.DataFrame, value: float) -> None:
    panel = panel.with_row_index().with_columns(
        pl.when(pl.col("index") == 0).then(value).otherwise(pl.col("Y")).cast(pl.Float64).alias("Y")
    )

    with pytest.raises(ValueError, match="finite"):
        MarketSelector(market_counts=[1]).select(panel)
