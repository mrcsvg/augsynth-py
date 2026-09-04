import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from augsynth_py.geoexp import MarketSelectionResults, MarketSelector


@pytest.fixture
def panel() -> pd.DataFrame:
    values = {
        "A": [1, 2, 4, 8, 7, 3],
        "B": [2, 3, 5, 7, 6, 5],
        "C": [4, 1, 3, 9, 2, 8],
        "D": [8, 2, 6, 1, 4, 0],
    }
    return pd.DataFrame(
        [
            {"time": time, "location": location, "Y": outcome}
            for time in range(1, 7)
            for location, outcomes in values.items()
            for outcome in [outcomes[time - 1]]
        ]
    )


def test_market_selector_returns_unique_sorted_treatment_groups(panel: pd.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1, 2]).select(panel)

    expected = pd.DataFrame(
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


def test_samples_follow_candidate_row_order_across_market_counts(panel: pd.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1, 2]).select(panel)

    candidates = selection.as_df()
    samples = selection.as_samples()

    assert list(samples) == [f"sample{number}" for number in range(1, 8)]
    for sample, candidate in zip(samples.values(), candidates.itertuples(), strict=True):
        assert sample == {
            "pod_A": list(candidate.treatment_markets),
            "pod_B": list(candidate.donor_markets),
        }


def test_stochastic_selector_samples_ranked_pairs_reproducibly(panel: pd.DataFrame) -> None:
    selector = MarketSelector(
        market_counts=[2],
        run_stochastic_process=True,
        rng=np.random.default_rng(0),
    )

    first = selector.select(panel).as_df()
    second = selector.select(panel).as_df()

    expected = pd.DataFrame(
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


def test_excluded_markets_are_not_selected_but_remain_donors(panel: pd.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1], exclude_markets=["d"]).select(panel)

    result = selection.as_df()
    assert result["treatment_markets"].tolist() == [("A",), ("B",), ("C",)]
    assert result.loc[0, "donor_markets"] == ("B", "C", "D")
    assert all("D" not in sample["pod_A"] for sample in selection.as_samples().values())
    assert all("D" in sample["pod_B"] for sample in selection.as_samples().values())


def test_samples_partition_every_market(panel: pd.DataFrame) -> None:
    samples = MarketSelector(market_counts=[1, 2]).select(panel).as_samples()

    for sample in samples.values():
        donors = set(sample["pod_A"])
        treatment = set(sample["pod_B"])
        assert donors.isdisjoint(treatment)
        assert donors | treatment == {"A", "B", "C", "D"}


def test_include_markets_filters_complete_candidates(panel: pd.DataFrame) -> None:
    result = MarketSelector(market_counts=[2], include_markets="a").select(panel).as_df()

    assert result["treatment_markets"].tolist() == [("A", "B"), ("A", "D")]


def test_numeric_market_identifiers_are_preserved(panel: pd.DataFrame) -> None:
    numeric_panel = panel.assign(location=panel["location"].map({"A": 1, "B": 2, "C": 3, "D": 4}))

    selection = MarketSelector(market_counts=[1]).select(numeric_panel)

    candidates = selection.as_df()
    assert candidates["treatment_markets"].tolist() == [(1,), (2,), (3,), (4,)]
    assert candidates.loc[0, "donor_markets"] == (2, 3, 4)
    assert selection.as_samples()["sample1"] == {"pod_A": [1], "pod_B": [2, 3, 4]}
    assert all(type(market) is int for group in candidates["treatment_markets"] for market in group)
    assert all(
        type(market) is int
        for sample in selection.as_samples().values()
        for pod in sample.values()
        for market in pod
    )


def test_result_conversions_return_defensive_copies(panel: pd.DataFrame) -> None:
    selection = MarketSelector(market_counts=[1]).select(panel)
    expected_candidates = selection.as_df()
    expected_samples = selection.as_samples()

    candidates = selection.as_df()
    candidates.loc[0, "treatment_markets"] = ("changed",)
    samples = selection.as_samples()
    samples["sample1"]["pod_A"].append("changed")
    samples["sample1"]["pod_B"].clear()
    samples.pop("sample2")

    assert_frame_equal(selection.as_df(), expected_candidates)
    assert selection.as_samples() == expected_samples


def test_selector_snapshots_configuration_and_does_not_mutate_data(panel: pd.DataFrame) -> None:
    market_counts = [1]
    include_markets = ["A"]
    selector = MarketSelector(
        market_counts=market_counts,
        include_markets=include_markets,
    )
    original = panel.copy(deep=True)

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
    panel: pd.DataFrame,
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

    assert result["market_count"].tolist() == [2, 2, 2]


def test_selector_supports_custom_panel_columns(panel: pd.DataFrame) -> None:
    custom_panel = panel.rename(columns={"location": "market", "time": "date", "Y": "revenue"})
    selector = MarketSelector(
        market_counts=[1],
        location_col="market",
        time_col="date",
        outcome_col="revenue",
    )

    result = selector.select(custom_panel).as_df()

    assert result["treatment_markets"].tolist() == [("A",), ("B",), ("C",), ("D",)]


def test_invalid_market_selection_inputs_raise_clear_errors(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not found"):
        MarketSelector(market_counts=[1], include_markets=["missing"]).select(panel)

    with pytest.raises(ValueError, match="duplicate"):
        MarketSelector(market_counts=[1]).select(
            panel.assign(time=panel["time"].where(panel.index != 0, 2))
        )

    with pytest.raises(ValueError, match="identifiers without missing values"):
        MarketSelector(market_counts=[1]).select(
            panel.assign(location=panel["location"].where(panel.index != 0))
        )

    with pytest.raises(ValueError, match="unique after case normalization"):
        MarketSelector(market_counts=[1]).select(
            panel.assign(location=panel["location"].replace("B", "a"))
        )


@pytest.mark.parametrize("column", ["location", "time", "Y"])
def test_required_columns_are_validated(panel: pd.DataFrame, column: str) -> None:
    missing = panel.drop(columns=column)

    with pytest.raises(ValueError, match="required columns"):
        MarketSelector(market_counts=[1]).select(missing)


def test_constant_market_is_rejected(panel: pd.DataFrame) -> None:
    panel = panel.copy()
    panel.loc[panel["location"] == "D", "Y"] = 1

    with pytest.raises(ValueError, match="constant"):
        MarketSelector(market_counts=[1]).select(panel)


def test_perfect_correlation_does_not_remove_anchor_candidates(panel: pd.DataFrame) -> None:
    panel = panel.copy()
    panel.loc[panel["location"] == "B", "Y"] = panel.loc[panel["location"] == "A", "Y"].to_numpy()

    result = MarketSelector(market_counts=[1]).select(panel).as_df()

    assert result["treatment_markets"].tolist() == [("A",), ("B",), ("C",), ("D",)]


def test_one_eligible_market_can_use_excluded_markets_as_donors(panel: pd.DataFrame) -> None:
    result = (
        MarketSelector(
            market_counts=[1],
            exclude_markets=["B", "C", "D"],
        )
        .select(panel)
        .as_df()
    )

    assert result.loc[0, "treatment_markets"] == ("A",)
    assert result.loc[0, "donor_markets"] == ("B", "C", "D")


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_non_finite_outcomes_are_rejected(panel: pd.DataFrame, value: float) -> None:
    panel = panel.astype({"Y": float})
    panel.loc[0, "Y"] = value

    with pytest.raises(ValueError, match="finite"):
        MarketSelector(market_counts=[1]).select(panel)
