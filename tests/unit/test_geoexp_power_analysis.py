import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from augsynth_py import DEFAULT_EFFECT_SIZES, PowerResults, Synth
from augsynth_py.geoexp import (
    GeoLiftPowerAnalysis,
    GeoLiftPowerAnalysisResults,
)


def make_panel() -> pd.DataFrame:
    outcomes = {
        "A": [20, 22, 21, 25, 24, 28, 27, 30, 32, 31, 34, 36],
        "B": [10, 11, 11, 13, 12, 14, 14, 15, 16, 16, 17, 18],
        "C": [7, 8, 7, 9, 9, 10, 9, 11, 11, 12, 12, 13],
        "outside": [100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1],
    }
    return pd.DataFrame(
        [
            {"time": time, "location": location, "Y": values[time - 1]}
            for time in range(1, 13)
            for location, values in outcomes.items()
        ]
    )


def test_evaluator_returns_enriched_results_for_a_pandas_panel() -> None:
    panel = make_panel()
    original = panel.copy(deep=True)
    samples = {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}}

    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=[2],
        effect_sizes=[0.0, 0.2],
        lookback_window=2,
        permutation_type="block",
        treatment_pod="pod_A",
    ).evaluate(panel, samples)

    simulations = results.as_df()

    assert isinstance(results, GeoLiftPowerAnalysisResults)
    assert isinstance(results.get_power_results("candidate"), PowerResults)
    assert simulations[["duration", "effect_size", "window"]].to_dict("records") == [
        {"duration": 2, "effect_size": 0.0, "window": 0},
        {"duration": 2, "effect_size": 0.0, "window": 1},
        {"duration": 2, "effect_size": 0.2, "window": 0},
        {"duration": 2, "effect_size": 0.2, "window": 1},
    ]
    assert simulations["sample"].tolist() == ["candidate"] * 4
    assert simulations["treatment_markets"].tolist() == [("A",)] * 4
    assert simulations["donor_markets"].tolist() == [("B", "C")] * 4
    assert simulations["treatment_pod"].tolist() == ["pod_A"] * 4
    assert simulations["detected_lift"].notna().all()
    assert set(results.get_power_results("candidate").params.treated) == {"A"}
    assert_frame_equal(panel, original)
    assert samples == {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}}


def test_detected_lift_uses_the_post_period_synthetic_total() -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0, 0.2],
        lookback_window=2,
        permutation_type="block",
        treatment_pod="pod_A",
    ).evaluate(
        make_panel(),
        {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}},
    )

    simulation = results.as_df(effect_size=0.2, window=0).iloc[0]

    assert simulation["injected_treated_post_total"] == pytest.approx(84.0)
    assert simulation["estimated_incrementality"] == pytest.approx(23.4)
    assert simulation["synthetic_post_total"] == pytest.approx(60.6)
    assert simulation["detected_lift"] == pytest.approx(0.3861386138613861)
    assert simulation["detected_lift"] != pytest.approx(simulation["att_pct"])

    design = results.evaluated_designs(target_power=0.1).iloc[0]
    assert design["h1_calibration_error"] == 0.164
    assert design["null_bias"] == pytest.approx(0.1368311833297224)


def test_collection_methods_filter_and_return_defensive_pandas_copies() -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=[2, 3],
        effect_sizes=[0.0, 0.2],
        lookback_window=2,
        permutation_type="block",
        treatment_pod="pod_A",
    ).evaluate(
        make_panel(),
        {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}},
    )

    simulations = results.as_df(duration=2, effect_size=0.2, window=1)
    curve = results.power_curve(duration=2, effect_size=0.2)
    mde = results.mde(target_power=0.1, duration=2)
    designs = results.evaluated_designs(target_power=0.1, duration=2)

    assert isinstance(simulations, pd.DataFrame) and len(simulations) == 1
    assert isinstance(curve, pd.DataFrame) and len(curve) == 1
    assert isinstance(mde, pd.DataFrame) and mde.to_dict("records") == [
        {"sample": "candidate", "duration": 2, "mde": 0.2}
    ]
    assert isinstance(designs, pd.DataFrame) and len(designs) == 1
    simulations.loc[0, "sample"] = "changed"
    curve.loc[0, "power"] = -1
    designs.loc[0, "n_failed"] = 99

    assert results.as_df().loc[0, "sample"] == "candidate"
    assert results.power_curve().loc[0, "power"] >= 0
    assert results.evaluated_designs(target_power=0.1).loc[0, "n_failed"] == 0
    assert results.as_df(sample="unknown").empty
    assert results.power_curve(sample="unknown").empty
    assert results.mde(sample="unknown").empty
    assert results.evaluated_designs(sample="unknown").empty
    with pytest.raises(KeyError, match="Unknown sample"):
        results.get_power_results("unknown")


def test_reordering_samples_does_not_change_iid_results() -> None:
    panel = make_panel()
    samples = {
        "first": {"pod_A": ["A"], "pod_B": ["B", "C"]},
        "second": {"pod_A": ["B"], "pod_B": ["A", "C"]},
    }
    evaluator = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0, 0.2],
        treatment_pod="pod_A",
        permutation_type="iid",
        ns=25,
        rng=np.random.default_rng(1987),
    )

    first = evaluator.evaluate(panel, samples).as_df()
    repeated = evaluator.evaluate(panel, samples).as_df()
    reversed_samples = evaluator.evaluate(panel, dict(reversed(samples.items()))).as_df()

    comparison_columns = [
        "sample",
        "duration",
        "effect_size",
        "window",
        "pvalue",
        "att",
        "detected_lift",
    ]
    sort_columns = ["sample", "duration", "effect_size", "window"]
    assert_frame_equal(
        first[comparison_columns].sort_values(sort_columns).reset_index(drop=True),
        repeated[comparison_columns].sort_values(sort_columns).reset_index(drop=True),
    )
    assert_frame_equal(
        first[comparison_columns].sort_values(sort_columns).reset_index(drop=True),
        reversed_samples[comparison_columns].sort_values(sort_columns).reset_index(drop=True),
    )
    assert reversed_samples["sample"].drop_duplicates().tolist() == ["second", "first"]


def test_samples_are_sequential_and_n_jobs_reaches_simulate_power(monkeypatch) -> None:
    calls = []

    def simulate_spy(panel, **kwargs):
        calls.append(
            {
                "markets": sorted(panel.get_column("location").unique().to_list()),
                "treated": kwargs["treated"],
                "n_jobs": kwargs["n_jobs"],
            }
        )
        return PowerResults(
            simulations=pl.DataFrame(
                {
                    "duration": [2],
                    "effect_size": [0.0],
                    "window": [0],
                    "treatment_start": [11],
                    "treatment_end": [12],
                    "n_pre_periods": [10],
                    "pvalue": [0.5],
                    "att": [1.0],
                    "att_pct": [0.1],
                    "rmspe_pre": [0.05],
                    "error": [None],
                },
                schema_overrides={"error": pl.String},
            ),
            alpha=kwargs["alpha"],
            effect_type=kwargs["effect_type"],
        )

    monkeypatch.setattr(
        "augsynth_py.geoexp.power_analysis.simulate_power",
        simulate_spy,
    )
    samples = {
        "first": {"pod_A": ["A"], "pod_B": ["B"]},
        "second": {"pod_A": ["C"], "pod_B": ["A"]},
    }

    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0],
        treatment_pod="pod_A",
        permutation_type="block",
        n_jobs=3,
    ).evaluate(make_panel(), samples)

    assert results.as_df()["sample"].tolist() == ["first", "second"]
    assert calls == [
        {"markets": ["A", "B"], "treated": ("A",), "n_jobs": 3},
        {"markets": ["A", "C"], "treated": ("C",), "n_jobs": 3},
    ]


def test_recorded_failures_remain_visible_in_evaluated_designs() -> None:
    configuration = {
        "estimator": Synth(),
        "durations": 2,
        "effect_sizes": [0.0, 0.2],
        "treatment_pod": "pod_A",
        "permutation_type": "block",
        "block_size": 20,
        "rng": np.random.default_rng(1),
    }
    samples = {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}}

    with pytest.raises(ValueError, match="block_size"):
        GeoLiftPowerAnalysis(**configuration).evaluate(make_panel(), samples)

    results = GeoLiftPowerAnalysis(**configuration, on_error="record").evaluate(
        make_panel(), samples
    )
    designs = results.evaluated_designs()

    assert results.as_df()["error"].notna().all()
    assert designs.loc[0, "mde"] is None or pd.isna(designs.loc[0, "mde"])
    assert designs.loc[0, "n_native_failed"] == 2
    assert designs.loc[0, "n_failed"] == 2
    assert results.rank_designs().loc[0, "sample"] == "candidate"


def test_custom_ranking_can_join_filter_and_sort_without_mutating_results() -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0, 0.2],
        treatment_pod="pod_A",
        permutation_type="block",
    ).evaluate(
        make_panel(),
        {
            "first": {"pod_A": ["A"], "pod_B": ["B", "C"]},
            "second": {"pod_A": ["B"], "pod_B": ["A", "C"]},
        },
    )
    external = pd.DataFrame({"sample": ["first", "second"], "outcome_share": [0.6, 0.2]})

    def eligible_lowest_mde(designs, *, market_data, maximum_share):
        designs.loc[:, "temporary"] = True
        return (
            designs.merge(market_data, on="sample")
            .query("outcome_share <= @maximum_share")
            .sort_values("mde")
        )

    ranked = results.rank_designs(
        eligible_lowest_mde,
        target_power=0.1,
        market_data=external,
        maximum_share=0.3,
    )
    best = results.get_best_design(
        eligible_lowest_mde,
        target_power=0.1,
        market_data=external,
        maximum_share=0.3,
    )

    assert ranked["sample"].tolist() == ["second"]
    assert best["sample"].tolist() == ["second"]
    assert "temporary" not in results.evaluated_designs(target_power=0.1)


@pytest.mark.parametrize(
    ("ranking", "error", "message"),
    [
        (lambda designs: [], TypeError, "pandas DataFrame"),
        (lambda designs: designs.drop(columns="sample"), ValueError, "required columns"),
    ],
)
def test_custom_ranking_contract_is_validated(ranking, error, message) -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0, 0.2],
        treatment_pod="pod_A",
        permutation_type="block",
    ).evaluate(
        make_panel(),
        {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}},
    )

    with pytest.raises(error, match=message):
        results.rank_designs(ranking, target_power=0.1)


def test_geolift_ranking_uses_treatment_markets_to_break_ties(monkeypatch) -> None:
    designs = pd.DataFrame(
        {
            "sample": ["C", "D", "A", "B", "missing", "failed"],
            "treatment_markets": [("a",), ("a",), ("z",), ("b",), ("c",), ("d",)],
            "duration": [3, 1, 2, 2, 2, 2],
            "mde": [0.1, 0.1, 0.1, 0.2, np.nan, 0.15],
            "power_at_mde": [0.8, 0.8, 0.8, 0.9, np.nan, 0.9],
            "h1_calibration_error": [0.05049, 0.05045, 0.0504, 0.1, np.nan, 0.0],
            "n_failed": [0, 0, 0, 0, 0, 1],
        }
    )
    results = GeoLiftPowerAnalysisResults({}, {}, pd.DataFrame(), "multiplicative")
    monkeypatch.setattr(
        GeoLiftPowerAnalysisResults,
        "evaluated_designs",
        lambda self, **kwargs: designs.copy(deep=True),
    )

    ranked = results.rank_designs()

    assert ranked["sample"].tolist() == ["C", "D", "A", "B", "failed", "missing"]
    assert ranked["rank"].tolist() == [1.0, 1.0, 1.0, 4.0, 5.0, 6.0]


def test_explicit_treatment_pod_controls_effect_injection() -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0],
        treatment_pod="pod_B",
        permutation_type="block",
    ).evaluate(
        make_panel(),
        {"candidate": {"pod_A": ["B", "C"], "pod_B": ["A"]}},
    )

    simulations = results.as_df()

    assert results.get_power_results("candidate").params.treated == ("A",)
    assert simulations.loc[0, "treatment_markets"] == ("A",)
    assert simulations.loc[0, "donor_markets"] == ("B", "C")
    assert simulations.loc[0, "treatment_pod"] == "pod_B"


def test_additive_effects_use_outcome_units_for_h1_calibration() -> None:
    results = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        effect_sizes=[0.0, 5.0],
        effect_type="additive",
        treatment_pod="pod_A",
        permutation_type="block",
    ).evaluate(
        make_panel(),
        {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}},
    )

    simulation = results.as_df(effect_size=5.0).iloc[0]
    design = results.evaluated_designs(target_power=0.1).iloc[0]

    assert simulation["injected_treated_post_total"] == pytest.approx(80.0)
    assert design["mde"] == 5.0
    assert design["h1_calibration_error"] == pytest.approx(
        round(abs(design["average_att"] - 5.0), 3)
    )
    with pytest.raises(ValueError, match="multiplicative"):
        results.rank_designs(target_power=0.1)


def test_evaluator_snapshots_its_estimator_configuration() -> None:
    estimator = Synth(fixedeff=True)
    evaluator = GeoLiftPowerAnalysis(
        estimator=estimator,
        durations=2,
        effect_sizes=[0.0],
        treatment_pod="pod_A",
        permutation_type="block",
    )
    estimator.fixedeff = False

    samples = {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}}
    result = evaluator.evaluate(make_panel(), samples).as_df()
    expected = (
        GeoLiftPowerAnalysis(
            estimator=Synth(fixedeff=True),
            durations=2,
            effect_sizes=[0.0],
            treatment_pod="pod_A",
            permutation_type="block",
        )
        .evaluate(make_panel(), samples)
        .as_df()
    )

    assert_frame_equal(result, expected)
    assert not hasattr(evaluator, "estimator")


def test_defaults_reuse_the_core_effect_grid_and_deterministic_permutations() -> None:
    evaluator = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        treatment_pod="pod_A",
    )

    assert evaluator.effect_sizes == DEFAULT_EFFECT_SIZES
    assert evaluator.permutation_type == "block"


def test_constructor_and_assignment_contracts_fail_clearly() -> None:
    with pytest.raises(ValueError, match="require rng"):
        GeoLiftPowerAnalysis(
            estimator=Synth(),
            durations=2,
            treatment_pod="pod_A",
            permutation_type="iid",
        )
    with pytest.raises(TypeError, match="rng"):
        GeoLiftPowerAnalysis(
            estimator=Synth(),
            durations=2,
            treatment_pod="pod_A",
            rng=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="treatment_pod"):
        GeoLiftPowerAnalysis(
            estimator=Synth(),
            durations=2,
            treatment_pod="A",
            permutation_type="block",
        )

    evaluator = GeoLiftPowerAnalysis(
        estimator=Synth(),
        durations=2,
        treatment_pod="pod_A",
        permutation_type="block",
    )
    with pytest.raises(ValueError, match="exactly pod_A and pod_B"):
        evaluator.evaluate(make_panel(), {"candidate": {"pod_A": ["A"]}})
    with pytest.raises(ValueError, match="both pods"):
        evaluator.evaluate(
            make_panel(),
            {"candidate": {"pod_A": ["A", "B"], "pod_B": ["B", "C"]}},
        )
    with pytest.raises(ValueError, match="not found"):
        evaluator.evaluate(
            make_panel(),
            {"candidate": {"pod_A": ["missing"], "pod_B": ["B", "C"]}},
        )
