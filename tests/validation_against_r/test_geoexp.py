"""Validation of the geo-experiment helpers against GeoLift.

These tests are deliberately separate from the unit suite. They require both
R and the GeoLift package, and the shared validation conftest marks them as
skipped when either dependency is unavailable. The unit tests still exercise
all Python behaviour in a normal augsynth-py installation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from augsynth_py import Synth
from augsynth_py.geoexp import GeoLiftPowerAnalysis, MarketSelector
from tests.validation_against_r._r_frames import from_r_data_frame, to_r_data_frame


def _panel() -> pl.DataFrame:
    values = {
        "A": [1, 2, 4, 8, 7, 3],
        "B": [2, 3, 5, 7, 6, 5],
        "C": [4, 1, 3, 9, 2, 8],
        "D": [8, 2, 6, 1, 4, 0],
    }
    return pl.DataFrame(
        [
            {"time": time, "location": location, "Y": outcomes[time - 1]}
            for time in range(1, 7)
            for location, outcomes in values.items()
        ]
    )


@pytest.mark.requires_r_pkg("GeoLift")
def test_market_selection_matches_geolift(r_session: Any) -> None:
    """Correlation-ranked treatment groups agree with GeoLift's selector."""
    panel = _panel()
    import rpy2.robjects as ro

    ro.globalenv["market_panel"] = to_r_data_frame(panel)

    r_session(
        "similarity <- GeoLift::MarketSelection("
        "market_panel, location_id = 'location', time_id = 'time', Y_id = 'Y')"
    )
    r_session(
        "selected <- GeoLift::stochastic_market_selector("
        "treatment_size = 2, similarity_matrix = similarity, "
        "run_stochastic_process = FALSE)"
    )
    r_session("selected_matrix <- matrix(selected, ncol = 2)")
    selected_matrix = r_session("selected_matrix")
    selected_values = np.asarray(selected_matrix, dtype=str).reshape(
        tuple(int(size) for size in selected_matrix.dim),
        order="F",
    )
    r_markets = {tuple(sorted(str(value).lower() for value in row)) for row in selected_values}

    python_markets = {
        tuple(sorted(str(market).lower() for market in group))
        for group in MarketSelector(market_counts=[2])
        .select(panel)
        .as_df()
        .get_column("treatment_markets")
    }
    assert python_markets == r_markets


@pytest.mark.requires_r_pkg("GeoLift")
def test_power_analysis_wrapper_matches_geolift_power(
    r_session: Any,
) -> None:
    """The wrapper preserves GeoLift's per-window power-analysis outputs."""
    panel = pl.DataFrame(
        [
            {"time": time, "location": location, "Y": value}
            for location, values in {
                "A": [20, 22, 21, 25, 24, 28, 27, 30, 32, 31, 34, 36],
                "B": [10, 11, 11, 13, 12, 14, 14, 15, 16, 16, 17, 18],
                "C": [7, 8, 7, 9, 9, 10, 9, 11, 11, 12, 12, 13],
            }.items()
            for time, value in enumerate(values, start=1)
        ]
    )
    import rpy2.robjects as ro

    r_panel = panel.with_columns(pl.col("location").str.to_lowercase())
    ro.globalenv["power_panel"] = to_r_data_frame(r_panel)
    r_session(
        "r_power <- GeoLift::GeoLiftPower("
        "power_panel, locations = 'a', effect_size = c(0, 0.2), "
        "treatment_periods = 2, lookback_window = 2, alpha = 0.11, "
        "model = 'none', fixed_effects = TRUE, parallel = FALSE, "
        "side_of_test = 'two_sided', conformal_type = 'block')"
    )
    r_session(
        "r_power_frame <- data.frame(duration = r_power$duration, "
        "effect_size = r_power$EffectSize, treatment_start = r_power$treatment_start, "
        "pvalue = r_power$pvalue, att = r_power$att_estimator, "
        "detected_lift = r_power$detected_lift)"
    )
    r_frame = from_r_data_frame(r_session("r_power_frame"))

    python_frame = (
        GeoLiftPowerAnalysis(
            estimator=Synth(fixedeff=True),
            durations=2,
            effect_sizes=[0.0, 0.2],
            lookback_window=2,
            alpha=0.11,
            treatment_pod="pod_A",
            permutation_type="block",
        )
        .evaluate(panel, {"candidate": {"pod_A": ["A"], "pod_B": ["B", "C"]}})
        .as_df()
    )
    columns = [
        "duration",
        "effect_size",
        "treatment_start",
        "pvalue",
        "att",
        "detected_lift",
    ]
    actual = python_frame.select(columns).sort(columns[:3])
    expected = r_frame.select(columns).sort(columns[:3])
    assert_frame_equal(
        actual,
        expected,
        check_dtypes=False,
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
