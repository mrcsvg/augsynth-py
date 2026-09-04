"""Fixtures and helpers for parity tests against the R implementations.

This module is the only place in the test suite that talks to R directly.
All R imports happen behind ``importorskip`` so that the unit suite continues
to work for developers without R installed.

Design notes
------------
* The fixtures are session-scoped because starting an R interpreter and loading
  packages is expensive (a few seconds). We only want to pay that once.
* We convert R outputs into plain numpy arrays / polars DataFrames at the
  fixture boundary. Tests should never touch ``rpy2`` types directly — that
  keeps the tests readable and decouples them from rpy2's API churn.
* The R panel ``GeoLift_PreTest`` (40 cities, 90 days) is the shared canonical
  fixture. If a test needs synthetic data, generate it in Python so the test
  is self-contained.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import polars as pl
import pytest

from tests.validation_against_r._r_frames import from_r_data_frame

# ---------------------------------------------------------------------------
# Skip-collection if the R toolchain is unavailable
# ---------------------------------------------------------------------------

# Skip before importing rpy2 when R itself is absent. An installed rpy2 wheel
# cannot initialize without libR, and that failure is not a missing-module error.
if shutil.which("R") is None:
    pytest.skip("R is not installed; skipping R validation.", allow_module_level=True)

# `importorskip` raises Skipped at collection time if rpy2 is missing.
# That cleanly removes the validation suite for developers without rpy2.
rpy2 = pytest.importorskip(
    "rpy2",
    reason="rpy2 not installed; install with `pip install -e .[validation]`.",
)
try:
    import rpy2.robjects as ro
except Exception as exc:
    pytest.skip(
        f"rpy2 could not initialize R: {exc}",
        allow_module_level=True,
    )


def _r_package_available(name: str) -> bool:
    """Return True if the R package ``name`` is installed in the linked R."""
    # `requireNamespace` returns its result invisibly; rpy2 drops invisible
    # values and we get NULL. Wrap in `isTRUE()` to force a visible logical.
    return bool(ro.r(f'isTRUE(requireNamespace("{name}", quietly = TRUE))')[0])


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests if required R packages are not installed.

    Each test can declare its R-side dependencies with the ``requires_r_pkg``
    marker, e.g.::

        @pytest.mark.requires_r_pkg("augsynth")
        def test_something(...): ...
    """
    for item in items:
        marker = item.get_closest_marker("requires_r_pkg")
        if marker is None:
            continue
        for pkg in marker.args:
            if not _r_package_available(pkg):
                item.add_marker(
                    pytest.mark.skip(reason=f"R package '{pkg}' is not installed in the linked R.")
                )
                break


# ---------------------------------------------------------------------------
# Session-scoped R session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def r_session() -> Any:  # returns the rpy2.robjects.r namespace
    """Return the singleton rpy2 R interpreter handle.

    Notes
    -----
    Loading rpy2 spins up an embedded R process. Doing this once per session
    keeps the suite fast.
    """
    # Quiet R output during tests; failures still raise.
    ro.r("options(warn = 1)")
    return ro.r


@pytest.fixture(scope="session")
def r_geolift_pretest(r_session: Any) -> pl.DataFrame:
    """Load the GeoLift_PreTest panel as a polars DataFrame.

    The panel has 40 US cities over 90 days (2021-01-01 through 2021-03-31)
    and is the canonical fixture for synthetic control parity tests.

    Returns
    -------
    pl.DataFrame
        Columns: ``location`` (str), ``Y`` (float), ``date`` (date),
        and any additional covariates present in the source dataset.
    """
    if not _r_package_available("GeoLift"):
        pytest.skip("R package 'GeoLift' is not installed.")

    r_session("suppressPackageStartupMessages(library(GeoLift))")
    r_session("data(GeoLift_PreTest)")
    return from_r_data_frame(r_session("GeoLift_PreTest"))


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    name: str = "array",
) -> None:
    """Assert two arrays match within tolerance, with an informative message.

    Default tolerances are strict (``atol=1e-6, rtol=1e-5``) and appropriate
    for synthetic-control weights and counterfactual paths. Estimators that
    legitimately need looser tolerances must justify them in a comment at the
    call site.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = np.max(np.abs(actual - expected))
        raise AssertionError(
            f"{name} differs from R reference: "
            f"max |actual - expected| = {max_abs:.3e} "
            f"(atol={atol}, rtol={rtol})"
        )
