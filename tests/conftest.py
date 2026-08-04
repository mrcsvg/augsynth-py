"""Shared pytest fixtures for the entire test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic RNG for tests that need randomness.

    Tests that need a different seed should use ``np.random.default_rng(seed)``
    explicitly. Using a fixture by default keeps the seed centralized.
    """
    return np.random.default_rng(seed=42)
