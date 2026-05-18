"""Smoke tests verifying the package imports and exposes a version."""

from __future__ import annotations

import re

import augsynth_py


def test_package_exposes_version() -> None:
    assert hasattr(augsynth_py, "__version__")
    # Must look like a PEP 440 version string.
    assert re.match(r"^\d+\.\d+\.\d+", augsynth_py.__version__)
