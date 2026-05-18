"""Domain-specific exceptions raised by ``augsynth_py``.

Generic input validation (wrong types, malformed arrays) raises ``ValueError``.
These custom exceptions cover panel-shape problems that are specific to the
synthetic-control setting and would be ambiguous as plain ``ValueError``.
"""

from __future__ import annotations


class AugsynthError(Exception):
    """Base class for all augsynth-py domain errors."""


class TreatedUnitNotFoundError(AugsynthError):
    """The unit identified as treated is not present in the panel."""


class EmptyDonorPoolError(AugsynthError):
    """No donor units remain after removing the treated unit."""


class SinglePeriodError(AugsynthError):
    """The panel has fewer than two time periods on at least one side of treatment."""
