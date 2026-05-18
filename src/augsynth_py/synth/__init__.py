"""Synthetic control estimators.

Implementations live here. Each estimator is in its own module and follows
the convention: a class with ``fit(X, y, ...)`` and ``predict(...)`` methods,
following the scikit-learn estimator contract loosely (no fit/transform split,
since synthetic control fits and predicts jointly).
"""

from augsynth_py.synth.classical import Synth

__all__ = ["Synth"]
