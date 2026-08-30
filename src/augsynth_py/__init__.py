"""augsynth-py: Augmented Synthetic Control Methods for Python."""

from augsynth_py._version import __version__
from augsynth_py.inference import (
    ConformalTestResult,
    adjust_pvalues,
    conformal_interval,
    conformal_pvalue,
    conformal_test,
)
from augsynth_py.power import (
    DEFAULT_EFFECT_SIZES,
    EffectType,
    PowerEstimator,
    PowerParams,
    PowerResults,
    simulate_power,
)
from augsynth_py.synth import AugSynth, Synth

__all__ = [
    "DEFAULT_EFFECT_SIZES",
    "AugSynth",
    "ConformalTestResult",
    "EffectType",
    "PowerEstimator",
    "PowerParams",
    "PowerResults",
    "Synth",
    "__version__",
    "adjust_pvalues",
    "conformal_interval",
    "conformal_pvalue",
    "conformal_test",
    "simulate_power",
]
