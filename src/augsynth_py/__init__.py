"""augsynth-py: Augmented Synthetic Control Methods for Python."""

from augsynth_py._version import __version__
from augsynth_py.inference import conformal_interval, conformal_pvalue
from augsynth_py.power import PowerResults, simulate_power
from augsynth_py.synth import AugSynth, Synth

__all__ = [
    "AugSynth",
    "PowerResults",
    "Synth",
    "__version__",
    "conformal_interval",
    "conformal_pvalue",
    "simulate_power",
]
