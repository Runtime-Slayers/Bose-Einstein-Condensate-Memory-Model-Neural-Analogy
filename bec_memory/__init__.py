"""
bec_memory — Bose-Einstein Condensate Memory Model
====================================================

A Python package implementing the BEC-Memory analogy framework:
using Bose-Einstein condensate physics as a mathematical model
for human memory formation, consolidation, and forgetting.

References
----------
Cornell & Wieman (1995) Science 269:198  — BEC of Rb-87
Ketterle (1995) PRL 75:3969             — BEC of Na-23
Ebbinghaus (1885)                        — Forgetting curves
Bagnato (1987) PRA 35:4354             — Harmonic-trap BEC statistics
Pitaevskii & Stringari (2003)           — BEC theory textbook

Quick Start
-----------
>>> from bec_memory import BECMemoryModel
>>> model = BECMemoryModel()
>>> frac = model.condensate_fraction(T=0.5)   # 0.646...
>>> times, retention = model.forgetting_curve(t_max=30.0)
"""

__version__ = "0.1.0"
__author__ = "Runtime-Slayers"
__license__ = "MIT"

from bec_memory.model import BECMemoryModel
from bec_memory.gpe import GPESolver
from bec_memory.forgetting import ForgettingCurve
from bec_memory.false_memory import BogoliubovSpectrum

__all__ = [
    "BECMemoryModel",
    "GPESolver",
    "ForgettingCurve",
    "BogoliubovSpectrum",
    "__version__",
]
