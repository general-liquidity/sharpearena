"""OpenOutcry — a leak-free, point-in-time Gym for trading agents.

This package bundles the native pyo3 binding (``openoutcry.openoutcry_py``) with a
gymnasium-compatible wrapper (:class:`OpenOutcryEnv`) and a PrimeIntellect ``verifiers``
environment (:mod:`openoutcry.verifiers_env`) whose rubric is scored by the real
SharpeBench kernel via :func:`score_run`.

The native binding exchanges the language-agnostic wire JSON at its boundary:
``TradingEnv.reset()`` returns an observation JSON string and ``TradingEnv.step()``
takes a decision JSON string. The pure-Python layers parse/build that JSON.
"""

from .openoutcry_py import TradingEnv, score_run
from .gym import OpenOutcryEnv

__all__ = ["TradingEnv", "score_run", "OpenOutcryEnv"]
__version__ = "0.1.0"
