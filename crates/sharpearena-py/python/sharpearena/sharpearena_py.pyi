# Type stub for the compiled pyo3 extension `sharpearena.sharpearena_py`.
#
# The wheel ships `py.typed`, so without this stub every symbol of the native
# module - where most of the API lives - is `Any` to a type checker. The stub is
# hand-maintained against `crates/sharpearena-py/src/lib.rs`; a two-way drift
# guard in `tests/test_stub_drift.py` fails when a runtime name or public class
# member is missing here, when the stub invents one, or when a callable's
# parameter names, ordering, kinds, or defaults drift from the native module.
#
# Types are honest about the wire contract: where the boundary exchanges JSON
# it takes and returns `str`. Numeric vectors cross as `Sequence[float]` in and
# `list[float]` out; the native module itself takes no numpy arrays (a numpy
# array of floats still satisfies `Sequence[float]`).

from collections.abc import Sequence
from typing import final

EVAL_SEED_BASE: int
MIN_SEALED_SALT_BYTES: int

class SharpeArenaError(ValueError):
    """Base class for every error the native engine raises."""

class InvalidArgument(SharpeArenaError):
    """A caller argument the engine cannot use."""

class InvalidJson(SharpeArenaError):
    """A wire-contract JSON payload that failed to deserialize."""

class InvalidSalt(SharpeArenaError):
    """A sealed-seed salt the derivation refuses."""

class DataUnavailable(SharpeArenaError):
    """The engine could not load the data it was pointed at."""

class EngineFailure(SharpeArenaError):
    """The engine failed while producing a result, not rejecting input."""

@final
class TradingEnv:
    def __init__(
        self,
        n_symbols: int = 4,
        n_days: int = 120,
        seed: int = 0,
        window_start: int | None = None,
        window_end: int | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        impact_bps: float | None = None,
        financing_bps: float | None = None,
        max_participation: float | None = None,
        distribution_mode: str = "calm",
        exec_seed: int | None = None,
        vol_clustering: float = 0.0,
        jump_burst_probability: float = 0.0,
        jump_burst_persistence: float = 0.0,
        jump_burst_size: float = 0.0,
    ) -> None: ...
    @classmethod
    def from_csv(
        cls,
        csv_text: str,
        seed: int = 0,
        window_start: int | None = None,
        window_end: int | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        impact_bps: float | None = None,
        financing_bps: float | None = None,
        max_participation: float | None = None,
        exec_seed: int | None = None,
    ) -> TradingEnv: ...
    @property
    def scenario_seed(self) -> int: ...
    @property
    def effective_config(self) -> str: ...
    def reset(self) -> str: ...
    def step(self, decision_json: str) -> tuple[str, float, bool, str]: ...
    def clone_state(self) -> str: ...
    def restore_state(self, state_json: str) -> None: ...

@final
class VecTradingEnv:
    def __init__(
        self,
        seeds: Sequence[int],
        n_symbols: int = 4,
        n_days: int = 120,
        window_start: int | None = None,
        window_end: int | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        impact_bps: float | None = None,
        financing_bps: float | None = None,
        max_participation: float | None = None,
        distribution_mode: str = "calm",
        exec_seed: int | None = None,
        autoreset_mode: str = "next_step",
        vol_clustering: float = 0.0,
        jump_burst_probability: float = 0.0,
        jump_burst_persistence: float = 0.0,
        jump_burst_size: float = 0.0,
    ) -> None: ...
    @classmethod
    def from_csv(
        cls,
        csv_text: str,
        seeds: Sequence[int],
        window_start: int | None = None,
        window_end: int | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        impact_bps: float | None = None,
        financing_bps: float | None = None,
        max_participation: float | None = None,
        autoreset_mode: str = "next_step",
    ) -> VecTradingEnv: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def scenario_seeds(self) -> list[int]: ...
    @property
    def autoreset_mode(self) -> str: ...
    def reset_batch(self) -> str: ...
    def step_batch(self, decisions_json: str) -> str: ...

@final
class PyMarketClearing:
    def __init__(
        self,
        n_symbols: int = 4,
        n_days: int = 120,
        seed: int = 0,
        n_agents: int = 2,
        capital: float = 1.0,
        kyle_lambda: float = 0.1,
        eta: float = 0.05,
        volume_scale: float = 1.0,
        vol_scale: float = 0.0,
        distribution_mode: str = "calm",
        richness: str = "standard",
        lambda_radius: float | None = None,
        eta_radius: float | None = None,
        uncertainty_correlation: float = 0.0,
        impact_exponent: float = 1.0,
    ) -> None: ...
    @property
    def scenario_seed(self) -> int: ...
    @property
    def uncertainty(self) -> str | None: ...
    def set_uncertainty(
        self,
        lambda_radius: float | None = None,
        eta_radius: float | None = None,
        correlation: float = 0.0,
    ) -> None: ...
    @property
    def impact_exponent(self) -> float: ...
    def set_impact_exponent(self, impact_exponent: float) -> None: ...
    @property
    def richness(self) -> str: ...
    @property
    def symbols(self) -> list[str]: ...
    @property
    def num_agents(self) -> int: ...
    @property
    def done(self) -> bool: ...
    def reset_market(self) -> str: ...
    def step_market(self, orders_json: str) -> str: ...

@final
class PyOrderBook:
    def __init__(self, tick_size: float = 0.01, levels: int = 10) -> None: ...
    def reset_book(self) -> str: ...
    def step_book(self, orders_json: str) -> str: ...
    def ladder(self) -> str: ...
    def uncross(self) -> str: ...
    def sweep_cost(self, side: str, qty: int) -> str: ...

def score_run(
    returns: Sequence[float],
    n_trials: int = 0,
    periods_per_year: float = 252.0,
) -> str: ...
def bootstrap_dsr_ci(
    per_seed_returns: Sequence[Sequence[float]],
    n_trials: int = 0,
    n_boot: int = 2000,
    resample_seed: int = 0x5BA7_2026,
    alpha: float = 0.05,
) -> str: ...
def paired_dsr_diff(
    a_per_seed_returns: Sequence[Sequence[float]],
    b_per_seed_returns: Sequence[Sequence[float]],
    n_trials: int = 0,
    n_boot: int = 2000,
    resample_seed: int = 0x5BA7_2026,
    alpha: float = 0.05,
) -> str: ...
def validate_decision_json(decision_json: str) -> bool: ...
def decision_schema_json() -> str: ...
def sample_mandate_json(
    seed: int, n_symbols: int = 4, allow_short: bool = True
) -> str: ...
def mandate_breach(
    mandate_json: str,
    returns: Sequence[float],
    weights: Sequence[Sequence[float]],
) -> float: ...
def perturb_action(
    seed: int,
    step_index: int,
    requested: Sequence[float],
    previous: Sequence[float],
    delay_prob: float = 0.0,
    slippage_bps: float = 0.0,
) -> list[float]: ...
def sealed_seed(salt: bytes, slot: int) -> int: ...
def spec_hash() -> str: ...
def _raise_coded(code: str, message: str) -> None: ...
def generate_scenario_json(
    seed: int,
    n_symbols: int = 4,
    n_days: int = 120,
    distribution_mode: str = "calm",
    vol_clustering: float = 0.0,
    jump_burst_probability: float = 0.0,
    jump_burst_persistence: float = 0.0,
    jump_burst_size: float = 0.0,
) -> str: ...
def calm_calibration_candidate_scenario_json(seed: int) -> str: ...
