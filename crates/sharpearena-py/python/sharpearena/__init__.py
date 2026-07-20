"""SharpeArena — a leak-free, point-in-time arena for trading agents.

This package bundles the native pyo3 binding (``sharpearena.sharpearena_py``) with a
gymnasium-compatible wrapper (:class:`SharpeArenaEnv`) and a PrimeIntellect ``verifiers``
environment (:mod:`sharpearena.verifiers_env`) whose rubric is scored by the real
SharpeBench kernel via :func:`score_run`.

The native binding exchanges the language-agnostic wire JSON at its boundary:
``TradingEnv.reset()`` returns an observation JSON string and ``TradingEnv.step()``
takes a decision JSON string. The pure-Python layers parse/build that JSON.
"""

from .sharpearena_py import TradingEnv, VecTradingEnv, score_run, validate_decision_json
from .gym import SharpeArenaEnv
from .vector import SharpeArenaVectorEnv
from .check_env import check_env, check_determinism_across_constructors
from .wrappers import (
    TimeLimit,
    CausalNormalizeObservation,
    CausalNormalizeReward,
    FrameStack,
    RecordEpisodeStatistics,
)
from .generalization import (
    train_test_seeds,
    evaluate_seeds,
    generalization_gap,
    cross_regime_transfer,
)
from .verifiers_env import SharpeArenaVerifiersEnv, load_environment, build_rubric
from .dataset import build_scenario_dataset, seed_ranges_disjoint
from .decision_parser import parse_decision
from .lookahead_guard import LookaheadGuard, LookaheadViolation, guarded, wrap_policy
from .trace import SCHEMA_VERSION, RolloutTraceWriter, load_trace, trace_to_returns
from .metrics import RunMetrics, cost_adjusted_score
from .wrappers_vector import (
    VectorCausalNormalizeObservation,
    VectorRecordEpisodeStatistics,
)
from .spaces import flatten_obs, unflatten_obs, flat_dim, FlattenObservation
from .execution_noise import ExecutionNoiseWrapper
from .mandate import (
    Mandate,
    sample_mandate,
    mandate_text,
    mandate_breach,
    mandate_from_dict,
    validate_mandate,
)
from .verifiers_env import mandate_reward
from .baselines import (
    run_baselines,
    leaderboard_markdown,
    MinVariancePolicy,
    MaxSharpePolicy,
    KellyVolTargetPolicy,
)
from .confidence import (
    deflated_sharpe_ci,
    paired_dsr_diff,
    pairwise_significance,
    significance_markdown,
)
from .rewards import (
    REWARD_SCHEMES,
    list_reward_schemes,
    build_scheme_rubric,
    differential_sharpe,
    sortino,
    drawdown_penalized,
    turnover_penalized,
    loss_averse,
)
from .indicators import CausalIndicatorObservation, INDICATORS, DEFAULT_INDICATORS
from .risk import DrawdownStopper, TurbulenceHalt, CrossSectionalDeleverage
from .failure_taxonomy import (
    FailureMode,
    classify_episode_failure,
    FailureRollup,
    rollup_failure_modes,
)
from .news import SyntheticNewsObservation, news_series
from .discrete import DiscreteAction
from .pairs import SpreadObservation, KalmanSpreadObservation
from .regime_eval import evaluate_per_regime, radar_score
from .portfolio_env import PortfolioEnv
from .market_making import (
    MarketMakingEnv,
    MMParams,
    analytically_optimal_policy,
    fixed_spread_policy,
    mm_regret,
)
from .execution import ExecutionEnv, execution_quality, twap_policy, immediate_policy
from .forecast import (
    calibrated_forecast,
    ForecastChannelObservation,
    forecast_skill_curve,
)
from .obs_extra import (
    MultiTimescaleMomentum,
    RollingCovarianceObservation,
    KalmanTrendObservation,
    TimeToHorizonObservation,
    CounterfactualInfo,
)
from .data_blocks import (
    find_continuous_blocks,
    block_windows,
    sample_block_window,
    make_block_env,
)
from .cascade import LiquidationCascadeEnv, cascade_survived, cascade_summary
from .lob_env import LOBMarketEnv, symmetric_quote_policy, noise_trader_policy
from .reward_misspecification import (
    MISSPECIFIED_REWARDS,
    MISSPECIFIED_PROXY_POLICIES,
    misspecification_gap,
    demonstrate_punishment,
)
from .minari_export import to_minari, to_minari_train_test
from .pettingzoo_env import MultiAgentSharpeArenaEnv, make_aec_env
from .market_env import EndogenousMarketEnv
from .checkpoint import CheckpointableEnv, CheckpointState
from .functional import SharpeArenaFuncEnv
from .curriculum import (
    CurriculumEnv,
    regime_curriculum,
    AdaptiveScheduler,
    AdaptiveCurriculumEnv,
)
from .preprocessing import (
    PreprocessingConfig,
    ExecutionNoiseConfig,
    CANONICAL_PREPROCESSING,
    make_preprocessed_env,
    describe_preprocessing,
)
from .eval_seeds import (
    EVAL_SEEDS,
    evaluate_eval_set,
    assert_no_regression,
    EVAL_SET_VERSION,
)
from .realism import (
    stylized_facts,
    certify_realism,
    RealismReport,
    DEFAULT_THRESHOLDS,
)
from .registration import register_envs

# Farama plugin convention: register the versioned env IDs at import time (idempotent).
register_envs()

__all__ = [
    "TradingEnv",
    "VecTradingEnv",
    "score_run",
    "validate_decision_json",
    "SharpeArenaEnv",
    "SharpeArenaVectorEnv",
    "LookaheadGuard",
    "LookaheadViolation",
    "guarded",
    "wrap_policy",
    "SCHEMA_VERSION",
    "RolloutTraceWriter",
    "load_trace",
    "trace_to_returns",
    "RunMetrics",
    "cost_adjusted_score",
    "check_env",
    "check_determinism_across_constructors",
    "TimeLimit",
    "CausalNormalizeObservation",
    "CausalNormalizeReward",
    "FrameStack",
    "RecordEpisodeStatistics",
    "train_test_seeds",
    "evaluate_seeds",
    "generalization_gap",
    "cross_regime_transfer",
    "SharpeArenaVerifiersEnv",
    "load_environment",
    "build_rubric",
    "build_scenario_dataset",
    "seed_ranges_disjoint",
    "parse_decision",
    "VectorCausalNormalizeObservation",
    "VectorRecordEpisodeStatistics",
    "flatten_obs",
    "unflatten_obs",
    "flat_dim",
    "FlattenObservation",
    "ExecutionNoiseWrapper",
    "Mandate",
    "sample_mandate",
    "mandate_text",
    "mandate_breach",
    "mandate_from_dict",
    "validate_mandate",
    "mandate_reward",
    "run_baselines",
    "leaderboard_markdown",
    "deflated_sharpe_ci",
    "paired_dsr_diff",
    "pairwise_significance",
    "significance_markdown",
    "to_minari",
    "to_minari_train_test",
    "MultiAgentSharpeArenaEnv",
    "make_aec_env",
    "EndogenousMarketEnv",
    "CheckpointableEnv",
    "CheckpointState",
    "SharpeArenaFuncEnv",
    "CurriculumEnv",
    "regime_curriculum",
    "AdaptiveScheduler",
    "AdaptiveCurriculumEnv",
    "PreprocessingConfig",
    "ExecutionNoiseConfig",
    "CANONICAL_PREPROCESSING",
    "make_preprocessed_env",
    "describe_preprocessing",
    "EVAL_SEEDS",
    "evaluate_eval_set",
    "assert_no_regression",
    "EVAL_SET_VERSION",
    "REWARD_SCHEMES",
    "list_reward_schemes",
    "build_scheme_rubric",
    "differential_sharpe",
    "sortino",
    "drawdown_penalized",
    "turnover_penalized",
    "loss_averse",
    "CausalIndicatorObservation",
    "INDICATORS",
    "DEFAULT_INDICATORS",
    "DrawdownStopper",
    "TurbulenceHalt",
    "CrossSectionalDeleverage",
    "FailureMode",
    "classify_episode_failure",
    "FailureRollup",
    "rollup_failure_modes",
    "SyntheticNewsObservation",
    "news_series",
    "DiscreteAction",
    "SpreadObservation",
    "KalmanSpreadObservation",
    "MinVariancePolicy",
    "MaxSharpePolicy",
    "KellyVolTargetPolicy",
    "evaluate_per_regime",
    "radar_score",
    "PortfolioEnv",
    "MarketMakingEnv",
    "MMParams",
    "analytically_optimal_policy",
    "fixed_spread_policy",
    "mm_regret",
    "ExecutionEnv",
    "execution_quality",
    "twap_policy",
    "immediate_policy",
    "calibrated_forecast",
    "ForecastChannelObservation",
    "forecast_skill_curve",
    "MultiTimescaleMomentum",
    "RollingCovarianceObservation",
    "KalmanTrendObservation",
    "TimeToHorizonObservation",
    "CounterfactualInfo",
    "find_continuous_blocks",
    "block_windows",
    "sample_block_window",
    "make_block_env",
    "LiquidationCascadeEnv",
    "cascade_survived",
    "cascade_summary",
    "LOBMarketEnv",
    "symmetric_quote_policy",
    "noise_trader_policy",
    "MISSPECIFIED_REWARDS",
    "MISSPECIFIED_PROXY_POLICIES",
    "misspecification_gap",
    "demonstrate_punishment",
    "stylized_facts",
    "certify_realism",
    "RealismReport",
    "DEFAULT_THRESHOLDS",
    "register_envs",
]
__version__ = "0.7.0"
