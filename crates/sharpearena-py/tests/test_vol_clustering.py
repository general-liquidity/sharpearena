"""The opt-in volatility-clustering driver (``vol_clustering``).

Three guarantees, matching the paper's F4 follow-up:

1. **Off is off**: ``vol_clustering=0.0`` (the default) is byte-identical to omitting the
   kwarg entirely, on both the scalar and vector surfaces.
2. **The driver drives**: clustered panels show strictly higher absolute-return
   autocorrelation than unclustered panels on the same seeds.
3. **The gate flips**: the realism gate's volatility-clustering fact
   (``abs_return_autocorr``) passes for clustered Hard panels on a seed set where it
   fails unclustered — the assertion that closes the paper's Finding 4.
"""

from __future__ import annotations

import numpy as np

from sharpearena import SharpeArenaEnv, TradingEnv, certify_realism, stylized_facts

SEEDS = list(range(8))
N_SYMBOLS = 4
N_DAYS = 120
STRENGTH = 0.5  # chosen empirically: flips the Hard clustering fact to 8/8 (see below)


def _rollout_obs(env: SharpeArenaEnv, seed: int, steps: int = 40) -> list[bytes]:
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(obs["closes"], dtype=np.float64).tobytes()]
    flat = np.zeros(N_SYMBOLS, dtype=np.float32)
    for _ in range(steps):
        obs, _reward, terminated, truncated, _info = env.step(flat)
        frames.append(np.asarray(obs["closes"], dtype=np.float64).tobytes())
        if terminated or truncated:
            break
    return frames


def _panel(tier: str, seed: int, vol_clustering: float) -> np.ndarray:
    env = SharpeArenaEnv(
        n_symbols=N_SYMBOLS,
        n_days=N_DAYS,
        seed=seed,
        distribution_mode=tier,
        vol_clustering=vol_clustering,
    )
    obs, _ = env.reset(seed=seed)
    closes = [np.asarray(obs["closes"], dtype=np.float64).reshape(-1)]
    flat = np.zeros(N_SYMBOLS, dtype=np.float32)
    for _ in range(512):
        obs, _reward, terminated, truncated, _info = env.step(flat)
        closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
        if terminated or truncated:
            break
    return np.stack(closes, axis=0)


def test_default_zero_is_byte_identical_native() -> None:
    for tier in ("calm", "hard", "extreme"):
        without = TradingEnv(
            n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=7, distribution_mode=tier
        )
        with_zero = TradingEnv(
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            seed=7,
            distribution_mode=tier,
            vol_clustering=0.0,
        )
        assert without.reset() == with_zero.reset(), tier


def test_default_zero_is_byte_identical_rollout() -> None:
    for seed in SEEDS[:3]:
        without = SharpeArenaEnv(
            n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode="hard"
        )
        with_zero = SharpeArenaEnv(
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            seed=seed,
            distribution_mode="hard",
            vol_clustering=0.0,
        )
        assert _rollout_obs(without, seed) == _rollout_obs(with_zero, seed)


def test_clustered_panels_raise_abs_return_autocorr() -> None:
    raised = 0
    for seed in SEEDS:
        base = stylized_facts(_panel("hard", seed, 0.0))["abs_return_autocorr"]
        clustered = stylized_facts(_panel("hard", seed, STRENGTH))["abs_return_autocorr"]
        if clustered > base:
            raised += 1
    assert raised >= 7, f"clustering should raise |r| autocorr on >=7/8 seeds, got {raised}"


def test_realism_clustering_fact_flips_to_pass_on_hard() -> None:
    """The F4 closer: the volatility-clustering fact fails unclustered and passes clustered."""
    unclustered_pass = 0
    clustered_pass = 0
    for seed in SEEDS:
        unclustered_pass += certify_realism(_panel("hard", seed, 0.0)).checks[
            "abs_return_autocorr"
        ]
        clustered_pass += certify_realism(_panel("hard", seed, STRENGTH)).checks[
            "abs_return_autocorr"
        ]
    assert unclustered_pass <= 2, (
        f"the default Hard tape should fail volatility clustering (got {unclustered_pass}/8);"
        " if this now passes, F4 in the paper is stale"
    )
    assert clustered_pass == len(SEEDS), (
        f"vol_clustering={STRENGTH} should pass volatility clustering on every seed,"
        f" got {clustered_pass}/{len(SEEDS)}"
    )
