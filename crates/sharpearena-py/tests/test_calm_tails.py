"""The reproducible Calm calibration candidate (not a certified preset).

Three guarantees, matching the paper's F4 calm-calibration follow-up:

1. **The canonical tape is untouched**: default Calm is byte-identical whether the
   knob kwargs are omitted or passed as zeros, and it still fails the gate (if this
   flips, F4 in the paper is stale).
2. **The candidate is honestly non-certified** under the finite-panel-calibrated ACF
   gate: it can be reproduced for follow-up, but it must not be promoted as a preset.
3. **The preset is still calm**: mean realized volatility stays within 25% of the
   default Calm tape on every diagnostic seed, and the tape replays deterministically.
"""

from __future__ import annotations

import numpy as np

from sharpearena import SharpeArenaEnv, TradingEnv, certify_realism
from sharpearena.realism import CALM_CALIBRATION_CANDIDATE_KNOBS

DIAGNOSTIC_SEEDS = list(range(8))
CONFIRMATION_SEEDS = list(range(100, 108))
N_SYMBOLS = 4
N_DAYS = 120
MIN_PASS = 7
VOL_RATIO_MAX = 1.25


def _panel(seed: int, **knobs: float) -> np.ndarray:
    env = SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode="calm", **knobs
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


def _realized_vol(panel: np.ndarray) -> float:
    return float((panel[1:] / panel[:-1] - 1.0).std(axis=0).mean())


def test_preset_knobs_are_the_calibrated_values() -> None:
    assert CALM_CALIBRATION_CANDIDATE_KNOBS == {
        "vol_clustering": 0.5,
        "jump_burst_probability": 0.02,
        "jump_burst_persistence": 0.0,
        "jump_burst_size": 0.015,
    }


def test_default_calm_is_byte_identical_and_still_uncertified() -> None:
    zeros = {k: 0.0 for k in CALM_CALIBRATION_CANDIDATE_KNOBS}
    for seed in DIAGNOSTIC_SEEDS[:3]:
        plain = TradingEnv(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode="calm")
        explicit = TradingEnv(
            n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode="calm", **zeros
        )
        assert plain.reset() == explicit.reset(), seed
    default_pass = sum(certify_realism(_panel(seed)).passed for seed in DIAGNOSTIC_SEEDS)
    assert default_pass == 0, (
        f"the default Calm tape should fail the realism gate on every diagnostic seed"
        f" (got {default_pass}/8); if this now passes, F4 in the paper is stale"
    )


def test_candidate_does_not_masquerade_as_a_certified_preset() -> None:
    diagnostic = sum(
        certify_realism(_panel(seed, **CALM_CALIBRATION_CANDIDATE_KNOBS)).passed
        for seed in DIAGNOSTIC_SEEDS
    )
    confirmation = sum(
        certify_realism(_panel(seed, **CALM_CALIBRATION_CANDIDATE_KNOBS)).passed
        for seed in CONFIRMATION_SEEDS
    )
    assert diagnostic < MIN_PASS or confirmation < MIN_PASS


def test_preset_stays_calm_and_replays() -> None:
    for seed in DIAGNOSTIC_SEEDS:
        base = _panel(seed)
        cert = _panel(seed, **CALM_CALIBRATION_CANDIDATE_KNOBS)
        again = _panel(seed, **CALM_CALIBRATION_CANDIDATE_KNOBS)
        assert np.array_equal(cert, again), seed
        assert np.all(np.isfinite(cert)) and np.all(cert > 0.0)
        ratio = _realized_vol(cert) / _realized_vol(base)
        assert ratio <= VOL_RATIO_MAX, f"seed {seed}: vol ratio {ratio:.3f} > {VOL_RATIO_MAX}"
