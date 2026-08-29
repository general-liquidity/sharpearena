"""Effective-config readback: what an arm actually ran, not what it asked for.

An evidence artifact that echoes the driver's own variables records an intention, not
a fact. The failure it cannot see is a harness bug that hands a labelled arm the wrong
tape: the run stays seed-reproducible and byte-identical, so fixed seeds, the cross
runtime FNV-1a goldens and the provenance digests all still agree, and the published
number is simply about a different market than its label claims.

This module closes that gap. :func:`env_effective_config` reads the configuration back
out of the environment that consumed it: the native binding reports the dimensions of
the panel it built, the window it clamped to, the execution seed left after default
resolution, and an FNV-1a fingerprint of the generated tape.
:func:`check_env_effective_config` compares that against what the caller says it
requested and raises :class:`EffectiveConfigError` on any disagreement, so a runner can
mark the arm unusable rather than publish it with a warning attached.

The scenario-generation knobs (seed, tier, clustering, jump bursts) are checked through
the fingerprint rather than by comparing labels: the panel is regenerated from the
requested values through :func:`~sharpearena.sharpearena_py.generate_scenario_json`, a
code path independent of environment construction, and the two fingerprints must agree.

**What this does not catch.** The user-seed split is restated here independently of
:class:`~sharpearena.gym.SharpeArenaEnv`, so the two disagreeing is a loud failure, but
a change in the published split rule itself is not something either side can detect.
Nor does the fingerprint speak for anything downstream of the tape: costs, execution
noise and policy wiring are outside its reach.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from .sharpearena_py import generate_scenario_json

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1

# Held-out band offset applied to an eval-mode seed. Mirrors ``dataset.EVAL_SEED_BASE``
# and ``gym._EVAL_SEED_BASE``; restated rather than imported, for the reason in the
# module docstring.
EVAL_SEED_BASE = 1_000_000


class EffectiveConfigError(RuntimeError):
    """The consumer is not running the configuration the caller requested."""


def fnv1a64_hex(text: str) -> str:
    """FNV-1a/64 over the UTF-8 bytes of ``text``, as 16 lowercase hex digits.

    The same fingerprint the Rust core and the WASM build pin their scenario goldens
    with, so a readback and a golden name one number for one panel.
    """
    h = _FNV_OFFSET
    for b in text.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _U64
    return f"{h:016x}"


def resolved_scenario_seed(user_seed: int, mode: str = "train") -> int:
    """The scenario seed a user seed lands on after the eval offset and the split.

    One user seed drives two independent streams (the price path and the execution
    noise) through ``numpy.random.SeedSequence``; only the first selects the tape.
    """
    if mode not in ("train", "eval"):
        raise ValueError("mode must be 'train' or 'eval'")
    offset = EVAL_SEED_BASE if mode == "eval" else 0
    state = np.random.SeedSequence(int(user_seed) + offset).generate_state(2)
    return int(state[0])


def scenario_fingerprint(
    *,
    seed: int,
    n_symbols: int = 4,
    n_days: int = 120,
    distribution_mode: str = "calm",
    vol_clustering: float = 0.0,
    jump_burst_probability: float = 0.0,
    jump_burst_persistence: float = 0.0,
    jump_burst_size: float = 0.0,
) -> str:
    """Fingerprint the panel a scenario configuration generates.

    ``seed`` is the scenario seed the generator consumes, not the user seed: pass
    :func:`resolved_scenario_seed` when checking a :class:`SharpeArenaEnv`.
    """
    return fnv1a64_hex(
        generate_scenario_json(
            seed=int(seed),
            n_symbols=int(n_symbols),
            n_days=int(n_days),
            distribution_mode=distribution_mode,
            vol_clustering=float(vol_clustering),
            jump_burst_probability=float(jump_burst_probability),
            jump_burst_persistence=float(jump_burst_persistence),
            jump_burst_size=float(jump_burst_size),
        )
    )


def env_effective_config(env: Any) -> dict:
    """Read one environment's effective configuration out of the native binding.

    Accepts a :class:`~sharpearena.gym.SharpeArenaEnv` or the native ``TradingEnv``
    underneath it.
    """
    native = getattr(env, "_env", env)
    raw = getattr(native, "effective_config", None)
    if raw is None:
        raise EffectiveConfigError(
            "environment exposes no effective_config readback; the installed native "
            "binding predates it, so the arm cannot be verified"
        )
    return json.loads(raw)


def check_env_effective_config(
    env: Any,
    *,
    seed: int,
    n_symbols: int = 4,
    n_days: int = 120,
    distribution_mode: str = "calm",
    vol_clustering: float = 0.0,
    jump_burst_probability: float = 0.0,
    jump_burst_persistence: float = 0.0,
    jump_burst_size: float = 0.0,
    mode: str = "train",
    scenario_seed: Optional[int] = None,
) -> dict:
    """Verify one environment against the configuration the caller requested.

    Returns the effective block (with the verified fingerprint) on success; raises
    :class:`EffectiveConfigError` naming every disagreement otherwise. ``scenario_seed``
    overrides the derived split for callers that hand the native binding a seed directly.
    """
    effective = env_effective_config(env)
    consumed_seed = (
        resolved_scenario_seed(seed, mode) if scenario_seed is None else int(scenario_seed)
    )
    expected = scenario_fingerprint(
        seed=consumed_seed,
        n_symbols=n_symbols,
        n_days=n_days,
        distribution_mode=distribution_mode,
        vol_clustering=vol_clustering,
        jump_burst_probability=jump_burst_probability,
        jump_burst_persistence=jump_burst_persistence,
        jump_burst_size=jump_burst_size,
    )

    problems = []
    if effective["n_symbols"] != int(n_symbols):
        problems.append(
            f"n_symbols requested {n_symbols}, environment built {effective['n_symbols']}"
        )
    if effective["n_bars"] != int(n_days):
        problems.append(f"n_days requested {n_days}, environment built {effective['n_bars']} bars")
    if effective["dataset_fnv1a64"] != expected:
        problems.append(
            "tape fingerprint {} does not match the panel the requested configuration "
            "generates ({}); seed={} mode={} distribution_mode={} vol_clustering={}".format(
                effective["dataset_fnv1a64"],
                expected,
                seed,
                mode,
                distribution_mode,
                vol_clustering,
            )
        )
    if problems:
        raise EffectiveConfigError(
            "effective configuration does not match the requested one: " + "; ".join(problems)
        )

    effective["scenario_seed"] = consumed_seed
    effective["verified"] = True
    return effective


def merge_effective_configs(blocks: dict[Any, dict]) -> dict:
    """Fold per-seed readbacks into one arm-level ``effective_config`` block.

    The shape-invariant fields must agree across every seed in the arm (they disagreeing
    means the arm is not one arm); the per-seed tape fingerprints are kept, since they
    are what a later reader can recheck without rerunning anything.
    """
    if not blocks:
        raise EffectiveConfigError("no effective-config readbacks to merge")
    shared_keys = ("n_symbols", "n_bars", "window_start", "window_end")
    items = list(blocks.items())
    first_seed, first = items[0]
    for seed, block in items[1:]:
        for key in shared_keys:
            if block[key] != first[key]:
                raise EffectiveConfigError(
                    f"seed {seed} ran with {key}={block[key]} but seed {first_seed} ran "
                    f"with {key}={first[key]}: this is not one arm"
                )
    steps = {b["steps_observed"] for b in blocks.values() if "steps_observed" in b}
    if len(steps) > 1:
        raise EffectiveConfigError(
            f"the arm's seeds ran different step counts {sorted(steps)}: this is not one arm"
        )
    merged_steps = {"steps_observed": steps.pop()} if steps else {}
    return {
        **merged_steps,
        "n_symbols": first["n_symbols"],
        "n_bars": first["n_bars"],
        "window_start": first["window_start"],
        "window_end": first["window_end"],
        "verified": all(bool(b.get("verified")) for b in blocks.values()),
        "readback": "panel dimensions, window and tape fingerprint read from the "
        "environment that consumed the configuration; the generation knobs are bound "
        "by the fingerprint, not echoed",
        "dataset_fnv1a64": {str(seed): blocks[seed]["dataset_fnv1a64"] for seed in blocks},
    }
