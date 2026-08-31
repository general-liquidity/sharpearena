#!/usr/bin/env python3
"""Sealed evaluation seeds: the predictability probe's band-scan adversary, re-run
against a salt-sealed band.

``make-predictability.py`` established that a bounded *public* seed band is
invertible by a table scan: matching one observed opening bar against every
candidate in a 2^16-wide band recovers the true seed 16/16 times in about a
second, after which the generator replays every future bar. The public held-out
set (``EVAL_SEEDS = EVAL_SEED_BASE + fixed offsets``) is exactly such a band.

This script re-runs the same adversary under two derivations of 16 named eval
slots and reports the recovery rate for each:

* public: ``EVAL_SEED_BASE + offset``, the committed public offsets for slots
  0..7 plus eight further offsets drawn uniformly from the same 2^16 band (a
  hypothetical extension of the public set, derived the same way).
* sealed: ``sealed_seed(salt, slot)`` with a 32-byte salt from ``os.urandom``.
  The adversary knows the band structure (every seed is >= EVAL_SEED_BASE),
  the slot names, the derivation function, and the generator; it does not know
  the salt.

The adversary is unchanged from the probe: build the first-bar-close table for
the 2^16 candidates at the band start, match the single observed opening bar
against it, and verify any match against a 30-bar prefix on the deployed tier.
The commit-reveal half of the workflow is also exercised: the salt commitment
(SHA-256) is recorded before the run, the salt is revealed at the end, and the
revealed salt is shown to replay every sealed scenario. Writes
``paper/evidence/sealed-seeds.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from sharpearena import TradingEnv, check_env_effective_config, merge_effective_configs
from sharpearena.dataset import EVAL_SEED_BASE
from sharpearena.eval_seeds import EVAL_SEEDS, sealed_eval_seeds

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"

N_SYMBOLS = 4
N_DAYS = 120
N_SLOTS = 16
TIER = "hard"  # the deployed tier used for prefix verification
WARMUP = 30
# Same adversary budget as the probe: a 2^16-wide band, here anchored at the
# public band start since every eval seed is known to be >= EVAL_SEED_BASE.
BAND_WIDTH = 1 << 16
MATCH_TOL = 1e-9
FLAT = json.dumps({"orders": []})
SLOTS = [f"held_out_{i:02d}" for i in range(N_SLOTS)]
_OPENING_READBACK: dict[int, dict] = {}
_TAPE_READBACK: dict[int, dict] = {}


def first_closes(
    seed: int, n_days: int = 2, *, record_readback: bool = False
) -> np.ndarray:
    """First-bar closes for a seed (tier-invariant: the post-passes never rewrite bar 0)."""
    env = TradingEnv(n_symbols=N_SYMBOLS, n_days=n_days, seed=seed, distribution_mode="calm")
    if record_readback and seed not in _OPENING_READBACK:
        _OPENING_READBACK[seed] = check_env_effective_config(
            env,
            seed=seed,
            n_symbols=N_SYMBOLS,
            n_days=n_days,
            distribution_mode="calm",
            scenario_seed=seed,
        )
    obs = json.loads(env.reset())
    return np.asarray([s["close_history"][-1] for s in obs["symbols"]])


def extract_closes(seed: int, mode: str) -> np.ndarray:
    """Replay the public environment with flat decisions and collect the tape."""
    env = TradingEnv(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=mode)
    if seed not in _TAPE_READBACK:
        _TAPE_READBACK[seed] = check_env_effective_config(
            env,
            seed=seed,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            distribution_mode=mode,
            scenario_seed=seed,
        )
    obs = json.loads(env.reset())
    rows = [[s["close_history"][-1] for s in obs["symbols"]]]
    done = False
    while not done:
        obs_json, _r, done, _info = env.step(FLAT)
        obs = json.loads(obs_json)
        rows.append([s["close_history"][-1] for s in obs["symbols"]])
    return np.asarray(rows, dtype=np.float64)


def build_table(band: range) -> tuple[np.ndarray, float]:
    """The adversary's table: first-bar closes for every candidate in the band."""
    t0 = time.time()
    table = np.stack([first_closes(c) for c in band])
    return table, time.time() - t0


def scan(observed_first: np.ndarray, band: range, table: np.ndarray) -> dict:
    t0 = time.time()
    hit = np.max(np.abs(table - observed_first[None, :]), axis=1) < MATCH_TOL
    matches = [int(band.start + i) for i in np.flatnonzero(hit)]
    return {"matches": matches, "candidates_checked": len(band), "elapsed_s": time.time() - t0}


def attack(seeds: dict[str, int], band: range, table: np.ndarray) -> dict:
    trials = []
    for name, seed in seeds.items():
        res = scan(first_closes(seed, record_readback=True), band, table)
        res["slot"] = name
        res["recovered"] = res["matches"] == [seed]
        res["prefix_verified"] = False
        if res["recovered"]:
            replay = extract_closes(res["matches"][0], TIER)
            truth = extract_closes(seed, TIER)
            res["prefix_verified"] = bool(
                np.max(np.abs(replay[:WARMUP] - truth[:WARMUP])) < MATCH_TOL
            )
        trials.append(res)
    return {
        "recovered": int(sum(t["recovered"] for t in trials)),
        "prefix_verified": int(sum(t["prefix_verified"] for t in trials)),
        "n_slots": len(seeds),
        "collisions": [t for t in trials if len(t["matches"]) > 1],
        "total_scan_s": float(sum(t["elapsed_s"] for t in trials)),
        "trials": trials,
    }


def main() -> None:
    _OPENING_READBACK.clear()
    _TAPE_READBACK.clear()
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    # Public derivation: committed offsets for the first 8 slots, then 8 further
    # offsets drawn uniformly from the same band with a recorded RNG seed.
    rng = np.random.default_rng(0)
    committed = [seed - EVAL_SEED_BASE for seed in EVAL_SEEDS.values()]
    extra = sorted(int(o) for o in rng.choice(BAND_WIDTH, size=N_SLOTS - len(committed), replace=False))
    public_offsets = committed + extra
    assert len(public_offsets) == N_SLOTS and len(set(public_offsets)) == N_SLOTS
    public_seeds = {name: EVAL_SEED_BASE + off for name, off in zip(SLOTS, public_offsets)}

    # Sealed derivation: commit to the salt before anything is observed.
    salt = os.urandom(32)
    commitment = hashlib.sha256(salt).hexdigest()
    sealed_seeds = sealed_eval_seeds(salt, names=SLOTS)
    assert all(s >= EVAL_SEED_BASE for s in sealed_seeds.values())

    band = range(EVAL_SEED_BASE, EVAL_SEED_BASE + BAND_WIDTH)
    table, table_s = build_table(band)

    public = attack(public_seeds, band, table)
    sealed = attack(sealed_seeds, band, table)

    # Reveal: with the salt, anyone recomputes the seeds and replays the scenarios.
    revealed = sealed_eval_seeds(salt, names=SLOTS)
    replay_ok = int(
        sum(
            np.max(np.abs(first_closes(revealed[n]) - first_closes(sealed_seeds[n]))) < MATCH_TOL
            for n in SLOTS
        )
    )
    assert revealed == sealed_seeds and hashlib.sha256(salt).hexdigest() == commitment

    band_span = 2**64 - EVAL_SEED_BASE
    evidence = {
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "n_slots": N_SLOTS,
            "slots": SLOTS,
            "verification_tier": TIER,
            "warmup_bars": WARMUP,
            "scan_band": [band.start, band.stop],
            "scan_band_width": BAND_WIDTH,
            "match_tolerance": MATCH_TOL,
            "adversary": (
                "predictability probe band scan: first-bar-close table over the 2^16 "
                "candidates at the public band start, one observed opening bar, "
                "prefix verification on the deployed tier"
            ),
            "public_extra_offsets_rng_seed": 0,
        },
        "effective_config": {
            "opening_bar_probe": merge_effective_configs(_OPENING_READBACK),
            "verification_tape": merge_effective_configs(_TAPE_READBACK),
        },
        "table_build_s": table_s,
        "public": {
            "derivation": "EVAL_SEED_BASE + offset",
            "seeds": public_seeds,
            **{k: v for k, v in public.items() if k != "trials"},
            "trials": public["trials"],
        },
        "sealed": {
            "derivation": "sealed_seed(salt, slot_index), salt = os.urandom(32)",
            "salt_commitment_sha256": commitment,
            "salt_revealed_hex": salt.hex(),
            "seeds": sealed_seeds,
            **{k: v for k, v in sealed.items() if k != "trials"},
            "reveal_replay_verified": replay_ok,
            "scan_coverage_of_band": BAND_WIDTH / band_span,
            "expected_recoveries_at_this_budget": N_SLOTS * BAND_WIDTH / band_span,
            "trials": sealed["trials"],
        },
        "note": (
            "the sealed band keeps the public band structure (every seed >= EVAL_SEED_BASE, "
            "disjoint from train by construction) but the concrete seeds are a keyed "
            "function of a salt the adversary does not hold; the same scan that recovers "
            "every public seed recovers none of the sealed ones, and revealing the salt "
            "afterwards replays the sealed evaluation exactly"
        ),
    }
    out = EVIDENCE / "sealed-seeds.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(
        f"public: {public['recovered']}/{N_SLOTS} recovered "
        f"({public['prefix_verified']} prefix-verified); "
        f"sealed: {sealed['recovered']}/{N_SLOTS} recovered; "
        f"reveal replay verified {replay_ok}/{N_SLOTS}; table {table_s:.2f}s"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
