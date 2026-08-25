"""Python-side cross-runtime golden-fingerprint test.

The Rust crate commits FNV-1a/64 fingerprints of canonical generated scenarios
(``crates/sharpearena/src/scenario_gen.rs``) and the WebAssembly crate asserts the same
values against its JSON kernel. This test closes the third runtime: the scenario JSON
returned through the pyo3 binding (``generate_scenario_json``, the native engine's own
serde serialization) must hash to the identical committed goldens, so a scenario
generated through Python is pinned to the byte against the native and wasm builds.

The constants below are copied verbatim from the committed pins in
``scenario_gen.rs``; if a generator change legitimately moves them, all three runtimes'
tests must be updated together.
"""

from __future__ import annotations

import pytest

from sharpearena.sharpearena_py import (
    calm_calibration_candidate_scenario_json,
    generate_scenario_json,
)

# Committed golden fingerprints from crates/sharpearena/src/scenario_gen.rs
# (GOLDEN_*_4X120_SEED7_FNV1A). The wasm crate asserts the calm and clustered values
# against its JSON kernel; this file asserts all of them through the Python binding.
GOLDEN_CALM_4X120_SEED7_FNV1A = 0xB7CF_976C_7121_9C52
GOLDEN_HARD_4X120_SEED7_FNV1A = 0x2EF5_AFF1_A716_05E6
GOLDEN_EXTREME_4X120_SEED7_FNV1A = 0xB082_0C4D_2C73_7F88
GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A = 0xA3D2_2742_4EF0_5868
GOLDEN_REGIME_4X120_SEED7_FNV1A = 0x8B82_2CF3_C9D3_038F
GOLDEN_HARD_CLUSTERED_4X120_SEED7_FNV1A = 0xA1D2_31F7_E114_A381
GOLDEN_CALM_CALIBRATION_CANDIDATE_4X120_SEED7_FNV1A = 0x32A6_3F8E_5743_EC93


def fnv1a(data: bytes) -> int:
    """Dependency-free FNV-1a/64, the same fingerprint the Rust and wasm tests use."""
    h = 0xCBF2_9CE4_8422_2325
    for b in data:
        h ^= b
        h = (h * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return h


@pytest.mark.parametrize(
    "mode,golden",
    [
        ("calm", GOLDEN_CALM_4X120_SEED7_FNV1A),
        ("hard", GOLDEN_HARD_4X120_SEED7_FNV1A),
        ("extreme", GOLDEN_EXTREME_4X120_SEED7_FNV1A),
        ("cointegrated_pairs", GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A),
        ("regime_shift", GOLDEN_REGIME_4X120_SEED7_FNV1A),
    ],
)
def test_scenario_fingerprint_matches_committed_golden(mode: str, golden: int) -> None:
    """The Calm/Hard/Extreme/structured 4x120 seed-7 scenarios generated through the
    Python binding must fingerprint to the goldens committed in the Rust crate."""
    json_bytes = generate_scenario_json(
        7, n_symbols=4, n_days=120, distribution_mode=mode
    ).encode()
    assert fnv1a(json_bytes) == golden, (
        f"Python-surface fingerprint for {mode} drifted from the committed "
        "scenario_gen.rs golden"
    )


def test_clustered_scenario_fingerprint_matches_committed_golden() -> None:
    """The opt-in vol_clustering=0.5 Hard scenario must match its committed golden,
    the same value the wasm kernel asserts."""
    json_bytes = generate_scenario_json(
        7, n_symbols=4, n_days=120, distribution_mode="hard", vol_clustering=0.5
    ).encode()
    assert fnv1a(json_bytes) == GOLDEN_HARD_CLUSTERED_4X120_SEED7_FNV1A


def test_calm_calibration_candidate_fingerprint_matches_committed_golden() -> None:
    """The pinned calm-calibration candidate preset (a reproducible negative-result
    configuration, not a certified preset) must match its committed golden."""
    json_bytes = calm_calibration_candidate_scenario_json(7).encode()
    assert fnv1a(json_bytes) == GOLDEN_CALM_CALIBRATION_CANDIDATE_4X120_SEED7_FNV1A


def test_scenario_json_is_deterministic_across_calls() -> None:
    """Two calls with the same arguments return byte-identical JSON."""
    a = generate_scenario_json(42, distribution_mode="extreme")
    b = generate_scenario_json(42, distribution_mode="extreme")
    assert a == b
