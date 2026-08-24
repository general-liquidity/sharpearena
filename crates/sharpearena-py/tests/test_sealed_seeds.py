"""Tests for sealed (salt-keyed) evaluation seeds.

The public ``EVAL_SEEDS`` band is a bounded public band, which the predictability probe
showed is invertible by a table scan. ``sealed_eval_seeds`` derives the same named slots
from a secret salt. These tests pin the contract: band membership (>= EVAL_SEED_BASE)
without needing the salt, determinism, salt and slot sensitivity, disjointness from the
train band, the native/Python pin, and that the public set is untouched.
"""

import importlib

import pytest

try:
    from sharpearena.dataset import EVAL_SEED_BASE
    from sharpearena.eval_seeds import (
        EVAL_SEEDS,
        EVAL_SET_VERSION,
        MIN_SEALED_SALT_BYTES,
        evaluate_eval_set,
        sealed_eval_seeds,
    )
    from sharpearena.sharpearena_py import sealed_seed

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding
    _HAVE_BINDING = False

requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)

SALT_A = b"operator-secret-salt-A-0123456789"
SALT_B = b"operator-secret-salt-B-0123456789"

# The public, committed set as of EVAL_SET_VERSION "sharpearena-eval-seeds-v1". Sealing
# must never move it: the regression snapshot in tests/data is keyed to these values.
PUBLIC_V1 = {
    "held_out_00": 1_000_000,
    "held_out_01": 1_000_007,
    "held_out_02": 1_000_013,
    "held_out_03": 1_000_029,
    "held_out_04": 1_000_101,
    "held_out_05": 1_000_257,
    "held_out_06": 1_001_024,
    "held_out_07": 1_004_099,
}


@requires_binding
def test_public_eval_seeds_are_unchanged():
    assert EVAL_SET_VERSION == "sharpearena-eval-seeds-v1"
    assert EVAL_SEEDS == PUBLIC_V1


@requires_binding
def test_sealed_seeds_keep_slot_names_and_land_in_the_held_out_band():
    sealed = sealed_eval_seeds(SALT_A)
    assert list(sealed) == list(EVAL_SEEDS)
    for name, seed in sealed.items():
        assert seed >= EVAL_SEED_BASE, f"{name}={seed} leaked into the train band"
        assert seed < 2**64
    assert len(set(sealed.values())) == len(sealed)


@requires_binding
def test_sealed_seeds_are_deterministic_and_pinned():
    assert sealed_eval_seeds(SALT_A) == sealed_eval_seeds(SALT_A)
    assert sealed_eval_seeds(bytearray(SALT_A)) == sealed_eval_seeds(SALT_A)
    # Same pin as the Rust unit test `sealed_seed_is_deterministic`.
    assert sealed_seed(SALT_A, 0) == 0x040A_380D_F918_05C2
    assert sealed_eval_seeds(SALT_A)["held_out_00"] == 0x040A_380D_F918_05C2


@requires_binding
def test_sealed_seeds_are_salt_sensitive():
    a, b = sealed_eval_seeds(SALT_A), sealed_eval_seeds(SALT_B)
    for name in a:
        assert a[name] != b[name]
    assert not set(a.values()) & set(b.values())
    flipped = bytes([SALT_A[0] ^ 1]) + SALT_A[1:]
    c = sealed_eval_seeds(flipped)
    for name in a:
        assert a[name] != c[name]


@requires_binding
def test_sealed_seeds_differ_from_the_public_set():
    sealed = sealed_eval_seeds(SALT_A)
    assert not set(sealed.values()) & set(EVAL_SEEDS.values())
    # Sealed seeds are spread over the whole 2^64 band; the probe's 2^16-wide scan
    # window at the band start covers none of them (2^-48 per seed).
    for seed in sealed.values():
        assert seed - EVAL_SEED_BASE >= 2**16


@requires_binding
def test_sealed_seeds_are_disjoint_from_train_band():
    train = range(0, EVAL_SEED_BASE)
    for seed in sealed_eval_seeds(SALT_A).values():
        assert seed not in train


@requires_binding
def test_short_salt_is_rejected():
    with pytest.raises(ValueError):
        sealed_eval_seeds(b"short")
    with pytest.raises(ValueError):
        sealed_eval_seeds(b"x" * (MIN_SEALED_SALT_BYTES - 1))
    with pytest.raises(ValueError):
        sealed_seed(b"short", 0)
    assert sealed_eval_seeds(b"x" * MIN_SEALED_SALT_BYTES)


@requires_binding
def test_custom_slot_names_and_duplicates():
    sealed = sealed_eval_seeds(SALT_A, names=["a", "b", "c"])
    assert list(sealed) == ["a", "b", "c"]
    assert sealed["a"] == sealed_seed(SALT_A, 0)
    assert sealed["c"] == sealed_seed(SALT_A, 2)
    with pytest.raises(ValueError):
        sealed_eval_seeds(SALT_A, names=["a", "a"])


@requires_binding
def test_evaluate_eval_set_sealed_runs_the_same_protocol():
    public = evaluate_eval_set(n_symbols=3, n_days=40)
    sealed = evaluate_eval_set(n_symbols=3, n_days=40, salt=SALT_A)
    sealed_again = evaluate_eval_set(n_symbols=3, n_days=40, salt=SALT_A)
    assert set(sealed) == set(EVAL_SEEDS)
    assert sealed == sealed_again
    assert sealed != public
    for scores in sealed.values():
        assert set(scores) == {"deflated_sharpe", "passed_k", "mean_return"}
    # Sealing is opt-in: the default call is byte-identical to before.
    assert evaluate_eval_set(n_symbols=3, n_days=40) == public
