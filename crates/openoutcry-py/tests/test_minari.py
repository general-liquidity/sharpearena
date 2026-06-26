"""Tests for the Minari offline-RL dataset exporter (:mod:`openoutcry.minari_export`).

The round-trip test needs the native ``openoutcry`` binding (for ``OpenOutcryEnv``), ``numpy``
and ``minari``; it skips when any is absent. The guard test loads ``minari_export`` standalone
(so it runs without the Rust toolchain too) and asserts the export raises a clear error when
``minari`` is unavailable — independent of whether ``minari`` is actually installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_PKGDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python", "openoutcry")
)

try:
    import openoutcry as oo  # noqa: F401
    from openoutcry.trace import RolloutTraceWriter
    from openoutcry import minari_export as me

    HAS_BINDING = True
except Exception:  # noqa: BLE001 - binding not built; fall back to standalone modules
    HAS_BINDING = False
    _pkg = types.ModuleType("oo_standalone")
    _pkg.__path__ = [_PKGDIR]  # type: ignore[attr-defined]
    sys.modules.setdefault("oo_standalone", _pkg)

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"oo_standalone.{name}", os.path.join(_PKGDIR, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    _trace = _load("trace")  # so oo_standalone.trace exists for minari_export's relative import
    RolloutTraceWriter = _trace.RolloutTraceWriter
    me = _load("minari_export")


# --- guard: module imports and raises cleanly without minari -----------------


def test_module_imports_without_minari():
    """`import openoutcry.minari_export` must succeed even when minari is absent."""
    assert hasattr(me, "to_minari")


def test_to_minari_raises_when_minari_missing(monkeypatch):
    """With minari unavailable, the export raises a clear, actionable RuntimeError."""
    monkeypatch.setattr(me, "_HAS_MINARI", False)
    with pytest.raises(RuntimeError, match="minari is not installed"):
        me.to_minari(
            [{"kind": "step", "step": 0, "observation": [0.0], "decision": [0.0], "reward": 0.0}],
            "openoutcry/guard-v0",
            observation_space=None,
            action_space=None,
        )


# --- round-trip: rollout -> trace -> Minari dataset --------------------------

needs_full = pytest.mark.skipif(
    not HAS_BINDING, reason="native openoutcry binding not built"
)


@needs_full
def test_rollout_exports_to_minari_and_round_trips(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("minari")
    from openoutcry import OpenOutcryEnv

    # Keep Minari's dataset store inside the test sandbox.
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))

    env = OpenOutcryEnv(n_symbols=3, n_days=40, seed=7)
    obs, _ = env.reset(seed=7)
    weights = np.full(len(env.symbols), 1.0 / len(env.symbols), dtype=np.float32)

    path = str(tmp_path / "run.jsonl")
    writer = RolloutTraceWriter(path, config={"policy": "equal_weight"}, n_trials=0)
    n_steps = 0
    for t in range(8):
        nobs, reward, terminated, truncated, info = env.step(weights)
        writer.record_step(step=t, observation=obs, decision=weights, reward=reward, info=info)
        obs = nobs
        n_steps += 1
        if terminated or truncated:
            break
    writer.finalize()
    writer.close()

    dataset = me.to_minari(
        path,
        "openoutcry/test-rollout-v0",
        observation_space=env.observation_space,
        action_space=env.action_space,
        author="openoutcry-tests",
    )

    # Episode/step counts match the recorded rollout.
    assert dataset.total_episodes == 1
    assert dataset.total_steps == n_steps

    # EpisodeData round-trips: n+1 observations vs n actions/rewards, and ends on a flag.
    ep = list(dataset.iterate_episodes())[0]
    closes = np.asarray(ep.observations["closes"])
    assert closes.shape == (n_steps + 1, len(env.symbols))
    assert np.asarray(ep.actions).shape == (n_steps, len(env.symbols))
    assert np.asarray(ep.rewards).shape == (n_steps,)
    terms = np.asarray(ep.terminations)
    truncs = np.asarray(ep.truncations)
    assert terms.shape == (n_steps,) and truncs.shape == (n_steps,)
    # Exactly one terminal flag on the final step; none before.
    assert bool(terms[-1]) ^ bool(truncs[-1])
    assert not terms[:-1].any() and not truncs[:-1].any()

    # Recorded actions field-mapped from the decisions (equal weights).
    assert np.asarray(ep.actions)[0] == pytest.approx(weights)


@needs_full
def test_terminated_flag_is_synthesized_per_run(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("minari")
    from openoutcry import OpenOutcryEnv

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))

    env = OpenOutcryEnv(n_symbols=2, n_days=30, seed=3)
    obs, _ = env.reset(seed=3)
    weights = np.zeros(len(env.symbols), dtype=np.float32)
    with RolloutTraceWriter(str(tmp_path / "r.jsonl")) as writer:
        for t in range(4):
            nobs, reward, _, _, info = env.step(weights)
            writer.record_step(step=t, observation=obs, decision=weights, reward=reward, info=info)
            obs = nobs
        writer.finalize()

    # Force a true-termination run-level outcome and confirm it lands on the final step.
    dataset = me.to_minari(
        str(tmp_path / "r.jsonl"),
        "openoutcry/test-term-v0",
        observation_space=env.observation_space,
        action_space=env.action_space,
        terminated=True,
        author="openoutcry-tests",
    )
    ep = list(dataset.iterate_episodes())[0]
    assert bool(np.asarray(ep.terminations)[-1]) is True
    assert bool(np.asarray(ep.truncations)[-1]) is False


@needs_full
def test_train_test_split_export_over_disjoint_seed_bands(tmp_path, monkeypatch):
    """`to_minari_train_test` emits `-train`/`-test` datasets stamped with disjoint-band
    provenance, each round-tripping, from traces on disjoint seed intervals."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("minari")
    from openoutcry import OpenOutcryEnv
    from openoutcry.dataset import EVAL_SEED_BASE

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))

    env = OpenOutcryEnv(n_symbols=2, n_days=30, seed=0)
    weights = np.full(len(env.symbols), 1.0 / len(env.symbols), dtype=np.float32)

    def _roll(seed: int, path: str) -> int:
        obs, _ = env.reset(seed=seed)
        with RolloutTraceWriter(path) as writer:
            n = 0
            for t in range(5):
                nobs, reward, terminated, truncated, info = env.step(weights)
                writer.record_step(
                    step=t, observation=obs, decision=weights, reward=reward, info=info
                )
                obs = nobs
                n += 1
                if terminated or truncated:
                    break
            writer.finalize()
        return n

    train_path = str(tmp_path / "train.jsonl")  # seed in [0, n)
    test_path = str(tmp_path / "test.jsonl")  # seed >= EVAL_SEED_BASE
    n_train = _roll(3, train_path)
    n_test = _roll(EVAL_SEED_BASE + 3, test_path)

    train_ds, test_ds = me.to_minari_train_test(
        train_path,
        test_path,
        "openoutcry/test-split-v0",
        observation_space=env.observation_space,
        action_space=env.action_space,
        author="openoutcry-tests",
    )

    assert train_ds.spec.dataset_id == "openoutcry/test-split-train-v0"
    assert test_ds.spec.dataset_id == "openoutcry/test-split-test-v0"
    assert train_ds.total_steps == n_train and test_ds.total_steps == n_test

    train_meta = train_ds.storage.metadata
    test_meta = test_ds.storage.metadata
    assert train_meta["split"] == "train" and test_meta["split"] == "test"
    assert train_meta["split_method"] == "disjoint_seed_interval"
    assert train_meta["eval_seed_base"] == EVAL_SEED_BASE
    assert "seed interval" in train_meta["split_note"]
    # The bands are provably disjoint across the EVAL_SEED_BASE boundary.
    assert train_meta["seed_band_end"] < EVAL_SEED_BASE
    assert test_meta["seed_band_start"] >= EVAL_SEED_BASE

    # Each side round-trips as an offline-RL dataset.
    assert list(train_ds.iterate_episodes())[0].rewards.shape == (n_train,)
    assert list(test_ds.iterate_episodes())[0].rewards.shape == (n_test,)


@needs_full
def test_train_test_split_rejects_overlapping_seed_bands(tmp_path, monkeypatch):
    """Sharing a scenario seed across the two traces is rejected — the split is the leak gate."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("minari")
    from openoutcry import OpenOutcryEnv

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))

    env = OpenOutcryEnv(n_symbols=2, n_days=30, seed=0)
    weights = np.zeros(len(env.symbols), dtype=np.float32)

    def _roll(seed: int, path: str) -> None:
        obs, _ = env.reset(seed=seed)
        with RolloutTraceWriter(path) as writer:
            for t in range(3):
                nobs, reward, _, _, info = env.step(weights)
                writer.record_step(
                    step=t, observation=obs, decision=weights, reward=reward, info=info
                )
                obs = nobs
            writer.finalize()

    same = str(tmp_path / "a.jsonl")
    other = str(tmp_path / "b.jsonl")
    _roll(11, same)
    _roll(11, other)  # same seed in both -> overlapping bands

    with pytest.raises(ValueError, match="disjoint seed"):
        me.to_minari_train_test(
            same,
            other,
            "openoutcry/test-overlap-v0",
            observation_space=env.observation_space,
            action_space=env.action_space,
        )


@needs_full
def test_data_collector_works_on_live_env(tmp_path, monkeypatch):
    """Documents that `minari.DataCollector(OpenOutcryEnv(...))` captures live rollouts."""
    np = pytest.importorskip("numpy")
    minari = pytest.importorskip("minari")
    from openoutcry import OpenOutcryEnv

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))

    env = minari.DataCollector(OpenOutcryEnv(n_symbols=2, n_days=30, seed=5))
    env.reset(seed=5)
    weights = np.zeros(2, dtype=np.float32)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(weights)
        if terminated or truncated:
            break
    dataset = env.create_dataset("openoutcry/test-collect-v0", author="openoutcry-tests")
    assert dataset.total_episodes >= 1
    assert dataset.total_steps >= 1
