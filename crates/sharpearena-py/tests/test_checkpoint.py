"""Tests for replay-based clone/restore (:mod:`sharpearena.checkpoint`).

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_checkpoint.py -q

The live tests need the native binding (``SharpeArenaEnv`` runs a real engine); they skip
cleanly if it isn't importable. The pure-data round-trip below the skip guard exercises
``CheckpointState`` serialization without the binding.
"""

import json
import pickle

import numpy as np
import pytest

# CheckpointState is pure data, but importing the module pulls in the engine binding (via
# .gym). Guard the whole live surface so the suite skips rather than errors when the native
# extension isn't built/current.
checkpoint = pytest.importorskip("sharpearena.checkpoint")
CheckpointableEnv = checkpoint.CheckpointableEnv
CheckpointState = checkpoint.CheckpointState

from sharpearena.gym import SharpeArenaEnv  # noqa: E402  (after importorskip)


def _equal_weight(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 1.0 / n, dtype=np.float32)


def _obs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _make(seed: int = 5) -> CheckpointableEnv:
    env = CheckpointableEnv(SharpeArenaEnv(n_symbols=3, n_days=60, seed=seed))
    env.reset()
    return env


def _roll(env, action, k: int):
    """Step ``action`` ``k`` times, returning the [(obs, reward)] trajectory."""
    traj = []
    for _ in range(k):
        obs, reward, terminated, truncated, _info = env.step(action)
        traj.append((obs, reward))
        if terminated or truncated:
            break
    return traj


# -- restore -----------------------------------------------------------------


def test_restore_returns_to_snapshot_point():
    """Roll a few steps, snapshot, roll further; restore -> the post-snapshot trajectory
    replays byte-identically."""
    env = _make()
    action = _equal_weight(env)

    _roll(env, action, 4)
    snap = env.clone_state()

    after = _roll(env, action, 6)  # the "ground truth" continuation from the snapshot
    assert after, "expected some post-snapshot steps"

    env.restore_state(snap)
    restored = _roll(env, action, 6)

    assert len(restored) == len(after)
    for (o1, r1), (o2, r2) in zip(after, restored):
        assert r1 == r2, "rewards must replay byte-identically"
        assert _obs_equal(o1, o2), "observations must replay byte-identically"


def test_restore_resets_recorded_prefix():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 3)
    snap = env.clone_state()
    assert snap.step == 3
    _roll(env, action, 5)

    env.restore_state(snap)
    again = env.clone_state()
    assert again.step == 3
    assert len(again.actions) == 3


@pytest.mark.parametrize("native", [False, True])
def test_checkpoint_preserves_every_scenario_control(native):
    controls = dict(vol_clustering=0.7, jump_burst_probability=0.5,
                    jump_burst_persistence=0.8, jump_burst_size=0.12)
    env = CheckpointableEnv(SharpeArenaEnv(
        n_symbols=2, n_days=80, seed=17, distribution_mode="extreme", **controls))
    env.reset()
    action = np.array([0.3, 0.2], dtype=np.float32)
    _roll(env, action, 3)
    snap = env.clone_state(native=native)
    expected = _roll(env, action, 10)
    fork = env.branch(snap)
    actual = _roll(fork, action, 10)
    assert len(actual) == len(expected) == 10
    assert all(r1 == r2 and _obs_equal(o1, o2)
               for (o1, r1), (o2, r2) in zip(expected, actual))
    assert all(snap.params[key] == value for key, value in controls.items())


def test_replay_checkpoint_after_native_restore_keeps_the_entire_prefix():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state(native=True)
    env.restore_state(snap)
    _roll(env, action, 2)
    second = env.clone_state()
    assert len(second.actions) == second.step == 6
    expected = _roll(env, action, 5)
    actual = _roll(env.branch(second), action, 5)
    assert len(actual) == len(expected) == 5
    assert all(r1 == r2 and _obs_equal(o1, o2)
               for (o1, r1), (o2, r2) in zip(expected, actual))


# -- branch ------------------------------------------------------------------


def test_branch_is_independent_of_original():
    """Stepping a branch must not perturb the parent env."""
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    branch = env.branch(snap)
    # Drive the branch far forward.
    _roll(branch, action, 8)

    # The parent, restored to the same snapshot, still replays the original continuation.
    parent_after = _roll(env.branch(snap), action, 5)
    env.restore_state(snap)
    parent_restored = _roll(env, action, 5)
    for (o1, r1), (o2, r2) in zip(parent_after, parent_restored):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_two_branches_same_actions_identical():
    """Determinism: two independent branches from one snapshot, fed identical actions,
    produce identical trajectories."""
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 5)
    snap = env.clone_state()

    b1 = env.branch(snap)
    b2 = env.branch(snap)
    t1 = _roll(b1, action, 7)
    t2 = _roll(b2, action, 7)

    assert len(t1) == len(t2) and t1
    for (o1, r1), (o2, r2) in zip(t1, t2):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_branch_does_not_share_action_list():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 3)
    snap = env.clone_state()
    branch = env.branch(snap)
    _roll(branch, action, 4)
    # Parent's recorded prefix is untouched by branch stepping.
    assert env.clone_state().step == 3
    assert branch.clone_state().step == 3 + 4


# -- CheckpointState serialization ------------------------------------------


def test_state_roundtrips_through_dict():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    restored = CheckpointState.from_dict(snap.to_dict(include_private=True))
    assert restored.params == snap.params
    assert restored.step == snap.step
    assert restored.include_rng == snap.include_rng
    assert np.allclose(np.array(restored.actions), np.array(snap.actions))


def test_state_is_picklable():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    blob = pickle.dumps(snap)
    back = pickle.loads(blob)
    assert isinstance(back, CheckpointState)
    assert back.to_dict(include_private=True) == snap.to_dict(include_private=True)

    # A pickled state restores a fresh env exactly.
    fresh = env.branch(back)
    a, b = _roll(fresh, action, 3), _roll(env.branch(snap), action, 3)
    for (o1, r1), (o2, r2) in zip(a, b):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_state_carries_no_env_handle():
    """Leak-safety: the captured params must not contain a dataset / env handle."""
    env = _make()
    snap = env.clone_state()
    for value in snap.params.values():
        assert not (
            callable(getattr(value, "reset", None))
            and callable(getattr(value, "step", None))
        ), "checkpoint params must not embed a live env/dataset handle"


def test_native_checkpoint_matches_replay():
    """The native snapshot (native=True) restores byte-identically and agrees with
    the replay path."""
    import numpy as np
    from sharpearena import SharpeArenaEnv, CheckpointableEnv

    def _act(env):
        n = env.action_space.shape[0]
        return np.full((n,), 0.2, dtype=np.float32)

    env = CheckpointableEnv(SharpeArenaEnv(n_symbols=3, n_days=60, seed=9))
    env.reset(seed=9)
    a = _act(env)
    for _ in range(8):
        env.step(a)
    snap_native = env.clone_state(native=True)
    snap_replay = env.clone_state(native=False)
    # Advance, then restore via the native no-replay path.
    after = [tuple(env.step(a)[0]["closes"]) for _ in range(4)]
    env.restore_state(snap_native)
    native_after = [tuple(env.step(a)[0]["closes"]) for _ in range(4)]
    # and via replay, on an independent branch
    branch = env.branch(snap_replay)
    replay_after = [tuple(branch.step(a)[0]["closes"]) for _ in range(4)]
    assert native_after == after
    assert native_after == replay_after
    # the native snapshot round-trips through to_dict/from_dict
    from sharpearena import CheckpointState
    rt = CheckpointState.from_dict(snap_native.to_dict(include_private=True))
    assert rt.native_state == snap_native.native_state


@pytest.mark.parametrize("native", [False, True])
def test_public_export_omits_future_csv_and_reconstruction_data(native):
    # The needle exists only in the last bar, not in the observed prefix. Checking
    # for Dataset objects alone would pass while exporting this entire string.
    future_price = "98765.432109"
    csv_text = "date,symbol,close\n" + "\n".join(
        f"2024-01-{day:02d},A,{future_price if day == 12 else 100 + day}"
        for day in range(1, 13)
    )
    env = CheckpointableEnv(SharpeArenaEnv(csv_text=csv_text, n_symbols=1))
    env.reset()
    env.step([0.25])
    snap = env.clone_state(native=native)
    assert future_price in snap.params["csv_text"], "exercise the actual CSV path"
    public = snap.to_dict()
    assert set(public) == {"schema", "step", "actions"}
    assert public["schema"] == "sharpearena.checkpoint-public.v1"
    assert public["step"] == 1
    assert public["actions"] == [[0.25]]
    assert future_price not in json.dumps(public)
    with pytest.raises(ValueError, match="public.*restor"):
        CheckpointState.from_dict(public)

    private = snap.to_dict(include_private=True)
    assert future_price in json.dumps(private)
    restored = CheckpointState.from_dict(json.loads(json.dumps(private)))
    actual = env.branch(restored).step([0.1])
    expected = env.step([0.1])
    assert _obs_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_private_export_and_import_are_detached():
    state = CheckpointState(params={"nested": {"values": [1, 2]}},
                            actions=[[0.25]], step=1)
    exported = state.to_dict(include_private=True)
    imported = CheckpointState.from_dict(exported)
    exported["params"]["nested"]["values"].append(3)
    exported["actions"][0][0] = 0.5
    assert state.params == imported.params == {"nested": {"values": [1, 2]}}
    assert state.actions == imported.actions == [[0.25]]


def test_nested_live_handle_is_refused_in_private_params():
    env = _make()
    with pytest.raises(TypeError, match="dataset/env handle"):
        checkpoint._assert_no_leak({"nested": [{"handle": env.env}]})


def test_legacy_private_payload_remains_restorable():
    env = _make()
    env.step(_equal_weight(env))
    private = env.clone_state().to_dict(include_private=True)
    del private["schema"]
    restored = CheckpointState.from_dict(private)
    assert env.branch(restored).clone_state().actions == private["actions"]


def test_unknown_checkpoint_schema_is_refused():
    with pytest.raises(ValueError, match="schema"):
        CheckpointState.from_dict({"schema": "unknown", "params": {}})


@pytest.mark.parametrize("action", [
    [[0.2, 0.2, 0.2]], ["0.2", "0.2", "0.2"], [True, False, True],
    [0.2, 0.2], [0.2, float("nan"), 0.2], [0.2, float("inf"), 0.2], [2, 0, 0],
])
def test_checkpoint_wrapper_rejects_invalid_actions_before_advance(action):
    env = _make()
    before = env.clone_state(native=True).to_dict(include_private=True)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.clone_state(native=True).to_dict(include_private=True) == before


def test_checkpoint_records_and_replays_the_exact_float64_action():
    env = _make()
    direct = SharpeArenaEnv(n_symbols=3, n_days=60, seed=5)
    direct.reset()
    action = np.array([0.123456789123, 0.234567891234, 0.345678912345])
    assert not np.array_equal(action, action.astype(np.float32))
    actual, expected = env.step(action), direct.step(action)
    snap = env.clone_state()
    assert snap.actions == [action.tolist()]
    assert actual[1:] == expected[1:]
    assert _obs_equal(actual[0], expected[0])
    fork = env.branch(snap)
    actual, expected = fork.step(action), direct.step(action)
    assert actual[1:] == expected[1:]
    assert _obs_equal(actual[0], expected[0])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("invalid", ["step", "string-step", "bool-step", "rng", "action"])
def test_malformed_restore_does_not_replace_or_advance_live_environment(native, invalid):
    env = _make()
    _roll(env, _equal_weight(env), 3)
    snap = env.clone_state(native=native)
    before = env.clone_state(native=True).to_dict(include_private=True)
    original = env.env
    if invalid == "step":
        snap.step = 2
    elif invalid == "string-step":
        snap.step = "3"
    elif invalid == "bool-step":
        snap.step = True
    elif invalid == "rng":
        snap.include_rng = "false"
    else:
        snap.actions[1] = [[0.2, 0.2, 0.2]]
    with pytest.raises((TypeError, ValueError)):
        env.restore_state(snap)
    assert env.env is original, "failure must not replace the live engine"
    assert env.clone_state(native=True).to_dict(include_private=True) == before


def test_corrupt_native_snapshot_does_not_replace_live_environment():
    env = _make()
    env.step(_equal_weight(env))
    before = env.clone_state(native=True).to_dict(include_private=True)
    original = env.env
    snap = env.clone_state(native=True)
    snap.native_state = "{broken"
    with pytest.raises(ValueError):
        env.restore_state(snap)
    assert env.env is original
    assert env.clone_state(native=True).to_dict(include_private=True) == before


@pytest.mark.parametrize("field,value", [("step", 1.5), ("step", -1),
                                       ("step", True), ("include_rng", "false")])
def test_private_decoder_refuses_coerced_metadata(field, value):
    private = _make().clone_state().to_dict(include_private=True)
    private[field] = value
    with pytest.raises((TypeError, ValueError)):
        CheckpointState.from_dict(private)
