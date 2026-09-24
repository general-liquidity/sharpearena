"""The TorchRL route: TensorDict layout, the dtype boundary, parity, and a collector.

The behaviour tests skip when ``torchrl`` is not installed; the rest of the package
works without it, and the optional-import test runs *only* when it is absent and
asserts that the module still imports and that construction refuses by name.

Three things are pinned here that nothing else in the suite can pin:

* the ``float64`` boundary. TorchRL is a torch pipeline and torch defaults to
  ``float32``; Gymnasium's ``Box.contains`` casts safely, so a narrowing would pass
  every space-membership assertion in the suite. The dtype tests compare dtypes and
  bits, never containment, which is the standard ``docs/rl-contract-coverage.md``
  sets for the native boundary.
* parity. The engine is replayed by ``integrations.parity`` with its own restatement
  of the wire contract, and the candidate path is driven entirely through the
  TensorDict API, so a mapping error in the tensor layout shows up as a mismatch
  rather than as agreement.
* the collector. A batch collected by ``torchrl.collectors.Collector`` must survive
  ``torch.save``/``torch.load`` unchanged, and replaying its recorded actions against
  a fresh engine must reproduce its rewards exactly.
"""

import importlib.util
import io

import numpy as np
import pytest

# Always safe: the module imports with or without torchrl, and the package must too.
import sharpearena  # noqa: F401
import sharpearena.torchrl_env as trl_env
from sharpearena.gym import SharpeArenaEnv
from sharpearena.integrations import parity
from sharpearena.torchrl_env import (
    OBSERVATION_KEYS,
    SharpeArenaTorchRLEnv,
    TorchRLUnavailable,
    equal_weight_policy,
)

_HAS_TORCHRL = importlib.util.find_spec("torchrl") is not None
needs_torchrl = pytest.mark.skipif(not _HAS_TORCHRL, reason="torchrl not installed")

_SCENARIO = {"n_symbols": 3, "n_days": 40, "seed": 7}


def _make(**overrides):
    return SharpeArenaTorchRLEnv(**{**_SCENARIO, **overrides})


class _TensorDictView:
    """The scalar Gymnasium shape ``integrations.parity`` consumes, over the TensorDict API.

    Every value it reports is read back out of the tensordict the TorchRL environment
    produced, so the comparison exercises the tensor layout rather than the
    ``SharpeArenaEnv`` underneath it. NAV has no spec key in that layout and is read
    from the engine's own step info, which is what ``adapter_rollout`` already does for
    the Gymnasium adapter.
    """

    def __init__(self, env: SharpeArenaTorchRLEnv) -> None:
        self._env_trl = env
        self._td = None

    @property
    def symbols(self):
        return self._env_trl.symbols

    @property
    def _env(self):
        # The native handle, so the parity record carries the effective config.
        return self._env_trl.gymnasium_env._env

    @staticmethod
    def _numpy_obs(td):
        return {key: td.get(key).detach().cpu().numpy() for key in OBSERVATION_KEYS}

    def reset(self):
        self._td = self._env_trl.reset()
        return self._numpy_obs(self._td), {}

    def step(self, action):
        import torch

        self._td.set(
            "action", torch.as_tensor(np.asarray(action), dtype=torch.float32)
        )
        out = self._env_trl.step(self._td)
        nxt = out.get("next")
        self._td = out.get("next").exclude("reward", "done", "terminated", "truncated").clone()
        return (
            self._numpy_obs(nxt),
            float(nxt.get("reward").item()),
            bool(nxt.get("terminated").item()),
            bool(nxt.get("truncated").item()),
            dict(self._env_trl.last_step_info),
        )


# -- the guarded optional dependency ---------------------------------------------


@pytest.mark.skipif(_HAS_TORCHRL, reason="this asserts the behaviour when torchrl is absent")
def test_construction_refuses_by_name_without_torchrl():
    with pytest.raises(TorchRLUnavailable) as excinfo:
        SharpeArenaTorchRLEnv(**_SCENARIO)
    assert "sharpearena[torchrl]" in str(excinfo.value)


def test_the_module_imports_whether_or_not_torchrl_is_installed():
    # The names exist either way; only construction is gated.
    assert trl_env.OBSERVATION_KEYS == ("closes", "positions", "cash")
    assert issubclass(TorchRLUnavailable, RuntimeError)


# -- the TorchRL contract ---------------------------------------------------------


@needs_torchrl
def test_the_upstream_spec_checker_passes():
    from torchrl.envs.utils import check_env_specs

    check_env_specs(_make())


@needs_torchrl
def test_the_step_output_lives_under_next_with_all_three_flags():
    import torch

    env = _make()
    td = env.reset()
    td.set("action", torch.full((3,), 1.0 / 3.0, dtype=torch.float32))
    out = env.step(td)

    nxt = out.get("next")
    assert set(OBSERVATION_KEYS) <= set(nxt.keys())
    for key in ("reward", "done", "terminated", "truncated"):
        assert key in nxt.keys(), key
    # Writing "done" alone would be read as termination upstream, which is wrong for a
    # window that ends by truncation.
    assert nxt.get("done").item() == (
        nxt.get("terminated").item() or nxt.get("truncated").item()
    )


@needs_torchrl
def test_running_out_of_bars_truncates_without_terminating():
    env = _make()
    policy = equal_weight_policy(env)
    rollout = env.rollout(1000, policy=policy)

    terminated = rollout.get(("next", "terminated")).squeeze(-1)
    truncated = rollout.get(("next", "truncated")).squeeze(-1)
    assert bool(truncated[-1]), "the window end must be reported as truncation"
    assert not bool(terminated[-1]), "an end-of-window step is not a blow-up"
    assert not terminated.any()


@needs_torchrl
def test_a_seed_handed_to_set_seed_selects_the_scenario():
    env_a, env_b, env_c = _make(), _make(), _make()
    env_a.set_seed(21)
    env_b.set_seed(21)
    env_c.set_seed(22)

    a = env_a.reset().get("closes")
    b = env_b.reset().get("closes")
    c = env_c.reset().get("closes")

    import torch

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# -- the dtype boundary -----------------------------------------------------------


@needs_torchrl
def test_observations_and_reward_cross_the_boundary_as_engine_float64():
    """Fails if ``float64`` is narrowed anywhere on the observation or reward path.

    Compared by dtype and by bits against the NumPy ``float64`` the Gymnasium adapter
    decodes, never by space membership: ``Box.contains`` casts safely and would accept
    a ``float32`` observation against a ``float64`` space without complaint.
    """
    import torch

    env = _make()
    reference = SharpeArenaEnv(**_SCENARIO)

    for key in OBSERVATION_KEYS:
        assert env.observation_spec[key].dtype is torch.float64, key
    assert env.reward_spec.dtype is torch.float64
    assert env.obs_dtype is torch.float64
    assert not env.narrows_observations

    td = env.reset()
    reference_obs, _ = reference.reset()
    for key in OBSERVATION_KEYS:
        tensor = td.get(key)
        assert tensor.dtype is torch.float64, key
        assert reference_obs[key].dtype == np.float64, key
        assert torch.equal(tensor, torch.from_numpy(reference_obs[key])), key

    action = np.full((3,), 1.0 / 3.0, dtype=np.float32)
    td.set("action", torch.as_tensor(action))
    nxt = env.step(td).get("next")
    ref_obs, ref_reward, _te, _tr, _info = reference.step(action)

    assert nxt.get("reward").dtype is torch.float64
    # Exact equality, not allclose: one path narrowing to float32 moves these bits.
    assert nxt.get("reward").item() == ref_reward
    for key in OBSERVATION_KEYS:
        assert torch.equal(nxt.get(key), torch.from_numpy(ref_obs[key])), key


@needs_torchrl
def test_float32_observations_are_opt_in_and_the_loss_is_locatable():
    """The narrowing is reachable, has to be named, and is shown to be lossy."""
    import torch

    narrowed = _make(obs_dtype=torch.float32)
    assert narrowed.narrows_observations
    assert narrowed.observation_spec["closes"].dtype is torch.float32
    assert narrowed.reward_spec.dtype is torch.float32

    exact = _make()
    lossy_closes = narrowed.reset().get("closes")
    exact_closes = exact.reset().get("closes")
    assert lossy_closes.dtype is torch.float32
    # Widened back it is a different number, which is exactly the precision that was
    # dropped; equality here would mean the narrowing had not happened.
    assert not torch.equal(lossy_closes.to(torch.float64), exact_closes)


@needs_torchrl
@pytest.mark.parametrize("bad", ["float16", "int64", "not-a-dtype"])
def test_an_undeclared_observation_width_is_refused_not_coerced(bad):
    with pytest.raises(ValueError, match="obs_dtype"):
        _make(obs_dtype=bad)


@needs_torchrl
def test_the_action_spec_is_the_advertised_box_and_no_wider():
    """The action is where precision is lost, and the spec states the width honestly."""
    import torch

    env = _make(max_weight=0.5)
    box = env.gymnasium_env.action_space

    assert env.action_spec.dtype is torch.float32
    assert box.dtype == np.float32
    assert torch.equal(env.action_spec.low, torch.from_numpy(box.low))
    assert torch.equal(env.action_spec.high, torch.from_numpy(box.high))
    # A weight the Box refuses must not be inside the spec either.
    assert not env.action_spec.is_in(torch.full((3,), 0.75, dtype=torch.float32))


# -- parity against the engine ----------------------------------------------------


@needs_torchrl
@pytest.mark.parametrize("fixture", parity.CORE_FIXTURES, ids=lambda f: f.name)
def test_the_tensordict_route_reproduces_the_engine(fixture):
    def make_env(fx):
        return _TensorDictView(
            SharpeArenaTorchRLEnv(
                seed=fx.seed,
                max_weight=fx.max_weight,
                allow_short=fx.allow_short,
                mode=fx.mode,
                **fx.scenario_kwargs(),
            )
        )

    report = parity.check_adapter_parity(
        fixture, make_env, source="sharpearena.torchrl_env.SharpeArenaTorchRLEnv"
    )
    report.require_parity()
    assert report.steps_compared > 0


@needs_torchrl
def test_narrowed_observations_fail_the_parity_digest():
    """The parity digest is the second gate on the dtype boundary.

    ``observation_digest`` hashes the float64 values, so a route that narrows them
    cannot report parity. This asserts the checker actually bites, which is what makes
    the passing parity test above evidence rather than a formality.
    """
    import torch

    fixture = parity.CORE_FIXTURES[0]

    def make_env(fx):
        return _TensorDictView(
            SharpeArenaTorchRLEnv(
                seed=fx.seed,
                max_weight=fx.max_weight,
                allow_short=fx.allow_short,
                mode=fx.mode,
                obs_dtype=torch.float32,
                **fx.scenario_kwargs(),
            )
        )

    report = parity.check_adapter_parity(fixture, make_env, source="narrowed")
    assert not report.ok
    assert any("observation" in mismatch for mismatch in report.mismatches)


# -- the collector round trip -----------------------------------------------------


@needs_torchrl
def test_a_collected_batch_survives_serialization_and_replays_unchanged():
    """The INT-12 round trip: collect, serialize, reload, and replay against the engine."""
    import torch
    from torchrl.collectors import Collector

    frames = 8
    collector = Collector(
        lambda: _make(),
        policy=equal_weight_policy(_make()),
        frames_per_batch=frames,
        total_frames=frames,
    )
    try:
        batch = next(iter(collector)).clone()
    finally:
        collector.shutdown()

    assert batch.shape == torch.Size([frames])
    assert batch.get(("next", "reward")).dtype is torch.float64

    buffer = io.BytesIO()
    torch.save(batch, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)

    assert set(restored.keys(include_nested=True)) == set(
        batch.keys(include_nested=True)
    )
    for key in ("action", ("next", "reward"), ("next", "closes"), ("next", "done")):
        assert torch.equal(restored.get(key), batch.get(key)), key

    # Replay the recorded actions against a fresh engine. The collector's rewards are
    # the engine's, not a rescaled or reordered copy of them.
    replay = SharpeArenaEnv(**_SCENARIO)
    replay.reset()
    for index in range(frames):
        action = restored.get("action")[index].numpy()
        obs, reward, _terminated, _truncated, _info = replay.step(action)
        assert reward == restored.get(("next", "reward"))[index].item(), index
        for key in OBSERVATION_KEYS:
            assert np.array_equal(
                obs[key], restored.get(("next", key))[index].numpy()
            ), (index, key)
