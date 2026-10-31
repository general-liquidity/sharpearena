"""TorchRL as a learner (INT-12): the runnable recipe, its checkpoint, and bootstrapping.

Separate from ``test_torchrl.py`` so the ``wheel-extras`` TorchRL cell can run this file
with skips refused. Budgets are tiny and nothing here says a policy learned anything.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from sharpearena._seed_bands import EVAL_SEED_BASE

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None, reason="torchrl not installed"
)

_RECIPE_PATH = Path(__file__).resolve().parents[3] / "examples" / "torchrl" / "train_ppo.py"
_N_SYMBOLS = 2
_N_DAYS = 10


@pytest.fixture(scope="module")
def recipe():
    spec = importlib.util.spec_from_file_location("torchrl_recipe", _RECIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parts(recipe):
    from torchrl.objectives import ClipPPOLoss
    import torch

    torch.manual_seed(0)
    env = recipe.make_env(0, _N_SYMBOLS, _N_DAYS)
    actor, critic = recipe.make_modules(env, hidden=8)
    loss = ClipPPOLoss(actor, critic, entropy_coeff=0.01, normalize_advantage=False)
    return env, actor, critic, loss, torch.optim.Adam(loss.parameters(), lr=1e-3)


def test_training_updates_both_networks_on_the_float32_side_of_the_boundary(recipe):
    import torch

    env, actor, critic, loss, optimizer = _parts(recipe)
    reset = env.reset()
    assert reset["observation"].dtype == torch.float32
    assert reset["observation"].shape == (2 * _N_SYMBOLS + 1,)
    assert env.base_env.reset()["closes"].dtype == torch.float64

    before = {
        name: [p.clone() for p in module.parameters()]
        for name, module in (("actor", actor), ("critic", critic))
    }
    consumed = recipe.train(actor, critic, loss, optimizer, env, 32, 16)

    assert consumed == 32
    for name, module in (("actor", actor), ("critic", critic)):
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before[name], module.parameters())
        ), f"{name} parameters did not move"


def test_gae_bootstraps_past_the_end_of_the_window(recipe):
    """The reason the adapter writes three end keys. The last step of a tape is a
    truncation, so its one-step advantage has to include the next state's value; were it
    read as a termination that term would vanish, silently, in every return."""
    import torch

    env, actor, critic, *_ = _parts(recipe)
    with torch.no_grad():
        rollout = env.rollout(_N_DAYS + 5, actor, break_when_any_done=True)
        assert bool(rollout["next", "truncated"][-1])
        assert not bool(rollout["next", "terminated"][-1])
        recipe.make_estimator(critic)(rollout)

    last = rollout[-1]
    bootstrap = last["next", "state_value"]
    assert abs(bootstrap.item()) > 1e-4, "a zero value cannot tell the two readings apart"
    expected = last["next", "reward"] + 0.99 * bootstrap - last["state_value"]
    assert last["advantage"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_checkpoint_restores_weights_and_optimizer_so_resumed_training_is_identical(
    recipe, tmp_path
):
    """Both arms restart from one file's worth of state on the same seed, so any drift
    between them is state the checkpoint lost; Adam's moments are the likely casualty."""
    import torch

    env, actor, critic, loss, optimizer = _parts(recipe)
    recipe.train(actor, critic, loss, optimizer, env, 32, 16)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

    restored_env, restored_actor, restored_critic, restored_loss, restored_optimizer = (
        _parts(recipe)
    )
    checkpoint = torch.load(path, weights_only=True)
    restored_actor.load_state_dict(checkpoint["actor"])
    restored_critic.load_state_dict(checkpoint["critic"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])

    for arm_env, arm_actor, arm_critic, arm_loss, arm_optimizer in (
        (env, actor, critic, loss, optimizer),
        (restored_env, restored_actor, restored_critic, restored_loss, restored_optimizer),
    ):
        torch.manual_seed(11)
        arm_env.reset()
        recipe.train(arm_actor, arm_critic, arm_loss, arm_optimizer, arm_env, 16, 16)

    for left, right in zip(actor.parameters(), restored_actor.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(critic.parameters(), restored_critic.parameters()):
        assert torch.equal(left, right)


def test_evaluation_is_deterministic_and_changes_no_parameter(recipe):
    import torch

    env, actor, *_ = _parts(recipe)
    before = [p.clone() for p in actor.parameters()]

    first = recipe.evaluate(actor, recipe.make_env(EVAL_SEED_BASE, _N_SYMBOLS, _N_DAYS), _N_DAYS)
    second = recipe.evaluate(actor, recipe.make_env(EVAL_SEED_BASE, _N_SYMBOLS, _N_DAYS), _N_DAYS)

    assert first == second
    assert all(torch.equal(old, new) for old, new in zip(before, actor.parameters()))


def test_runnable_recipe_trains_checkpoints_resumes_and_evaluates(tmp_path):
    """The recipe is what `support-status.md` calls workflow-tested, so it runs here,
    in its own process, against the installed package."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_RECIPE_PATH),
            "--frames=32",
            "--frames-per-batch=16",
            "--n-symbols=2",
            "--n-days=10",
            f"--out-dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"evaluation seed: {EVAL_SEED_BASE}" in completed.stdout
    assert "frames:           48" in completed.stdout
    assert (tmp_path / "ppo.pt").is_file()
