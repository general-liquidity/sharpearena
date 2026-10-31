"""The installed SB3 cell constrains CPU Torch and cannot inherit source imports."""

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import optional_extras  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "wheel_check", SCRIPTS / "check-wheel-install.py"
)
wheel_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_check)


def test_sb3_is_an_exercised_declared_extra_not_an_exclusion():
    assert "sb3" in optional_extras.validate_coverage()
    assert "sb3" in optional_extras.installable_extras()
    assert "sb3" not in optional_extras.INSTALL_MATRIX_EXCLUDED


def test_torchrl_is_an_exercised_declared_extra_not_an_exclusion():
    assert "torchrl" in optional_extras.validate_coverage()
    assert "torchrl" in optional_extras.installable_extras()
    assert "torchrl" not in optional_extras.INSTALL_MATRIX_EXCLUDED


def test_ray_is_an_exercised_declared_extra_not_an_exclusion():
    assert "ray" in optional_extras.validate_coverage()
    assert "ray" in optional_extras.installable_extras()
    assert "ray" not in optional_extras.INSTALL_MATRIX_EXCLUDED


@pytest.mark.parametrize("extra", ["sb3", "torchrl"])
def test_cpu_torch_is_constrained_and_source_import_paths_are_removed(
    monkeypatch, extra
):
    calls = []
    monkeypatch.setenv("PYTHONPATH", "source-tree-must-not-leak")
    monkeypatch.setenv("PYTHONHOME", "foreign-interpreter")
    monkeypatch.setattr(wheel_check, "venv_python", lambda path: path / "python")

    def capture(command, **kwargs):
        command = [str(part) for part in command]
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTHONHOME" not in kwargs["env"]
        if "--constraint" in command:
            path = Path(command[command.index("--constraint") + 1])
            assert path.read_text() == "torch==2.9.0+cpu\n"
        calls.append(command)

    monkeypatch.setattr(wheel_check, "run", capture)
    wheel_check.verify(
        Path("sharpearena-0.31.0-cp312-test.whl"),
        [extra],
        [f"scripts/check-optional-extras.py --extra {extra}"],
        True,
    )
    assert any("torch==2.9.0+cpu" in call and "--index-url" in call for call in calls)
    assert any(
        "--constraint" in call and call[-1].endswith(f".whl[{extra}]") for call in calls
    )
    assert any(
        "import torch; assert torch.version.cuda is None" in call for call in calls
    )


def _ci_branch(extra):
    workflow = (SCRIPTS.parent / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    return workflow.split(f'"$EXTRA" = {extra} ]', 1)[1].split("; then", 1)[1].split(
        "\n          el", 1
    )[0].split("\n          fi", 1)[0]


@pytest.mark.parametrize(
    "extra,test_file",
    [
        ("sb3", "test_sb3_learner.py"),
        ("torchrl", "test_torchrl_learner.py"),
        ("minari", "test_minari_learner.py"),
        ("ray", "test_rllib_learner.py"),
    ],
)
def test_learner_file_runs_in_the_cell_that_installs_its_extra(extra, test_file):
    """The learner files skip without their extra, so a file no cell runs is never run
    at all, and nothing else in CI would notice."""
    branch = _ci_branch(extra)
    assert f"scripts/run-learner-tests.py {test_file}" in branch
    assert "--pytest" in branch


def test_ray_cell_keeps_its_own_resolution_with_torch_and_pettingzoo():
    """`ray[rllib]` pins gymnasium==1.2.2 below the source suite's pin, so the RLlib
    learner route can only run in this separately resolved cell."""
    assert "--extra pettingzoo --cpu-torch" in _ci_branch("ray")
    assert ["ray", "pettingzoo"] in wheel_check.CPU_TORCH_EXTRAS
    requirements = SCRIPTS.parent / "crates" / "sharpearena-py" / "ci-requirements.txt"
    pinned = {line.split("==")[0] for line in requirements.read_text().splitlines()}
    assert "ray" not in pinned


def _load_runner(monkeypatch, tmp_path, body):
    spec = importlib.util.spec_from_file_location(
        "learner_runner", SCRIPTS / "run-learner-tests.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    package = tmp_path / "site-packages" / "sharpearena"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.delitem(sys.modules, "sharpearena", raising=False)
    (tmp_path / "test_gated_case.py").write_text(body)
    monkeypatch.setattr(runner, "TESTS", tmp_path)
    return runner


def test_learner_runner_fails_a_cell_whose_tests_skipped(monkeypatch, tmp_path):
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        "import pytest\n\ndef test_gated():\n    pytest.skip('extra absent')\n",
    )
    with pytest.raises(SystemExit, match="skipped in a cell"):
        runner.main(["test_gated_case.py"])


def test_learner_runner_passes_a_cell_whose_tests_ran(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch, tmp_path, "def test_gated():\n    pass\n")
    assert runner.main(["test_gated_case.py"]) == 0


def test_learner_runner_refuses_a_source_tree_import(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch, tmp_path, "def test_gated():\n    pass\n")
    source = tmp_path / "checkout" / "sharpearena"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(source.parent))
    with pytest.raises(SystemExit, match="not an installed distribution"):
        runner.main(["test_gated_case.py"])
