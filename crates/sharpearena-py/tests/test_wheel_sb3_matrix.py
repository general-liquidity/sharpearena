"""The installed SB3 cell constrains CPU Torch and cannot inherit source imports."""

import importlib.util
from pathlib import Path
import sys


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


def test_cpu_torch_is_constrained_and_source_import_paths_are_removed(monkeypatch):
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
        ["sb3"],
        ["scripts/check-optional-extras.py --extra sb3"],
        True,
    )
    assert any("torch==2.9.0+cpu" in call and "--index-url" in call for call in calls)
    assert any(
        "--constraint" in call and call[-1].endswith(".whl[sb3]") for call in calls
    )
    assert any(
        "import torch; assert torch.version.cuda is None" in call for call in calls
    )
