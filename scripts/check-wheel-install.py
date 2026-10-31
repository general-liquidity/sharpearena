#!/usr/bin/env python3
"""Build the SharpeArena wheel, install it into a clean virtual environment, and run
checks against it from outside the checkout.

The compatibility matrix runs this on every operating system and every Python version
the package declares support for, so the install and the checks cannot be shell built:
a virtual environment lays itself out under `Scripts` on Windows and `bin` everywhere
else, and a Windows path does not survive a POSIX glob. Doing it here keeps one code
path for every cell of the matrix.

    python scripts/check-wheel-install.py --build --github-env
    python scripts/check-wheel-install.py --wheel "$WHEEL" \
        --run "scripts/check-installed-wheel.py" \
        --run "scripts/check-packaged-adapter.py"
    python scripts/check-wheel-install.py --wheel "$WHEEL" --extra minari \
        --run "scripts/check-optional-extras.py --extra minari"

Each `--run` script is executed by the virtual environment's interpreter with the
working directory set to an empty temporary directory, so an import can only resolve to
the installed distribution.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CRATE = REPO_ROOT / "crates" / "sharpearena-py"


def run(command: list, **kwargs) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def build_wheel(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--requirement",
            CRATE / "build-requirements.txt",
        ]
    )
    # --locked: the crate's committed Cargo.lock is what the published wheel is built
    # from, so a manifest that has moved past it fails here instead of resolving fresh.
    run(
        [sys.executable, "-m", "maturin", "build", "--locked", "--out", out_dir],
        cwd=CRATE,
    )
    wheels = sorted(out_dir.glob("sharpearena-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {out_dir}, found {wheels}")
    if wheels[0].stat().st_size == 0:
        raise SystemExit(f"{wheels[0]} is empty")
    return wheels[0]


def venv_python(venv: Path) -> Path:
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no interpreter under {venv}")


def wheel_version(wheel: Path) -> str:
    parts = wheel.name.split("-")
    if len(parts) < 2:
        raise SystemExit(f"cannot read a version out of {wheel.name}")
    return parts[1]


def pinned_pytest() -> str:
    """The pytest the source suite pins, so a learner cell runs the same runner."""
    for line in (CRATE / "ci-requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("pytest=="):
            return line.strip()
    raise SystemExit("ci-requirements.txt pins no pytest")


# The extra sets a cell may constrain to CPU Torch. Ray declares no learning framework, so
# its cell brings Torch itself for RLlib training, and PettingZoo for the multi-agent route.
CPU_TORCH_EXTRAS = (["sb3"], ["torchrl"], ["ray", "pettingzoo"])


def verify(
    wheel: Path,
    extras: list,
    checks: list,
    cpu_torch: bool = False,
    pytest_runner: bool = False,
) -> None:
    requirement = str(wheel)
    if extras:
        requirement = f"{requirement}[{','.join(extras)}]"
    # What the installed distribution has to report back, so a sharpearena already on the
    # interpreter cannot answer for the wheel this cell built.
    environment = dict(os.environ, SHARPEARENA_EXPECTED_VERSION=wheel_version(wheel))
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        venv = workdir / "consumer"
        run([sys.executable, "-m", "venv", str(venv)], env=environment)
        python = venv_python(venv)
        run(
            [python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            env=environment,
        )
        constraints = []
        if cpu_torch:
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "torch==2.9.0+cpu",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                env=environment,
            )
            constraint = workdir / "cpu-torch.txt"
            constraint.write_text("torch==2.9.0+cpu\n", encoding="utf-8")
            constraints = ["--constraint", constraint]
        run(
            [python, "-m", "pip", "install", "--quiet", *constraints, requirement],
            env=environment,
        )
        if pytest_runner:
            run(
                [python, "-m", "pip", "install", "--quiet", pinned_pytest()],
                env=environment,
            )
        if cpu_torch:
            run(
                [python, "-c", "import torch; assert torch.version.cuda is None"],
                env=environment,
            )
        elsewhere = workdir / "elsewhere"
        elsewhere.mkdir()
        for check in checks:
            command = shlex.split(check)
            script = REPO_ROOT / command[0]
            if not script.is_file():
                raise SystemExit(f"{script} does not exist")
            run([python, script, *command[1:]], cwd=elsewhere, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build the wheel and stop")
    parser.add_argument("--out", default=None, help="where --build writes the wheel")
    parser.add_argument(
        "--wheel", default=None, help="an already built wheel to verify"
    )
    parser.add_argument("--extra", action="append", default=[], dest="extras")
    parser.add_argument("--run", action="append", default=[], dest="checks")
    parser.add_argument(
        "--cpu-torch",
        action="store_true",
        help="install and constrain Torch 2.9.0+cpu for a learner wheel cell",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="install the pinned pytest, for scripts/run-learner-tests.py",
    )
    parser.add_argument(
        "--github-env",
        action="store_true",
        help="write WHEEL=<path> to $GITHUB_ENV after --build",
    )
    arguments = parser.parse_args()

    if arguments.build == bool(arguments.wheel):
        raise SystemExit("pass exactly one of --build and --wheel")

    if arguments.build:
        out = Path(arguments.out or os.environ.get("RUNNER_TEMP", ".")) / "dist"
        wheel = build_wheel(out)
        print(f"built {wheel}")
        github_env = os.environ.get("GITHUB_ENV")
        if arguments.github_env and github_env:
            with open(github_env, "a", encoding="utf-8") as handle:
                handle.write(f"WHEEL={wheel}\n")
        return 0

    wheel = Path(arguments.wheel)
    if not wheel.is_file():
        raise SystemExit(f"{wheel} is not a file")
    checks = list(arguments.checks)
    if not checks:
        raise SystemExit("--wheel needs at least one --run check to be worth anything")
    if arguments.cpu_torch and arguments.extras not in CPU_TORCH_EXTRAS:
        raise SystemExit(f"--cpu-torch supports exactly the extras {CPU_TORCH_EXTRAS}")
    verify(wheel, arguments.extras, checks, arguments.cpu_torch, arguments.pytest)
    print(f"verified {wheel.name} with extras {arguments.extras or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
