"""Exercise command forwarding only, never the throughput experiment."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _driver():
    source = Path(__file__).resolve().parents[3] / "paper/src/make-throughput.py"
    spec = importlib.util.spec_from_file_location(
        "throughput_driver_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_node_script_is_a_process_argument_not_a_shell_parameter(monkeypatch, platform):
    driver = _driver()
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='{"probe": true}')

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.run_wasm() == {"probe": True}
    assert len(observed) == 1
    argv, kwargs = observed[0]
    assert argv == ["node", "bench/throughput.js"]
    assert kwargs["shell"] is False
    assert kwargs["check"] is True
    assert kwargs["cwd"] == driver.REPO / "npm" / "sharpearena"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for argv probe")
def test_real_node_receives_script_in_a_path_with_spaces_and_shell_metacharacters(
    tmp_path, monkeypatch
):
    driver = _driver()
    root = tmp_path / "repo with spaces & symbols"
    bench = root / "npm" / "sharpearena" / "bench"
    bench.mkdir(parents=True)
    script = bench / "throughput.js"
    script.write_text(
        "console.log(JSON.stringify({argv: process.argv.slice(1), probe: 42}));\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(driver, "REPO", root)
    original = subprocess.run

    def closed_input(argv, **kwargs):
        # The old shell=True path launches bare Node on POSIX; close stdin so
        # that mutant exits instead of waiting for interactive input.
        return original(argv, stdin=subprocess.DEVNULL, timeout=10, **kwargs)

    monkeypatch.setattr(driver.subprocess, "run", closed_input)
    result = driver.run_wasm()
    assert result == {"probe": 42, "argv": [str(script)]}


def test_node_failure_is_not_a_successful_empty_measurement(monkeypatch):
    driver = _driver()

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(7, argv, stderr="fixture failure")

    monkeypatch.setattr(driver.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        driver.run_wasm()
    assert failure.value.returncode == 7
