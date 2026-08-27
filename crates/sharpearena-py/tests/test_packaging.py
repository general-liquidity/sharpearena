"""The declared optional dependencies must produce a working environment.

CI installed only ``pytest numpy gymnasium verifiers`` for most of this package's life,
so every Minari test skipped silently; installing ``ci-requirements.txt`` unmasked them
and each missing transitive dependency surfaced one at a time. That file is not shipped
in the wheel, so a user following the documented ``pip install sharpearena[minari]`` got
the broken environment CI had. These tests pin the extra against what the Minari path
actually imports.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import tomllib
from pathlib import Path

import pytest


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
CI_REQUIREMENTS = Path(__file__).resolve().parents[1] / "ci-requirements.txt"

# Imported by the Minari path this package exercises: h5py by minari's default HDF5
# storage, jax by `EpisodeBuffer.add_step_data` under `DataCollector`, and PIL by
# `minari/dataset/_storages/hdf5_storage.py`, which imports it at module scope without
# declaring it.
MINARI_RUNTIME_IMPORTS = {"h5py": "h5py", "jax": "jax", "pillow": "PIL"}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    return _canonical(re.split(r"[\[<>=!~;\s]", requirement, maxsplit=1)[0])


def _requested_extras(requirement: str) -> set[str]:
    match = re.search(r"\[([^\]]+)\]", requirement)
    return {part.strip() for part in match.group(1).split(",")} if match else set()


def _extras() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _closure(requirements: list[str]) -> set[str]:
    """Distribution names ``pip install`` would resolve from these requirements.

    One level deep and extras-only, which is all this needs: the question is whether the
    extra reaches the packages listed above, not whether pip resolves the whole graph.
    """
    names = set()
    for requirement in requirements:
        name = _requirement_name(requirement)
        names.add(name)
        wanted = _requested_extras(requirement)
        if not wanted:
            continue
        for nested in metadata.requires(name) or []:
            marker = re.search(r'extra\s*==\s*[\'"]([^\'"]+)[\'"]', nested)
            if marker and marker.group(1) in wanted:
                names.add(_requirement_name(nested))
    return names


def test_the_minari_extra_reaches_every_package_the_minari_path_imports() -> None:
    pytest.importorskip("minari", reason="the extra is resolved against minari's own metadata")
    reachable = _closure(_extras()["minari"])
    missing = sorted(name for name in MINARI_RUNTIME_IMPORTS if name not in reachable)
    assert not missing, (
        f"pip install sharpearena[minari] would not install {missing}; "
        "the Minari tests import them, so the documented install is broken"
    )


def test_ci_pins_every_package_the_minari_extra_promises() -> None:
    """``ci-requirements.txt`` is the environment the suite is actually green against."""
    pinned = {
        _requirement_name(line)
        for line in CI_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert set(MINARI_RUNTIME_IMPORTS) <= pinned


def test_the_installed_environment_can_import_them() -> None:
    """A skip here is the silent-skip hole itself, so it fails once minari is present."""
    pytest.importorskip("minari", reason="the Minari path is not installed at all")
    for module in sorted(MINARI_RUNTIME_IMPORTS.values()):
        __import__(module)
