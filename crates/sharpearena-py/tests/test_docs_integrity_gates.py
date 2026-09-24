"""Tests for the two documentation-integrity gates in `scripts/`.

Both gates are checks on the repository's own prose, so both can pass vacuously: a
check that never fires is indistinguishable from a check that cannot fire. Every test
here that matters plants a defect in a throwaway git checkout and asserts the gate
catches it, then plants the closest TRUE statement it can and asserts the gate stays
quiet. A gate with no failing case is not evidence.

`check-absence-claims.py` exists because this repository shipped three false absence
claims in one day: an inventory row reading "Nothing in the tree" for HUD and Harbor
after both had merged, and a logo record saying Harbor published no logo and that MCP
published no logo asset, when both publish one. The record already told everyone to grep
before claiming a gap. Per the poka-yoke rule, a document saying "do not do X" is
training, not a device, and training degrades; these are the devices.

`check-logo-assets.py` exists because `docs/assets/logos/README.md` records a SHA-256 for
each of the twenty-one vendored assets and asserts every file is byte-identical to what
its project publishes, and until now nothing read those hashes back.

On the exemption mechanism, which `test_an_exemption_*` covers: an exemption names one
file, quotes the excused claim verbatim and gives a written reason. That shape was
chosen over an inline marker or a per-file opt-out because it cannot be applied
sideways or in bulk, because editing the prose retires it automatically, and because an
exemption that matches nothing is itself a failure, so dead ones cannot accumulate. It
cannot stop a human determined to silence a true finding; nothing short of review can,
and a check claiming otherwise would be making the same unverified absence claim the
gate exists to catch. What it guarantees is that silencing costs one legible line in one
small central file, naming the exact sentence and the reason, in a diff a reviewer sees.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
ABSENCE = SCRIPTS / "check-absence-claims.py"
LOGOS = SCRIPTS / "check-logo-assets.py"
LOGO_DIR = REPO / "docs" / "assets" / "logos"


def run(script: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A throwaway git checkout. Both gates read the tree through `git ls-files`, so a
    fixture has to be a real repository with real commits, not a loose directory."""
    root = tmp_path / "tree"
    root.mkdir()
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True, env=env)
    (root / "examples" / "hud").mkdir(parents=True)
    (root / "examples" / "hud" / "episode.py").write_text("# fixture\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("# notes\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "seed"], check=True, env=env)
    return root


def commit(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    subprocess.run(["git", "-C", str(root), "add", relative], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", relative], check=True, env=env)


# --- the repository as it stands -------------------------------------------------


def test_the_tree_satisfies_the_absence_gate() -> None:
    result = run(ABSENCE, REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_tree_satisfies_the_logo_gate() -> None:
    result = run(LOGOS, REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_correction_section_does_not_trip_the_gate() -> None:
    """`docs/assets/logos/README.md` deliberately restates two claims now known false,
    under "A correction worth keeping", so the record of the error survives. Both are
    excused by exact quote in the exemption file; this asserts the section is quiet and
    that it is quiet BECAUSE of those entries rather than by luck."""
    text = (REPO / "docs" / "assets" / "logos" / "README.md").read_text(encoding="utf-8")
    assert "## A correction worth keeping" in text

    entries = json.loads(
        (SCRIPTS / "absence-claim-exemptions.json").read_text(encoding="utf-8")
    )["exemptions"]
    # The gate matches a claim against a unit of prose, which is a paragraph's lines
    # joined by single spaces, so presence in the file is checked the same way rather
    # than against the hard-wrapped source.
    def flat(source: str) -> str:
        return " ".join(source.split())

    for entry in entries:
        assert entry["claim"] in flat((REPO / entry["file"]).read_text(encoding="utf-8")), entry
    excused = [e for e in entries if e["file"] == "docs/assets/logos/README.md"]
    assert len(excused) == 2
    for entry in excused:
        assert entry["claim"] in flat(text)


# --- rule A: the path named as the object of an absence claim ----------------------


def test_rule_a_catches_a_planted_false_absence_claim(sandbox: Path) -> None:
    commit(sandbox, "docs/notes.md", "There is no `examples/hud/` directory in this tree.\n")
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "examples/hud" in result.stderr


def test_rule_a_stays_quiet_on_a_true_absence_claim(sandbox: Path) -> None:
    commit(sandbox, "docs/notes.md", "There is no `examples/harbor/` directory in this tree.\n")
    assert run(ABSENCE, sandbox).returncode == 0


def test_rule_a_catches_the_path_first_wording(sandbox: Path) -> None:
    commit(sandbox, "docs/notes.md", "The fixture `examples/hud/episode.py` does not exist.\n")
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "examples/hud/episode.py" in result.stderr


def test_rule_a_ignores_absence_of_a_property_rather_than_a_path(sandbox: Path) -> None:
    """The distinction rule A turns on. An audit trail is full of true sentences saying
    a file that exists has no particular field in it, and an earlier draft that fired on
    any absence word sharing a sentence with any existing path reported forty of those
    before it reported anything real."""
    commit(
        sandbox,
        "docs/notes.md",
        "`examples/hud/episode.py` has no `cost` field, and there is no `build.rs` for it.\n",
    )
    assert run(ABSENCE, sandbox).returncode == 0


def test_rule_a_ignores_a_path_that_only_draws_the_row(sandbox: Path) -> None:
    """A row illustrating itself with an image is not asserting anything about the tree,
    and counting markup attributes would fire on every true "Nothing in the tree" row
    that carries a mark."""
    commit(
        sandbox,
        "docs/notes.md",
        "| Thing | Present |\n|---|---|\n"
        '| <img src="../examples/hud/episode.py" alt=""> Something | '
        "There is no such file | \n",
    )
    assert run(ABSENCE, sandbox).returncode == 0


# --- rule B: the row subject a tree path is named after ---------------------------


def test_rule_b_catches_the_row_that_shipped(sandbox: Path) -> None:
    """The defect verbatim: a status cell saying the subject is absent, beside a name
    the tree answers with a directory. Rule A cannot reach this one, because the row
    names no path at all; the missing path is the defect."""
    commit(
        sandbox,
        "docs/inventory.md",
        "| Ecosystem | Present | Where |\n|---|---|---|\n"
        "| HUD, Harbor | No | Nothing in the tree |\n",
    )
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "HUD" in result.stderr


def test_rule_b_stays_quiet_when_the_row_is_true(sandbox: Path) -> None:
    commit(
        sandbox,
        "docs/inventory.md",
        "| Ecosystem | Present | Where |\n|---|---|---|\n"
        "| CleanRL, PufferLib | No | Nothing in the tree |\n",
    )
    assert run(ABSENCE, sandbox).returncode == 0


def test_rule_b_does_not_read_a_vendored_logo_as_an_integration(sandbox: Path) -> None:
    """`docs/assets/logos/` is named after third parties by construction. Reading
    `torchrl.png` as evidence of a TorchRL integration would fire on a row that is true,
    which is how a gate gets switched off."""
    commit(sandbox, "docs/assets/logos/torchrl.png", "not really a png\n")
    commit(
        sandbox,
        "docs/inventory.md",
        "| Ecosystem | Present | Where |\n|---|---|---|\n"
        "| TorchRL | No | Nothing in the tree |\n",
    )
    assert run(ABSENCE, sandbox).returncode == 0


# --- rule C: a project said to publish nothing, whose asset is vendored here -------


def test_rule_c_catches_the_logo_claim_that_shipped(sandbox: Path) -> None:
    commit(sandbox, "docs/assets/logos/harbor.png", "vendored bytes\n")
    commit(sandbox, "docs/notes.md", "Harbor publishes no logo in its repository or on its site.\n")
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "harbor.png" in result.stderr


def test_rule_c_stays_quiet_for_a_project_with_nothing_vendored(sandbox: Path) -> None:
    commit(sandbox, "docs/assets/logos/harbor.png", "vendored bytes\n")
    commit(sandbox, "docs/notes.md", "CleanRL publishes no logo in its repository or on its site.\n")
    assert run(ABSENCE, sandbox).returncode == 0


# --- the exemption mechanism ------------------------------------------------------


def test_an_exemption_excuses_exactly_the_claim_it_quotes(sandbox: Path) -> None:
    commit(sandbox, "docs/notes.md", "There is no `examples/hud/` directory in this tree.\n")
    assert run(ABSENCE, sandbox).returncode == 1

    commit(
        sandbox,
        "scripts/absence-claim-exemptions.json",
        json.dumps(
            {
                "exemptions": [
                    {
                        "file": "docs/notes.md",
                        "claim": "There is no `examples/hud/` directory",
                        "reason": "kept as a record of a claim now known to be false",
                    }
                ]
            }
        ),
    )
    assert run(ABSENCE, sandbox).returncode == 0


def test_editing_the_prose_retires_its_exemption(sandbox: Path) -> None:
    """The property that makes the mechanism safe to have: an exemption is bound to
    exact wording, so it cannot drift onto a claim nobody reviewed."""
    commit(
        sandbox,
        "scripts/absence-claim-exemptions.json",
        json.dumps(
            {
                "exemptions": [
                    {
                        "file": "docs/notes.md",
                        "claim": "There is no `examples/hud/` directory",
                        "reason": "kept as a record of a claim now known to be false",
                    }
                ]
            }
        ),
    )
    commit(sandbox, "docs/notes.md", "There is no `examples/hud/` folder in this tree.\n")
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "stale" in result.stderr


def test_an_exemption_that_matches_nothing_is_itself_a_failure(sandbox: Path) -> None:
    commit(
        sandbox,
        "scripts/absence-claim-exemptions.json",
        json.dumps(
            {
                "exemptions": [
                    {
                        "file": "docs/notes.md",
                        "claim": "a sentence nobody ever wrote",
                        "reason": "left behind after the prose it excused was deleted",
                    }
                ]
            }
        ),
    )
    result = run(ABSENCE, sandbox)
    assert result.returncode == 1
    assert "stale" in result.stderr


def test_an_exemption_without_a_reason_is_refused(sandbox: Path) -> None:
    commit(sandbox, "docs/notes.md", "There is no `examples/hud/` directory in this tree.\n")
    commit(
        sandbox,
        "scripts/absence-claim-exemptions.json",
        json.dumps(
            {
                "exemptions": [
                    {
                        "file": "docs/notes.md",
                        "claim": "There is no `examples/hud/` directory",
                        "reason": "because",
                    }
                ]
            }
        ),
    )
    result = run(ABSENCE, sandbox)
    assert result.returncode != 0
    assert "too thin to review" in result.stderr


# --- guard 2: vendored logo integrity ---------------------------------------------


def _logo_sandbox(root: Path, *, record_extra: str = "", drop: str = "") -> None:
    commit(root, "docs/assets/logos/alpha.svg", "<svg>alpha</svg>\n")
    commit(root, "docs/assets/logos/beta.png", "beta bytes\n")
    digests = {
        name: hashlib.sha256((root / "docs/assets/logos" / name).read_bytes()).hexdigest()
        for name in ("alpha.svg", "beta.png")
        if name != drop
    }
    body = "\n".join(f"{digest}  {name}" for name, digest in digests.items())
    commit(
        root,
        "docs/assets/logos/README.md",
        "# logos\n\n```\n" + body + ("\n" + record_extra if record_extra else "") + "\n```\n",
    )


def test_logo_gate_passes_when_every_hash_matches(sandbox: Path) -> None:
    _logo_sandbox(sandbox)
    result = run(LOGOS, sandbox)
    assert result.returncode == 0, result.stdout + result.stderr


def test_logo_gate_catches_a_changed_file(sandbox: Path) -> None:
    """The invariant the README asserts and nothing checked: recolouring, cropping or
    rescaling a vendored mark leaves the page still claiming it is byte-identical to
    what upstream publishes."""
    _logo_sandbox(sandbox)
    commit(sandbox, "docs/assets/logos/beta.png", "beta bytes, quietly recoloured\n")
    result = run(LOGOS, sandbox)
    assert result.returncode == 1
    assert "beta.png" in result.stderr


def test_logo_gate_catches_an_asset_with_no_recorded_hash(sandbox: Path) -> None:
    _logo_sandbox(sandbox)
    commit(sandbox, "docs/assets/logos/gamma.svg", "<svg>gamma</svg>\n")
    result = run(LOGOS, sandbox)
    assert result.returncode == 1
    assert "no recorded SHA-256" in result.stderr


def test_logo_gate_catches_a_recorded_hash_with_no_asset(sandbox: Path) -> None:
    _logo_sandbox(
        sandbox,
        record_extra="0" * 64 + "  delta.svg",
    )
    result = run(LOGOS, sandbox)
    assert result.returncode == 1
    assert "delta.svg" in result.stderr


def test_every_vendored_asset_in_this_repository_is_covered(sandbox: Path) -> None:
    """Belt and braces on the real tree: the gate's own pass could in principle come
    from finding nothing to check."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "docs/assets/logos"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assets = [path for path in tracked if not path.endswith(".md")]
    assert len(assets) == 21, assets
    record = (LOGO_DIR / "README.md").read_text(encoding="utf-8")
    for path in assets:
        name = path.rsplit("/", 1)[-1]
        digest = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in record, name
