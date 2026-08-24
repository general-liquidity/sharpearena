"""Write a deterministic content manifest for the paper's evidence snapshot.

The manifest identifies a dirty-but-reviewable candidate by hashes of every
producer/input source and every evidence artifact; it does not guess historical
package versions from file dates.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "evidence" / "provenance.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(patterns: tuple[str, ...]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(found, key=lambda path: path.as_posix())


source_paths = files(
    (
        "Cargo.toml",
        "Cargo.lock",
        "crates/**/*.toml",
        "crates/**/*.rs",
        "crates/sharpearena-py/python/**/*.py",
        "paper/src/*.py",
    )
)
source_records = [
    {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
    for path in source_paths
]
snapshot = hashlib.sha256(
    "".join(f"{item['sha256']}  {item['path']}\n" for item in source_records).encode()
).hexdigest()

artifact_paths = files(("paper/evidence/*.json", "paper/figures/*.pdf"))
artifact_paths = [path for path in artifact_paths if path != OUT]
artifacts = [
    {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
    for path in artifact_paths
]

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()
manifest = {
    "schema_version": 1,
    "repository_head": head,
    "source_snapshot_sha256": snapshot,
    "source_snapshot_scope": [
        "Cargo.toml",
        "Cargo.lock",
        "crates/**/*.toml",
        "crates/**/*.rs",
        "crates/sharpearena-py/python/**/*.py",
        "paper/src/*.py",
    ],
    "reproduction_entrypoint": "commands in paper/sections/A-commands.tex",
    "source_files": source_records,
    "artifacts": artifacts,
}
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}: {len(source_records)} sources, {len(artifacts)} artifacts")
