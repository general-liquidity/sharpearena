#!/usr/bin/env python3
"""Create and verify a SharpeArena release without a stale provenance tag.

The release tag is deliberately one commit after the version bump.  Its commit may
change only ``paper/evidence/provenance.json`` and that manifest must name the tag's
parent as a clean generation.  This avoids the impossible self-reference of asking a
manifest to contain the hash of the commit that contains the manifest while still
making the tag's complete in-scope tree verifiable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "src"))

from provenance_common import (
    ARTIFACT_SCOPE,
    EXCLUDED_DIR_NAMES,
    MODEL_ARTIFACT_SCOPE,
    MODEL_IDENTITY_FIELDS,
    SOURCE_SCOPE,
    digest_bytes,
    identity_summaries,
    manifest_rule_problems,
    snapshot_digest,
)

MANIFEST_PATH = "paper/evidence/provenance.json"
TAG_RE = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")
VERSION_RE = re.compile(r'(?m)^version\s*=\s*"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"')


class ReleaseError(RuntimeError):
    """A release invariant failed."""


def run(
    root: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stdout + completed.stderr
        raise ReleaseError(f"command failed ({' '.join(args)}):\n{detail.rstrip()}")
    return completed


def git(root: Path, *args: str) -> str:
    return run(root, "git", *args).stdout.strip()


def git_bytes(root: Path, object_name: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", object_name],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseError(
            f"cannot read {object_name}: {completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def git_blobs(root: Path, commit: str, paths: list[str]) -> dict[str, bytes | None]:
    """Read many paths from one commit without spawning one git process per file."""
    if not paths:
        return {}
    request = "".join(f"{commit}:{path}\n" for path in paths).encode()
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=request,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseError(
            "git cat-file --batch failed: "
            + completed.stderr.decode(errors="replace").strip()
        )
    result: dict[str, bytes | None] = {}
    output = completed.stdout
    offset = 0
    for path in paths:
        newline = output.find(b"\n", offset)
        if newline < 0:
            raise ReleaseError("git cat-file returned a truncated header")
        header = output[offset:newline].split(b" ")
        if header[-1] in (b"missing", b"ambiguous"):
            result[path] = None
            offset = newline + 1
            continue
        if len(header) != 3:
            raise ReleaseError("git cat-file returned an invalid header")
        size = int(header[2])
        result[path] = output[newline + 1 : newline + 1 + size]
        offset = newline + 1 + size + 1
    return result


def tag_version(tag: str) -> str:
    matched = TAG_RE.fullmatch(tag)
    if matched is None:
        raise ReleaseError(f"release tag must be vMAJOR.MINOR.PATCH, got {tag!r}")
    return matched.group("version")


def parse_toml_version(data: bytes, path: str) -> str:
    matched = VERSION_RE.search(data.decode("utf-8"))
    if matched is None:
        raise ReleaseError(f"cannot find a literal version in {path}")
    return matched.group("version")


def version_problems(root: Path, commit: str, expected: str) -> list[str]:
    problems: list[str] = []

    def compare(path: str, actual: str) -> None:
        if actual != expected:
            problems.append(f"{path} reports {actual}, tag requires {expected}")

    cargo = git_bytes(root, f"{commit}:Cargo.toml")
    compare("Cargo.toml", parse_toml_version(cargo, "Cargo.toml"))

    for path in (
        "npm/sharpearena/package.json",
        "npm/sharpearena/pkg/package.json",
    ):
        payload = json.loads(git_bytes(root, f"{commit}:{path}").decode("utf-8"))
        compare(path, str(payload.get("version")))

    for path in (
        "crates/sharpearena-py/pyproject.toml",
        "crates/sharpearena-py/Cargo.toml",
    ):
        compare(path, parse_toml_version(git_bytes(root, f"{commit}:{path}"), path))

    init_path = "crates/sharpearena-py/python/sharpearena/__init__.py"
    init_text = git_bytes(root, f"{commit}:{init_path}").decode("utf-8")
    init_match = re.search(
        r'(?m)^__version__\s*=\s*"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"',
        init_text,
    )
    if init_match is None:
        problems.append(f"{init_path} has no literal __version__")
    else:
        compare(init_path, init_match.group("version"))

    dependency_paths = (
        "crates/sharpearena-wasm/Cargo.toml",
        "crates/sharpearena-py/Cargo.toml",
    )
    dependency_re = re.compile(
        r'sharpearena\s*=\s*\{[^\n}]*version\s*=\s*"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"'
    )
    for path in dependency_paths:
        text = git_bytes(root, f"{commit}:{path}").decode("utf-8")
        matched = dependency_re.search(text)
        if matched is None:
            problems.append(f"{path} has no literal sharpearena dependency version")
        else:
            compare(f"{path} sharpearena dependency", matched.group("version"))
    return problems


def _is_excluded(path: str, excludes: set[str]) -> bool:
    return any(part in excludes for part in PurePosixPath(path).parts)


def _matches(path: str, pattern: str) -> bool:
    """Match a repository-root glob with ``*`` kept inside one path component."""
    translated: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*" and index + 1 < len(pattern) and pattern[index + 1] == "*":
            index += 2
            if index < len(pattern) and pattern[index] == "/":
                translated.append("(?:.*/)?")
                index += 1
            else:
                translated.append(".*")
            continue
        if char == "*":
            translated.append("[^/]*")
        elif char == "?":
            translated.append("[^/]")
        else:
            translated.append(re.escape(char))
        index += 1
    translated.append("$")
    return re.fullmatch("".join(translated), path) is not None


def expanded_scope(
    paths: list[str], patterns: list[str], excludes: set[str]
) -> list[str]:
    return sorted(
        path
        for path in paths
        if not _is_excluded(path, excludes)
        and any(_matches(path, pattern) for pattern in patterns)
    )


def manifest_problems(root: Path, commit: str, manifest: dict) -> list[str]:
    problems = manifest_rule_problems(manifest)
    if problems:
        return problems
    groups = (
        ("source", manifest.get("source_files", [])),
        ("artifact", manifest.get("artifacts", [])),
        ("model artifact", manifest.get("model_artifacts", [])),
    )
    paths_to_read = [
        record.get("path")
        for _, records in groups
        for record in records
        if isinstance(record.get("path"), str)
    ]
    blobs = git_blobs(root, commit, paths_to_read)
    for name, records in groups:
        for record in records:
            path = record.get("path")
            recorded = record.get("sha256")
            if not isinstance(path, str) or not isinstance(recorded, str):
                problems.append(f"{name} has an invalid path/digest record")
                continue
            blob = blobs.get(path)
            if blob is None:
                problems.append(f"{name}: MISSING {path} from the tagged tree")
                continue
            actual = digest_bytes(blob)
            if actual != recorded:
                problems.append(
                    f"{name}: DIGEST {path} differs in the tagged tree\n"
                    f"    recorded {recorded}\n"
                    f"    tagged   {actual}"
                )

    seen_models: set[str] = set()
    for record in manifest.get("model_artifacts", []):
        path = record.get("path")
        if not isinstance(path, str):
            continue
        blob = blobs.get(path)
        if not isinstance(blob, bytes):
            continue
        try:
            actual_identities = identity_summaries(
                json.loads(blob.decode("utf-8")), MODEL_IDENTITY_FIELDS
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            problems.append(f"model artifact: INVALID {path}: {error}")
            continue
        if actual_identities != record.get("identities"):
            problems.append(f"model artifact: IDENTITY SUMMARY {path}")
        for identity in actual_identities:
            model = identity["model"]
            if model in seen_models:
                problems.append(f"model artifact: DUPLICATE MODEL {model}")
            seen_models.add(model)

    source_records = manifest.get("source_files", [])
    if snapshot_digest(source_records) != manifest.get("source_snapshot_sha256"):
        problems.append("source_snapshot_sha256 does not match the source records")

    paths = git(root, "ls-tree", "-r", "--name-only", commit).splitlines()
    excludes = set(EXCLUDED_DIR_NAMES)
    recorded_sources = [record["path"] for record in source_records]
    actual_sources = expanded_scope(paths, list(SOURCE_SCOPE), excludes)
    if recorded_sources != actual_sources:
        problems.append(
            "source scope does not exactly match the files in the tagged tree"
        )

    recorded_artifacts = [record["path"] for record in manifest.get("artifacts", [])]
    actual_artifacts = [
        path
        for path in expanded_scope(paths, list(ARTIFACT_SCOPE), excludes)
        if path != MANIFEST_PATH
    ]
    if recorded_artifacts != actual_artifacts:
        problems.append(
            "artifact scope does not exactly match the files in the tagged tree"
        )

    recorded_models = [record["path"] for record in manifest.get("model_artifacts", [])]
    actual_models = expanded_scope(paths, list(MODEL_ARTIFACT_SCOPE), excludes)
    if recorded_models != actual_models:
        problems.append("model-artifact scope does not exactly match the tagged tree")
    return problems


def verify_tag(
    root: Path, tag: str, *, _allow_prospective: bool = False
) -> tuple[list[str], str | None]:
    expected_version = tag_version(tag)
    problems: list[str] = []
    object_type = run(root, "git", "cat-file", "-t", tag, check=False)
    if object_type.returncode != 0:
        return [f"release tag {tag} does not exist"], None
    if object_type.stdout.strip() != "tag":
        problems.append(f"release tag {tag} must be annotated")

    commit = git(root, "rev-parse", f"{tag}^{{commit}}")
    origin_main = run(
        root,
        "git",
        "rev-parse",
        "--verify",
        "refs/remotes/origin/main^{commit}",
        check=False,
    )
    if origin_main.returncode != 0:
        problems.append(
            "refs/remotes/origin/main is absent; tag lineage is unverifiable"
        )
    else:
        main_commit = origin_main.stdout.strip()
        main_before_tag = (
            run(
                root,
                "git",
                "merge-base",
                "--is-ancestor",
                main_commit,
                commit,
                check=False,
            ).returncode
            == 0
        )
        tag_before_main = (
            run(
                root,
                "git",
                "merge-base",
                "--is-ancestor",
                commit,
                main_commit,
                check=False,
            ).returncode
            == 0
        )
        if not tag_before_main and not (_allow_prospective and main_before_tag):
            relation = (
                "is on a history diverged from origin/main"
                if not main_before_tag
                else "is not an ancestor of origin/main"
            )
            problems.append(f"release tag {tag} {relation}")
    parents = git(root, "rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) != 2:
        problems.append(f"release tag {tag} must point at a single-parent commit")
        return problems, commit
    parent = parents[1]
    changed = git(root, "diff", "--name-only", parent, commit).splitlines()
    extras = [path for path in changed if path != MANIFEST_PATH]
    if changed != [MANIFEST_PATH]:
        problems.append(
            f"release tag {tag} must point at a provenance-only rebind commit"
        )
        problems.extend(f"rebind commit also changes {path}" for path in extras)

    try:
        manifest = json.loads(
            git_bytes(root, f"{commit}:{MANIFEST_PATH}").decode("utf-8")
        )
    except (ReleaseError, UnicodeDecodeError, json.JSONDecodeError) as error:
        problems.append(f"cannot read the tagged provenance manifest: {error}")
        return problems, commit

    if manifest.get("generated_at_head_dirty") is not False:
        problems.append("tagged provenance manifest records a dirty generation")
    if manifest.get("generated_at_head") != parent:
        problems.append(
            "tagged provenance manifest must name the tag commit's parent "
            f"({parent}), got {manifest.get('generated_at_head')}"
        )

    problems += manifest_problems(root, commit, manifest)
    problems += version_problems(root, commit, expected_version)
    return problems, commit


def require_clean(root: Path) -> None:
    status = git(root, "status", "--porcelain")
    if status:
        raise ReleaseError(f"working tree is not clean:\n{status}")


def workspace_version(root: Path) -> str:
    return parse_toml_version((root / "Cargo.toml").read_bytes(), "Cargo.toml")


def next_version(current: str, bump: str) -> str:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", bump):
        return bump
    major, minor, patch = (int(part) for part in current.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError("bump must be patch, minor, major, or MAJOR.MINOR.PATCH")


def prepare_changelog(root: Path, current: str, target: str) -> None:
    """Move the current Unreleased notes into a dated release section."""
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    if f"## [{target}]" in text:
        return
    heading = "## [Unreleased]\n"
    if text.count(heading) != 1:
        raise ReleaseError("CHANGELOG.md must contain exactly one Unreleased heading")
    compare = (
        f"[Unreleased]: https://github.com/general-liquidity/sharpearena/compare/"
        f"v{target}...HEAD"
    )
    text, links = re.subn(r"(?m)^\[Unreleased\]: .+$", compare, text, count=1)
    if links != 1:
        raise ReleaseError("CHANGELOG.md has no Unreleased comparison link")
    text = text.replace(
        heading,
        f"{heading}\n## [{target}] - {datetime.now(UTC).date().isoformat()}\n",
        1,
    )
    release_link = (
        f"[{target}]: https://github.com/general-liquidity/sharpearena/compare/"
        f"v{current}...v{target}\n"
    )
    marker = f"[{current}]:"
    index = text.find(marker)
    if index < 0:
        raise ReleaseError(f"CHANGELOG.md has no link definition for {current}")
    text = text[:index] + release_link + text[index:]
    path.write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "--", "CHANGELOG.md")
    git(
        root,
        "commit",
        "-m",
        f"docs(changelog): prepare v{target}",
        "--",
        "CHANGELOG.md",
    )
    require_clean(root)


BASE_REF = "refs/remotes/origin/main"
WORKTREE_PREFIX = "sharpearena-release-"


def resolve_base(root: Path, base_ref: str) -> str:
    resolved = run(
        root, "git", "rev-parse", "--verify", f"{base_ref}^{{commit}}", check=False
    )
    if resolved.returncode != 0:
        raise ReleaseError(f"release base {base_ref} does not resolve to a commit")
    return resolved.stdout.strip()


def remove_release_worktree(root: Path, parent: Path, tree: Path) -> None:
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    if tree.exists():
        return
    run(root, "git", "worktree", "prune", check=False)
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def execute_release(
    root: Path,
    bump: str,
    *,
    push: bool,
    base_ref: str = BASE_REF,
    fetch: bool = True,
) -> str:
    """Cut the release in a throwaway worktree checked out from ``base_ref``.

    The working checkout is never the release tree. A stale local ``main`` and a dirty
    or partially staged working tree therefore cannot reach the release at all, rather
    than being caught by an assertion that has to be remembered. The tag and the pushed
    commits come from the freshly fetched base, and the caller's checkout is left
    exactly as it was found, local ``main`` included.
    """

    if fetch:
        run(root, "git", "fetch", "--quiet", "origin", "main", "--tags")
    base = resolve_base(root, base_ref)
    current = parse_toml_version(git_bytes(root, f"{base}:Cargo.toml"), "Cargo.toml")
    target = next_version(current, bump)
    target_tag = f"v{target}"
    if (
        run(root, "git", "rev-parse", "--verify", target_tag, check=False).returncode
        == 0
    ):
        raise ReleaseError(f"tag {target_tag} already exists")
    release_branch = f"release-{target_tag}"
    if (
        run(
            root,
            "git",
            "rev-parse",
            "--verify",
            f"refs/heads/{release_branch}",
            check=False,
        ).returncode
        == 0
    ):
        raise ReleaseError(f"release branch {release_branch} already exists")
    parent = Path(tempfile.mkdtemp(prefix=WORKTREE_PREFIX))
    tree = parent / target_tag
    run(
        root,
        "git",
        "worktree",
        "add",
        "--quiet",
        "-b",
        release_branch,
        str(tree),
        base,
    )
    try:
        return cut_release(tree, bump, current, target, release_branch, push=push)
    finally:
        remove_release_worktree(root, parent, tree)
        run(root, "git", "branch", "-D", release_branch, check=False)


def cut_release(
    root: Path,
    bump: str,
    current: str,
    target: str,
    release_branch: str,
    *,
    push: bool,
) -> str:
    require_clean(root)
    branch = git(root, "branch", "--show-current")
    if branch != release_branch:
        raise ReleaseError(
            f"release worktree must be on {release_branch}, found {branch!r}"
        )
    run(root, "cargo", "release", "--version")
    run(root, "cargo", "release", bump, "--allow-branch", release_branch)
    prepare_changelog(root, current, target)
    run(
        root,
        "cargo",
        "release",
        bump,
        "--execute",
        "--no-confirm",
        "--no-push",
        "--allow-branch",
        release_branch,
    )
    require_clean(root)
    release_commit = git(root, "rev-parse", "HEAD")
    version = workspace_version(root)
    tag = f"v{version}"
    if version != target:
        raise ReleaseError(f"cargo-release produced {version}, expected {target}")

    run(root, sys.executable, "paper/src/make-provenance.py")
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("generated_at_head") != release_commit:
        raise ReleaseError("provenance writer did not bind the release commit")
    if manifest.get("generated_at_head_dirty") is not False:
        raise ReleaseError("provenance writer did not record a clean release tree")
    changed = git(root, "status", "--porcelain").splitlines()
    if len(changed) != 1 or not changed[0].endswith(MANIFEST_PATH):
        raise ReleaseError(
            "provenance rebind changed something other than the manifest:\n"
            + "\n".join(changed)
        )
    git(root, "add", "--", MANIFEST_PATH)
    git(
        root,
        "commit",
        "-m",
        f"chore(provenance): rebind on the {tag} release tree",
        "--",
        MANIFEST_PATH,
    )
    require_clean(root)
    git(root, "tag", "-a", tag, "-m", f"SharpeArena {tag}")

    problems, tag_commit = verify_tag(root, tag, _allow_prospective=True)
    if problems:
        raise ReleaseError("prospective release tag failed:\n" + "\n".join(problems))
    assert tag_commit is not None
    run(root, sys.executable, "paper/src/check-provenance.py")

    if push:
        # Main and tag become visible together. A tag without its checked rebind commit,
        # or a branch update without the tag, cannot escape this process.
        run(root, "git", "push", "--atomic", "origin", "HEAD:main", f"refs/tags/{tag}")
        git(root, "update-ref", "refs/remotes/origin/main", tag_commit)
        published_problems, _ = verify_tag(root, tag)
        if published_problems:
            raise ReleaseError(
                "published release tag failed post-push verification:\n"
                + "\n".join(published_problems)
            )
    return tag


def clone_rehearsal_tree(root: Path, clone: Path) -> None:
    """Clone the exact reviewed HEAD while retaining origin/main for lineage checks.

    GitHub checks pull requests out detached. A plain local clone follows the source
    repository's ``main`` ref and can therefore rehearse the base branch instead of
    the commit under review. Make the reviewed object the clone's local ``main``
    explicitly, while copying the source's remote-tracking main as the trust anchor.
    """
    reviewed_commit = git(root, "rev-parse", "HEAD^{commit}")
    origin_main = git(root, "rev-parse", "refs/remotes/origin/main^{commit}")
    run(
        root,
        "git",
        "clone",
        "--quiet",
        "--local",
        "--no-hardlinks",
        "--no-checkout",
        str(root),
        str(clone),
    )
    git(clone, "update-ref", "refs/remotes/origin/main", origin_main)
    git(clone, "checkout", "--quiet", "--force", "-B", "main", reviewed_commit)
    require_clean(clone)


def rehearse(root: Path, bump: str) -> str:
    require_clean(root)
    with tempfile.TemporaryDirectory(prefix="sharpearena-release-rehearsal-") as temp:
        clone = Path(temp) / "repo"
        clone_rehearsal_tree(root, clone)
        identity = dict(os.environ)
        identity.setdefault("GIT_AUTHOR_NAME", "SharpeArena release rehearsal")
        identity.setdefault("GIT_AUTHOR_EMAIL", "release-rehearsal@example.invalid")
        identity.setdefault("GIT_COMMITTER_NAME", identity["GIT_AUTHOR_NAME"])
        identity.setdefault("GIT_COMMITTER_EMAIL", identity["GIT_AUTHOR_EMAIL"])
        completed = run(
            clone,
            sys.executable,
            "scripts/release.py",
            "execute",
            bump,
            "--no-push",
            "--no-fetch",
            "--base-ref",
            "refs/heads/main",
            "--repo",
            str(clone),
            env=identity,
        )
        created = re.search(r"\bcreated (v[0-9]+\.[0-9]+\.[0-9]+)\b", completed.stdout)
        if created is None:
            raise ReleaseError(
                "isolated release completed without reporting the prospective tag"
            )
        return created.group(1)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-tag", help="verify an existing release tag")
    verify.add_argument("tag")
    verify.add_argument("--repo", type=Path, default=Path.cwd())
    verify.add_argument("--github-output", type=Path)

    rehearsal = commands.add_parser(
        "rehearse", help="exercise the full release in an isolated local clone"
    )
    rehearsal.add_argument("bump", help="patch, minor, major, or an exact version")
    rehearsal.add_argument("--repo", type=Path, default=Path.cwd())

    execute = commands.add_parser(
        "execute", help="cut a checked release and atomically push main plus tag"
    )
    execute.add_argument("bump", help="patch, minor, major, or an exact version")
    execute.add_argument("--no-push", action="store_true", help=argparse.SUPPRESS)
    execute.add_argument("--no-fetch", action="store_true", help=argparse.SUPPRESS)
    execute.add_argument("--base-ref", default=BASE_REF, help=argparse.SUPPRESS)
    execute.add_argument("--repo", type=Path, default=Path.cwd())
    return cli


def main() -> int:
    args = parser().parse_args()
    root = args.repo.resolve()
    try:
        if args.command == "verify-tag":
            problems, commit = verify_tag(root, args.tag)
            if problems:
                for problem in problems:
                    print(f"release: {problem}")
                print(f"FAIL: release tag {args.tag} is not provenance-safe")
                return 1
            assert commit is not None
            if args.github_output is not None:
                with args.github_output.open(
                    "a", encoding="utf-8", newline="\n"
                ) as out:
                    out.write(f"validated_commit={commit}\n")
            print(f"OK: release tag {args.tag} is bound to its exact in-scope tree")
            return 0
        if args.command == "rehearse":
            tag = rehearse(root, args.bump)
            print(f"OK: rehearsed {tag} without pushing")
            return 0
        tag = execute_release(
            root,
            args.bump,
            push=not args.no_push,
            base_ref=args.base_ref,
            fetch=not args.no_fetch,
        )
        suffix = " without pushing" if args.no_push else " and pushed atomically"
        print(f"OK: created {tag}{suffix}")
        return 0
    except (OSError, ReleaseError, json.JSONDecodeError) as error:
        print(f"release: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
