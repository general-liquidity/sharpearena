#!/usr/bin/env python3
"""Move SharpeArena's exact SharpeBench pin to a published release, and rebind SPEC_HASH.

Arena pins every `sharpebench-*` crate at one exact published version. Moving it is
one change set, and this script makes all of it:

1. rewrites `=OLD` to `=NEW` on every SharpeBench dependency in
   `crates/sharpearena/Cargo.toml` and `crates/sharpearena-py/Cargo.toml`;
2. re-resolves both lockfiles (`Cargo.lock`, `crates/sharpearena-py/Cargo.lock`);
3. checks that every top-level `CompositeScore` key of the new `sharpebench-core`
   that ends in `_error` or `_refusal` is classified by `sharpearena.kernel_score`;
4. builds the engine and reads the new `SHARPEARENA_SPEC_HASH` from its build script
   (the exact pins are an input of the hash, so it moves with every bump);
5. rebinds the three committed copies of the hash:
   `crates/sharpearena/contract/attestation/spec-hash.json`,
   `npm/sharpearena/src/specHash.ts` and
   `crates/sharpearena-py/python/sharpearena/_spec_hash.py`;
6. runs the spec-hash tests and checks the pyo3 crate against the new pin;
7. rebuilds the committed wasm bundle with `scripts/build-sharpearena-wasm.sh` and
   holds it to `scripts/check-wasm-bundle.mjs`.

Steps 1 to 6 run in a staged copy of the tree, so a failure leaves the checkout
untouched. Only when all pass are the changed files copied back, and only then is
the wasm bundle rebuilt in place.

    # Preview against a local SharpeBench checkout whose version is NEW. Nothing in
    # the checkout changes; the override never reaches a tracked file.
    python scripts/bump-bench-pin.py --to 0.29.0 --dry-run --bench-source ../sharpebench

    # The release round, once sharpebench 0.29.0 is on crates.io.
    python scripts/bump-bench-pin.py --to 0.29.0

`--to` equal to the current pin is a rebind: it recomputes SPEC_HASH from the tree
and rebinds the copies, which is the step a spec-source edit needs. A real run
refuses `--bench-source` and refuses a version crates.io does not serve unyanked,
so the script cannot leave a path or unpublished dependency behind. It does not
commit, edit CHANGELOG.md or prose, or bump Arena's own version; it lists the
remaining mentions of the old version for review.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_compat import (  # noqa: E402
    LOCKFILES,
    BenchCompatError,
    kernel_key_problems,
    patch_flags,
    published_crates,
    workspace_version,
)

ROOT = Path(__file__).resolve().parent.parent
PIN_MANIFESTS = ("crates/sharpearena/Cargo.toml", "crates/sharpearena-py/Cargo.toml")
PYTHON_CRATE = "crates/sharpearena-py"
SPEC_RECORD = "crates/sharpearena/contract/attestation/spec-hash.json"
#: Every committed copy of the spec hash, each holding it exactly once in quotes.
SPEC_HASH_COPIES = (
    SPEC_RECORD,
    "npm/sharpearena/src/specHash.ts",
    "crates/sharpearena-py/python/sharpearena/_spec_hash.py",
)
#: What the staged copy needs to resolve, build and test the pin. Everything else in
#: the repository is irrelevant to the bump and stays out of the copy.
STAGED = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "crates", "npm/sharpearena/src")
STAGE_IGNORE = shutil.ignore_patterns(
    "target", ".venv", "__pycache__", ".pytest_cache", "node_modules", "*.pyd", "*.so"
)
VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
SPEC_HASH_ENV = "SHARPEARENA_SPEC_HASH"
HASH = re.compile(r"^[0-9a-f]{16}$")
#: Where an old version may still be named on purpose (history, frozen records and
#: their register), or is rewritten by this script already.
MENTION_EXCLUDES = (
    ":!CHANGELOG.md",
    ":!docs/audits",
    ":!docs/evidence-register.*",
    ":!*.lock",
    ":!paper/evidence",
    *(f":!{path}" for path in PIN_MANIFESTS),
)


class BumpError(RuntimeError):
    """The bump cannot be made as asked; nothing in the checkout has changed."""


def version_key(version: str) -> tuple[int, int, int]:
    matched = VERSION.fullmatch(version)
    if matched is None:
        raise BumpError(f"{version!r} is not a MAJOR.MINOR.PATCH version")
    return tuple(int(part) for part in matched.groups())  # type: ignore[return-value]


def bench_pins(manifest_text: str) -> dict[str, str]:
    """Each SharpeBench dependency of a manifest and its exact version, in any table."""
    manifest = tomllib.loads(manifest_text)
    tables = [manifest, *manifest.get("target", {}).values()]
    pins: dict[str, str] = {}
    for table in tables:
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, spec in table.get(section, {}).items():
                if not name.startswith("sharpebench"):
                    continue
                if isinstance(spec, dict) and {"path", "git"} & spec.keys():
                    raise BumpError(f"{name} is taken from outside crates.io: {spec!r}")
                text = spec if isinstance(spec, str) else spec.get("version")
                if not isinstance(text, str) or not text.startswith("="):
                    raise BumpError(f"{name} must be pinned as =X.Y.Z, found {spec!r}")
                version_key(text[1:])
                pins[f"{section}.{name}"] = text[1:]
    return pins


def current_pin(root: Path) -> str:
    """The one SharpeBench version every pin manifest names."""
    versions: set[str] = set()
    for path in PIN_MANIFESTS:
        pins = bench_pins((root / path).read_text(encoding="utf-8"))
        if not pins:
            raise BumpError(f"{path} has no SharpeBench dependency")
        versions |= set(pins.values())
    if len(versions) != 1:
        raise BumpError(f"the SharpeBench pins name more than one version: {sorted(versions)}")
    return versions.pop()


def rewrite_pins(manifest_text: str, old: str, new: str) -> str:
    """Rewrite `=old` to `=new` on every SharpeBench dependency line, and only there."""
    expected = len(bench_pins(manifest_text))
    pattern = re.compile(
        r'(?m)^(?P<head>sharpebench[A-Za-z0-9_-]*\s*=\s*(?:\{[^}\n]*?\bversion\s*=\s*)?")='
        + re.escape(old)
        + r'(?P<tail>")'
    )
    rewritten, count = pattern.subn(lambda m: f"{m['head']}={new}{m['tail']}", manifest_text)
    if count != expected:
        raise BumpError(
            f"found {count} rewritable SharpeBench pins at ={old} but the manifest "
            f"declares {expected}; edit it by hand"
        )
    if old != new and bench_pins(rewritten) != {k: new for k in bench_pins(manifest_text)}:
        raise BumpError("the rewritten manifest does not pin every SharpeBench crate at the new version")
    return rewritten


def rebind_text(text: str, old_hash: str, new_hash: str, path: str) -> str:
    """Replace the one quoted copy of the old hash; refuse zero or several."""
    count = text.count(f'"{old_hash}"')
    if count != 1:
        raise BumpError(f"{path} holds the spec hash {old_hash} {count} times, expected once")
    return text.replace(f'"{old_hash}"', f'"{new_hash}"')


def spec_hash_from_cargo_messages(lines: list[str]) -> str:
    """The SPEC_HASH the `sharpearena` build script emitted, from `--message-format=json`."""
    found: set[str] = set()
    for line in lines:
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("reason") != "build-script-executed":
            continue
        for key, value in message.get("env", []):
            if key == SPEC_HASH_ENV:
                found.add(value)
    if len(found) != 1 or not HASH.fullmatch(next(iter(found))):
        raise BumpError(f"expected one 16-hex-digit {SPEC_HASH_ENV} from the build, got {sorted(found)}")
    return found.pop()


def locked_bench(lock_text: str) -> dict[str, tuple[str, str | None]]:
    """SharpeBench packages in a lockfile: name to (version, source)."""
    lock = tomllib.loads(lock_text)
    return {
        str(package["name"]): (str(package["version"]), package.get("source"))
        for package in lock.get("package", [])
        if str(package.get("name", "")).startswith("sharpebench")
    }


def lock_problems(lock_text: str, version: str, *, require_registry: bool) -> list[str]:
    locked = locked_bench(lock_text)
    if not locked:
        return ["locks no SharpeBench crate"]
    problems = []
    for name, (found, source) in sorted(locked.items()):
        if found != version:
            problems.append(f"locks {name} {found}, the pin requires {version}")
        elif require_registry and source != CRATES_IO:
            problems.append(f"does not resolve {name} from crates.io (source {source!r})")
    return problems


def run(command: list[str], cwd: Path, *, capture: bool = False) -> str:
    print("+", " ".join(command), f"  (in {cwd})", flush=True)
    result = subprocess.run(
        command, cwd=cwd, check=False, text=True, encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        raise BumpError(f"{command[0]} {command[1] if len(command) > 1 else ''} exited {result.returncode}")
    return result.stdout or ""


def stage(root: Path, dest: Path) -> None:
    for name in STAGED:
        source = root / name
        target = dest / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=STAGE_IGNORE)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def registry_problems(crates: list[str], version: str) -> list[str]:
    """Each pinned crate must be on crates.io at `version`, unyanked."""
    import release  # the release driver owns the index lookup and its offline fixture

    lookup = release.default_bench_index()
    problems = []
    for name in sorted(crates):
        document = lookup(name)
        record = None if document is None else release.published_bench_release(document, version)
        if record is None:
            problems.append(f"{name} {version} is not published on crates.io")
        elif record.get("yanked") is not False:
            problems.append(f"{name} {version} is yanked")
    return problems


def bump_in(tree: Path, old: str, new: str, flags: list[str], *, registry: bool) -> dict[str, str]:
    """Make the whole change set inside `tree`; return the old and new spec hash."""
    for path in PIN_MANIFESTS:
        manifest = tree / path
        manifest.write_text(rewrite_pins(manifest.read_text(encoding="utf-8"), old, new), encoding="utf-8", newline="")

    # `cargo metadata` re-resolves only the requirements that changed and writes the
    # lock; every other locked package keeps its version.
    metadata = json.loads(run(["cargo", "metadata", *flags, "--format-version", "1"], tree, capture=True))
    run(["cargo", "metadata", *flags, "--format-version", "1"], tree / PYTHON_CRATE, capture=True)
    problems = []
    for lock in LOCKFILES:
        problems += [f"{lock} {p}" for p in lock_problems((tree / lock).read_text(encoding="utf-8"), new, require_registry=registry)]
    core = [p for p in metadata["packages"] if p["name"] == "sharpebench-core"]
    if len(core) != 1:
        problems.append(f"expected one resolved sharpebench-core, found {len(core)}")
    else:
        composite = Path(core[0]["manifest_path"]).parent / "src" / "composite.rs"
        problems += kernel_key_problems(composite, tree / "crates/sharpearena-py/python/sharpearena/kernel_score.py")
    if problems:
        raise BumpError("; ".join(problems))

    record = json.loads((tree / SPEC_RECORD).read_text(encoding="utf-8"))
    old_hash = str(record["spec_hash"])
    lines = run(
        ["cargo", "build", *flags, "-p", "sharpearena", "--lib", "--message-format=json"], tree, capture=True
    ).splitlines()
    new_hash = spec_hash_from_cargo_messages(lines)
    if new_hash != old_hash:
        for path in SPEC_HASH_COPIES:
            file = tree / path
            file.write_text(rebind_text(file.read_text(encoding="utf-8"), old_hash, new_hash, path), encoding="utf-8", newline="")

    run(["cargo", "test", *flags, "-p", "sharpearena", "--lib", "spec_hash"], tree)
    run(["cargo", "test", *flags, "-p", "sharpearena-wasm", "--lib", "spec_hash"], tree)
    run(["cargo", "test", *flags, "-p", "sharpearena", "--test", "spec_dependency_contract"], tree)
    run(["cargo", "check", *flags, "--all-targets"], tree / PYTHON_CRATE)
    return {"old": old_hash, "new": new_hash}


def changed_files(root: Path, tree: Path) -> list[str]:
    candidates = [*PIN_MANIFESTS, *LOCKFILES, *SPEC_HASH_COPIES]
    return [p for p in candidates if (root / p).read_bytes() != (tree / p).read_bytes()]


def report(root: Path, tree: Path, changed: list[str], *, overridden: bool) -> None:
    for path in changed:
        if path in LOCKFILES:
            before = locked_bench((root / path).read_text(encoding="utf-8"))
            after = locked_bench((tree / path).read_text(encoding="utf-8"))
            print(f"--- {path} (SharpeBench packages{'; sources are the local override' if overridden else ''})")
            for name in sorted(set(before) | set(after)):
                print(f"    {name}: {before.get(name, ('absent', None))[0]} -> {after.get(name, ('absent', None))[0]}")
            continue
        diff = difflib.unified_diff(
            (root / path).read_text(encoding="utf-8").splitlines(keepends=True),
            (tree / path).read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=0,
        )
        sys.stdout.writelines(diff)


def old_mentions(root: Path, old: str) -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "-F", old, "--", ".", *MENTION_EXCLUDES],
        cwd=root, text=True, encoding="utf-8", stdout=subprocess.PIPE, check=False,
    )
    return [line if len(line) <= 160 else line[:157] + "..." for line in result.stdout.splitlines()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", required=True, help="the published SharpeBench version to pin")
    parser.add_argument("--dry-run", action="store_true", help="stage, verify and print the change set; change nothing")
    parser.add_argument("--bench-source", type=Path, help="dry run only: a SharpeBench checkout at --to to overlay")
    parser.add_argument("--skip-wasm", action="store_true", help="real run: leave the wasm rebuild to the caller")
    args = parser.parse_args(argv)

    try:
        version_key(args.to)
        old = current_pin(ROOT)
        if version_key(args.to) < version_key(old):
            raise BumpError(f"--to {args.to} is older than the current pin {old}")
        flags: list[str] = []
        if args.bench_source is not None:
            if not args.dry_run:
                raise BumpError("--bench-source is a dry-run preview; a real bump resolves from crates.io only")
            bench = args.bench_source.resolve()
            found = workspace_version(bench)
            if found != args.to:
                raise BumpError(f"{bench} is SharpeBench {found}, not {args.to}")
            flags = patch_flags(published_crates(bench))
        elif not args.dry_run:
            crates = {name.split(".", 1)[1] for path in PIN_MANIFESTS for name in bench_pins((ROOT / path).read_text(encoding="utf-8"))}
            problems = registry_problems(sorted(crates), args.to)
            if problems:
                raise BumpError("; ".join(problems))
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", *PIN_MANIFESTS, *LOCKFILES, *SPEC_HASH_COPIES, "npm/sharpearena/pkg"],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True,
            ).stdout.strip()
            if dirty:
                raise BumpError(f"files this bump writes have uncommitted changes:\n{dirty}")

        with tempfile.TemporaryDirectory(prefix="arena-pin-") as staging:
            tree = Path(staging)
            stage(ROOT, tree)
            hashes = bump_in(tree, old, args.to, flags, registry=not flags)
            changed = changed_files(ROOT, tree)
            print(f"\nSharpeBench pin ={old} -> ={args.to}; SPEC_HASH {hashes['old']} -> {hashes['new']}")
            report(ROOT, tree, changed, overridden=bool(flags))
            if args.dry_run:
                print("\ndry run: the checkout is unchanged")
            else:
                for path in changed:
                    shutil.copyfile(tree / path, ROOT / path)
                print(f"\nwrote {len(changed)} files")
        wasm_moves = hashes["old"] != hashes["new"]
        if args.dry_run or args.skip_wasm:
            if wasm_moves:
                print("pending: bash scripts/build-sharpearena-wasm.sh && node scripts/check-wasm-bundle.mjs")
        elif wasm_moves:
            # PATH order, not the Windows system directory first, so Git Bash (and the
            # wasm-pack on its PATH) is found rather than a WSL launcher.
            bash = shutil.which("bash") or "bash"
            try:
                run([bash, "scripts/build-sharpearena-wasm.sh"], ROOT)
                run(["node", "scripts/check-wasm-bundle.mjs"], ROOT)
            except BumpError as error:
                raise BumpError(
                    f"{error}; the pins and hash copies are written, rebuild the bundle with "
                    "scripts/build-sharpearena-wasm.sh and check it with scripts/check-wasm-bundle.mjs"
                ) from error
        if old != args.to:
            mentions = old_mentions(ROOT, old)
            if mentions:
                print(f"\nreview: {len(mentions)} remaining mentions of {old} (prose is not rewritten):")
                for line in mentions:
                    print(f"    {line}")
    except (BumpError, BenchCompatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
