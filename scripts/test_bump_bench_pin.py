#!/usr/bin/env python3
"""Tests for scripts/bump-bench-pin.py and scripts/bench_compat.py.

The pure steps of the pin bump are tested here on planted inputs: pin parsing and
rewriting, the spec-hash rebind, reading the build script's hash, the lockfile
checks, the CompositeScore key audit and the lockfile guard of the overlay. The
end-to-end bump needs cargo and a SharpeBench at the new version, so it is run by
hand in dry-run mode (RELEASING.md); the argument refusals that need neither are
tested here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_compat  # noqa: E402


def load_bump():
    spec = importlib.util.spec_from_file_location("bump_bench_pin", ROOT / "scripts" / "bump-bench-pin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump = load_bump()

MANIFEST = """[package]
name = "x"

[dependencies]
sharpebench-sim = "=0.28.0"
sharpebench-core = { version = "=0.28.0", default-features = false }
serde = "=0.28.0"

[dev-dependencies]
sharpebench-attest = "=0.28.0"

[target.'cfg(unix)'.dependencies]
sharpebench-protocol = "=0.28.0"
"""

COMPOSITE = """
pub struct Other {
    pub other_error: Option<String>,
}

/// The scored result.
#[derive(Serialize)]
pub struct CompositeScore {
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deflation_error: Option<String>,
    pub deflation_refusal: Option<String>,
    pub psr_standard_error: PsrStandardError,
    pub dsr_se: Option<f64>,
}

impl CompositeScore {
    pub fn later_error(&self) {}
}
"""


class VersionAndPins(unittest.TestCase):
    def test_versions_parse_and_order_numerically(self):
        self.assertLess(bump.version_key("0.9.0"), bump.version_key("0.10.0"))
        for bad in ("0.28", "v0.28.0", "0.28.0-rc1", "00.1.0", ""):
            with self.assertRaises(bump.BumpError):
                bump.version_key(bad)

    def test_pins_are_read_from_every_table(self):
        pins = bump.bench_pins(MANIFEST)
        self.assertEqual(
            pins,
            {
                "dependencies.sharpebench-sim": "0.28.0",
                "dependencies.sharpebench-core": "0.28.0",
                "dev-dependencies.sharpebench-attest": "0.28.0",
                "dependencies.sharpebench-protocol": "0.28.0",
            },
        )

    def test_a_path_git_or_inexact_bench_dependency_is_refused(self):
        for spec in (
            '{ path = "../sim", version = "=0.28.0" }',
            '{ git = "https://example.invalid/sb" }',
            '"0.28.0"',
            '"^0.28.0"',
        ):
            with self.assertRaises(bump.BumpError):
                bump.bench_pins(f"[dependencies]\nsharpebench-sim = {spec}\n")

    def test_rewrite_moves_every_bench_pin_and_nothing_else(self):
        out = bump.rewrite_pins(MANIFEST, "0.28.0", "0.29.0")
        self.assertEqual(set(bump.bench_pins(out).values()), {"0.29.0"})
        self.assertIn('serde = "=0.28.0"', out)
        self.assertIn("default-features = false", out)
        self.assertEqual(out.count("0.29.0"), 4)

    def test_rewrite_refuses_a_pin_it_cannot_reach(self):
        # A dotted-table pin is a real pin the line rewrite does not reach; the count
        # check refuses rather than leave it behind at the old version.
        dotted = MANIFEST + '\n[build-dependencies.sharpebench-stats]\nversion = "=0.28.0"\n'
        self.assertEqual(len(bump.bench_pins(dotted)), 5)
        with self.assertRaises(bump.BumpError):
            bump.rewrite_pins(dotted, "0.28.0", "0.29.0")
        with self.assertRaises(bump.BumpError):
            bump.rewrite_pins(MANIFEST, "0.27.0", "0.29.0")

    def test_the_committed_manifests_name_one_pin_and_round_trip(self):
        pin = bump.current_pin(ROOT)
        bump.version_key(pin)
        for path in bump.PIN_MANIFESTS:
            text = (ROOT / path).read_text(encoding="utf-8")
            there = bump.rewrite_pins(text, pin, "9.9.9")
            self.assertEqual(bump.rewrite_pins(there, "9.9.9", pin), text)


class SpecHash(unittest.TestCase):
    def test_rebind_replaces_the_one_quoted_copy(self):
        text = 'export const SPEC_HASH = "c5cdeccb4e3c9cfe";\n'
        self.assertEqual(
            bump.rebind_text(text, "c5cdeccb4e3c9cfe", "0123456789abcdef", "f"),
            'export const SPEC_HASH = "0123456789abcdef";\n',
        )

    def test_rebind_refuses_zero_or_several_copies(self):
        for text in ("no hash here", '"aaaaaaaaaaaaaaaa" "aaaaaaaaaaaaaaaa"', "aaaaaaaaaaaaaaaa unquoted"):
            with self.assertRaises(bump.BumpError):
                bump.rebind_text(text, "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "f")

    def test_every_committed_copy_holds_the_recorded_hash_once(self):
        record = json.loads((ROOT / bump.SPEC_RECORD).read_text(encoding="utf-8"))
        for path in bump.SPEC_HASH_COPIES:
            text = (ROOT / path).read_text(encoding="utf-8")
            self.assertEqual(text.count(f'"{record["spec_hash"]}"'), 1, path)

    def test_the_hash_is_read_from_the_build_script_message(self):
        message = {
            "reason": "build-script-executed",
            "env": [["SHARPEARENA_SPEC_INPUTS", "a,b"], ["SHARPEARENA_SPEC_HASH", "3aa2092de1d7c70d"]],
        }
        lines = ["   Compiling x", json.dumps({"reason": "compiler-artifact"}), json.dumps(message)]
        self.assertEqual(bump.spec_hash_from_cargo_messages(lines), "3aa2092de1d7c70d")

    def test_no_or_conflicting_or_malformed_hash_is_refused(self):
        def msg(value):
            return json.dumps({"reason": "build-script-executed", "env": [["SHARPEARENA_SPEC_HASH", value]]})

        for lines in ([], [msg("0" * 16), msg("1" * 16)], [msg("C5CDECCB4E3C9CFE")], [msg("abc")]):
            with self.assertRaises(bump.BumpError):
                bump.spec_hash_from_cargo_messages(lines)


class Lockfiles(unittest.TestCase):
    LOCK = """version = 4

[[package]]
name = "sharpebench-core"
version = "0.29.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "sharpebench-sim"
version = "0.29.0"

[[package]]
name = "serde"
version = "1.0.0"
"""

    def test_lock_versions_and_registry_source_are_checked(self):
        self.assertEqual(bump.lock_problems(self.LOCK, "0.29.0", require_registry=False), [])
        self.assertEqual(
            bump.lock_problems(self.LOCK, "0.29.0", require_registry=True),
            ["does not resolve sharpebench-sim from crates.io (source None)"],
        )
        self.assertEqual(len(bump.lock_problems(self.LOCK, "0.30.0", require_registry=False)), 2)
        self.assertEqual(
            bump.lock_problems("version = 4\n", "0.29.0", require_registry=False),
            ["locks no SharpeBench crate"],
        )

    def test_overlay_lockfiles_are_restored_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "crates" / "sharpearena-py").mkdir(parents=True)
            originals = {}
            for name in bench_compat.LOCKFILES:
                originals[name] = f"lock {name}\r\n".encode()
                (root / name).write_bytes(originals[name])
            with self.assertRaises(RuntimeError):
                with bench_compat.lockfiles_preserved(root):
                    for name in bench_compat.LOCKFILES:
                        (root / name).write_text("[[patch.unused]]\n")
                    raise RuntimeError("build failed")
            for name in bench_compat.LOCKFILES:
                self.assertEqual((root / name).read_bytes(), originals[name])


class KernelKeys(unittest.TestCase):
    def test_only_composite_score_fields_are_listed(self):
        self.assertEqual(
            bench_compat.composite_score_fields(COMPOSITE),
            ["agent_id", "deflation_error", "deflation_refusal", "psr_standard_error", "dsr_se"],
        )

    def test_a_renamed_or_flattened_field_is_refused(self):
        for attribute in ('#[serde(rename = "x")]', "#[serde(flatten)]", '#[serde(default, alias = "y")]'):
            planted = COMPOSITE.replace("    pub dsr_se", f"    {attribute}\n    pub dsr_se")
            with self.assertRaises(bench_compat.BenchCompatError):
                bench_compat.composite_score_fields(planted)
        with self.assertRaises(bench_compat.BenchCompatError):
            bench_compat.composite_score_fields("pub struct Other {\n}\n")

    def test_every_suffixed_key_must_be_classified(self):
        module = bench_compat.load_kernel_score()
        fields = bench_compat.composite_score_fields(COMPOSITE)
        self.assertEqual(bench_compat.unclassified_kernel_keys(fields, module), [])
        planted = [*fields, "drift_refusal", "zz_error", "error_count"]
        self.assertEqual(
            bench_compat.unclassified_kernel_keys(planted, module), ["drift_refusal", "zz_error"]
        )

    def test_key_problems_name_the_file_and_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            composite = pathlib.Path(tmp) / "composite.rs"
            composite.write_text(COMPOSITE.replace("pub dsr_se", "pub carry_error: Option<String>,\n    pub dsr_se"), encoding="utf-8")
            problems = bench_compat.kernel_key_problems(composite)
        self.assertEqual(len(problems), 1)
        self.assertIn("'carry_error'", problems[0])


class Arguments(unittest.TestCase):
    def run_main(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = bump.main(list(argv))
        return code, err.getvalue()

    def test_a_real_bump_refuses_a_local_override(self):
        code, err = self.run_main("--to", bump.current_pin(ROOT), "--bench-source", str(ROOT))
        self.assertEqual(code, 1)
        self.assertIn("dry-run preview", err)

    def test_an_older_version_is_refused(self):
        code, err = self.run_main("--to", "0.0.1", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("older than the current pin", err)

    def test_an_override_at_another_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = pathlib.Path(tmp)
            (bench / "Cargo.toml").write_text('[workspace.package]\nversion = "0.1.0"\n', encoding="utf-8")
            code, err = self.run_main("--to", "99.0.0", "--dry-run", "--bench-source", str(bench))
        self.assertEqual(code, 1)
        self.assertIn("is SharpeBench 0.1.0, not 99.0.0", err)


if __name__ == "__main__":
    unittest.main()
