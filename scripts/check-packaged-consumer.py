#!/usr/bin/env python3
"""Build a fresh downstream crate against the normalized `cargo package` archive.

`check-packaged-spec.py` already runs one of the crate's own tests inside Cargo's package
staging tree. That proves the SPEC_HASH assertion survives the manifest rewrite; it does
not prove that anyone outside this repository can depend on the crate. A `mod` that only
resolved through the workspace, a data file the include rules dropped, or an API only
reachable from an integration test all pass that check and fail for the first person who
runs `cargo add sharpearena`.

This extracts the published `.crate` archive and compiles and RUNS a throwaway crate that
has never seen this repository and depends only on the extracted bytes. It also replays
the versioned conformance kit from inside the archive, so a release that ships the engine
without the wire contract fails here rather than being discovered by an integrator.

No registry publication, no model calls, no market data.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib

TARGET = "sharpearena"

CONSUMER_MAIN = """\
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use sharpearena::{Action, Agent, BuyAndHold, Decision, MarketObservation, CONTRACT_VERSION};

fn main() {
    // The archive root, handed in by the driver: a consumer reads the contract kit out of
    // the package it installed, not out of a checkout it does not have.
    let root = PathBuf::from(std::env::args().nth(1).expect("archive root argument"));
    let contract = root.join("contract");

    let kit: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(contract.join("conformance-kit.v1.json"))
            .expect("the packaged crate ships the versioned conformance kit"),
    )
    .expect("the kit is valid JSON");
    assert_eq!(
        kit["contract_version"].as_str(),
        Some(CONTRACT_VERSION),
        "the packaged kit certifies a different wire version than the packaged crate"
    );

    let names = kit["fixtures"].as_array().expect("fixtures is an array");
    assert!(!names.is_empty(), "the packaged kit lists no fixtures");
    for name in names {
        let name = name.as_str().expect("fixture name is a string");
        let text = fs::read_to_string(contract.join("conformance").join(name))
            .unwrap_or_else(|e| panic!("{name} is named by the kit but missing from the archive: {e}"));
        let value: serde_json::Value = serde_json::from_str(&text).expect("fixture is valid JSON");
        let observation: MarketObservation = serde_json::from_value(value["observation"].clone())
            .unwrap_or_else(|e| panic!("{name}: observation does not parse: {e}"));
        let observed: BTreeSet<String> =
            observation.symbols.iter().map(|s| s.symbol.clone()).collect();

        let mut agent = BuyAndHold;
        let decision = agent.decide(&observation);
        assert_eq!(
            decision.orders.len(),
            observation.symbols.len(),
            "{name}: the packaged reference agent skipped an observed symbol"
        );
        for order in &decision.orders {
            assert!(
                matches!(
                    order.action,
                    Action::Buy | Action::Sell | Action::Hold | Action::Close
                ),
                "{name}: action outside the enum"
            );
            assert!(order.target_weight.is_finite(), "{name}: non-finite weight");
            assert!(observed.contains(&order.symbol), "{name}: unobserved symbol");
        }

        if let Some(legacy) = value.get("legacy_decision") {
            let parsed: Decision = serde_json::from_value(legacy.clone())
                .unwrap_or_else(|e| panic!("{name}: legacy_decision does not parse: {e}"));
            assert!(!parsed.orders.is_empty(), "{name}: empty legacy decision");
        }
    }

    println!("packaged consumer replayed {} fixtures from the archive", names.len());
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="package uncommitted changes")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"][
        "package"
    ]["version"]
    command = [
        "cargo",
        "package",
        "--no-verify",
        "-p",
        TARGET,
        "--target-dir",
        str(root / "target"),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    # --no-verify: the verification build is what check-packaged-spec.py already drives.
    # Here the consumer build is the verification, and it is a stricter one.
    subprocess.run(command, cwd=root, check=True)

    archive = root / "target" / "package" / f"{TARGET}-{version}.crate"
    if not archive.is_file():
        raise SystemExit(f"cargo package produced no archive for {TARGET} {version}")

    with tempfile.TemporaryDirectory(prefix="sharpearena-consumer-") as work:
        extracted = Path(work) / "archive"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extracted, filter="data")
        packaged = extracted / f"{TARGET}-{version}"
        if not (packaged / "Cargo.toml").is_file():
            raise SystemExit(f"{archive.name} does not contain {TARGET}-{version}/Cargo.toml")

        consumer = Path(work) / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "Cargo.toml").write_text(
            "[package]\n"
            'name = "packaged-consumer"\n'
            'version = "0.0.0"\n'
            'edition = "2021"\n'
            "publish = false\n\n"
            "# Empty table: this crate is deliberately not a member of any workspace.\n"
            "[workspace]\n\n"
            "[dependencies]\n"
            f"{TARGET} = {{ path = {json.dumps(str(packaged))} }}\n"
            'serde_json = "1"\n',
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(CONSUMER_MAIN, encoding="utf-8")
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--", str(packaged)],
            cwd=consumer,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(f"the packaged-archive consumer failed with {result.returncode}")
        if "packaged consumer replayed" not in result.stdout:
            raise SystemExit("the consumer binary did not report a completed replay")


if __name__ == "__main__":
    main()
    sys.exit(0)
