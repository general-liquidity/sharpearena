#!/usr/bin/env python3
"""Generate deterministic SharpeArena evidence for the SharpeBench tutorial.

The example uses logical clocks and synthetic outcomes. It demonstrates the
artifact boundary between the products; it is not an empirical agent result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

from sharpearena import (
    BINARY_BRIER,
    PROBABILITY,
    ForecastContract,
    ForecastLedger,
    ForecastRunIdentity,
    InformationExposure,
    Outcome,
    write_forecast_evidence,
)


OUTCOMES = (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
ALPHA_PREDICTIONS = (0.82, 0.18, 0.76, 0.71, 0.24, 0.31, 0.68, 0.27)
BETA_PREDICTIONS = (0.58, 0.61, 0.47, 0.55, 0.63, 0.44, 0.52, 0.57)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(agent_id: str) -> ForecastRunIdentity:
    return ForecastRunIdentity(
        agent_id=agent_id,
        model_id=f"tutorial-model-{agent_id}",
        model_sha256=_digest(f"model:{agent_id}:v1"),
        scaffold_id="forecast-quality-tutorial",
        scaffold_sha256=_digest("scaffold:forecast-quality-tutorial:v1"),
        prompt_sha256=_digest(f"prompt:{agent_id}:v1"),
        operator_id="sharpe-suite-tutorial",
        config_sha256=_digest(f"config:{agent_id}:v1"),
    )


def _contract(index: int) -> ForecastContract:
    resolves_at = 30 if index < 4 else 40
    return ForecastContract(
        contract_id=f"tutorial-binary-{index + 1:02d}",
        question=f"Will synthetic instrument {index + 1:02d} close above its frozen reference?",
        instrument=f"SYNTH-{index + 1:02d}",
        target="close_above_frozen_reference",
        kind=PROBABILITY,
        opens_at=10,
        deadline=20,
        resolves_at=resolves_at,
        observation_source="tutorial:synthetic-market-v1",
        open_definition="value in the frozen observation at logical clock 10",
        close_definition=f"value in the frozen resolution at logical clock {resolves_at}",
        unit="binary",
        scoring_rule=BINARY_BRIER,
        boundary_ownership="an equal close resolves false",
        missing_data_policy="cancel the contract",
        fallback_policy="no fallback source",
    )


def _exposure(agent_id: str, index: int, submitted_at: int) -> InformationExposure:
    consensus_visible = agent_id == "agent-beta" and index == 0
    return InformationExposure(
        observed_at=submitted_at,
        market_snapshot_sha256=_digest(f"market:{index}:clock:{submitted_at}"),
        consensus_visible=consensus_visible,
        consensus_snapshot_sha256=(
            _digest("tutorial-consensus:clock:12") if consensus_visible else None
        ),
        source_ids=(
            ("synthetic-market", "tutorial-consensus")
            if consensus_visible
            else ("synthetic-market",)
        ),
    )


def _submit_initial(
    ledger: ForecastLedger,
    *,
    agent_id: str,
    index: int,
    prediction: float,
) -> None:
    contract = _contract(index)
    claim_id = f"claim-{index + 1:02d}"
    if agent_id == "agent-beta" and index == 4:
        ledger.submit(
            claim_id=claim_id,
            contract=contract,
            prediction=prediction,
            confidence=prediction,
            rationale="pre-open tutorial attempt retained for audit",
            submitted_at=9,
            idempotency_key=f"{agent_id}:{claim_id}:pre-open",
            exposure=_exposure(agent_id, index, 9),
        )
        ledger.submit(
            claim_id=claim_id,
            contract=contract,
            prediction=prediction,
            confidence=prediction,
            rationale="forecast after the contract opened",
            submitted_at=12,
            idempotency_key=f"{agent_id}:{claim_id}:eligible",
            expected_revision=0,
            revision_reason="the contract is now open",
            exposure=_exposure(agent_id, index, 12),
        )
        return
    first_prediction = 0.55 if agent_id == "agent-alpha" and index == 2 else prediction
    ledger.submit(
        claim_id=claim_id,
        contract=contract,
        prediction=first_prediction,
        confidence=first_prediction,
        rationale="forecast from the frozen synthetic observation",
        submitted_at=12,
        idempotency_key=f"{agent_id}:{claim_id}:initial",
        exposure=_exposure(agent_id, index, 12),
    )
    if agent_id == "agent-alpha" and index == 2:
        ledger.submit(
            claim_id=claim_id,
            contract=contract,
            prediction=prediction,
            confidence=prediction,
            rationale="eligible revision after a declared public update",
            submitted_at=18,
            idempotency_key=f"{agent_id}:{claim_id}:revision",
            expected_revision=0,
            trigger_event_id="tutorial-public-update",
            revision_reason="public update changed the forecast",
            exposure=_exposure(agent_id, index, 18),
        )
        ledger.submit(
            claim_id=claim_id,
            contract=contract,
            prediction=0.01,
            confidence=0.99,
            rationale="late attempt retained but never scored",
            submitted_at=21,
            idempotency_key=f"{agent_id}:{claim_id}:late",
            expected_revision=1,
            revision_reason="deadline has passed",
            exposure=_exposure(agent_id, index, 21),
        )


def build_evidence(output_dir: Path) -> tuple[Path, Path, Path]:
    """Write two complete ledgers and their content-address manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for agent_id, predictions in (
        ("agent-alpha", ALPHA_PREDICTIONS),
        ("agent-beta", BETA_PREDICTIONS),
    ):
        ledger = ForecastLedger(_identity(agent_id))
        for index, prediction in enumerate(predictions):
            _submit_initial(
                ledger,
                agent_id=agent_id,
                index=index,
                prediction=prediction,
            )
        outcomes = [
            Outcome(
                claim_id=f"claim-{index + 1:02d}",
                value=outcome,
                available_at=_contract(index).resolves_at,
            )
            for index, outcome in enumerate(OUTCOMES)
        ]
        path = output_dir / f"{agent_id}.json"
        write_forecast_evidence(path, ledger.evidence(outcomes, generated_at=41))
        paths.append(path)

    manifest = {
        "schema_version": "sharpe-suite.forecast-quality-tutorial.v1",
        "purpose": "deterministic interface fixture, not empirical agent evidence",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return paths[0], paths[1], manifest_path


def run_sharpebench(
    sharpebench_dir: Path,
    evidence_paths: Sequence[Path],
    report_path: Path,
) -> None:
    """Run the independent consumer with the tutorial's frozen analysis settings."""

    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "sharpebench",
        "--",
        "forecast-quality",
        *(str(path.resolve()) for path in evidence_paths),
        "--bootstrap-samples",
        "400",
        "--seed",
        "23",
        "--confidence",
        "0.9",
        "--alpha",
        "0.05",
        "--bins",
        "5",
        "--json",
    ]
    completed = subprocess.run(
        command,
        cwd=sharpebench_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    report_path.write_text(completed.stdout, encoding="utf-8", newline="\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("fixtures"),
        help="directory for the two evidence files and manifest",
    )
    parser.add_argument(
        "--sharpebench-dir",
        type=Path,
        help="optional SharpeBench checkout to run as the independent consumer",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="report path used with --sharpebench-dir (default: OUTPUT/report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    agent_alpha, agent_beta, _ = build_evidence(args.output_dir)
    if args.sharpebench_dir is not None:
        report = args.report or args.output_dir / "report.json"
        run_sharpebench(args.sharpebench_dir, (agent_alpha, agent_beta), report)
    elif args.report is not None:
        raise SystemExit("--report requires --sharpebench-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
