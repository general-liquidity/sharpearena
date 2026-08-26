"""Inspect or run one guarded local forward paper-trading decision.

The command can read public Binance data or authenticated Alpaca market data. Its
only remote order destination is Alpaca's paper endpoint, and that path requires an
explicit ``--allow-remote-paper-submit`` switch. Credentials are read from the
environment and are never accepted in, or written beside, the plan.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Sequence

from .paper_trading import (
    AlpacaMarketData,
    AlpacaPaperBroker,
    BinancePublicData,
    ForwardEvidenceJournal,
    InMemoryPaperBroker,
    LifecycleStore,
    PaperAccount,
    PaperRiskConfig,
    PaperRiskGuard,
    PaperTradingSession,
    forward_window_from_preimage,
    prepare_forward_window_commitment,
    prepare_forward_window_reveal,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _reject_unknown(payload: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the selected Alpaca adapter")
    return value


@dataclass(frozen=True)
class PaperExecutionPlan:
    agent_id: str
    model_digest: str
    symbols: tuple[str, ...]
    market_data: dict[str, Any]
    broker: dict[str, Any]
    account: PaperAccount
    risk: PaperRiskConfig

    @property
    def plan_sha256(self) -> str:
        return sha256(
            _canonical_bytes(
                {
                    "agent_id": self.agent_id,
                    "model_digest": self.model_digest,
                    "symbols": self.symbols,
                    "market_data": self.market_data,
                    "broker": self.broker,
                    "account": asdict(self.account),
                    "risk": asdict(self.risk),
                }
            )
        ).hexdigest()


def load_execution_plan(path: Path) -> PaperExecutionPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paper plan must be an object")
    _reject_unknown(
        payload,
        {
            "agent_id",
            "model_digest",
            "symbols",
            "market_data",
            "broker",
            "account",
            "risk",
        },
        "paper plan",
    )
    for name in ("market_data", "broker", "account", "risk"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"{name} must be an object")
    symbols = payload.get("symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(symbol, str) and symbol for symbol in symbols)
    ):
        raise ValueError("symbols must be a non-empty string array")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    if any(symbol.upper() != symbol for symbol in symbols):
        raise ValueError("symbols must use canonical uppercase spelling")
    agent_id = payload.get("agent_id")
    model_digest = payload.get("model_digest")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(model_digest, str) or not model_digest:
        raise ValueError("model_digest must be a non-empty string")

    market_data = dict(payload["market_data"])
    _reject_unknown(
        market_data, {"provider", "interval", "timeframe", "limit"}, "market_data"
    )
    if market_data.get("provider") not in {"binance-public", "alpaca"}:
        raise ValueError("market_data.provider must be binance-public or alpaca")
    limit = market_data.get("limit", 120)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("market_data.limit must be a positive integer")
    market_data["limit"] = limit

    broker = dict(payload["broker"])
    _reject_unknown(broker, {"provider"}, "broker")
    if broker.get("provider") not in {"in-memory", "alpaca-paper"}:
        raise ValueError("broker.provider must be in-memory or alpaca-paper")

    account_payload = dict(payload["account"])
    _reject_unknown(
        account_payload,
        {"cash", "equity", "session_start_equity", "peak_equity", "positions"},
        "account",
    )
    risk_payload = dict(payload["risk"])
    _reject_unknown(
        risk_payload,
        {
            "allowed_symbols",
            "max_order_notional",
            "max_gross_exposure",
            "max_daily_loss",
            "max_drawdown",
            "allow_shorting",
            "kill_switch",
        },
        "risk",
    )
    allowed = risk_payload.get("allowed_symbols")
    if not isinstance(allowed, list):
        raise ValueError("risk.allowed_symbols must be an array")
    risk_payload["allowed_symbols"] = tuple(allowed)
    return PaperExecutionPlan(
        agent_id=agent_id,
        model_digest=model_digest,
        symbols=tuple(symbols),
        market_data=market_data,
        broker=broker,
        account=PaperAccount(**account_payload),
        risk=PaperRiskConfig(**risk_payload),
    )


def _run_execute(args: argparse.Namespace) -> int:
    plan = load_execution_plan(args.plan)
    if args.inspect:
        print(
            json.dumps(
                {
                    "plan_sha256": plan.plan_sha256,
                    "agent_id": plan.agent_id,
                    "model_digest": plan.model_digest,
                    "symbols": plan.symbols,
                    "market_data": plan.market_data,
                    "broker": plan.broker,
                    "risk": asdict(plan.risk),
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    provider = plan.market_data["provider"]
    if provider == "binance-public":
        source = BinancePublicData(plan.market_data.get("interval", "1h"))
    else:
        source = AlpacaMarketData(
            _required_environment("ALPACA_API_KEY_ID"),
            _required_environment("ALPACA_API_SECRET_KEY"),
            plan.market_data.get("timeframe", "1Day"),
        )
    latest = {}
    for symbol in plan.symbols:
        bars = source.recent_bars(symbol, limit=plan.market_data["limit"])
        if not bars:
            raise RuntimeError(f"market-data adapter returned no bars for {symbol}")
        latest[symbol] = bars[-1]

    risk = PaperRiskGuard(plan.risk)
    if plan.broker["provider"] == "in-memory":
        broker = InMemoryPaperBroker(risk)
    else:
        if not args.allow_remote_paper_submit:
            raise ValueError(
                "Alpaca paper submission requires --allow-remote-paper-submit"
            )
        broker = AlpacaPaperBroker(
            _required_environment("ALPACA_API_KEY_ID"),
            _required_environment("ALPACA_API_SECRET_KEY"),
            risk,
        )
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    window = (
        None
        if args.window_preimage is None
        else forward_window_from_preimage(args.window_preimage)
    )
    session = PaperTradingSession(
        broker,
        ForwardEvidenceJournal(args.evidence),
        agent_id=plan.agent_id,
        model_digest=plan.model_digest,
        lifecycles=LifecycleStore(args.lifecycle_state),
        window=window,
        # A broker that can report its own book is the authority on the account.
        # The plan file only seeds it.
        account_source=broker if hasattr(broker, "account_snapshot") else None,
    )
    session.reconcile_all()
    responses = session.execute_decision(decision, plan.symbols, plan.account, latest)
    session.lifecycles.save()
    print(
        json.dumps(
            {"plan_sha256": plan.plan_sha256, "paper_responses": responses},
            sort_keys=True,
        )
    )
    return 0


def _run_commit(args: argparse.Namespace) -> int:
    salt = _required_environment(args.salt_env)
    manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    result = prepare_forward_window_commitment(
        args.agent_id,
        args.target_window,
        manifest,
        salt,
        commitment_path=args.commitment,
        private_preimage_path=args.private_preimage,
    )
    print(json.dumps(result["commitment"], sort_keys=True))
    return 0


def _run_reveal(args: argparse.Namespace) -> int:
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    commitment = json.loads(args.commitment.read_text(encoding="utf-8"))
    preimage = json.loads(args.private_preimage.read_text(encoding="utf-8"))
    entry = prepare_forward_window_reveal(submission, commitment, preimage)
    payload = json.dumps([entry], sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "entries": 1}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute", help="inspect or execute one decision")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--decision", type=Path, required=True)
    execute.add_argument("--evidence", type=Path, required=True)
    execute.add_argument("--inspect", action="store_true")
    execute.add_argument("--allow-remote-paper-submit", action="store_true")
    execute.add_argument(
        "--lifecycle-state",
        type=Path,
        default=None,
        help="durable order-lifecycle table; without it an unknown submission "
        "cannot be reconciled after a restart",
    )
    execute.add_argument(
        "--window-preimage",
        type=Path,
        default=None,
        help="private reveal preimage whose commitment stamps every record",
    )

    commit = commands.add_parser("commit", help="prepare a forward-window commitment")
    commit.add_argument("--agent-id", required=True)
    commit.add_argument("--target-window", required=True)
    commit.add_argument("--artifact-manifest", type=Path, required=True)
    commit.add_argument("--commitment", type=Path, required=True)
    commit.add_argument("--private-preimage", type=Path, required=True)
    commit.add_argument("--salt-env", default="SHARPEARENA_FORWARD_SALT")

    reveal = commands.add_parser(
        "reveal", help="open a commitment into a SharpeBench RevealedEntry array"
    )
    reveal.add_argument("--submission", type=Path, required=True)
    reveal.add_argument("--commitment", type=Path, required=True)
    reveal.add_argument("--private-preimage", type=Path, required=True)
    reveal.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "execute":
        return _run_execute(args)
    if args.command == "commit":
        return _run_commit(args)
    return _run_reveal(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PaperExecutionPlan", "load_execution_plan"]
