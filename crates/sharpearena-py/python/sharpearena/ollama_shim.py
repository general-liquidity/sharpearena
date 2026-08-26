"""NDJSON stdio adapter from SharpeBench ``ExternalAgent`` to local Ollama.

Run with ``python -m sharpearena.ollama_shim --model MODEL``. Each stdin line is one
wire ``MarketObservation`` and each stdout line is one canonical wire ``Decision``.
Errors are written to stderr and close the process non-zero; they are never converted
to holds. Only loopback Ollama endpoints are accepted by :class:`OllamaClient`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import IO, Optional, Sequence

from .local_agents import ModelRunConfig, OllamaClient, PromptRenderer, SamplingConfig


def run_stdio(
    client: OllamaClient,
    config: ModelRunConfig,
    source: IO[str],
    sink: IO[str],
    errors: IO[str],
    *,
    identity_path: Optional[Path] = None,
) -> int:
    """Drive the line protocol; factored for a no-network CI smoke test."""
    try:
        identity = client.identity(config)
        if identity_path is not None:
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text(
                json.dumps(asdict(identity), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        renderer = PromptRenderer()
        previous = None
        logical_step = 0
        for line in source:
            if not line.strip():
                continue
            observation = json.loads(line)
            if not isinstance(observation, dict):
                raise ValueError("observation line must be a JSON object")
            inferred = logical_step % config.decision_cadence == 0 or previous is None
            if inferred:
                result = client.decide(
                    observation,
                    config,
                    renderer,
                    sampling_seed=config.sampling.seed,
                )
                previous = result.decision
            emitted = json.loads(json.dumps(previous))
            if not inferred:
                # Reusing a target allocation is free; charging the original model
                # call on every intervening bar would multiply compute cost by cadence.
                emitted.pop("cost", None)
            sink.write(
                json.dumps(emitted, sort_keys=True, separators=(",", ":")) + "\n"
            )
            sink.flush()
            logical_step += 1
        return 0
    except Exception as error:  # noqa: BLE001 - the process boundary reports every fault
        errors.write(
            json.dumps(
                {"error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
            )
            + "\n"
        )
        errors.flush()
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact installed Ollama tag")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--decision-cadence", type=int, default=1)
    parser.add_argument("--identity-out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = ModelRunConfig(
        model=args.model,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            context_tokens=args.context_tokens,
            thinking=args.thinking,
        ),
        decision_cadence=args.decision_cadence,
    )
    client = OllamaClient(args.base_url, timeout_seconds=args.timeout_seconds)
    return run_stdio(
        client,
        config,
        sys.stdin,
        sys.stdout,
        sys.stderr,
        identity_path=args.identity_out,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
