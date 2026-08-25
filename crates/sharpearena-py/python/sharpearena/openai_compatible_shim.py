"""NDJSON adapter from SharpeBench to a local OpenAI-compatible model server.

The backend may be llama.cpp, vLLM, SGLang, or another implementation with the
same strict chat-completions surface. The endpoint must be loopback-only and the
identity manifest must state backend/version/model/digest/quantization/offload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .local_agents import (
    ModelRunConfig,
    OpenAICompatibleClient,
    SamplingConfig,
    load_identity_manifest,
)
from .ollama_shim import run_stdio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact served model ID")
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--supports-thinking",
        action="store_true",
        help="declare support for chat_template_kwargs.enable_thinking",
    )
    parser.add_argument("--decision-cadence", type=int, default=1)
    parser.add_argument("--identity-out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    identities = load_identity_manifest(args.identity_manifest)
    config = ModelRunConfig(
        model=args.model,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
        ),
        decision_cadence=args.decision_cadence,
    )
    client = OpenAICompatibleClient(
        identities,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        supports_thinking=args.supports_thinking,
    )
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
