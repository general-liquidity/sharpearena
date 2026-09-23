"""Canonical-evidence grading for the HUD fixture.

The grade is computed from what the engine recorded, not from what the agent said.
The agent's answer is carried through as `agent_report` so a later diagnostic pass
can compare the two, and it never enters the score. No language-model judge is
involved at any point.

Reward shape follows the existing `episode_outcomes` gate: an incomplete or
process-blocked episode takes the floor rather than credit for a favourable prefix,
and an unavailable scoring kernel stays typed instead of collapsing to zero.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

from sharpearena.episode_outcomes import is_process_block
from sharpearena.kernel_score import is_kernel_score_unavailable, kernel_score_or_unavailable
from sharpearena.mandate import mandate_breach
from sharpearena.sharpearena_py import score_run

#: Floor for an episode that did not complete its contract. Matches the primary
#: floor already used by the verifiers rubric, so the two surfaces agree.
INCOMPLETE_FLOOR = -1.0


def composite_score(returns: list[float], n_trials: int) -> dict:
    """The SharpeBench composite for a realized return series, or an empty dict.

    Fewer than two bars is not a short series, it is no series: returning `{}`
    keeps that distinguishable from a computed score of zero.
    """
    if len(returns) < 2:
        return {}
    return json.loads(score_run(returns, n_trials))


def grade_episode(episode: Any, agent_report: Optional[str], *, n_trials: int = 1) -> dict:
    """Build the wire grade frame from the episode's canonical evidence."""
    from episode import episode_identity  # fixture-local import

    counts = episode.attempt_counts()
    completed = episode.bars_advanced >= episode.horizon
    blocking = [event for event in episode.events if is_process_block(event)]
    composite = composite_score(episode.returns, n_trials)
    kernel = kernel_score_or_unavailable(composite) if composite else None
    breach = mandate_breach(episode.mandate, episode.returns, episode.events)

    if not completed:
        status = "incomplete"
    elif blocking:
        status = "process_block"
    else:
        status = "completed"

    if status == "completed":
        realized = math.prod(1.0 + r for r in episode.returns) - 1.0
        score = max(-1.0, min(1.0, realized))
    else:
        score = INCOMPLETE_FLOOR

    info: dict[str, Any] = {
        "status": status,
        "identity": episode_identity(episode),
        "counts": counts,
        "bars_advanced": episode.bars_advanced,
        "returns": [float(r) for r in episode.returns],
        "process_events": episode.events,
        "process_block_count": len(blocking),
        "mandate_breach": float(breach),
        "composite": composite,
        # `KernelScore` is a float or the `unavailable_scoring_kernel_*` string.
        # The string is kept verbatim: an unavailable estimate is not a zero.
        "deflated_sharpe": kernel,
        "deflated_sharpe_available": None if kernel is None else not is_kernel_score_unavailable(
            kernel
        ),
        # Diagnostic only. Never an input to `score`.
        "agent_report": agent_report,
    }
    return {"score": float(score), "done": True, "info": info}


def export_evidence(grade: dict, destination: Path) -> Path:
    """Write one grade frame so an interrupted export leaves no readable run.

    Serialize fully, write to a sibling temporary file, then rename. A process
    killed during the write leaves the temporary path behind and `destination`
    either absent or still holding the previous export. Writing straight to
    `destination` would instead leave a truncated file that a later reader can
    parse far enough to mistake for a short run.
    """
    payload = json.dumps(grade, sort_keys=True, indent=2)
    temporary = destination.with_name(destination.name + ".partial")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination
