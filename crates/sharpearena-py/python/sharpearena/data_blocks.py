"""Gap-aware contiguous-block episode sampling for frozen real-data CSVs.

Synthetic SharpeArena paths are continuous by construction, but a real CSV
(``date,symbol,close[,dividend]``) has gaps — exchange downtime, new-listing
starts, missing bars. An episode window straddling a discontinuity silently
injects a fabricated jump, breaking the point-in-time, leak-free contract on
real data.

This module segments a frozen dataset's date axis into continuous blocks and
constrains every sampled episode window to lie entirely inside a single block,
so a reset never crosses a gap. The block math is a pure function of
``(timestamps, max_gap)`` — deterministic and binding-free; only
:func:`make_block_env` touches the native env.

A usable block must hold at least ``horizon + warmup + 1`` bars (the episode's
``warmup`` lookback plus ``horizon`` decisions need one extra bar to realize the
final return); shorter blocks are discarded, matching the source repo.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from typing import Optional, Sequence, Union

import numpy as np

Block = tuple[int, int]
Window = dict[str, int]
Timestamp = Union[str, int, float]


def _to_ordinal(ts: Timestamp) -> float:
    """Map a timestamp (numeric index, ISO date, or ISO datetime) to a real number
    on a monotone axis. Only differences are used downstream, so the absolute
    origin/unit is irrelevant as long as it is consistent across the sequence."""
    if isinstance(ts, bool):
        raise TypeError("timestamps must be numbers or strings, not bool")
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(date.fromisoformat(s).toordinal())
    except ValueError:
        return datetime.fromisoformat(s).timestamp()


def _ordinals(timestamps: Sequence[Timestamp]) -> np.ndarray:
    return np.asarray([_to_ordinal(t) for t in timestamps], dtype=np.float64)


def infer_cadence(timestamps: Sequence[Timestamp]) -> float:
    """The median positive step between consecutive timestamps — the typical bar
    cadence. Scale by a tolerance to derive a ``max_gap`` threshold."""
    ords = _ordinals(timestamps)
    if ords.size < 2:
        raise ValueError("need >= 2 timestamps to infer cadence")
    diffs = np.diff(ords)
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        raise ValueError("timestamps are non-increasing; cannot infer cadence")
    return float(np.median(positive))


def find_continuous_blocks(
    timestamps: Sequence[Timestamp], *, max_gap: float
) -> list[Block]:
    """Segment ordered ``timestamps`` into ``[start, end)`` index blocks where every
    consecutive pair differs by ``<= max_gap``. A larger jump starts a new block.

    Deterministic pure function of ``(timestamps, max_gap)``. ``max_gap`` is in the
    same unit the timestamps reduce to (days for ISO dates, seconds for ISO
    datetimes, raw units for numeric indices) — see :func:`infer_cadence`.
    """
    if max_gap <= 0:
        raise ValueError("max_gap must be > 0")
    ords = _ordinals(timestamps)
    n = ords.size
    if n == 0:
        return []
    blocks: list[Block] = []
    start = 0
    for i in range(1, n):
        if ords[i] - ords[i - 1] > max_gap:
            blocks.append((start, i))
            start = i
    blocks.append((start, n))
    return blocks


def block_windows(
    blocks: Sequence[Block], *, horizon: int, warmup: int = 0
) -> list[Window]:
    """Every valid ``[window_start, window_end)`` episode window that fits entirely
    inside a single block. A window spans ``horizon + warmup + 1`` bars; blocks too
    short to hold one are discarded, so no returned window ever straddles a gap.

    Each window is a dict keyed ``window_start`` / ``window_end`` — the indices
    :class:`~sharpearena.gym.SharpeArenaEnv` consumes.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    window_len = horizon + warmup + 1
    out: list[Window] = []
    for start_b, end_b in blocks:
        if end_b - start_b < window_len:
            continue
        for start in range(start_b, end_b - window_len + 1):
            out.append(
                {"window_start": int(start), "window_end": int(start + window_len)}
            )
    return out


def sample_block_window(
    blocks: Sequence[Block], *, horizon: int, warmup: int = 0, seed: int
) -> Window:
    """Deterministically pick one in-block window from ``seed``. Reproducible via
    ``SeedSequence`` (the eval-seed style); raises if no block is long enough."""
    windows = block_windows(blocks, horizon=horizon, warmup=warmup)
    if not windows:
        raise ValueError(
            f"no block holds an episode window of {horizon + warmup + 1} bars "
            f"(horizon={horizon}, warmup={warmup}); longest block has "
            f"{max((e - s for s, e in blocks), default=0)} bars"
        )
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    return windows[int(rng.integers(len(windows)))]


def _csv_dates(csv_text: str) -> list[str]:
    """The sorted, unique date axis of a long-format ``date,symbol,close[,dividend]``
    CSV — the bar index that ``window_start`` / ``window_end`` address."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    if not rows:
        raise ValueError("empty CSV")
    header = [h.strip().lower() for h in rows[0]]
    col = header.index("date") if "date" in header else 0
    seen: set[str] = set()
    for row in rows[1:]:
        if col < len(row):
            value = row[col].strip()
            if value:
                seen.add(value)
    return sorted(seen)


def make_block_env(
    csv_text: str,
    *,
    horizon: int,
    warmup: int = 0,
    seed: int,
    max_gap: Optional[float] = None,
    tolerance: float = 1.5,
    **env_kwargs,
):
    """Construct an :class:`~sharpearena.gym.SharpeArenaEnv` over a seed-chosen in-block
    window of a frozen CSV, so a reset never straddles a gap.

    ``max_gap`` defaults to the inferred bar cadence times ``tolerance``. Extra
    ``env_kwargs`` pass through to the env constructor. Raises if no continuous
    block is long enough for ``horizon + warmup + 1`` bars.
    """
    dates = _csv_dates(csv_text)
    gap = float(max_gap) if max_gap is not None else infer_cadence(dates) * float(tolerance)
    blocks = find_continuous_blocks(dates, max_gap=gap)
    window = sample_block_window(blocks, horizon=horizon, warmup=warmup, seed=seed)
    try:
        from .gym import SharpeArenaEnv
    except ImportError:
        from sharpearena.gym import SharpeArenaEnv
    return SharpeArenaEnv(
        csv_text=csv_text,
        window_start=window["window_start"],
        window_end=window["window_end"],
        seed=int(seed),
        **env_kwargs,
    )


__all__ = [
    "find_continuous_blocks",
    "block_windows",
    "sample_block_window",
    "make_block_env",
    "infer_cadence",
]
