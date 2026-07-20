"""Sequential / curriculum scenario chaining over :class:`SharpeArenaEnv`.

Procgen's ``use_sequential_levels`` chains level seeds (``current_level_seed + 997``)
into a continuous curriculum instead of resampling i.i.d. The trading analog: walk an
ordered list of scenario seeds (or a deterministic seed-chaining rule) so an agent
experiences a fixed sequence of regimes — e.g. calm → shock → recovery — as successive
episodes, rather than independently sampled ones.

:class:`CurriculumEnv` is a thin :class:`gymnasium.Wrapper` that, on every ``reset()``,
points the wrapped env at the *next* scenario seed in its schedule. The schedule is
fixed up front (an explicit seed list or a pure ``base + step`` rule), so the same
constructor args always yield the same episode sequence and seeds are never drawn from
future bars (leak-free).

:func:`regime_curriculum` rotates the synthetic ``distribution_mode`` (calm/hard/extreme)
across episodes. Because :class:`SharpeArenaEnv` fixes ``distribution_mode`` at
construction, the wrapper rebuilds the underlying env per ``reset()`` via an
``env_factory`` — the documented approach for changing a construction-time parameter
between episodes.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import gymnasium as gym

from .gym import SharpeArenaEnv

# Procgen's sequential-level stride; chaining seeds by a fixed coprime-ish step keeps
# successive scenarios far apart in seed space while staying a pure function of k.
_CHAIN_STEP = 997

EnvFactory = Callable[[int], gym.Env]


class CurriculumEnv(gym.Wrapper):
    """Advance through an ordered schedule of scenario seeds, one per ``reset()``.

    Parameters
    ----------
    seeds:
        Explicit ordered scenario seeds to walk. If ``None``, a deterministic chain of
        length ``n_episodes`` is generated from the base seed (``env_kwargs['seed']``,
        default 0) per ``schedule``.
    schedule:
        ``"sequential"`` → ``base + k``; ``"chained"`` → Procgen's ``base + 997*k``.
        When ``seeds`` is given this is only a descriptive label surfaced in ``info``.
    n_episodes:
        Length of the generated chain (required when ``seeds`` is ``None``).
    loop:
        If ``True`` the schedule wraps around at the end; if ``False`` the schedule is
        consumed once, after which each further ``reset()`` clamps to the last seed and
        flags ``info['curriculum_exhausted'] = True``.
    env_factory:
        Optional ``(episode_index) -> gym.Env`` builder used to rebuild the underlying
        env per episode (e.g. to rotate a construction-time ``distribution_mode``). When
        ``None`` a single :class:`SharpeArenaEnv` is built from ``env_kwargs`` and reused
        (``reset(seed=...)`` re-points it at the next scenario).
    **env_kwargs:
        Forwarded to :class:`SharpeArenaEnv` when ``env_factory`` is ``None``.
    """

    def __init__(
        self,
        seeds: Optional[Sequence[int]] = None,
        *,
        schedule: str = "sequential",
        n_episodes: Optional[int] = None,
        loop: bool = True,
        env_factory: Optional[EnvFactory] = None,
        **env_kwargs,
    ) -> None:
        if seeds is not None:
            self._seeds: list[int] = [int(s) for s in seeds]
            if not self._seeds:
                raise ValueError("seeds must be non-empty")
        else:
            if schedule not in ("sequential", "chained"):
                raise ValueError("schedule must be 'sequential' or 'chained'")
            if n_episodes is None or n_episodes < 1:
                raise ValueError("n_episodes (>= 1) is required when seeds is None")
            base = int(env_kwargs.get("seed", 0))
            step = 1 if schedule == "sequential" else _CHAIN_STEP
            self._seeds = [base + k * step for k in range(int(n_episodes))]

        self._schedule = schedule
        self._loop = bool(loop)
        self._env_factory = env_factory
        self._episode = 0
        self.curriculum_exhausted = False

        env = env_factory(0) if env_factory is not None else SharpeArenaEnv(**env_kwargs)
        super().__init__(env)

    @property
    def seeds(self) -> list[int]:
        """The resolved scenario-seed schedule (copy)."""
        return list(self._seeds)

    @property
    def schedule(self) -> str:
        return self._schedule

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset onto the next scheduled scenario seed (the curriculum drives the seed;
        any externally supplied ``seed`` is ignored so the sequence stays deterministic).
        """
        n = len(self._seeds)
        pos = self._episode
        exhausted = False
        if pos >= n:
            if self._loop:
                idx = pos % n
            else:
                exhausted = True
                idx = n - 1
        else:
            idx = pos
        self.curriculum_exhausted = exhausted

        active_seed = self._seeds[idx]
        if self._env_factory is not None:
            self.env = self._env_factory(idx)
        obs, info = self.env.reset(seed=active_seed)

        curriculum = {"index": idx, "seed": active_seed, "schedule": self._schedule}
        mode = getattr(self.env, "_distribution_mode", None)
        if mode is not None:
            curriculum["distribution_mode"] = mode
        info["curriculum"] = curriculum
        if exhausted:
            info["curriculum_exhausted"] = True

        self._episode += 1
        return obs, info


def regime_curriculum(
    base_seed: int,
    length: int,
    *,
    distribution_modes: Sequence[str] = ("calm", "hard", "extreme"),
    schedule: str = "chained",
    **env_kwargs,
) -> CurriculumEnv:
    """A :class:`CurriculumEnv` whose successive episodes rotate the difficulty tier.

    Episode ``k`` runs ``distribution_modes[k % len(distribution_modes)]`` over a
    chained scenario seed — the literal "calm → shock → recovery" curriculum. Because
    ``distribution_mode`` is fixed at :class:`SharpeArenaEnv` construction, each episode
    gets a freshly built env (via ``env_factory``); the seed for the episode is then
    applied by the wrapper's ``reset``.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if not distribution_modes:
        raise ValueError("distribution_modes must be non-empty")
    if schedule not in ("sequential", "chained"):
        raise ValueError("schedule must be 'sequential' or 'chained'")

    step = 1 if schedule == "sequential" else _CHAIN_STEP
    seeds = [int(base_seed) + k * step for k in range(int(length))]
    modes = list(distribution_modes)

    def _factory(idx: int) -> gym.Env:
        return SharpeArenaEnv(
            distribution_mode=modes[idx % len(modes)],
            seed=seeds[idx],
            **env_kwargs,
        )

    return CurriculumEnv(
        seeds=seeds,
        schedule="regime",
        loop=True,
        env_factory=_factory,
    )


# -- adaptive difficulty-targeting curriculum (Prioritized Level Replay) ----------


class AdaptiveScheduler:
    """Prioritized-Level-Replay difficulty targeting over a fixed candidate level set.

    A fixed rotation replays trivially-solved levels and hopeless ones in equal measure.
    PLR instead spends the next episode on a level in the agent's *zone of proximal
    development*: one it solves *sometimes* (the 30-70%-solve band), where the learning
    signal is richest. This tracks a per-level solve rate from recorded outcomes and
    scores each level by the ZPD weight ``p * (1 - p)`` (Bernoulli variance): maximal at
    ``p = 0.5``, decaying to zero as a level becomes trivially easy (``p -> 1``) or
    hopeless (``p -> 0``). :meth:`select_next` is a **pure deterministic function of the
    recorded history** (argmax weight, ties broken by lowest index, no RNG). Unseen levels
    take a ``prior`` pseudo-rate (default ``0.5``, the peak) so each is explored once
    before the mid band is replayed.
    """

    def __init__(self, levels: Sequence[int], *, prior: float = 0.5) -> None:
        deduped: list[int] = []
        for x in levels:
            xi = int(x)
            if xi not in deduped:
                deduped.append(xi)
        if not deduped:
            raise ValueError("levels must be non-empty")
        self._levels = deduped
        self._solves: dict[int, int] = {x: 0 for x in deduped}
        self._attempts: dict[int, int] = {x: 0 for x in deduped}
        self._prior = float(prior)

    @property
    def levels(self) -> list[int]:
        """The scheduled candidate levels (seeds), in tie-break order (copy)."""
        return list(self._levels)

    def success_rate(self, level: int) -> float:
        """Observed ``solves / attempts`` for ``level``, or the ``prior`` when unseen."""
        level = int(level)
        if level not in self._attempts:
            raise KeyError(f"level {level} is off-schedule")
        a = self._attempts[level]
        return self._prior if a == 0 else self._solves[level] / a

    def weight(self, level: int) -> float:
        """ZPD replay weight ``p * (1 - p)`` (peaks at ``p = 0.5``, zero at both tails)."""
        p = self.success_rate(level)
        return p * (1.0 - p)

    def record(self, level: int, solved: bool) -> None:
        """Record one episode outcome for ``level`` (``solved`` = success criterion met)."""
        level = int(level)
        if level not in self._attempts:
            raise KeyError(f"level {level} is off-schedule")
        self._attempts[level] += 1
        self._solves[level] += 1 if solved else 0

    def select_next(self) -> int:
        """The next level to replay: highest-weight candidate, ties broken by lowest index."""
        best = self._levels[0]
        best_w = self.weight(best)
        for lv in self._levels[1:]:
            w = self.weight(lv)
            if w > best_w:
                best, best_w = lv, w
        return best


class AdaptiveCurriculumEnv(gym.Wrapper):
    """A curriculum whose next scenario seed is chosen adaptively by the agent's online
    success rate (Prioritized Level Replay) rather than a fixed rotation.

    On every ``reset()`` the wrapper asks an :class:`AdaptiveScheduler` for the
    highest-learning-signal (mid-difficulty) level and points the env at that seed; as the
    episode runs it accumulates reward, and on episode end it records a solved/failed
    outcome (``solved_fn(total_reward)``, default: a net-positive episode return) back into
    the scheduler. The seed choice is deterministic given the observed outcome history, so
    the same run replays identically.

    Parameters
    ----------
    levels:
        The candidate scenario seeds to target adaptively.
    solved_fn:
        ``(total_episode_return) -> bool`` success criterion. Defaults to "made money"
        (``total > 0``), a deterministic proxy for a trading "solve".
    prior:
        Unseen-level pseudo success rate (default ``0.5``, the ZPD peak).
    env_factory:
        Optional ``(seed) -> gym.Env`` builder rebuilt per episode (e.g. to fix a
        construction-time ``distribution_mode``). When ``None`` a single
        :class:`SharpeArenaEnv` is built from ``env_kwargs`` and re-pointed via
        ``reset(seed=...)``.
    **env_kwargs:
        Forwarded to :class:`SharpeArenaEnv` when ``env_factory`` is ``None``.
    """

    def __init__(
        self,
        levels: Sequence[int],
        *,
        solved_fn: Optional[Callable[[float], bool]] = None,
        prior: float = 0.5,
        env_factory: Optional[EnvFactory] = None,
        **env_kwargs,
    ) -> None:
        self._scheduler = AdaptiveScheduler(levels, prior=prior)
        self._solved_fn = solved_fn or (lambda total: total > 0.0)
        self._env_factory = env_factory
        self._active_seed: Optional[int] = None
        self._episode_return = 0.0

        first = self._scheduler.levels[0]
        env = env_factory(first) if env_factory is not None else SharpeArenaEnv(**env_kwargs)
        super().__init__(env)

    @property
    def scheduler(self) -> AdaptiveScheduler:
        return self._scheduler

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset onto the scheduler's next (mid-difficulty) seed; any external ``seed`` is
        ignored so the adaptive sequence stays deterministic in the outcome history."""
        active = self._scheduler.select_next()
        self._active_seed = active
        self._episode_return = 0.0
        if self._env_factory is not None:
            self.env = self._env_factory(active)
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=active)
        info["curriculum"] = {
            "seed": active,
            "success_rate": self._scheduler.success_rate(active),
            "weight": self._scheduler.weight(active),
        }
        return obs, info

    def step(self, action):
        """Advance one bar; on episode end record the solved/failed outcome for the seed."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        if bool(terminated) or bool(truncated):
            solved = bool(self._solved_fn(self._episode_return))
            self._scheduler.record(self._active_seed, solved)
            info["curriculum_solved"] = solved
        return obs, reward, terminated, truncated, info


__all__ = [
    "CurriculumEnv",
    "regime_curriculum",
    "AdaptiveScheduler",
    "AdaptiveCurriculumEnv",
]
