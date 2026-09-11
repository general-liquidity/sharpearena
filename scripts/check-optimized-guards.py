"""Check the package's published guarantees under `python -O`, with no pytest.

`assert` statements are removed outright by the CPython compiler under `-O` or
`PYTHONOPTIMIZE`, so a guarantee carried by an assertion does not exist in a
configuration a user can run. ARENA-REVIEW A2 found the train/test seed disjointness
of `train_test_seeds` carried that way, together with the range check that would have
caught the input which breaks it.

Run this file with `python -O`. It refuses to run without the flag, because a check
for stripped assertions that runs unstripped proves nothing. It uses no `assert`
statements itself for the same reason: every check here is an `if` and a `raise`, so
the script cannot pass vacuously under the flag it exists to exercise.

    python -O scripts/check-optimized-guards.py
"""

from __future__ import annotations

import sys


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if sys.flags.optimize < 1:
        fail("run this under `python -O`; without the flag it checks nothing")

    from sharpearena.generalization import train_test_seeds

    # A negative gap collapses the two bands onto each other. Under the old bare
    # `assert` the range check and the disjointness check vanished together and this
    # returned two identical 256-seed bands labelled train and test.
    try:
        train, test = train_test_seeds(256, 256, 0, -256)
    except ValueError:
        pass
    else:
        overlap = len(set(train) & set(test))
        fail(
            f"train_test_seeds(256, 256, 0, -256) returned bands sharing "
            f"{overlap} seeds instead of refusing"
        )

    # The refusal did not turn a legal split into a failure.
    train, test = train_test_seeds(8, 4, 0, 10_000)
    if len(train) != 8 or len(test) != 4 or set(train) & set(test):
        fail(f"a legal split was mangled: train={train} test={test}")

    import sharpearena.eval_seeds as eval_seeds

    # Importing the module already ran the band guard over the committed mapping.
    # Re-run it over a tampered one to show the guard is present, not merely quiet.
    below = dict(eval_seeds.EVAL_SEEDS)
    below["held_out_00"] = eval_seeds.EVAL_SEED_BASE - 1
    try:
        eval_seeds._require_held_out_band(below)
    except ValueError:
        pass
    else:
        fail("a named eval seed below EVAL_SEED_BASE was accepted")

    from sharpearena.mandate import MandateError, require_mandate
    from sharpearena.verifiers_env import mandate_reward

    # ARENA-REVIEW A6. A present-but-malformed mandate used to read as "no mandate",
    # which the reward layer turns into full credit. The refusal is an `if` and a
    # `raise`, so it must survive the flag that strips assertions. The payload violates
    # exactly one rule (the style), so nothing else here can be what refused it.
    malformed = {"style": "momentum_v2"}
    try:
        require_mandate(malformed)
    except MandateError:
        pass
    else:
        fail("a mandate with an unrecognized style was accepted")
    try:
        reward = mandate_reward(state={"mandate": malformed})
    except MandateError:
        pass
    else:
        fail(f"a malformed mandate scored {reward} instead of refusing")

    # The absent case is a different state and keeps its vacuous full credit, so the
    # check above is about the malformed payload and not about mandates in general.
    if mandate_reward(state={}) != 1.0:
        fail("a scenario with no mandate must stay vacuously satisfied")

    from sharpearena.preprocessing import ExecutionNoiseConfig

    # ARENA-REVIEW A8. A NaN execution-noise knob passed every guard in the core and
    # reported itself as enabled through `nan != 0.0`. One knob per case, the other left
    # at its in-range default.
    for kwargs in ({"delay_prob": float("nan")}, {"slippage_bps": -0.1}):
        try:
            ExecutionNoiseConfig(**kwargs)
        except ValueError:
            continue
        fail(f"ExecutionNoiseConfig({kwargs}) was accepted")
    if ExecutionNoiseConfig(slippage_bps=25.0).enabled is not True:
        fail("a legal execution-noise knob was mangled by the refusal")

    print("OK: published guarantees still refuse under -O")


if __name__ == "__main__":
    main()
