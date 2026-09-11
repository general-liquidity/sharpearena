"""Guards on published guarantees must survive ``python -O``.

``assert`` statements are removed outright by the CPython compiler under ``-O`` or
``PYTHONOPTIMIZE``, so a guard written as ``assert`` is absent from the configuration
a user can run. Two of this package's published guarantees were carried that way:
the train/test seed disjointness of ``train_test_seeds`` (ARENA-REVIEW A2) and the
held-out band membership of the named eval seeds.

pytest itself never runs under ``-O`` (it rewrites assertions and refuses the flag's
effects), so these tests re-enter a fresh interpreter with the flag set. That is the
point: an in-process test proves nothing about the configuration the defect lived in.
Each subprocess asserts on the *refusal*, printed as a single token, so the only thing
that can make the test pass is the guard firing. A subprocess that died for any other
reason prints nothing and fails the comparison, and its stderr is surfaced.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("numpy")


def _run_optimized(body: str) -> str:
    """Execute ``body`` in a fresh ``python -O`` and return its stdout, stripped."""
    proc = subprocess.run(
        [sys.executable, "-O", "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"the -O subprocess exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout.strip()


def test_train_test_seeds_refuses_a_negative_gap_under_dash_o():
    """A negative gap collapses the bands onto each other; -O must still refuse it.

    Under the old ``assert`` the range check and the disjointness check vanished
    together and this call returned two identical 256-seed bands, so the
    generalization gap over them was zero by construction.
    """
    out = _run_optimized(
        """
        import sys
        if sys.flags.optimize < 1:
            raise SystemExit("the -O flag did not reach the child")
        # Proof the stripping is real: this assert never fires under -O.
        assert False, "unreachable under -O"
        from sharpearena.generalization import train_test_seeds
        try:
            train, test = train_test_seeds(256, 256, 0, -256)
        except ValueError as exc:
            print("REFUSED:" + type(exc).__name__)
        else:
            print("RETURNED:overlap=" + str(len(set(train) & set(test))))
        """
    )
    assert out == "REFUSED:ValueError", out


def test_train_test_seeds_still_splits_under_dash_o():
    """The refusal did not turn a legal split into a failure."""
    out = _run_optimized(
        """
        from sharpearena.generalization import train_test_seeds
        train, test = train_test_seeds(8, 4, 0, 10_000)
        print(str(len(train)) + "," + str(len(test)) + "," +
              str(len(set(train) & set(test))))
        """
    )
    assert out == "8,4,0", out


def test_eval_seed_band_membership_is_enforced_under_dash_o():
    """The named eval seeds are checked against the held-out band at import time."""
    out = _run_optimized(
        """
        import sharpearena.eval_seeds as es
        # The committed mapping imported cleanly, so the guard accepts what ships.
        es._require_held_out_band(es.EVAL_SEEDS)
        below = dict(es.EVAL_SEEDS)
        below["held_out_00"] = es.EVAL_SEED_BASE - 1
        try:
            es._require_held_out_band(below)
        except ValueError as exc:
            print("REFUSED:" + type(exc).__name__)
        else:
            print("ACCEPTED")
        """
    )
    assert out == "REFUSED:ValueError", out
