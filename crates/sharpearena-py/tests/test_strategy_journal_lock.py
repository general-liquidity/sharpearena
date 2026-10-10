"""Bounded acquisition and unchanged ownership for strategy evidence journals."""

import errno
import os
import threading
import time
import types
import sys

import pytest
from sharpearena import strategy_generation as sg


def test_default_acquisition_has_a_deadline(tmp_path, monkeypatch):
    """A fake clock makes indefinite retry fail immediately, without hanging a test."""
    elapsed = [0.0]
    attempts = []
    monkeypatch.setattr(sg.time, "monotonic", lambda: elapsed[0])

    def pause(seconds):
        elapsed[0] += seconds
        if elapsed[0] > 6:
            raise AssertionError("acquisition continued beyond its default deadline")

    monkeypatch.setattr(sg.time, "sleep", pause)
    if os.name == "nt":
        import msvcrt

        def busy(fd, mode, size):
            attempts.append(mode)
            raise OSError(errno.EACCES, "busy")

        monkeypatch.setattr(msvcrt, "locking", busy)
    else:
        import fcntl

        def busy(fd, mode):
            assert mode & fcntl.LOCK_NB, "acquisition must not block inside the OS"
            attempts.append(mode)
            raise OSError(errno.EAGAIN, "busy")

        monkeypatch.setattr(fcntl, "flock", busy)
    with pytest.raises(TimeoutError, match="remained busy") as error:
        with sg._journal_lock(tmp_path / "journal.jsonl"):
            pytest.fail("busy owner was bypassed")
    assert error.value.code == "journal_busy"
    assert elapsed[0] == pytest.approx(5.0)
    assert attempts


def test_real_owner_is_preserved_after_contender_times_out(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("existing evidence\n", encoding="utf-8")
    outcomes = []

    def contender():
        start = time.monotonic()
        try:
            with sg._journal_lock(path, timeout=0.1):
                outcomes.append("unexpected acquisition")
        except TimeoutError as error:
            outcomes.append((error.code, time.monotonic() - start))

    with sg._journal_lock(path):
        worker = threading.Thread(target=contender, daemon=True)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(outcomes) == 1
        assert outcomes[0][0] == "journal_busy"
        assert 0.08 <= outcomes[0][1] < 2
        with pytest.raises(TimeoutError):
            with sg._journal_lock(path, timeout=0):
                pytest.fail("timed out contender broke the owner's lock")
    assert path.read_text(encoding="utf-8") == "existing evidence\n"
    with sg._journal_lock(path, timeout=0):
        pass


@pytest.mark.parametrize("timeout", [-1, True, "1", float("nan"), float("inf")])
def test_invalid_timeout_is_rejected_before_creating_lock(tmp_path, timeout):
    path = tmp_path / "journal.jsonl"
    with pytest.raises(ValueError, match="finite non-negative"):
        with sg._journal_lock(path, timeout=timeout):
            pass
    assert not path.with_name(path.name + ".lock").exists()


def test_non_contention_errors_propagate_without_waiting(tmp_path, monkeypatch):
    expected = OSError(errno.EIO, "device failure")

    def broken(*args):
        raise expected

    if os.name == "nt":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", broken)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", broken)
    monkeypatch.setattr(
        sg.time, "sleep", lambda seconds: pytest.fail("unrelated error retried")
    )
    with pytest.raises(OSError) as error:
        with sg._journal_lock(tmp_path / "journal.jsonl"):
            pass
    assert error.value is expected


def test_busy_append_exposes_unpersisted_outcome_without_touching_census(tmp_path):
    path = tmp_path / "journal.jsonl"
    record = {"status": "completed", "result": "synthetic"}
    with sg._journal_lock(path):
        with pytest.raises(TimeoutError) as error:
            sg._append_with_census(path, record, {}, (0, 1, 1), True, 1, lock_timeout=0)
    assert error.value.pending_record is record
    assert "test_split_census" not in record
    assert not path.exists()


def test_runner_timeout_is_configurable_and_validated():
    runner = sg.StrategySearchRunner(object(), journal_lock_timeout=0.25)
    assert runner.journal_lock_timeout == 0.25
    with pytest.raises(ValueError, match="finite non-negative"):
        sg.StrategySearchRunner(object(), journal_lock_timeout=float("nan"))


def test_posix_acquisition_uses_nonblocking_operation_and_releases(
    tmp_path, monkeypatch
):
    """Exercise the POSIX branch even when the local platform is Windows."""
    operations = []

    def flock(fd, operation):
        operations.append(operation)
        if len(operations) == 1:
            raise OSError(errno.EAGAIN, "busy")

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=flock),
    )
    monkeypatch.setattr(sg, "os", types.SimpleNamespace(name="posix"))
    monkeypatch.setattr(sg.time, "sleep", lambda seconds: None)
    with sg._journal_lock(tmp_path / "journal.jsonl"):
        assert operations == [6, 6]
    assert operations == [6, 6, 8]
