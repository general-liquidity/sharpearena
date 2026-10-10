# Evidence write boundaries

`StrategySearchRunner(..., journal_lock_timeout=5.0)` bounds journal lock
acquisition using a monotonic clock. A finite nonnegative timeout is required;
zero makes one immediate acquisition attempt. The existing owner retains its
lock. Non-contention IO errors propagate without retrying.

If the journal remains busy, `JournalBusyError` reports `code="journal_busy"`
and exposes `pending_record`: that outcome has not been appended or counted in
the test-split census. Retain it before deciding whether to retry persistence.
Do not rerun generation merely to make an IO retry appear successful. The
deadline covers acquisition, not evaluation or IO while holding the lock.

`compile_benchmark_evidence` requires every attempt to record a finite,
nonnegative numeric cost and its dataset to record finite positive
`periods_per_year`. Boolean values and numeric strings are not measurements.
An explicitly recorded zero remains zero; an absent cost is not substituted.
The manifest still states `local_compute_cost_included=False`: validating the
reported provider cost does not measure hardware, energy or opportunity cost.

`tests/test_paper_producer_regressions.py` discovers `paper/src/test_producer_*.py`
and executes their unittest cases from the canonical package test suite. Import
failures, empty suites, skipped cases and failing cases fail this check. These
tests use synthetic fixtures or frozen-input renders redirected to temporary
files, not new empirical benchmark fields or replacement published evidence.
