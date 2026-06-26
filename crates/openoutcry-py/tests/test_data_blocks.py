"""Tests for gap-aware contiguous-block episode sampling.

The block math is binding-free, so the module is loaded directly from its file
when the native package can't import (no maturin build). The env-construction
test skip-guards on the native binding::

    python -m pytest tests/test_data_blocks.py -q
"""

import numpy as np
import pytest

try:
    from openoutcry import data_blocks
except Exception:  # native binding not built — load the pure module by path
    import importlib.util
    import pathlib

    _path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "python"
        / "openoutcry"
        / "data_blocks.py"
    )
    _spec = importlib.util.spec_from_file_location("openoutcry_data_blocks", _path)
    data_blocks = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(data_blocks)

find_continuous_blocks = data_blocks.find_continuous_blocks
block_windows = data_blocks.block_windows
sample_block_window = data_blocks.sample_block_window
infer_cadence = data_blocks.infer_cadence

_GAPPED_CSV = (
    "date,symbol,close\n"
    "2025-01-01,AAA,10\n2025-01-01,BBB,20\n"
    "2025-01-02,AAA,11\n2025-01-02,BBB,19\n"
    "2025-01-03,AAA,12\n2025-01-03,BBB,21\n"
    "2025-01-04,AAA,13\n2025-01-04,BBB,22\n"
    "2025-01-05,AAA,14\n2025-01-05,BBB,23\n"
    "2025-06-01,AAA,30\n2025-06-01,BBB,40\n"  # large gap
    "2025-06-02,AAA,31\n2025-06-02,BBB,41\n"
)


# -- find_continuous_blocks -------------------------------------------------


def test_blocks_split_on_gap():
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-10", "2025-01-11"]
    assert find_continuous_blocks(dates, max_gap=1) == [(0, 3), (3, 5)]


def test_no_gap_is_single_block():
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert find_continuous_blocks(dates, max_gap=1) == [(0, 3)]


def test_numeric_index_blocks():
    assert find_continuous_blocks([0, 1, 2, 3, 10, 11], max_gap=1) == [(0, 4), (4, 6)]


def test_empty_timestamps():
    assert find_continuous_blocks([], max_gap=1) == []


def test_bad_max_gap_rejected():
    with pytest.raises(ValueError):
        find_continuous_blocks([0, 1, 2], max_gap=0)


def test_infer_cadence():
    assert infer_cadence(["2025-01-01", "2025-01-02", "2025-01-03"]) == 1.0
    assert infer_cadence([0, 2, 4, 6]) == 2.0


# -- block_windows ----------------------------------------------------------


def _within_one_block(window, blocks) -> bool:
    s, e = window["window_start"], window["window_end"]
    return any(bs <= s and e <= be for bs, be in blocks)


def test_windows_never_straddle_a_gap():
    blocks = [(0, 5), (5, 12)]
    windows = block_windows(blocks, horizon=2, warmup=1)  # window_len = 4
    assert windows
    for w in windows:
        s, e = w["window_start"], w["window_end"]
        assert e - s == 4
        assert _within_one_block(w, blocks)
        # the gap boundary sits at index 5; no window may span it
        assert not (s < 5 < e)


def test_short_blocks_discarded():
    blocks = [(0, 3), (5, 12)]  # first block too short for window_len 4
    windows = block_windows(blocks, horizon=2, warmup=1)
    assert windows
    assert all(w["window_start"] >= 5 for w in windows)


def test_window_length_includes_warmup_and_realization_bar():
    windows = block_windows([(0, 10)], horizon=3, warmup=2)
    assert all(w["window_end"] - w["window_start"] == 6 for w in windows)


# -- sample_block_window ----------------------------------------------------


def test_sample_is_deterministic_and_in_block():
    blocks = [(0, 5), (5, 30)]
    w1 = sample_block_window(blocks, horizon=3, warmup=2, seed=42)
    w2 = sample_block_window(blocks, horizon=3, warmup=2, seed=42)
    assert w1 == w2
    assert w1["window_end"] - w1["window_start"] == 6
    assert _within_one_block(w1, blocks)


def test_sample_explores_multiple_windows_across_seeds():
    blocks = [(0, 40)]
    picks = {
        (w["window_start"], w["window_end"])
        for w in (
            sample_block_window(blocks, horizon=3, warmup=2, seed=s) for s in range(50)
        )
    }
    assert len(picks) > 1
    assert all(0 <= s and e <= 40 for s, e in picks)


def test_sample_raises_when_no_block_fits():
    with pytest.raises(ValueError):
        sample_block_window([(0, 3), (4, 6)], horizon=5, warmup=2, seed=0)


# -- CSV segmentation + env construction ------------------------------------


def test_csv_segments_into_blocks():
    dates = data_blocks._csv_dates(_GAPPED_CSV)
    assert len(dates) == 7
    blocks = find_continuous_blocks(dates, max_gap=infer_cadence(dates) * 1.5)
    assert blocks == [(0, 5), (5, 7)]


def test_make_block_env_steps_without_crossing_gap():
    pytest.importorskip("openoutcry.openoutcry_py")
    env = data_blocks.make_block_env(_GAPPED_CSV, horizon=2, warmup=0, seed=3)
    obs, info = env.reset()
    for _ in range(2):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
