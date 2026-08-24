"""The exact-output inversion result, and the deliberate observation boundary."""

from sharpearena.splitmix_inversion import (
    candidate_states_from_unit,
    mix64,
    next_word,
    unmix64,
)


def test_mix64_is_exactly_invertible_on_representative_words():
    for word in (0, 1, 7, 0x123456789ABCDEF0, (1 << 64) - 1):
        assert unmix64(mix64(word)) == word


def test_one_published_unit_leaves_exactly_2048_candidate_states():
    state, output = next_word(0x123456789ABCDEF0)
    unit = (output >> 11) / float(1 << 53)
    candidates = candidate_states_from_unit(unit)
    assert len(candidates) == 2048
    assert state in candidates
    assert all(0 <= s < 1 << 64 for s in candidates)


def test_candidate_states_reproduce_the_observed_upper_53_bits():
    state, output = next_word(987654321)
    unit = (output >> 11) / float(1 << 53)
    expected = output >> 11
    assert all(mix64(s) >> 11 == expected for s in candidate_states_from_unit(unit))
