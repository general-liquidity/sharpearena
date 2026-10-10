"""Joint Fisher-Yates construction coverage without probabilistic pass criteria.

Each legal choice tuple is exercised through the actual seat encoder. Uniform
independent choices imply uniform permutations by this bijection. This does not
prove SplitMix outputs independent uniform words over an arbitrary seed sample.
The finite 53-bit mapping's bucket imbalance is characterized separately.
"""

from collections import Counter
from itertools import permutations, product

import pytest

import sharpearena.lob_env as lob

WORDS = 1 << 53


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_every_shuffle_choice_tuple_encodes_one_distinct_permutation(monkeypatch, n):
    env = lob.LOBMarketEnv(n, n_steps=1, seed=17, priority="seeded_shuffle")
    env.reset(seed=17)
    noise_state = env._rng
    seen = Counter()
    bounds = list(range(n, 1, -1))
    for choices in product(*(range(bound) for bound in bounds)):
        words = iter(
            _ceildiv(choice * WORDS, bound) << 11
            for choice, bound in zip(choices, bounds)
        )
        calls = []

        def draw(state):
            calls.append(state)
            return state + 1, next(words)

        monkeypatch.setattr(lob, "_splitmix_bits", draw)
        codes = env._seat_codes()
        assert len(calls) == n - 1
        assert sorted(code // n for code in codes) == list(range(n))
        assert [code % n for code in codes] == list(range(n))
        assert [env._owner(code) for code in codes] == list(range(n))
        order = tuple(sorted(range(n), key=codes.__getitem__))
        seen[order] += 1
        assert env._rng == noise_state
    assert seen == Counter({order: 1 for order in permutations(range(n))})


@pytest.mark.parametrize("bound", [2, 3, 4, 5, 6])
def test_53bit_choice_mapping_has_all_buckets_with_at_most_one_word_imbalance(bound):
    sizes = []
    for choice in range(bound):
        start = _ceildiv(choice * WORDS, bound)
        stop = _ceildiv((choice + 1) * WORDS, bound)
        assert (start * bound) >> 53 == choice
        assert ((stop - 1) * bound) >> 53 == choice
        sizes.append(stop - start)
    assert sum(sizes) == WORDS
    assert min(sizes) > 0
    assert max(sizes) - min(sizes) <= 1
    assert (max(sizes) == min(sizes)) == (WORDS % bound == 0)


def test_first_seat_and_pairwise_marginals_do_not_identify_the_joint_law():
    # Uniformly selecting only even four-seat permutations has perfect old
    # marginals but omits half the permutations. This is a coverage witness,
    # not a claim that the production generator has this distribution.
    even = [
        order
        for order in permutations(range(4))
        if sum(order[a] > order[b] for a in range(4) for b in range(a + 1, 4)) % 2 == 0
    ]
    assert len(even) == 12
    assert Counter(order[0] for order in even) == Counter(
        {seat: 3 for seat in range(4)}
    )
    for a in range(4):
        for b in range(a + 1, 4):
            assert sum(order.index(a) < order.index(b) for order in even) == 6
    assert len(set(even)) == 12 < 24
