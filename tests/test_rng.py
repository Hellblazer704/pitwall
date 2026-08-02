"""Seed discipline.

The whole project's reproducibility claim rests on this module, so these tests
are about guarantees rather than coverage.
"""

from __future__ import annotations

import numpy as np

from pitwall.rng import SeedBank, label_to_int, stable_index


def test_same_seed_same_stream() -> None:
    a = SeedBank(42).generator("degradation").standard_normal(16)
    b = SeedBank(42).generator("degradation").standard_normal(16)
    assert np.array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = SeedBank(42).generator("degradation").standard_normal(16)
    b = SeedBank(43).generator("degradation").standard_normal(16)
    assert not np.array_equal(a, b)


def test_labels_are_independent_streams() -> None:
    bank = SeedBank(42)
    assert not np.array_equal(
        bank.generator("degradation").standard_normal(16),
        bank.generator("simulator").standard_normal(16),
    )


def test_label_stream_does_not_depend_on_call_order() -> None:
    """A subsystem's stream must not shift because another one was built first.

    This is what allows the Monte Carlo work to be reordered or parallelised
    without changing results.
    """
    first = SeedBank(9)
    a_then_b = (
        first.generator("alpha").standard_normal(8),
        first.generator("beta").standard_normal(8),
    )
    second = SeedBank(9)
    b_then_a = (
        second.generator("beta").standard_normal(8),
        second.generator("alpha").standard_normal(8),
    )
    assert np.array_equal(a_then_b[0], b_then_a[1])
    assert np.array_equal(a_then_b[1], b_then_a[0])


def test_spawned_generators_are_independent_and_ordered() -> None:
    bank = SeedBank(11)
    gens = bank.generators("mc", 4)
    draws = [g.standard_normal(32) for g in gens]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])

    # Same request, same streams, in the same order.
    again = [g.standard_normal(32) for g in SeedBank(11).generators("mc", 4)]
    for expected, actual in zip(draws, again, strict=True):
        assert np.array_equal(expected, actual)


def test_label_to_int_is_stable_across_processes() -> None:
    """Not Python's salted hash(), which changes between interpreter runs."""
    assert label_to_int("degradation") == label_to_int("degradation")
    assert label_to_int("degradation") != label_to_int("simulator")


def test_stable_index_is_order_independent() -> None:
    assert stable_index(["b", "a", "c"]) == stable_index(["c", "b", "a"])
    assert stable_index(["b", "a"]) == {"a": 0, "b": 1}
