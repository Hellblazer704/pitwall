"""Seed management.

Everything stochastic in pitwall takes a generator from here rather than
touching the global numpy random state. The rule is that a run is defined by
one integer in the Hydra config, and any two runs with that integer equal
produce byte-identical outputs regardless of how the work was split across
processes.

The mechanism is :class:`numpy.random.SeedSequence` spawning. A parent sequence
is derived from the config seed and a stable string label per subsystem, and
children are spawned by index. Spawning is deterministic and does not depend on
the order in which children are consumed, which is what makes the Monte Carlo
ensembles parallelisable without changing results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

__all__ = ["SeedBank", "label_to_int", "spawn_generators"]


def label_to_int(label: str) -> int:
    """Map a subsystem label to a stable 64-bit integer.

    ``hash()`` is salted per process, so it cannot be used for anything that
    has to reproduce across runs.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def spawn_generators(seed_sequence: np.random.SeedSequence, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent generators from ``seed_sequence``."""
    return [np.random.default_rng(child) for child in seed_sequence.spawn(n)]


class SeedBank:
    """Hands out independent generators keyed by subsystem label.

    Two different labels never share a stream, and the same label always
    returns the same stream for a given root seed.
    """

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._root = np.random.SeedSequence(self._seed)

    @property
    def seed(self) -> int:
        return self._seed

    def sequence(self, label: str) -> np.random.SeedSequence:
        """A child sequence for ``label``, derived from the root seed."""
        return np.random.SeedSequence(entropy=self._seed, spawn_key=(label_to_int(label),))

    def generator(self, label: str) -> np.random.Generator:
        """A single generator for ``label``."""
        return np.random.default_rng(self.sequence(label))

    def generators(self, label: str, n: int) -> list[np.random.Generator]:
        """``n`` independent generators for ``label``, e.g. one per MC worker."""
        return spawn_generators(self.sequence(label), n)

    def chain_generators(self, label: str, chains: int) -> list[np.random.Generator]:
        """Per-chain generators for an MCMC run."""
        return self.generators(f"{label}/chains", chains)

    def child(self, label: str) -> SeedBank:
        """A sub-bank, so a component can namespace its own labels."""
        bank = SeedBank.__new__(SeedBank)
        bank._seed = self._seed
        bank._root = self.sequence(label)
        return bank

    def __repr__(self) -> str:
        return f"SeedBank(seed={self._seed})"


def as_generator(source: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce a seed-ish thing into a generator.

    Convenience for library functions that want to accept either. ``None``
    means "unseeded", which is only acceptable in interactive use, never in a
    run that gets reported.
    """
    if isinstance(source, np.random.Generator):
        return source
    return np.random.default_rng(source)


def stable_index(values: Sequence[str]) -> dict[str, int]:
    """Sorted label to contiguous index.

    Model coefficient vectors are indexed by circuit, compound and driver. If
    that mapping depended on the order rows happened to arrive in, a refit on
    reordered data would produce a different-looking posterior for the same
    model, and cached fits would silently misalign.
    """
    return {value: i for i, value in enumerate(sorted(set(values)))}
