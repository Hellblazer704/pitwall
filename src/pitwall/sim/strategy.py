"""Strategy representation.

A strategy is a planned sequence of stints: which compound to start on, when to
stop, and what to fit each time. The simulator treats this as an *intention*
rather than a script -- a reactive policy can override it mid-race, and a car
that is still in the pit lane when the race ends obviously does not complete
the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

__all__ = ["Strategy", "plan_to_matrix"]


@dataclass(frozen=True)
class Strategy:
    """A planned stint sequence.

    ``compounds`` has one entry per stint, so it is always one longer than
    ``stops``. ``stops[i]`` is the lap at the *end* of which the car pits, so
    a stop on lap 20 means laps 1-20 on the first compound and lap 21 onwards
    on the second.
    """

    compounds: tuple[str, ...]
    stops: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.compounds) != len(self.stops) + 1:
            raise ValueError(
                f"{len(self.compounds)} compounds needs {len(self.compounds) - 1} stops, "
                f"got {len(self.stops)}"
            )
        if list(self.stops) != sorted(self.stops):
            raise ValueError(f"stop laps must be increasing, got {self.stops}")
        if len(set(self.stops)) != len(self.stops):
            raise ValueError(f"duplicate stop laps in {self.stops}")

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    def stint_lengths(self, race_laps: int) -> list[int]:
        edges = [0, *self.stops, race_laps]
        return [b - a for a, b in pairwise(edges)]

    def is_legal(self, race_laps: int, min_stint: int = 1, two_compounds: bool = True) -> bool:
        """Regulation and viability check.

        The two-compound rule is real: a dry race requires at least two
        different slick specifications, and ignoring it is the single easiest
        way to produce a strategy recommendation that would be disqualified.
        """
        if self.stops and max(self.stops) >= race_laps:
            return False
        if min(self.stint_lengths(race_laps)) < min_stint:
            return False
        return not (two_compounds and len(set(self.compounds)) < 2)

    def label(self) -> str:
        order = "-".join(compound[0] for compound in self.compounds)
        laps = ",".join(str(stop) for stop in self.stops)
        return f"{self.n_stops}stop {order} @{laps}" if laps else f"0stop {order}"

    def __str__(self) -> str:
        return self.label()


def plan_to_matrix(
    strategies: list[Strategy],
    race_laps: int,
    compound_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Expand strategies into per-lap arrays for the vectorised engine.

    Returns ``(pit_after_lap, compound_by_lap)``, both ``(n_strategies,
    race_laps)``. ``pit_after_lap[s, l]`` is True when strategy ``s`` pits at
    the end of lap ``l+1``; ``compound_by_lap[s, l]`` is the compound index the
    car is running *during* lap ``l+1``.
    """
    n = len(strategies)
    pit_after = np.zeros((n, race_laps), dtype=bool)
    compound = np.zeros((n, race_laps), dtype=np.int64)

    for s, strategy in enumerate(strategies):
        edges = [0, *strategy.stops, race_laps]
        for stint, (start, end) in enumerate(pairwise(edges)):
            index = compound_index[strategy.compounds[stint]]
            compound[s, start:end] = index
        for stop in strategy.stops:
            if 0 < stop <= race_laps:
                pit_after[s, stop - 1] = True

    return pit_after, compound
