"""Enumerating candidate strategies.

The search space is stop count x pit laps x compound sequence. Enumerated
exhaustively it is large: a 57-lap race with two stops has around 1,500 legal
pit-lap pairs, times the compound sequences, times three stop counts. Scoring
all of that at 12,000 races each would take hours.

So the search is coarse-to-fine. Candidates are first laid out on a lap grid
(every third lap by default), screened with a smaller ensemble, and then the
best handful are refined lap by lap in a window around their screened optimum
and re-scored with the full ensemble. This is not guaranteed to find the global
optimum, but the objective is smooth in pit lap -- moving a stop by one lap
changes expected points by a few hundredths -- so the risk of the grid stepping
over a sharp peak is low. Where it is not smooth is around a safety car, and
that is precisely what the reactive policy exists to handle.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence

from pitwall.sim.strategy import Strategy

log = logging.getLogger(__name__)

__all__ = ["enumerate_candidates", "refine_around"]


def _compound_sequences(
    compounds: Sequence[str], n_stints: int, require_two: bool
) -> list[tuple[str, ...]]:
    """All compound orderings for a given number of stints.

    The two-compound rule is applied here rather than filtered later, so the
    screening ensemble is never spent on strategies that would be excluded from
    the results.
    """
    sequences = []
    for combo in itertools.product(compounds, repeat=n_stints):
        if require_two and len(set(combo)) < 2:
            continue
        sequences.append(combo)
    return sequences


def enumerate_candidates(
    race_laps: int,
    stop_counts: Sequence[int],
    compounds: Sequence[str],
    lap_grid_step: int = 3,
    min_stint_laps: int = 6,
    require_two_compounds: bool = True,
    start_compound: str | None = None,
    allowed_sequences: Sequence[tuple[str, ...]] | None = None,
) -> list[Strategy]:
    """Candidate strategies on a coarse lap grid.

    ``start_compound`` pins the opening stint, which is what you want when the
    car starts on the tyre it qualified on and has no choice about it.
    ``allowed_sequences`` restricts the compound orderings to those that
    survived :func:`compound_shortlist`.
    """
    candidates: list[Strategy] = []
    allowed = set(allowed_sequences) if allowed_sequences is not None else None

    for n_stops in stop_counts:
        if n_stops < 0:
            continue
        n_stints = n_stops + 1
        if n_stints * min_stint_laps > race_laps:
            continue

        # Legal pit laps, on the grid, leaving room for a minimum stint either
        # side of every stop.
        #
        # The grid is coarsened in proportion to the stop count. The number of
        # lap combinations is C(grid_points, n_stops), so a grid fine enough to
        # be sensible for a one-stopper produces 560 combinations for a
        # three-stopper and the screening pass stops being affordable. Widening
        # the step keeps the count flat across stop counts, and stage 3 refines
        # the winner lap by lap regardless.
        step = max(1, lap_grid_step) * max(1, n_stops)
        earliest = min_stint_laps
        latest = race_laps - min_stint_laps
        grid = list(range(earliest, latest + 1, step))

        for stops in itertools.combinations(grid, n_stops):
            lengths = _lengths(stops, race_laps)
            if min(lengths) < min_stint_laps:
                continue
            for sequence in _compound_sequences(compounds, n_stints, require_two_compounds):
                if start_compound is not None and sequence[0] != start_compound:
                    continue
                if allowed is not None and sequence not in allowed:
                    continue
                candidates.append(Strategy(compounds=sequence, stops=tuple(stops)))

    log.info(
        "enumerated %d candidates (%d laps, stops=%s, grid step %d)",
        len(candidates),
        race_laps,
        list(stop_counts),
        lap_grid_step,
    )
    return candidates


def _lengths(stops: Sequence[int], race_laps: int) -> list[int]:
    edges = [0, *stops, race_laps]
    return [b - a for a, b in itertools.pairwise(edges)]


def even_split_stops(race_laps: int, n_stops: int) -> tuple[int, ...]:
    """Pit laps that divide the race into equal stints."""
    return tuple(round(race_laps * (i + 1) / (n_stops + 1)) for i in range(n_stops))


def compound_shortlist(
    race_laps: int,
    stop_counts: Sequence[int],
    compounds: Sequence[str],
    require_two_compounds: bool = True,
    start_compound: str | None = None,
) -> list[Strategy]:
    """One representative strategy per (stop count, compound sequence).

    The full product of stop counts, pit laps and compound sequences is far too
    large to screen: three stops on a 57-lap race with a three-lap grid is over
    forty thousand candidates before anything has been simulated. But the two
    choices are close to separable. Which compounds to run is driven by their
    relative pace and wear, and *when* to stop is driven by pit loss and
    traffic; the interaction between them is second order.

    So the search is staged. Here every compound sequence is evaluated once at
    an even stint split, which is enough to rank sequences. The winners then get
    a full pit-lap search in :func:`enumerate_candidates`, and the best of those
    get a lap-by-lap refinement. That turns a product into a sum and takes the
    candidate count from tens of thousands to a few hundred.

    The cost is that a compound sequence which is only good at an unusual stint
    split can be screened out. In practice sequences are ranked robustly at the
    even split, and the refinement stage recovers the lap placement.
    """
    out: list[Strategy] = []
    for n_stops in stop_counts:
        if n_stops < 0:
            continue
        stops = even_split_stops(race_laps, n_stops)
        if len(set(stops)) != len(stops) or (stops and max(stops) >= race_laps):
            continue
        for sequence in _compound_sequences(compounds, n_stops + 1, require_two_compounds):
            if start_compound is not None and sequence[0] != start_compound:
                continue
            out.append(Strategy(compounds=sequence, stops=stops))
    return out


def refine_around(
    strategy: Strategy,
    race_laps: int,
    radius: int = 3,
    min_stint_laps: int = 6,
) -> list[Strategy]:
    """Every lap-level perturbation of ``strategy`` within ``radius``.

    Compound sequence is held fixed: it is a discrete choice the screening pass
    has already made, and re-opening it here would multiply the refinement cost
    by the number of sequences for no benefit.
    """
    if not strategy.stops:
        return [strategy]

    windows = [
        range(max(1, stop - radius), min(race_laps - 1, stop + radius) + 1)
        for stop in strategy.stops
    ]

    out: list[Strategy] = []
    for stops in itertools.product(*windows):
        if list(stops) != sorted(stops) or len(set(stops)) != len(stops):
            continue
        if min(_lengths(stops, race_laps)) < min_stint_laps:
            continue
        out.append(Strategy(compounds=strategy.compounds, stops=tuple(stops)))
    return out
