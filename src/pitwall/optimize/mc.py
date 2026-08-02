"""Monte Carlo evaluation of candidate strategies.

Two things here matter more than the search itself.

**Common random numbers.** Every candidate is scored against the same sampled
races: the same safety-car schedule, the same degradation draws, the same
execution noise. Without that, comparing two strategies that differ by half a
tenth of expected points across 12,000 noisy races is comparing two noise
draws. With it, the difference between candidates is driven by the strategies
and the paired variance is a fraction of the unpaired variance, which is worth
roughly an order of magnitude in ensemble size.

**Distributions, not expectations.** The output is the full finishing-position
distribution. A strategy with a better mean finish can easily be the wrong
call: a team fighting for a championship cares about the left tail, a team
needing a result cares about P(podium), and the two can point in opposite
directions. Reporting a single number throws away the part of the answer a
strategist actually argues about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitwall.sim.engine import simulate_ensemble
from pitwall.sim.events import NeutralisationSchedule
from pitwall.sim.params import SimParams
from pitwall.sim.strategy import Strategy, plan_to_matrix

log = logging.getLogger(__name__)

__all__ = ["POINTS", "StrategyEvaluation", "evaluate_candidates"]

# Championship points for the top ten. This is the objective teams actually
# maximise, and it is very far from linear in finishing position: the gap
# between P1 and P2 is worth more than the gap between P6 and P10.
POINTS = np.array([25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0])


def points_for(positions: np.ndarray) -> np.ndarray:
    table = np.zeros(int(positions.max()) + 1)
    table[1 : min(len(POINTS), len(table) - 1) + 1] = POINTS[: len(table) - 1]
    return table[positions]


@dataclass
class StrategyEvaluation:
    """Scored outcome of one candidate over an ensemble."""

    strategy: Strategy
    positions: np.ndarray  # (n_races,)
    n_cars: int
    expected_points: float
    mean_position: float
    median_position: float
    p_win: float
    p_podium: float
    p_points: float
    position_distribution: np.ndarray
    # Monte Carlo standard error on expected points, so a reported difference
    # between candidates can be judged against the noise in the estimate.
    points_se: float
    extras: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.label(),
            "n_stops": self.strategy.n_stops,
            "compounds": "-".join(self.strategy.compounds),
            "stops": ",".join(str(s) for s in self.strategy.stops),
            "expected_points": self.expected_points,
            "points_se": self.points_se,
            "mean_position": self.mean_position,
            "median_position": self.median_position,
            "p_win": self.p_win,
            "p_podium": self.p_podium,
            "p_points": self.p_points,
            **self.extras,
        }


def _summarise(
    strategy: Strategy, positions: np.ndarray, n_cars: int, extras: dict[str, float] | None = None
) -> StrategyEvaluation:
    points = points_for(positions)
    counts = np.bincount(positions, minlength=n_cars + 1)[1 : n_cars + 1]
    distribution = counts / max(counts.sum(), 1)

    return StrategyEvaluation(
        strategy=strategy,
        positions=positions,
        n_cars=n_cars,
        expected_points=float(points.mean()),
        mean_position=float(positions.mean()),
        median_position=float(np.median(positions)),
        p_win=float((positions == 1).mean()),
        p_podium=float((positions <= 3).mean()),
        p_points=float((positions <= 10).mean()),
        position_distribution=distribution,
        points_se=float(points.std(ddof=1) / np.sqrt(max(points.size, 1))),
        extras=extras or {},
    )


def evaluate_candidates(
    params: SimParams,
    candidates: list[Strategy],
    field_strategies: list[Strategy],
    focal_car: int,
    grid: np.ndarray,
    pace_offsets: np.ndarray,
    n_races: int,
    seed: int,
    schedule: NeutralisationSchedule | None = None,
    compound_index: dict[str, int] | None = None,
) -> list[StrategyEvaluation]:
    """Score every candidate for ``focal_car``, holding the rest of the field fixed.

    A fresh generator seeded identically per candidate is what delivers the
    common random numbers. It works because the engine draws the same number of
    random values in the same order regardless of which strategy it is running
    -- see the note on the unconditional pit-noise draw in
    :mod:`pitwall.sim.engine`.
    """
    posterior = params.posterior
    index = compound_index or {c: posterior.compound_index(c) for c in posterior.compounds}

    if schedule is None:
        from pitwall.sim.engine import _sample_schedule

        schedule = _sample_schedule(params, np.random.default_rng(seed), n_races)

    results: list[StrategyEvaluation] = []
    for candidate in candidates:
        plans = list(field_strategies)
        plans[focal_car] = candidate
        pit_after, compound_by_lap = plan_to_matrix(plans, params.race_laps, index)

        outcome = simulate_ensemble(
            params,
            np.random.default_rng(seed),
            n_races,
            grid,
            pace_offsets,
            pit_after,
            compound_by_lap,
            schedule=schedule,
        )
        results.append(
            _summarise(
                candidate,
                outcome.positions_for(focal_car),
                params.n_cars,
                extras={
                    "p_retired": float(outcome.state.retired[:, focal_car].mean()),
                    "mean_stops": float(outcome.state.stops_done[:, focal_car].mean()),
                },
            )
        )
    return results


def rank(
    evaluations: list[StrategyEvaluation], objective: str = "expected_points"
) -> list[StrategyEvaluation]:
    """Best first. Position-based objectives sort ascending, points descending."""
    ascending = objective in {"mean_position", "median_position"}
    return sorted(evaluations, key=lambda e: getattr(e, objective), reverse=not ascending)


def to_frame(evaluations: list[StrategyEvaluation]) -> pd.DataFrame:
    return pd.DataFrame([e.as_row() for e in evaluations])


def distribution_frame(evaluations: list[StrategyEvaluation], top: int = 5) -> pd.DataFrame:
    """Finishing-position distributions for the leading candidates."""
    rows = []
    for evaluation in evaluations[:top]:
        for position, probability in enumerate(evaluation.position_distribution, start=1):
            rows.append(
                {
                    "strategy": evaluation.strategy.label(),
                    "position": position,
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows)
