"""The online policy: re-optimising mid-race when a neutralisation deploys.

This is the decision a strategist is actually paid to make. The pre-race plan
is a starting point, but the moment the safety-car boards come out the cost of
a pit stop roughly halves and the whole plan has to be re-evaluated against the
state the race is genuinely in -- not the state it was forecast to be in.

How the policy is evaluated
---------------------------

Running the ensemble forward under a fixed plan is easy. Evaluating a *policy*
is harder, because the decision has to be made from inside the race, with only
the information available at that moment, and because different sampled races
face the decision at different laps.

The loop here does this:

1. Advance the whole ensemble one lap at a time under the pre-race plan.
2. When a neutralisation deploys, collect the races that are facing the
   decision now, and that can still act on it: a stop left in the plan, and
   enough laps remaining for the stop to pay back.
3. Branch. Copy the state for exactly those races, and run *both* futures to
   the flag: stay on the plan, or box this lap onto each available compound.
4. Compare expected points across the branches and take the best, but only if
   it beats staying out by a margin. Without that threshold the policy churns
   on Monte Carlo noise and reacts to differences it cannot actually resolve.
5. Write the decision back and carry on.

The decision is made once per deployment lap and applied to every race facing
it, rather than per sampled race. That is deliberate: a strategist seeing a
safety car on lap 23 makes one call for the situation. Letting each sampled
race pick its own branch would use knowledge of how that particular race turns
out, which is clairvoyance, and it would flatter the policy badly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pitwall.optimize.mc import points_for
from pitwall.sim.engine import EnsembleState, simulate_ensemble
from pitwall.sim.events import NeutralisationSchedule
from pitwall.sim.params import SimParams
from pitwall.sim.strategy import Strategy, plan_to_matrix

log = logging.getLogger(__name__)

__all__ = ["ReactiveConfig", "ReactiveOutcome", "run_reactive_policy"]


@dataclass(frozen=True)
class ReactiveConfig:
    min_laps_remaining: int = 8
    switch_threshold_points: float = 0.15
    enabled: bool = True
    # Target number of forward simulations per branch at each decision point,
    # reached by replicating the races that are facing the decision.
    decision_budget: int = 3000


@dataclass
class ReactiveOutcome:
    """Result of running the policy over an ensemble."""

    positions: np.ndarray  # (n_races,) for the focal car
    expected_points: float
    n_decisions: int
    n_switches: int
    decisions: list[dict[str, object]] = field(default_factory=list)
    baseline_points: float = 0.0

    @property
    def gain_vs_static(self) -> float:
        return self.expected_points - self.baseline_points


def _replan_pit_now(
    pit_plan: np.ndarray,
    comp_plan: np.ndarray,
    car: int,
    lap: int,
    compound_idx: int,
    race_laps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite a plan so the car boxes at the end of ``lap``.

    The next planned stop is removed, because the point of taking a cheap stop
    under a neutralisation is to bring the *existing* stop forward, not to add
    one. Everything after the new stop runs on the compound fitted here.
    """
    pit = pit_plan.copy()
    comp = comp_plan.copy()

    future = pit[:, car, lap + 1 :]
    next_stop = np.argmax(future, axis=1)
    has_next = future.any(axis=1)
    rows = np.nonzero(has_next)[0]
    pit[rows, car, lap + 1 + next_stop[rows]] = False

    pit[:, car, lap] = True
    comp[:, car, lap + 1 : race_laps] = compound_idx
    return pit, comp


def _forward_points(
    params: SimParams,
    state: EnsembleState,
    schedule: NeutralisationSchedule,
    pit_plan: np.ndarray,
    comp_plan: np.ndarray,
    grid: np.ndarray,
    pace_offsets: np.ndarray,
    focal_car: int,
    start_lap: int,
    seed: int,
) -> np.ndarray:
    """Run a branch to the flag and return the focal car's points per race."""
    outcome = simulate_ensemble(
        params,
        np.random.default_rng(seed),
        n_races=state.cum_time.shape[0],
        grid=grid,
        pace_offsets=pace_offsets,
        pit_after_lap=pit_plan,
        compound_by_lap=comp_plan,
        schedule=schedule,
        start_lap=start_lap,
        state=state,
    )
    return points_for(outcome.positions_for(focal_car))


def run_reactive_policy(
    params: SimParams,
    base_strategy: Strategy,
    field_strategies: list[Strategy],
    focal_car: int,
    grid: np.ndarray,
    pace_offsets: np.ndarray,
    n_races: int,
    seed: int,
    config: ReactiveConfig,
    schedule: NeutralisationSchedule | None = None,
    compound_index: dict[str, int] | None = None,
) -> ReactiveOutcome:
    """Simulate the ensemble under the online policy."""
    posterior = params.posterior
    index = compound_index or {c: posterior.compound_index(c) for c in posterior.compounds}
    race_laps = params.race_laps

    plans = list(field_strategies)
    plans[focal_car] = base_strategy
    base_pit, base_comp = plan_to_matrix(plans, race_laps, index)

    if schedule is None:
        from pitwall.sim.engine import _sample_schedule

        schedule = _sample_schedule(params, np.random.default_rng(seed), n_races)

    # Per-race plans, so a decision can apply to some races and not others.
    pit_plan = np.repeat(base_pit[None, ...], n_races, axis=0)
    comp_plan = np.repeat(base_comp[None, ...], n_races, axis=0)

    state: EnsembleState | None = None
    decisions: list[dict[str, object]] = []
    n_switches = 0
    simulated_to = 0

    # A single generator advanced across the whole race. The main ensemble is
    # simulated once, in segments, rather than replayed from lap 0 at every
    # decision point: a decision only ever rewrites the plan for laps that have
    # not been run yet, so the already-simulated prefix stays valid.
    main_rng = np.random.default_rng(seed)

    lap = 0
    while lap < race_laps:
        deployed = schedule.deploy_lap[:, lap]
        laps_remaining = race_laps - lap

        actionable = (
            config.enabled
            and laps_remaining >= config.min_laps_remaining
            and bool(deployed.any())
            and lap + 1 < race_laps
        )
        if not actionable:
            lap += 1
            continue

        if simulated_to < lap + 1:
            segment = simulate_ensemble(
                params,
                main_rng,
                n_races,
                grid,
                pace_offsets,
                pit_plan,
                comp_plan,
                schedule=schedule,
                start_lap=simulated_to,
                state=state,
                stop_after_lap=lap + 1,
            )
            state = segment.state
            simulated_to = lap + 1

        assert state is not None
        facing = deployed & ~state.retired[:, focal_car]
        facing &= pit_plan[:, focal_car, lap + 1 :].any(axis=1)
        if not facing.any():
            lap += 1
            continue

        # Only a slice of the ensemble sees a deployment on any given lap --
        # typically a few dozen races out of a few thousand -- and comparing
        # branches on that few samples is comparing noise: the spread on
        # expected points across 60 races is comfortably larger than the
        # margins the decision turns on.
        #
        # So the facing races are replicated before branching. Each replica
        # starts from the same real race state and is run forward under its own
        # random draws, which is exactly the Monte Carlo the decision wants:
        # many possible futures from one actual situation.
        facing_idx = np.nonzero(facing)[0]
        replicas = int(np.clip(config.decision_budget // max(facing_idx.size, 1), 1, 64))
        tiled = np.tile(facing_idx, replicas)

        sub_schedule = schedule.subset(tiled)
        sub_state = state.subset(tiled)
        sub_pit, sub_comp = pit_plan[tiled], comp_plan[tiled]
        branch_seed = seed + 1000 + lap

        stay_points = _forward_points(
            params,
            sub_state,
            sub_schedule,
            sub_pit,
            sub_comp,
            grid,
            pace_offsets,
            focal_car,
            lap + 1,
            branch_seed,
        )

        best_gain = 0.0
        best_compound: str | None = None
        best_plans: tuple[np.ndarray, np.ndarray] | None = None

        for compound, compound_idx in index.items():
            trial_pit, trial_comp = _replan_pit_now(
                sub_pit, sub_comp, focal_car, lap, compound_idx, race_laps
            )
            box_points = _forward_points(
                params,
                state.subset(tiled),
                sub_schedule,
                trial_pit,
                trial_comp,
                grid,
                pace_offsets,
                focal_car,
                lap + 1,
                branch_seed,
            )
            gain = float(box_points.mean() - stay_points.mean())
            if gain > best_gain:
                best_gain = gain
                best_compound = compound
                # Applied to the real ensemble rows, not the replicated ones.
                best_plans = _replan_pit_now(
                    pit_plan[facing_idx],
                    comp_plan[facing_idx],
                    focal_car,
                    lap,
                    compound_idx,
                    race_laps,
                )

        took_it = best_gain > config.switch_threshold_points and best_plans is not None
        decisions.append(
            {
                "lap": lap + 1,
                "n_races_facing": int(facing.sum()),
                "stay_points": float(stay_points.mean()),
                "best_gain_points": best_gain,
                "compound": best_compound,
                "switched": bool(took_it),
            }
        )

        if took_it and best_plans is not None:
            pit_plan[facing_idx], comp_plan[facing_idx] = best_plans
            n_switches += 1
            log.info(
                "lap %d: neutralisation, boxing onto %s (+%.3f pts over staying out, %d races)",
                lap + 1,
                best_compound,
                best_gain,
                int(facing.sum()),
            )

        lap += 1

    # The reported outcome is a single clean replay under the final plan, so it
    # uses one degradation draw per race throughout. The segment simulations
    # that produced the decision states drew their own, which means each
    # decision was taken under a re-sampled view of tyre behaviour rather than
    # the one the race then plays out with. That is deliberate: a strategist on
    # the wall does not know the true degradation either, and re-drawing at the
    # decision point is what representing their uncertainty looks like.
    final = simulate_ensemble(
        params,
        np.random.default_rng(seed),
        n_races,
        grid,
        pace_offsets,
        pit_plan,
        comp_plan,
        schedule=schedule,
    )
    positions = final.positions_for(focal_car)

    static = simulate_ensemble(
        params,
        np.random.default_rng(seed),
        n_races,
        grid,
        pace_offsets,
        base_pit,
        base_comp,
        schedule=schedule,
    )

    return ReactiveOutcome(
        positions=positions,
        expected_points=float(points_for(positions).mean()),
        n_decisions=len(decisions),
        n_switches=n_switches,
        decisions=decisions,
        baseline_points=float(points_for(static.positions_for(focal_car)).mean()),
    )
