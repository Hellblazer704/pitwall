"""The lap-by-lap race engine, vectorised over an ensemble of races.

Why the vectorisation runs this way
-----------------------------------

The obvious way to write a race simulator is a loop over laps containing a loop
over cars, and then to call it 10,000 times. That is about 14 million Python
iterations per candidate strategy, which puts a single optimiser run into the
tens of minutes and makes the reactive policy in Component 3 impossible.

Instead the state is ``(n_races, n_cars)`` and the *ensemble* advances one lap
at a time. The lap loop is Python; the work inside it is numpy over all races
at once. The one genuinely sequential part is resolving traffic, which has to
walk the field from the leader backwards because whether a car is held up
depends on what the car ahead just did. That walk is ``n_cars`` iterations of
numpy over ``n_races`` -- so a 12,000-race ensemble over 57 laps costs about
1,100 vectorised steps rather than 13.7 million scalar ones.

Model of a lap
--------------

For each car, in order:

1. **Free-air lap time**: base pace, plus degradation sampled from the
   Component 1 posterior for this race, plus the fuel term, plus execution
   noise. Under a neutralisation the whole field runs to a delta instead.
2. **Pit loss**, if the car stops at the end of this lap, discounted if the
   stop happens under a neutralisation.
3. **Traffic**: a car within the dirty-air threshold of the car ahead loses
   time, scaled by how close it is.
4. **Position resolution**: a car cannot simply drive through the one in
   front. If its unimpeded lap would put it ahead, it must either complete an
   overtake -- probability depending on pace delta and how hard the circuit is
   to pass at -- or it is held to a minimum following gap.
5. **Retirement**, mechanical or incident.

Step 4 is what separates this from a spreadsheet. Summing lap times and sorting
lets every faster car through for free, which systematically overvalues
strategies that emerge behind traffic. That is one of the ablations in
Component 5.

Known limitations, stated rather than hidden
--------------------------------------------

*Lapped traffic* is not modelled as a distinct phenomenon. Cars are ordered by
cumulative time, so a car a lap down sorts behind the lead-lap field and the
following-gap constraint applies to it as if it were racing. Blue flags mean
real lapped cars yield, so the simulator slightly overstates how much the
leaders lose to backmarkers.

*Neutralisations are exogenous* -- see :mod:`pitwall.sim.events`.

*Tyre warm-up* is folded into the pit loss rather than modelled as a separate
out-lap deficit, because the pit-loss estimate in
:mod:`pitwall.ingest.circuits` is measured across the in-lap and out-lap pair
and therefore already contains it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pitwall.sim.events import GREEN, SAFETY_CAR, VSC, NeutralisationSchedule
from pitwall.sim.params import SimParams

log = logging.getLogger(__name__)

__all__ = ["EnsembleResult", "EnsembleState", "simulate_ensemble"]

# Cumulative-time sentinel for a retired car. Large enough to sort behind any
# real race time, and offset by laps completed so retirements are still
# classified in the right order relative to each other.
_RETIRED_BASE = 1.0e6
_RETIRED_LAP_WEIGHT = 1.0e3


@dataclass
class EnsembleState:
    """Mutable per-(race, car) state. Also the resume point for a mid-race restart."""

    cum_time: np.ndarray  # (R, C) seconds; sentinel value once retired
    tyre_age: np.ndarray  # (R, C) laps on the current set
    compound: np.ndarray  # (R, C) compound index
    retired: np.ndarray  # (R, C) bool
    laps_done: np.ndarray  # (R, C) laps completed
    stops_done: np.ndarray  # (R, C) pit stops taken
    just_pitted: np.ndarray  # (R, C) bool, pitted at the end of the previous lap

    def copy(self) -> EnsembleState:
        return EnsembleState(
            cum_time=self.cum_time.copy(),
            tyre_age=self.tyre_age.copy(),
            compound=self.compound.copy(),
            retired=self.retired.copy(),
            laps_done=self.laps_done.copy(),
            stops_done=self.stops_done.copy(),
            just_pitted=self.just_pitted.copy(),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.cum_time.shape  # type: ignore[return-value]

    def subset(self, mask: np.ndarray) -> EnsembleState:
        """The state for a subset of races, as an independent copy.

        Used by the reactive policy to branch a decision on just the races that
        are actually facing it.
        """
        return EnsembleState(
            cum_time=self.cum_time[mask].copy(),
            tyre_age=self.tyre_age[mask].copy(),
            compound=self.compound[mask].copy(),
            retired=self.retired[mask].copy(),
            laps_done=self.laps_done[mask].copy(),
            stops_done=self.stops_done[mask].copy(),
            just_pitted=self.just_pitted[mask].copy(),
        )

    def assign(self, mask: np.ndarray, other: EnsembleState) -> None:
        """Write ``other`` back into the rows selected by ``mask``."""
        self.cum_time[mask] = other.cum_time
        self.tyre_age[mask] = other.tyre_age
        self.compound[mask] = other.compound
        self.retired[mask] = other.retired
        self.laps_done[mask] = other.laps_done
        self.stops_done[mask] = other.stops_done
        self.just_pitted[mask] = other.just_pitted


@dataclass
class EnsembleResult:
    """Outcome of simulating an ensemble."""

    finish_position: np.ndarray  # (R, C) 1-based
    state: EnsembleState
    schedule: NeutralisationSchedule
    car_pace: np.ndarray  # (R, C) base pace used, for diagnostics
    meta: dict[str, float] = field(default_factory=dict)

    def positions_for(self, car: int) -> np.ndarray:
        return self.finish_position[:, car]

    def position_distribution(self, car: int, n_cars: int | None = None) -> np.ndarray:
        """Probability of each finishing position, 1..n_cars."""
        size = n_cars or int(self.finish_position.shape[1])
        counts = np.bincount(self.positions_for(car), minlength=size + 1)[1 : size + 1]
        return counts / max(counts.sum(), 1)


def _classify(state: EnsembleState) -> np.ndarray:
    """Finishing positions from laps completed then elapsed time."""
    # Retired cars already carry a sentinel cumulative time ordered by laps
    # completed, so a single sort on cumulative time reproduces the FIA
    # classification: everyone who finished, in time order, then retirements
    # in reverse order of distance covered.
    order = np.argsort(state.cum_time, axis=1, kind="stable")
    positions = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    positions[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
    return positions


def _draw_degradation(
    params: SimParams, rng: np.random.Generator, n_races: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-race degradation coefficients and residual sd.

    One posterior sample per race, not per car: the compounds' curves at a
    circuit are estimated from overlapping data and their errors are
    correlated, so drawing them independently would understate the risk of a
    compound switch.
    """
    posterior = params.posterior
    n_samples = posterior.n_samples

    if params.deg_use_posterior_mean:
        # Ablation path: collapse to a point estimate, discarding model
        # uncertainty entirely.
        mean_coefs = posterior.coefficients(params.circuit, slice(None), rng=rng).mean(axis=0)
        coefs = np.repeat(mean_coefs[None, ...], n_races, axis=0)
        sigma = np.full(n_races, float(np.median(posterior.sigma)))
        return coefs, sigma

    draw = rng.integers(0, n_samples, size=n_races)
    coefs = posterior.coefficients(params.circuit, draw, rng=rng)
    sigma = posterior.sigma[draw]
    return coefs, sigma


def _initial_state(
    params: SimParams,
    rng: np.random.Generator,
    n_races: int,
    grid: np.ndarray,
    start_compound: np.ndarray,
    start_tyre_age: np.ndarray,
) -> EnsembleState:
    n_cars = params.n_cars
    shape = (n_races, n_cars)

    # Grid position is converted to a starting time deficit. The gap between
    # grid slots is a real, roughly constant quantity; anything else about the
    # start (bogged down, a good launch) is the start_position_sd noise.
    grid_gap_s = 0.75
    deficit = (grid[None, :] - 1) * grid_gap_s
    jitter = rng.normal(0.0, params.start_position_sd * grid_gap_s, size=shape)

    return EnsembleState(
        cum_time=deficit + jitter,
        tyre_age=np.repeat(start_tyre_age[None, :], n_races, axis=0).astype(float),
        compound=np.repeat(start_compound[None, :], n_races, axis=0).astype(np.int64),
        retired=np.zeros(shape, dtype=bool),
        laps_done=np.zeros(shape, dtype=np.int64),
        stops_done=np.zeros(shape, dtype=np.int64),
        just_pitted=np.zeros(shape, dtype=bool),
    )


def _dirty_air_loss(gap: np.ndarray, params: SimParams) -> np.ndarray:
    """Time lost per lap to the wake of the car ahead.

    Linear decay from the full loss when glued to the gearbox, to nothing at
    the threshold. The real relationship is closer to inverse-square in gap,
    but the linear form is within the noise of what the lap data can resolve
    and has one fewer parameter to justify.
    """
    if params.dirty_air_max_loss_s <= 0 or params.dirty_air_threshold_s <= 0:
        return np.zeros_like(gap)
    closeness = np.clip(1.0 - gap / params.dirty_air_threshold_s, 0.0, 1.0)
    return params.dirty_air_max_loss_s * closeness


def _overtake_probability(pace_delta: np.ndarray, params: SimParams) -> np.ndarray:
    """P(pass completed this lap) given the attacker's pace advantage."""
    logit = params.overtake_intercept + params.overtake_pace_coef * pace_delta
    base = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    return np.clip(base * params.overtake_difficulty, 0.0, 1.0)


def simulate_ensemble(
    params: SimParams,
    rng: np.random.Generator,
    n_races: int,
    grid: np.ndarray,
    pace_offsets: np.ndarray,
    pit_after_lap: np.ndarray,
    compound_by_lap: np.ndarray,
    schedule: NeutralisationSchedule | None = None,
    start_lap: int = 0,
    state: EnsembleState | None = None,
    stop_after_lap: int | None = None,
    forced_pit: np.ndarray | None = None,
) -> EnsembleResult:
    """Simulate ``n_races`` races.

    ``pit_after_lap`` and ``compound_by_lap`` are ``(n_cars, race_laps)``
    plans, or ``(n_races, n_cars, race_laps)`` if strategies differ per race
    (which is how the reactive policy expresses a decision it made mid-race).

    ``start_lap``, ``state`` and ``stop_after_lap`` let a race be run in
    segments. That is what makes genuine online re-optimisation possible: run
    to the deployment lap, branch, evaluate each branch to the flag, then
    continue with the winner.
    """
    n_cars = params.n_cars
    race_laps = params.race_laps
    shape = (n_races, n_cars)

    if pit_after_lap.ndim == 2:
        pit_plan = np.repeat(pit_after_lap[None, ...], n_races, axis=0)
        comp_plan = np.repeat(compound_by_lap[None, ...], n_races, axis=0)
    else:
        pit_plan, comp_plan = pit_after_lap, compound_by_lap

    if schedule is None:
        schedule = _sample_schedule(params, rng, n_races)

    if state is None:
        state = _initial_state(
            params,
            rng,
            n_races,
            grid,
            start_compound=comp_plan[0, :, 0],
            start_tyre_age=np.zeros(n_cars),
        )

    coefs, resid_sd = _draw_degradation(params, rng, n_races)
    # (R, n_compounds) views, so the per-lap gather is a take_along_axis.
    coef_offset, coef_linear, coef_quad = coefs[:, :, 0], coefs[:, :, 1], coefs[:, :, 2]

    car_pace = params.base_lap_s + pace_offsets[None, :] + np.zeros(shape)
    posterior = params.posterior
    age_center, age_scale = posterior.age_center_laps, posterior.age_scale_laps

    # Beyond the oldest tyre age this circuit and compound were actually run
    # to, the fitted quadratic is pure extrapolation and can curve anywhere.
    # Past that point the curve is continued linearly at the slope it had
    # reached, which is a claim the data supports; a quadratic there is not.
    if params.circuit in posterior.circuits:
        max_age_row = posterior.max_age[posterior.circuits.index(params.circuit)]
    else:
        max_age_row = np.median(posterior.max_age, axis=0)
    z_cap = (max_age_row - age_center) / age_scale  # (n_compounds,)

    burn_per_lap = params.fuel_start_mass_kg / max(race_laps, 1)
    last_lap = race_laps if stop_after_lap is None else min(stop_after_lap, race_laps)

    for lap in range(start_lap, last_lap):
        regime = schedule.regime[:, lap]  # (R,)
        alive = ~state.retired

        # -- 1. free-air lap time -------------------------------------------
        z_raw = (state.tyre_age - age_center) / age_scale
        cap = z_cap[state.compound]
        z = np.minimum(z_raw, cap)
        overshoot = np.maximum(z_raw - cap, 0.0)

        linear = np.take_along_axis(coef_linear, state.compound, axis=1)
        quad = np.take_along_axis(coef_quad, state.compound, axis=1)
        deg = (
            np.take_along_axis(coef_offset, state.compound, axis=1)
            + linear * z
            + quad * z * z
            # Linear continuation past the observed age range, at the slope the
            # fitted curve had reached there.
            + (linear + 2.0 * quad * z) * overshoot
        )
        fuel = params.fuel_s_per_kg * burn_per_lap * (race_laps - lap)

        noise_sd = params.driver_noise_sd_s
        if params.deg_stochastic:
            noise_sd = np.sqrt(noise_sd**2 + resid_sd[:, None] ** 2)
        lap_time = car_pace + deg + fuel + rng.normal(0.0, 1.0, size=shape) * noise_sd

        # Under a neutralisation the field runs to a delta, so car pace and
        # tyre state stop mattering and everyone laps at the same reduced rate.
        neutral_time = np.where(
            regime == SAFETY_CAR,
            params.base_lap_s * params.sc_lap_time_multiplier,
            params.base_lap_s * params.vsc_lap_time_multiplier,
        )
        under_neutral = regime != GREEN
        lap_time = np.where(under_neutral[:, None], neutral_time[:, None], lap_time)

        # -- 2. pit stops ----------------------------------------------------
        pits = pit_plan[:, :, lap].copy()
        if forced_pit is not None:
            pits |= forced_pit[:, :, lap] if forced_pit.ndim == 3 else forced_pit
        pits &= alive

        # Drawn unconditionally, even on laps where nobody stops. Guarding this
        # behind `if pits.any()` would make the number of random draws depend on
        # the strategy being evaluated, which silently destroys the common
        # random numbers the optimiser relies on: two candidates would then be
        # compared against different sampled races and the difference between
        # them would be mostly Monte Carlo noise.
        stop_noise = rng.normal(0.0, params.stop_time_sd_s, size=shape)
        botched = rng.random(shape) < params.botch_prob
        botch_cost = botched * rng.exponential(params.botch_extra_mean_s, size=shape)
        loss = params.pit_loss_s + stop_noise + botch_cost

        discount = np.where(
            regime == SAFETY_CAR,
            params.sc_loss_multiplier,
            np.where(regime == VSC, params.vsc_loss_multiplier, 1.0),
        )
        lap_time = lap_time + np.where(pits, loss * discount[:, None], 0.0)

        # -- 3 & 4. traffic and position resolution --------------------------
        # The field is gathered into track order once, so the walk from the
        # leader backwards touches contiguous columns instead of doing a
        # fancy-index gather per car per lap. That indexing was the single
        # hottest thing in the simulator; hoisting it out is worth about 4x.
        order = np.argsort(state.cum_time, axis=1, kind="stable")
        cum_ord = np.take_along_axis(state.cum_time, order, axis=1)
        time_ord = np.take_along_axis(lap_time, order, axis=1)
        pitted_ord = np.take_along_axis(state.just_pitted, order, axis=1)

        resolved_ord = np.empty_like(cum_ord)
        resolved_ord[:, 0] = cum_ord[:, 0] + time_ord[:, 0]
        behind_reference = resolved_ord[:, 0]

        noise = rng.random((n_races, max(n_cars - 1, 1)))

        for position in range(1, n_cars):
            start_gap = cum_ord[:, position] - cum_ord[:, position - 1]
            car_time = time_ord[:, position]

            # Dirty air, and the extra cost of rejoining into a pack.
            in_wake = (start_gap < params.dirty_air_threshold_s) & ~under_neutral
            penalty = _dirty_air_loss(start_gap, params)
            penalty = penalty + pitted_ord[:, position] * params.emergence_penalty_s
            car_time = car_time + np.where(in_wake, penalty, 0.0)

            unimpeded = cum_ord[:, position] + car_time
            limit = behind_reference + params.min_following_gap_s
            blocked = unimpeded < limit

            # An overtake is only on if the follower is genuinely quicker, is
            # close enough to attack, and the race is green.
            pace_delta = time_ord[:, position - 1] - time_ord[:, position]
            attacking = (
                blocked
                & ~under_neutral
                & (start_gap < params.min_gap_to_attempt_s)
                & (pace_delta > 0)
            )
            probability = np.where(attacking, _overtake_probability(pace_delta, params), 0.0)
            passed = noise[:, position - 1] < probability

            resolved = np.where(
                passed,
                unimpeded,
                np.where(blocked, limit + params.failed_attempt_cost_s * attacking, unimpeded),
            )
            resolved_ord[:, position] = resolved
            # The car physically at the back of everything processed so far is
            # whichever of the two is later, which is the passed car after a
            # successful move and the follower otherwise.
            behind_reference = np.maximum(behind_reference, resolved)

        new_cum = np.empty_like(cum_ord)
        np.put_along_axis(new_cum, order, resolved_ord, axis=1)

        # -- 5. bookkeeping and retirements ----------------------------------
        state.cum_time = np.where(alive, new_cum, state.cum_time)
        state.laps_done = state.laps_done + alive
        state.tyre_age = state.tyre_age + alive

        if params.reliability_enabled:
            hazard = np.full(shape, params.mechanical_per_lap + params.incident_per_lap)
            if lap == 0:
                hazard = hazard + params.lap1_incident_prob
            newly_retired = alive & (rng.random(shape) < hazard)
            if newly_retired.any():
                state.retired |= newly_retired
                sentinel = _RETIRED_BASE - state.laps_done * _RETIRED_LAP_WEIGHT
                state.cum_time = np.where(newly_retired, sentinel, state.cum_time)

        # Fit the new tyre for anyone who pitted at the end of this lap.
        if lap + 1 < race_laps:
            next_compound = comp_plan[:, :, lap + 1]
            state.compound = np.where(pits, next_compound, state.compound)
        state.tyre_age = np.where(pits, 0.0, state.tyre_age)
        state.stops_done = state.stops_done + pits
        state.just_pitted = pits

        # -- safety car bunching ---------------------------------------------
        deployed_now = schedule.deploy_lap[:, lap] & (regime == SAFETY_CAR)
        if deployed_now.any():
            state.cum_time = _bunch_field(state, deployed_now, params)

    return EnsembleResult(
        finish_position=_classify(state),
        state=state,
        schedule=schedule,
        car_pace=car_pace,
        meta={"n_races": float(n_races), "race_laps": float(race_laps)},
    )


def _bunch_field(state: EnsembleState, deployed: np.ndarray, params: SimParams) -> np.ndarray:
    """Compress gaps behind the leader when a full safety car deploys.

    This is the mechanism that makes a safety car worth so much. A car that
    stops under it rejoins having lost only the pit-lane time rather than the
    pit-lane time plus a whole lap's worth of gap to the field, and everyone
    who has already stopped watches their hard-won lead evaporate. Getting the
    bunching right matters more to a strategy recommendation than almost
    anything else in the simulator.
    """
    cum = state.cum_time.copy()
    order = np.argsort(cum, axis=1, kind="stable")
    rows = np.arange(cum.shape[0])[:, None]

    ordered = np.take_along_axis(cum, order, axis=1)
    leader = ordered[:, :1]
    running = ~np.take_along_axis(state.retired, order, axis=1)

    # Position within the queue of still-running cars, so retirements do not
    # open gaps in the bunched pack.
    queue_index = np.cumsum(running, axis=1) - 1
    bunched = leader + np.maximum(queue_index, 0) * params.sc_bunch_gap_s
    bunched = np.where(running, bunched, ordered)

    updated = np.empty_like(cum)
    updated[rows, order] = bunched
    return np.where(deployed[:, None], updated, cum)


def _sample_schedule(
    params: SimParams, rng: np.random.Generator, n_races: int
) -> NeutralisationSchedule:
    from pitwall.sim.events import sample_neutralisations

    return sample_neutralisations(
        rng,
        n_races=n_races,
        race_laps=params.race_laps,
        sc_per_race=params.sc_per_race,
        vsc_per_race=params.vsc_per_race,
        lap1_multiplier=params.sc_lap1_multiplier,
        decay=params.sc_decay,
        sc_duration_mean=params.sc_duration_mean,
        sc_duration_sd=params.sc_duration_sd,
        vsc_duration_mean=params.vsc_duration_mean,
        vsc_duration_sd=params.vsc_duration_sd,
        enabled=params.sc_enabled,
    )
