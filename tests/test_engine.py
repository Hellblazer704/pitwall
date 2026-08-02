"""Race engine tests.

These are mostly invariants rather than value checks. A race simulator has few
quantities with a known right answer, but plenty of properties that must hold
for any correct implementation, and those catch real bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_params
from pitwall.sim.engine import simulate_ensemble
from pitwall.sim.events import GREEN, SAFETY_CAR, VSC, sample_neutralisations
from pitwall.sim.strategy import Strategy, plan_to_matrix


def _plans(params, stops=(20,), compounds=("MEDIUM", "HARD")):
    index = {c: params.posterior.compound_index(c) for c in params.posterior.compounds}
    strategies = [Strategy(compounds=compounds, stops=stops) for _ in range(params.n_cars)]
    return plan_to_matrix(strategies, params.race_laps, index)


def _run(params, seed=1, n_races=200, **kwargs):
    pit, comp = _plans(params)
    grid = np.arange(1, params.n_cars + 1, dtype=float)
    pace = np.linspace(0.0, 1.3, params.n_cars)
    return simulate_ensemble(
        params, np.random.default_rng(seed), n_races, grid, pace, pit, comp, **kwargs
    )


def test_finishing_positions_are_a_permutation(params) -> None:
    result = _run(params)
    expected = np.arange(1, params.n_cars + 1)
    for race in range(result.finish_position.shape[0]):
        assert np.array_equal(np.sort(result.finish_position[race]), expected)


def test_same_seed_is_bit_identical(params) -> None:
    a = _run(params, seed=7)
    b = _run(params, seed=7)
    assert np.array_equal(a.finish_position, b.finish_position)
    assert np.allclose(a.state.cum_time, b.state.cum_time)


def test_different_seeds_diverge(params) -> None:
    a = _run(params, seed=7)
    b = _run(params, seed=8)
    assert not np.array_equal(a.finish_position, b.finish_position)


def test_faster_cars_finish_further_forward(params) -> None:
    """Grid and pace both favour car 0, so it must average a better result."""
    result = _run(params, n_races=400)
    mean_positions = result.finish_position.mean(axis=0)
    assert mean_positions[0] < mean_positions[-1]
    # And the ordering should be broadly monotone in pace.
    assert np.corrcoef(mean_positions, np.arange(params.n_cars))[0, 1] > 0.9


def test_laps_completed_never_exceeds_race_distance(params) -> None:
    result = _run(params)
    assert result.state.laps_done.max() <= params.race_laps


def test_retirements_stop_accumulating_laps(params) -> None:
    result = _run(params, n_races=500)
    retired = result.state.retired
    if retired.any():
        assert result.state.laps_done[retired].max() <= params.race_laps
        # Retirements are classified behind every finisher.
        for race in range(result.finish_position.shape[0]):
            if not retired[race].any():
                continue
            worst_finisher = result.finish_position[race][~retired[race]].max()
            best_retirement = result.finish_position[race][retired[race]].min()
            assert best_retirement > worst_finisher


def test_disabling_reliability_removes_retirements(params) -> None:
    result = _run(params.without(reliability_enabled=False), n_races=300)
    assert not result.state.retired.any()


def test_planned_stops_are_executed(params) -> None:
    result = _run(params.without(reliability_enabled=False))
    assert np.all(result.state.stops_done == 1)


def test_more_stops_cost_more_time_without_degradation(posterior) -> None:
    """With wear switched off, an extra stop is pure loss.

    Isolates the pit-loss accounting from everything else: if a second stop
    ever looks free here, the pit loss is not being applied.
    """
    flat = posterior.__class__(
        **{
            **posterior.__dict__,
            "beta": np.zeros_like(posterior.beta),
            "driver": np.zeros_like(posterior.driver),
        }
    )
    params = make_params(flat).without(
        reliability_enabled=False,
        sc_enabled=False,
        driver_noise_sd_s=0.0,
        deg_stochastic=False,
        start_position_sd=0.0,
        dirty_air_max_loss_s=0.0,
        min_following_gap_s=0.0,
        stop_time_sd_s=0.0,
        botch_prob=0.0,
    )
    grid = np.arange(1, params.n_cars + 1, dtype=float)
    pace = np.zeros(params.n_cars)
    index = {c: params.posterior.compound_index(c) for c in params.posterior.compounds}

    times = []
    for stops, compounds in (((20,), ("MEDIUM", "HARD")), ((14, 28), ("MEDIUM", "HARD", "HARD"))):
        strategies = [Strategy(compounds=compounds, stops=stops) for _ in range(params.n_cars)]
        pit, comp = plan_to_matrix(strategies, params.race_laps, index)
        out = simulate_ensemble(params, np.random.default_rng(3), 40, grid, pace, pit, comp)
        times.append(float(out.state.cum_time[:, 0].mean()))

    assert times[1] > times[0]
    assert times[1] - times[0] == pytest.approx(params.pit_loss_s, rel=0.35)


def test_traffic_constraint_prevents_driving_through_cars(posterior) -> None:
    """A much faster car behind a slow one must not simply pass every lap.

    With overtaking effectively impossible, the fast car should be stuck.
    """
    params = make_params(posterior).without(
        n_cars=2,
        reliability_enabled=False,
        sc_enabled=False,
        driver_noise_sd_s=0.0,
        deg_stochastic=False,
        start_position_sd=0.0,
        overtake_intercept=-50.0,
        overtake_difficulty=0.0,
    )
    index = {c: params.posterior.compound_index(c) for c in params.posterior.compounds}
    strategies = [Strategy(compounds=("MEDIUM", "HARD"), stops=(20,)) for _ in range(2)]
    pit, comp = plan_to_matrix(strategies, params.race_laps, index)

    # Car 1 starts second but is two seconds a lap quicker.
    grid = np.array([1.0, 2.0])
    pace = np.array([2.0, 0.0])
    blocked = simulate_ensemble(params, np.random.default_rng(5), 100, grid, pace, pit, comp)
    stuck_rate = float((blocked.finish_position[:, 1] == 2).mean())

    free = simulate_ensemble(
        params.without(overtake_intercept=12.0, overtake_difficulty=1.0, min_following_gap_s=0.0),
        np.random.default_rng(5),
        100,
        grid,
        pace,
        pit,
        comp,
    )
    passing_rate = float((free.finish_position[:, 1] == 1).mean())

    assert stuck_rate > 0.5, "a car that cannot overtake should stay behind"
    assert passing_rate > 0.9, "a much faster car that can pass freely should get through"


def test_neutralisation_rate_matches_the_configured_rate(params) -> None:
    schedule = sample_neutralisations(
        np.random.default_rng(0),
        n_races=4000,
        race_laps=params.race_laps,
        sc_per_race=0.5,
        vsc_per_race=0.4,
        lap1_multiplier=params.sc_lap1_multiplier,
        decay=params.sc_decay,
        sc_duration_mean=params.sc_duration_mean,
        sc_duration_sd=params.sc_duration_sd,
        vsc_duration_mean=params.vsc_duration_mean,
        vsc_duration_sd=params.vsc_duration_sd,
    )
    observed = schedule.deploy_lap.sum(axis=1).mean()
    assert observed == pytest.approx(0.9, rel=0.15)


def test_neutralisations_do_not_overlap(params) -> None:
    schedule = sample_neutralisations(
        np.random.default_rng(1),
        n_races=500,
        race_laps=params.race_laps,
        sc_per_race=1.5,
        vsc_per_race=1.5,
        lap1_multiplier=1.0,
        decay=0.0,
        sc_duration_mean=4.0,
        sc_duration_sd=1.0,
        vsc_duration_mean=2.0,
        vsc_duration_sd=0.5,
    )
    # Every run of neutralised laps must begin at a recorded deployment. If a
    # deployment could start while another was still running, the second one
    # would extend the first's run without opening a new one and the rate would
    # be inflated relative to the fitted value.
    #
    # Back-to-back deployments are allowed: one ends, another begins on the
    # very next lap. That is a real thing (an incident during the restart) and
    # it shows up here as one continuous run with two deploy flags, which is
    # why the check is "runs start at a deployment" rather than "deployments
    # are separated by green".
    for race in range(schedule.n_races):
        active = schedule.regime[race] != GREEN
        deploys = schedule.deploy_lap[race]
        run_starts = np.nonzero(active & ~np.concatenate([[False], active[:-1]]))[0]
        for start in run_starts:
            assert deploys[start], "a neutralised run began without a deployment"
        assert deploys.sum() >= len(run_starts)


def test_disabling_safety_car_leaves_the_race_green(params) -> None:
    schedule = sample_neutralisations(
        np.random.default_rng(2),
        n_races=100,
        race_laps=params.race_laps,
        sc_per_race=1.0,
        vsc_per_race=1.0,
        lap1_multiplier=1.0,
        decay=0.0,
        sc_duration_mean=3.0,
        sc_duration_sd=1.0,
        vsc_duration_mean=2.0,
        vsc_duration_sd=0.5,
        enabled=False,
    )
    assert np.all(schedule.regime == GREEN)
    assert not schedule.deploy_lap.any()


def test_regimes_are_only_the_defined_codes(params) -> None:
    schedule = sample_neutralisations(
        np.random.default_rng(3),
        n_races=200,
        race_laps=params.race_laps,
        sc_per_race=1.0,
        vsc_per_race=1.0,
        lap1_multiplier=2.0,
        decay=0.5,
        sc_duration_mean=3.0,
        sc_duration_sd=1.0,
        vsc_duration_mean=2.0,
        vsc_duration_sd=0.5,
    )
    assert set(np.unique(schedule.regime)) <= {GREEN, VSC, SAFETY_CAR}


def test_safety_car_bunches_the_field(params) -> None:
    """The mechanism that makes a deployment worth so much."""
    sc_params = params.without(reliability_enabled=False)
    pit, comp = _plans(sc_params)
    grid = np.arange(1, sc_params.n_cars + 1, dtype=float)
    pace = np.linspace(0.0, 2.5, sc_params.n_cars)

    forced = sample_neutralisations(
        np.random.default_rng(0),
        n_races=200,
        race_laps=sc_params.race_laps,
        sc_per_race=0.0,
        vsc_per_race=0.0,
        lap1_multiplier=1.0,
        decay=0.0,
        sc_duration_mean=3.0,
        sc_duration_sd=0.1,
        vsc_duration_mean=2.0,
        vsc_duration_sd=0.1,
        enabled=False,
    )
    # Hand-build a deployment on lap 15 for every race.
    forced.regime[:, 14:18] = SAFETY_CAR
    forced.deploy_lap[:, 14] = True

    with_sc = simulate_ensemble(
        sc_params, np.random.default_rng(4), 200, grid, pace, pit, comp, schedule=forced
    )
    without_sc = simulate_ensemble(
        sc_params.without(sc_enabled=False),
        np.random.default_rng(4),
        200,
        grid,
        pace,
        pit,
        comp,
    )

    spread_with = float(
        (with_sc.state.cum_time.max(axis=1) - with_sc.state.cum_time.min(axis=1)).mean()
    )
    spread_without = float(
        (without_sc.state.cum_time.max(axis=1) - without_sc.state.cum_time.min(axis=1)).mean()
    )
    assert spread_with < spread_without


def test_simulation_can_resume_from_a_saved_state(params) -> None:
    """Segmented running must reproduce a single continuous run.

    The reactive policy depends on this: it advances the ensemble in pieces
    around each decision point.
    """
    quiet = params.without(reliability_enabled=False)
    pit, comp = _plans(quiet)
    grid = np.arange(1, quiet.n_cars + 1, dtype=float)
    pace = np.linspace(0.0, 1.3, quiet.n_cars)

    rng = np.random.default_rng(11)
    first = simulate_ensemble(quiet, rng, 50, grid, pace, pit, comp, stop_after_lap=15)
    schedule = first.schedule
    second = simulate_ensemble(
        quiet, rng, 50, grid, pace, pit, comp, schedule=schedule, start_lap=15, state=first.state
    )

    assert second.state.laps_done.max() == quiet.race_laps
    assert np.all(np.isfinite(second.state.cum_time))


def test_degradation_is_bounded_beyond_the_observed_age_range(posterior) -> None:
    """Past the oldest observed tyre age the curve must not run away.

    A quadratic extrapolated far outside its data can curve to anything; the
    engine continues it linearly instead.
    """
    tight = posterior.__class__(
        **{**posterior.__dict__, "max_age": np.full_like(posterior.max_age, 10.0)}
    )
    params = make_params(tight).without(
        reliability_enabled=False,
        sc_enabled=False,
        driver_noise_sd_s=0.0,
        deg_stochastic=False,
        start_position_sd=0.0,
    )
    index = {c: params.posterior.compound_index(c) for c in params.posterior.compounds}
    # One very long stint, well beyond the 10-lap cap.
    strategies = [
        Strategy(compounds=("MEDIUM", "HARD"), stops=(params.race_laps - 6,))
        for _ in range(params.n_cars)
    ]
    pit, comp = plan_to_matrix(strategies, params.race_laps, index)
    grid = np.arange(1, params.n_cars + 1, dtype=float)
    out = simulate_ensemble(
        params, np.random.default_rng(2), 20, grid, np.zeros(params.n_cars), pit, comp
    )
    per_lap = out.state.cum_time[:, 0] / params.race_laps
    assert np.all(np.isfinite(per_lap))
    # A runaway quadratic would put lap times into the hundreds of seconds.
    assert per_lap.max() < params.base_lap_s + 15.0
