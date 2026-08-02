"""Optimiser tests, mostly about the properties the search relies on."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_params
from pitwall.optimize.mc import POINTS, evaluate_candidates, points_for, rank, to_frame
from pitwall.optimize.reactive import ReactiveConfig, run_reactive_policy
from pitwall.sim.engine import _sample_schedule
from pitwall.sim.strategy import Strategy


def _field(params):
    grid = np.arange(1, params.n_cars + 1, dtype=float)
    pace = np.linspace(0.0, 1.3, params.n_cars)
    strategies = [Strategy(compounds=("MEDIUM", "HARD"), stops=(20,)) for _ in range(params.n_cars)]
    return grid, pace, strategies


def test_points_table_matches_the_regulations() -> None:
    assert points_for(np.array([1]))[0] == 25.0
    assert points_for(np.array([2]))[0] == 18.0
    assert points_for(np.array([10]))[0] == 1.0
    assert points_for(np.array([11]))[0] == 0.0
    assert POINTS.sum() == 101.0


def test_common_random_numbers_make_identical_strategies_identical(params) -> None:
    """The property the whole comparison rests on."""
    grid, pace, strategies = _field(params)
    schedule = _sample_schedule(params, np.random.default_rng(0), 300)
    candidate = Strategy(compounds=("MEDIUM", "HARD"), stops=(20,))

    results = evaluate_candidates(
        params, [candidate, candidate], strategies, 0, grid, pace, 300, seed=5, schedule=schedule
    )
    assert np.array_equal(results[0].positions, results[1].positions)
    assert results[0].expected_points == results[1].expected_points


def test_common_random_numbers_reduce_paired_variance(params) -> None:
    """Paired comparison should be far less noisy than independent runs.

    This is the reason CRN is worth the constraint it puts on the engine.
    """
    grid, pace, strategies = _field(params)
    a = Strategy(compounds=("MEDIUM", "HARD"), stops=(18,))
    b = Strategy(compounds=("MEDIUM", "HARD"), stops=(22,))

    paired, independent = [], []
    for trial in range(6):
        schedule = _sample_schedule(params, np.random.default_rng(100 + trial), 400)
        both = evaluate_candidates(
            params, [a, b], strategies, 0, grid, pace, 400, seed=200 + trial, schedule=schedule
        )
        paired.append(both[0].expected_points - both[1].expected_points)

        sched_a = _sample_schedule(params, np.random.default_rng(300 + trial), 400)
        sched_b = _sample_schedule(params, np.random.default_rng(900 + trial), 400)
        ea = evaluate_candidates(
            params, [a], strategies, 0, grid, pace, 400, seed=400 + trial, schedule=sched_a
        )[0]
        eb = evaluate_candidates(
            params, [b], strategies, 0, grid, pace, 400, seed=700 + trial, schedule=sched_b
        )[0]
        independent.append(ea.expected_points - eb.expected_points)

    assert np.std(paired) < np.std(independent)


def test_strategy_ranking_orders_correctly(params) -> None:
    grid, pace, strategies = _field(params)
    candidates = [
        Strategy(compounds=("MEDIUM", "HARD"), stops=(20,)),
        Strategy(compounds=("SOFT", "HARD"), stops=(12,)),
    ]
    results = evaluate_candidates(params, candidates, strategies, 0, grid, pace, 400, seed=1)

    by_points = rank(results, "expected_points")
    assert by_points[0].expected_points >= by_points[-1].expected_points

    by_position = rank(results, "mean_position")
    assert by_position[0].mean_position <= by_position[-1].mean_position


def test_position_distribution_is_a_probability_vector(params) -> None:
    grid, pace, strategies = _field(params)
    result = evaluate_candidates(
        params,
        [Strategy(compounds=("MEDIUM", "HARD"), stops=(20,))],
        strategies,
        0,
        grid,
        pace,
        300,
        seed=2,
    )[0]
    distribution = result.position_distribution
    assert distribution.shape == (params.n_cars,)
    assert distribution.sum() == pytest.approx(1.0)
    assert np.all(distribution >= 0)


def test_reported_probabilities_are_consistent(params) -> None:
    grid, pace, strategies = _field(params)
    result = evaluate_candidates(
        params,
        [Strategy(compounds=("MEDIUM", "HARD"), stops=(20,))],
        strategies,
        0,
        grid,
        pace,
        500,
        seed=3,
    )[0]
    assert result.p_win <= result.p_podium <= result.p_points
    assert 1.0 <= result.mean_position <= params.n_cars


def test_monte_carlo_standard_error_shrinks_with_ensemble_size(params) -> None:
    grid, pace, strategies = _field(params)
    candidate = Strategy(compounds=("MEDIUM", "HARD"), stops=(20,))
    small = evaluate_candidates(params, [candidate], strategies, 0, grid, pace, 200, seed=4)[0]
    large = evaluate_candidates(params, [candidate], strategies, 0, grid, pace, 2000, seed=4)[0]
    assert large.points_se < small.points_se


def test_result_frame_has_the_reported_columns(params) -> None:
    grid, pace, strategies = _field(params)
    results = evaluate_candidates(
        params,
        [Strategy(compounds=("MEDIUM", "HARD"), stops=(20,))],
        strategies,
        0,
        grid,
        pace,
        100,
        seed=6,
    )
    frame = to_frame(results)
    assert {"strategy", "expected_points", "p_win", "p_podium", "p_points"} <= set(frame.columns)


def test_reactive_policy_runs_and_reports_decisions(params) -> None:
    grid, pace, strategies = _field(params)
    schedule = _sample_schedule(params, np.random.default_rng(0), 400)
    outcome = run_reactive_policy(
        params,
        Strategy(compounds=("MEDIUM", "HARD"), stops=(20,)),
        strategies,
        0,
        grid,
        pace,
        n_races=400,
        seed=11,
        config=ReactiveConfig(min_laps_remaining=8, switch_threshold_points=0.1),
        schedule=schedule,
    )
    assert outcome.positions.shape == (400,)
    assert outcome.n_decisions >= 0
    assert np.isfinite(outcome.expected_points)
    for decision in outcome.decisions:
        assert 1 <= int(decision["lap"]) <= params.race_laps


def test_reactive_policy_is_inert_when_disabled(params) -> None:
    """With the policy off, the outcome must be the static plan exactly."""
    grid, pace, strategies = _field(params)
    schedule = _sample_schedule(params, np.random.default_rng(0), 300)
    outcome = run_reactive_policy(
        params,
        Strategy(compounds=("MEDIUM", "HARD"), stops=(20,)),
        strategies,
        0,
        grid,
        pace,
        n_races=300,
        seed=12,
        config=ReactiveConfig(enabled=False),
        schedule=schedule,
    )
    assert outcome.n_switches == 0
    assert outcome.expected_points == pytest.approx(outcome.baseline_points)


def test_ablation_overrides_actually_change_the_simulator(posterior) -> None:
    from pitwall.ablation.study import ABLATIONS, _ablated

    base = make_params(posterior)
    assert _ablated(base, "calibrated") is base
    assert not _ablated(base, "no_safety_car").sc_enabled
    assert _ablated(base, "no_traffic").dirty_air_max_loss_s == 0.0
    assert _ablated(base, "no_pit_variance").botch_prob == 0.0
    assert _ablated(base, "deterministic_degradation").deg_use_posterior_mean
    assert not _ablated(base, "naive_all").reliability_enabled
    assert set(ABLATIONS) >= {"calibrated", "naive_all"}
