"""Wiring the strategy search together for one race."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pitwall.degradation.fit import load_posterior
from pitwall.optimize.candidates import (
    compound_shortlist,
    enumerate_candidates,
    refine_around,
)
from pitwall.optimize.mc import (
    StrategyEvaluation,
    distribution_frame,
    evaluate_candidates,
    rank,
    to_frame,
)
from pitwall.optimize.reactive import ReactiveConfig, ReactiveOutcome, run_reactive_policy
from pitwall.paths import Paths
from pitwall.rng import SeedBank
from pitwall.sim.engine import _sample_schedule
from pitwall.sim.field import RaceField, build_field
from pitwall.sim.params import SimParams, load_sim_params

log = logging.getLogger(__name__)

__all__ = ["OptimizationResult", "optimize_for_field", "optimize_race"]


@dataclass
class OptimizationResult:
    field: RaceField
    driver: str
    focal_car: int
    grid: float
    ranked: list[StrategyEvaluation]
    screened: int
    refined: int
    reactive: ReactiveOutcome | None

    @property
    def best(self) -> StrategyEvaluation:
        return self.ranked[0]

    def table(self, top: int = 10) -> pd.DataFrame:
        return to_frame(self.ranked[:top])

    def distributions(self, top: int = 5) -> pd.DataFrame:
        return distribution_frame(self.ranked, top=top)

    def render(self, top: int = 8) -> str:
        lines = [
            f"Race    : {self.field.summary()}",
            f"Driver  : {self.driver} (car index {self.focal_car}), grid P{int(self.grid)}",
            f"Search  : {self.screened} screened, {self.refined} refined",
            "",
            "Top candidates by expected points:",
            self.table(top).to_string(index=False, float_format=lambda v: f"{v:.3f}"),
            "",
        ]

        best = self.best
        lines.append(f"Best: {best.strategy.label()}")
        lines.append(
            f"  expected points {best.expected_points:.3f} (MC se {best.points_se:.3f}), "
            f"mean finish {best.mean_position:.2f}"
        )
        lines.append(
            f"  P(win) {best.p_win:.3f}  P(podium) {best.p_podium:.3f}  "
            f"P(points) {best.p_points:.3f}  P(retire) {best.extras.get('p_retired', 0):.3f}"
        )

        lines.append("")
        lines.append("Finishing-position distribution for the best strategy:")
        lines.append(_histogram(best.position_distribution))

        if self.reactive is not None:
            r = self.reactive
            lines += [
                "",
                "Reactive policy (re-optimises when a neutralisation deploys):",
                f"  static plan      {r.baseline_points:.3f} expected points",
                f"  reactive policy  {r.expected_points:.3f} expected points",
                f"  gain             {r.gain_vs_static:+.3f}",
                f"  {r.n_decisions} decision points, switched at {r.n_switches}",
            ]
            if r.decisions:
                frame = pd.DataFrame(r.decisions)
                lines.append("")
                lines.append(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return "\n".join(lines)


def _histogram(distribution: np.ndarray, width: int = 44) -> str:
    peak = float(distribution.max()) or 1.0
    lines = []
    for position, probability in enumerate(distribution, start=1):
        if probability < 0.001:
            continue
        bar = "#" * round(width * probability / peak)
        lines.append(f"  P{position:<2d} {probability:6.3f} {bar}")
    return "\n".join(lines)


def optimize_for_field(
    cfg: DictConfig,
    params: SimParams,
    field: RaceField,
    focal_car: int,
    seed: int,
    grid_override: int | None = None,
) -> OptimizationResult:
    """Screen, refine and (optionally) evaluate the reactive policy."""
    search = cfg.optimizer.search
    mc = cfg.optimizer.monte_carlo
    posterior = params.posterior
    compound_index = {c: posterior.compound_index(c) for c in posterior.compounds}

    grid = field.grid.astype(float).copy()
    if grid_override is not None:
        grid[focal_car] = float(grid_override)

    objective = str(cfg.optimizer.objective.primary)
    n_screen = int(mc.n_races_screen)
    n_final = int(mc.n_races_final)

    # One schedule per pass, shared by every candidate in it. This is the
    # backbone of the common random numbers.
    schedule_screen = _sample_schedule(params, np.random.default_rng(seed), n_screen)
    schedule_final = _sample_schedule(params, np.random.default_rng(seed + 1), n_final)

    def score(candidates: list, n_races: int, schedule: object, pass_seed: int) -> list:
        return evaluate_candidates(
            params,
            candidates,
            field.strategies,
            focal_car,
            grid,
            field.pace_offsets,
            n_races=n_races,
            seed=pass_seed,
            schedule=schedule,  # type: ignore[arg-type]
            compound_index=compound_index,
        )

    # -- stage 1: which compounds, at an even stint split ---------------------
    shortlist = compound_shortlist(
        race_laps=params.race_laps,
        stop_counts=list(search.stop_counts),
        compounds=list(posterior.compounds),
        require_two_compounds=bool(search.enforce_two_compound_rule),
    )
    if not shortlist:
        raise ValueError(f"no legal strategies for a {params.race_laps}-lap race")

    log.info("stage 1: ranking %d compound sequences", len(shortlist))
    sequence_ranked = rank(score(shortlist, n_screen, schedule_screen, seed), objective)
    keep = int(search.keep_sequences)
    best_sequences = [e.strategy.compounds for e in sequence_ranked[:keep]]
    log.info("stage 1 kept: %s", ["-".join(s) for s in best_sequences])

    # -- stage 2: when to stop, for the surviving sequences -------------------
    candidates = enumerate_candidates(
        race_laps=params.race_laps,
        stop_counts=list(search.stop_counts),
        compounds=list(posterior.compounds),
        lap_grid_step=int(search.lap_grid_step),
        min_stint_laps=int(search.min_stint_laps),
        require_two_compounds=bool(search.enforce_two_compound_rule),
        allowed_sequences=best_sequences,
    )
    log.info("stage 2: screening %d candidates over %d races each", len(candidates), n_screen)
    ordered = rank(score(candidates, n_screen, schedule_screen, seed), objective)

    top_k = int(search.refine_top_k)
    radius = int(search.refine_radius_laps)
    refined: list = []
    seen = set()
    for evaluation in ordered[:top_k]:
        for candidate in refine_around(
            evaluation.strategy, params.race_laps, radius, int(search.min_stint_laps)
        ):
            key = (candidate.compounds, candidate.stops)
            if key not in seen:
                seen.add(key)
                refined.append(candidate)

    # -- stage 3: lap-by-lap refinement at full ensemble size ------------------
    max_refine = int(search.max_refine_candidates)
    if len(refined) > max_refine:
        log.info("refinement set trimmed from %d to %d", len(refined), max_refine)
        refined = refined[:max_refine]

    log.info("stage 3: refining %d candidates over %d races each", len(refined), n_final)
    final_ranked = rank(score(refined, n_final, schedule_final, seed + 1), objective)

    reactive_outcome = None
    if bool(cfg.optimizer.reactive.enabled):
        reactive_cfg = ReactiveConfig(
            min_laps_remaining=int(cfg.optimizer.reactive.min_laps_remaining),
            switch_threshold_points=float(cfg.optimizer.reactive.switch_threshold_points),
            enabled=True,
            decision_budget=int(cfg.optimizer.reactive.n_races_decision),
        )
        n_decision = int(cfg.optimizer.reactive.n_races_decision)
        schedule_reactive = _sample_schedule(params, np.random.default_rng(seed + 2), n_decision)
        reactive_outcome = run_reactive_policy(
            params,
            final_ranked[0].strategy,
            field.strategies,
            focal_car,
            grid,
            field.pace_offsets,
            n_races=n_decision,
            seed=seed + 2,
            config=reactive_cfg,
            schedule=schedule_reactive,
            compound_index=compound_index,
        )

    return OptimizationResult(
        field=field,
        driver=field.drivers[focal_car],
        focal_car=focal_car,
        grid=float(grid[focal_car]),
        ranked=final_ranked,
        screened=len(candidates),
        refined=len(refined),
        reactive=reactive_outcome,
    )


def optimize_race(
    cfg: DictConfig,
    season: int,
    round_no: int,
    driver: str | None = None,
    grid: int | None = None,
) -> OptimizationResult:
    paths = Paths.from_config(cfg).ensure()
    posterior = load_posterior(paths, list(cfg.data.train_seasons))
    field = build_field(season, round_no, paths, posterior)

    params = load_sim_params(
        cfg, paths, posterior, circuit=field.circuit, race_laps=field.race_laps
    )
    params = params.without(n_cars=field.n_cars)

    focal_car = field.index_of(driver) if driver else 0
    bank = SeedBank(int(cfg.seed))
    seed = int(bank.generator(f"optimize/{season}/{round_no}").integers(0, 2**31 - 1))

    result = optimize_for_field(cfg, params, field, focal_car, seed, grid_override=grid)

    print(result.render())
    target = paths.artifacts / f"strategy_{season}_r{round_no:02d}_{result.driver}.csv"
    result.table(top=25).to_csv(target, index=False)
    result.distributions().to_csv(target.with_name(f"{target.stem}_distributions.csv"), index=False)
    log.info("wrote %s", target)
    return result
