"""The ablation: removing realism one feature at a time.

The claim being tested is specific. A simulator that leaves out safety cars,
treats degradation as deterministic, lets cars pass each other freely and
assumes pit stops take exactly the same time every race does not merely
produce noisier answers. It produces *systematically different and worse
recommendations*, and it is confident about them.

The mechanism in each case is the same: the missing feature is a source of
downside that falls disproportionately on aggressive strategies. Remove it and
aggression looks free.

- **No safety car.** This one cuts both ways and is the most interesting. A
  neutralisation is a cheap pit stop, which rewards having a stop still in
  hand, and it bunches the field, which destroys a lead built by track
  position. A simulator without it misprices both.
- **Deterministic degradation.** Using the posterior mean throws away the
  spread of tyre behaviour. A long stint's value depends on the tyre lasting;
  with the uncertainty removed, the tail where it does not simply disappears.
- **No traffic.** Without dirty air and without the constraint that a car
  cannot drive through the one in front, emerging from a pit stop into a pack
  costs nothing, so any strategy that gives up track position looks free.
- **No pit-loss variance.** Every stop becomes exactly average: no slow wheel
  gun, no bad release. Each extra stop is a fixed cost with no tail, so
  multi-stop strategies stop carrying execution risk.

Two numbers are reported per ablation. **Strategy divergence** is whether the
degraded simulator recommends something different at all. **Regret** is the
cost of that: the recommendation from the degraded model, scored under the
*calibrated* model, against the calibrated model's own best. Regret is the
honest measure, because a naive model always rates its own choice highly --
the question is how that choice performs under the model we believe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pitwall.degradation.fit import load_posterior
from pitwall.optimize.candidates import compound_shortlist, enumerate_candidates
from pitwall.optimize.mc import evaluate_candidates, rank
from pitwall.paths import Paths
from pitwall.rng import SeedBank
from pitwall.sim.engine import _sample_schedule
from pitwall.sim.field import build_field
from pitwall.sim.params import SimParams, circuit_profiles, load_sim_params
from pitwall.sim.strategy import Strategy

log = logging.getLogger(__name__)

__all__ = ["ABLATIONS", "AblationResult", "run_ablation"]


# name -> SimParams overrides that switch the feature off.
ABLATIONS: dict[str, dict[str, object]] = {
    "calibrated": {},
    "no_safety_car": {"sc_enabled": False, "sc_per_race": 0.0, "vsc_per_race": 0.0},
    "deterministic_degradation": {"deg_stochastic": False, "deg_use_posterior_mean": True},
    "no_traffic": {
        "dirty_air_max_loss_s": 0.0,
        "emergence_penalty_s": 0.0,
        "min_following_gap_s": 0.0,
        # Passing becomes free: the overtake probability saturates at 1 so the
        # position constraint never binds.
        "overtake_intercept": 12.0,
    },
    "no_pit_variance": {"pit_loss_sd_s": 0.0, "stop_time_sd_s": 0.0, "botch_prob": 0.0},
    "no_reliability": {"reliability_enabled": False},
    "naive_all": {
        "sc_enabled": False,
        "sc_per_race": 0.0,
        "vsc_per_race": 0.0,
        "deg_stochastic": False,
        "deg_use_posterior_mean": True,
        "dirty_air_max_loss_s": 0.0,
        "emergence_penalty_s": 0.0,
        "min_following_gap_s": 0.0,
        "overtake_intercept": 12.0,
        "pit_loss_sd_s": 0.0,
        "stop_time_sd_s": 0.0,
        "botch_prob": 0.0,
        "reliability_enabled": False,
    },
}


@dataclass
class AblationResult:
    table: pd.DataFrame
    per_race: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "Realism ablation",
            "=" * 96,
            "",
            "Each row is a simulator with one feature removed. It picks its own best",
            "strategy, and that strategy is then scored under the calibrated simulator.",
            "'regret' is what the naive recommendation costs, in expected championship",
            "points, against what the calibrated model would have chosen.",
            "",
            self.table.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
            "",
        ]

        naive = self.table.loc[self.table["ablation"] == "naive_all"]
        if not naive.empty:
            row = naive.iloc[0]
            lines += [
                "Stripping every realism feature at once changes the recommended stop",
                f"count in {row['stop_count_differs']:.0%} of cases and costs",
                f"{row['regret_points']:.3f} expected points against the calibrated model,",
                "while the naive model believes its own choice is worth",
                f"{row['self_reported_points']:.3f} points -- an overstatement of",
                f"{row['optimism_points']:.3f}.",
            ]
        return "\n".join(lines)


def _ablated(params: SimParams, name: str) -> SimParams:
    overrides = ABLATIONS[name]
    return params.without(**overrides) if overrides else params


def _search(
    params: SimParams,
    field_data: object,
    focal_car: int,
    seed: int,
    cfg: DictConfig,
    n_races: int,
    schedule: object,
) -> list:
    """A compact version of the Component 3 search, shared by every arm."""
    search = cfg.optimizer.search
    posterior = params.posterior
    index = {c: posterior.compound_index(c) for c in posterior.compounds}
    fd = field_data

    shortlist = compound_shortlist(
        race_laps=params.race_laps,
        stop_counts=list(search.stop_counts),
        compounds=list(posterior.compounds),
        require_two_compounds=bool(search.enforce_two_compound_rule),
    )
    ranked = rank(
        evaluate_candidates(
            params,
            shortlist,
            fd.strategies,  # type: ignore[attr-defined]
            focal_car,
            fd.grid,  # type: ignore[attr-defined]
            fd.pace_offsets,  # type: ignore[attr-defined]
            n_races=n_races,
            seed=seed,
            schedule=schedule,  # type: ignore[arg-type]
            compound_index=index,
        ),
        str(cfg.optimizer.objective.primary),
    )
    best_sequences = [e.strategy.compounds for e in ranked[: int(search.keep_sequences)]]

    candidates = enumerate_candidates(
        race_laps=params.race_laps,
        stop_counts=list(search.stop_counts),
        compounds=list(posterior.compounds),
        lap_grid_step=int(search.lap_grid_step),
        min_stint_laps=int(search.min_stint_laps),
        require_two_compounds=bool(search.enforce_two_compound_rule),
        allowed_sequences=best_sequences,
    )
    return rank(
        evaluate_candidates(
            params,
            candidates,
            fd.strategies,  # type: ignore[attr-defined]
            focal_car,
            fd.grid,  # type: ignore[attr-defined]
            fd.pace_offsets,  # type: ignore[attr-defined]
            n_races=n_races,
            seed=seed,
            schedule=schedule,  # type: ignore[arg-type]
            compound_index=index,
        ),
        str(cfg.optimizer.objective.primary),
    )


def run_ablation(
    cfg: DictConfig,
    races: list[tuple[int, int]] | None = None,
    n_races: int | None = None,
    cars: tuple[int, ...] = (0, 4, 9),
) -> AblationResult:
    """Run every ablation arm over a set of races and score the regret."""
    paths = Paths.from_config(cfg).ensure()
    posterior = load_posterior(paths, list(cfg.data.train_seasons))
    profiles = circuit_profiles(cfg, paths)
    bank = SeedBank(int(cfg.seed))
    ensemble = int(n_races or cfg.optimizer.monte_carlo.n_races_screen)

    if races is None:
        holdout = list(cfg.data.holdout_seasons)
        races = [(holdout[0], r) for r in (1, 5, 9, 13, 17, 21)]

    rows: list[dict[str, object]] = []

    for season, round_no in races:
        try:
            field_data = build_field(season, round_no, paths, posterior)
        except Exception as exc:
            log.warning("skipping %s r%02d: %s", season, round_no, exc)
            continue

        calibrated = load_sim_params(
            cfg,
            paths,
            posterior,
            circuit=field_data.circuit,
            race_laps=field_data.race_laps,
            profiles=profiles,
        ).without(n_cars=field_data.n_cars)

        index = {c: posterior.compound_index(c) for c in posterior.compounds}

        for car in cars:
            if car >= field_data.n_cars:
                continue
            seed = int(bank.generator(f"ablate/{season}/{round_no}/{car}").integers(0, 2**31 - 1))
            # One schedule for the whole comparison at this race, so every arm
            # is judged against the same sampled races.
            schedule = _sample_schedule(calibrated, np.random.default_rng(seed), ensemble)

            truth_ranked = _search(calibrated, field_data, car, seed, cfg, ensemble, schedule)
            truth_best: Strategy = truth_ranked[0].strategy
            truth_best_points = truth_ranked[0].expected_points

            for name in ABLATIONS:
                arm_params = _ablated(calibrated, name)
                arm_schedule = (
                    schedule
                    if arm_params.sc_enabled
                    else _sample_schedule(arm_params, np.random.default_rng(seed), ensemble)
                )
                arm_ranked = _search(arm_params, field_data, car, seed, cfg, ensemble, arm_schedule)
                arm_best: Strategy = arm_ranked[0].strategy

                # The whole point of the exercise: score the naive model's pick
                # under the calibrated model, not under the one that chose it.
                under_truth = evaluate_candidates(
                    calibrated,
                    [arm_best],
                    field_data.strategies,
                    car,
                    field_data.grid,
                    field_data.pace_offsets,
                    n_races=ensemble,
                    seed=seed,
                    schedule=schedule,
                    compound_index=index,
                )[0].expected_points
                rows.append(
                    {
                        "season": season,
                        "round": round_no,
                        "event": field_data.event,
                        "driver": field_data.drivers[car],
                        "grid": field_data.grid[car],
                        "ablation": name,
                        "recommended": arm_best.label(),
                        "recommended_stops": arm_best.n_stops,
                        "calibrated_recommended": truth_best.label(),
                        "calibrated_stops": truth_best.n_stops,
                        "self_reported_points": arm_ranked[0].expected_points,
                        "points_under_calibrated": under_truth,
                        "calibrated_best_points": truth_best_points,
                        "regret_points": truth_best_points - under_truth,
                        "optimism_points": arm_ranked[0].expected_points - under_truth,
                        "strategy_differs": arm_best.label() != truth_best.label(),
                        "stop_count_differs": arm_best.n_stops != truth_best.n_stops,
                    }
                )
                log.info(
                    "%s r%02d %s [%s] -> %s (regret %.3f)",
                    season,
                    round_no,
                    field_data.drivers[car],
                    name,
                    arm_best.label(),
                    truth_best_points - under_truth,
                )

    per_race = pd.DataFrame(rows)
    if per_race.empty:
        raise RuntimeError("ablation produced no rows; are the races ingested?")

    table = (
        per_race.groupby("ablation", as_index=False)
        .agg(
            n=("regret_points", "size"),
            regret_points=("regret_points", "mean"),
            self_reported_points=("self_reported_points", "mean"),
            points_under_calibrated=("points_under_calibrated", "mean"),
            optimism_points=("optimism_points", "mean"),
            strategy_differs=("strategy_differs", "mean"),
            stop_count_differs=("stop_count_differs", "mean"),
            mean_stops=("recommended_stops", "mean"),
        )
        .sort_values("regret_points", ascending=False)
        .reset_index(drop=True)
    )

    result = AblationResult(
        table=table,
        per_race=per_race,
        summary={"n_races": float(per_race[["season", "round"]].drop_duplicates().shape[0])},
    )

    target = paths.artifacts / "ablation"
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / "summary.csv", index=False)
    per_race.to_csv(target / "per_race.csv", index=False)
    (target / "report.txt").write_text(result.render(), encoding="utf-8")
    log.info("wrote ablation artefacts to %s", target)

    print(result.render())
    return result
