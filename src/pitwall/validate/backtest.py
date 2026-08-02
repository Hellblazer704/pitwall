"""Backtesting on a held-out season.

Nothing in the model has seen 2025. The degradation posterior, the per-circuit
pit-loss and neutralisation estimates and the overtaking index are all fitted
on 2022-2024 only.

Three questions, in increasing order of how much they tell you:

1. **Is the simulator calibrated?** Replay each race with the strategies the
   teams actually ran and compare the simulated finishing order to the real
   one. This tests the race model in isolation, because the strategy input is
   the true one. If this is wrong, nothing downstream can be right.

2. **Are the reported probabilities honest?** A model that says "38% chance of
   a podium" should be right about 38% of the time when it says that. Checked
   with a reliability curve and a Brier score, because a simulator can have a
   good mean finishing position and still be badly overconfident.

3. **Does the optimiser's recommendation beat what the teams ran?** Run the
   search for each car and compare the recommended strategy against the actual
   one, in the simulator. This is the weakest of the three tests and the most
   easily misread: the comparison happens inside the model, so it measures
   whether the recommendation is better *according to the simulator*, and a
   simulator with a bias will happily rate its own preferred strategies highly.
   It is reported alongside how often the recommendation matched the real stop
   count, which is a check against reality rather than against itself.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pitwall.degradation.fit import load_posterior
from pitwall.ingest.fetch import RaceTables
from pitwall.optimize.mc import points_for
from pitwall.paths import Paths
from pitwall.rng import SeedBank
from pitwall.sim.engine import simulate_ensemble
from pitwall.sim.field import RaceField, build_field
from pitwall.sim.params import circuit_profiles, load_sim_params
from pitwall.sim.strategy import plan_to_matrix

log = logging.getLogger(__name__)

__all__ = ["BacktestResult", "run_backtest"]


@dataclass
class BacktestResult:
    per_race: pd.DataFrame
    per_car: pd.DataFrame
    calibration: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "Backtest on held-out races",
            "=" * 78,
            "",
            f"races        : {int(self.summary.get('n_races', 0))}",
            f"cars scored  : {int(self.summary.get('n_cars', 0))}",
            "",
            "Replaying the strategies teams actually ran:",
            f"  mean absolute position error : {self.summary.get('mae_position', np.nan):.2f}",
            f"  median absolute error        : {self.summary.get('medae_position', np.nan):.2f}",
            f"  Spearman rank correlation    : {self.summary.get('spearman', float('nan')):.3f}",
            f"  exact position hit rate      : {self.summary.get('exact_rate', float('nan')):.3f}",
            f"  within one place             : {self.summary.get('within_one', float('nan')):.3f}",
            "",
            "Probability calibration:",
            f"  Brier score, P(points)  : {self.summary.get('brier_points', float('nan')):.4f}",
            f"  Brier score, P(podium)  : {self.summary.get('brier_podium', float('nan')):.4f}",
            f"  baseline (climatology)  : {self.summary.get('brier_baseline', float('nan')):.4f}",
            "",
            self.calibration.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
            "",
            "Worst-predicted races (mean absolute position error):",
            self.per_race.sort_values("mae_position", ascending=False)
            .head(8)
            .to_string(index=False, float_format=lambda v: f"{v:.2f}"),
        ]
        return "\n".join(lines)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _reliability(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Predicted probability against observed frequency, in bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_rate": float(outcomes[mask].mean()),
                "gap": float(outcomes[mask].mean() - probabilities[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def replay_race(
    cfg: DictConfig,
    paths: Paths,
    field_data: RaceField,
    seed: int,
    n_races: int,
    profiles: pd.DataFrame,
    posterior: object,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate one real race with the strategies the teams actually ran."""
    params = load_sim_params(
        cfg,
        paths,
        posterior,  # type: ignore[arg-type]
        circuit=field_data.circuit,
        race_laps=field_data.race_laps,
        profiles=profiles,
    ).without(n_cars=field_data.n_cars)

    index = {c: params.posterior.compound_index(c) for c in params.posterior.compounds}
    pit_after, compound_by_lap = plan_to_matrix(field_data.strategies, field_data.race_laps, index)

    outcome = simulate_ensemble(
        params,
        np.random.default_rng(seed),
        n_races,
        field_data.grid,
        field_data.pace_offsets,
        pit_after,
        compound_by_lap,
    )

    positions = outcome.finish_position  # (n_races, n_cars)
    predicted_mean = positions.mean(axis=0)
    p_points = (positions <= 10).mean(axis=0)
    p_podium = (positions <= 3).mean(axis=0)
    p_win = (positions == 1).mean(axis=0)

    actual = field_data.actual_finish.astype(float)
    # Unclassified cars have no finishing position. They are dropped from the
    # position-error metrics (there is no truth to compare against) but the
    # simulator's own retirement rate is reported separately.
    valid = np.isfinite(actual)

    frame = pd.DataFrame(
        {
            "season": field_data.season,
            "round": field_data.round,
            "event": field_data.event,
            "circuit": field_data.circuit,
            "driver": field_data.drivers,
            "team": field_data.teams,
            "grid": field_data.grid,
            "actual_finish": actual,
            "actual_classified": field_data.finished,
            "predicted_mean_position": predicted_mean,
            "predicted_median_position": np.median(positions, axis=0),
            "p_win": p_win,
            "p_podium": p_podium,
            "p_points": p_points,
            "p_retired": outcome.state.retired.mean(axis=0),
            "strategy": [s.label() for s in field_data.strategies],
            "abs_error": np.where(valid, np.abs(predicted_mean - actual), np.nan),
        }
    )

    stats = {
        "mae_position": float(np.nanmean(frame["abs_error"])),
        "spearman": _spearman(predicted_mean[valid], actual[valid]),
        "n_valid": int(valid.sum()),
    }
    return frame, stats


def run_backtest(
    cfg: DictConfig,
    seasons: list[int],
    limit: int | None = None,
    n_races: int | None = None,
) -> BacktestResult:
    paths = Paths.from_config(cfg).ensure()
    posterior = load_posterior(paths, list(cfg.data.train_seasons))
    profiles = circuit_profiles(cfg, paths)
    bank = SeedBank(int(cfg.seed))
    ensemble = int(n_races or cfg.optimizer.monte_carlo.n_races_final)

    rows: list[pd.DataFrame] = []
    race_rows: list[dict[str, object]] = []

    for season in seasons:
        season_dir = paths.raw / str(season)
        if not season_dir.is_dir():
            log.warning("no ingested races for %s", season)
            continue
        rounds = sorted(
            int(d.name.split("_")[1]) for d in season_dir.iterdir() if RaceTables.complete(d)
        )
        if limit is not None:
            rounds = rounds[:limit]

        for round_no in rounds:
            try:
                field_data = build_field(season, round_no, paths, posterior)
            except Exception as exc:
                log.warning("skipping %s r%02d: %s", season, round_no, exc)
                continue
            if field_data.race_laps < 10 or field_data.n_cars < 10:
                log.warning("skipping %s r%02d: too few laps or cars", season, round_no)
                continue

            seed = int(bank.generator(f"backtest/{season}/{round_no}").integers(0, 2**31 - 1))
            frame, stats = replay_race(cfg, paths, field_data, seed, ensemble, profiles, posterior)
            rows.append(frame)
            race_rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "event": field_data.event,
                    "circuit": field_data.circuit,
                    "n_cars": field_data.n_cars,
                    **stats,
                }
            )
            log.info(
                "%s r%02d %-28s MAE %.2f  rho %.3f",
                season,
                round_no,
                field_data.event[:28],
                stats["mae_position"],
                stats["spearman"],
            )

    if not rows:
        raise FileNotFoundError(f"no ingested races found for seasons {seasons}")

    per_car = pd.concat(rows, ignore_index=True)
    per_race = pd.DataFrame(race_rows)

    valid = per_car["abs_error"].notna()
    scored = per_car.loc[valid]
    errors = scored["abs_error"].to_numpy()

    points_outcome = (scored["actual_finish"] <= 10).to_numpy(dtype=float)
    podium_outcome = (scored["actual_finish"] <= 3).to_numpy(dtype=float)
    p_points = scored["p_points"].to_numpy()
    p_podium = scored["p_podium"].to_numpy()

    calibration = _reliability(p_points, points_outcome)
    calibration.insert(0, "event", "P(points)")

    summary = {
        "n_races": float(len(per_race)),
        "n_cars": float(len(scored)),
        "mae_position": float(errors.mean()),
        "medae_position": float(np.median(errors)),
        "spearman": float(per_race["spearman"].mean()),
        "exact_rate": float(
            (np.abs(scored["predicted_median_position"] - scored["actual_finish"]) < 0.5).mean()
        ),
        "within_one": float(
            (np.abs(scored["predicted_median_position"] - scored["actual_finish"]) <= 1.5).mean()
        ),
        "brier_points": float(np.mean((p_points - points_outcome) ** 2)),
        "brier_podium": float(np.mean((p_podium - podium_outcome) ** 2)),
        # Climatology: predicting the base rate for everyone. A model that
        # cannot beat this has learned nothing race-specific.
        "brier_baseline": float(np.mean((points_outcome.mean() - points_outcome) ** 2)),
    }

    result = BacktestResult(
        per_race=per_race, per_car=per_car, calibration=calibration, summary=summary
    )

    target = paths.artifacts / "backtest"
    target.mkdir(parents=True, exist_ok=True)
    per_car.to_csv(target / "per_car.csv", index=False)
    per_race.to_csv(target / "per_race.csv", index=False)
    calibration.to_csv(target / "calibration.csv", index=False)
    (target / "report.txt").write_text(result.render(), encoding="utf-8")
    log.info("wrote backtest artefacts to %s", target)

    print(result.render())
    return result


def strategy_comparison(
    cfg: DictConfig,
    seasons: list[int],
    limit: int | None = None,
    drivers_per_race: int = 4,
) -> pd.DataFrame:
    """Optimiser recommendation against what the team actually ran.

    Scored inside the simulator, so this answers "is the recommendation better
    according to the model" and not "would it have been better in reality".
    The stop-count agreement column is the part that can be checked against the
    world.
    """
    from pitwall.optimize.run import optimize_for_field

    paths = Paths.from_config(cfg).ensure()
    posterior = load_posterior(paths, list(cfg.data.train_seasons))
    profiles = circuit_profiles(cfg, paths)
    bank = SeedBank(int(cfg.seed))
    rows = []

    for season in seasons:
        season_dir = paths.raw / str(season)
        if not season_dir.is_dir():
            continue
        rounds = sorted(
            int(d.name.split("_")[1]) for d in season_dir.iterdir() if RaceTables.complete(d)
        )
        if limit is not None:
            rounds = rounds[:limit]

        for round_no in rounds:
            try:
                field_data = build_field(season, round_no, paths, posterior)
            except Exception as exc:
                log.warning("skipping %s r%02d: %s", season, round_no, exc)
                continue

            params = load_sim_params(
                cfg,
                paths,
                posterior,
                circuit=field_data.circuit,
                race_laps=field_data.race_laps,
                profiles=profiles,
            ).without(n_cars=field_data.n_cars)
            index = {c: posterior.compound_index(c) for c in posterior.compounds}

            for car in range(min(drivers_per_race, field_data.n_cars)):
                seed = int(
                    bank.generator(f"compare/{season}/{round_no}/{car}").integers(0, 2**31 - 1)
                )
                try:
                    result = optimize_for_field(cfg, params, field_data, car, seed)
                except Exception as exc:
                    log.warning("optimise failed %s r%02d car %d: %s", season, round_no, car, exc)
                    continue

                actual = field_data.strategies[car]
                from pitwall.optimize.mc import evaluate_candidates

                actual_eval = evaluate_candidates(
                    params,
                    [actual],
                    field_data.strategies,
                    car,
                    field_data.grid,
                    field_data.pace_offsets,
                    n_races=int(cfg.optimizer.monte_carlo.n_races_final),
                    seed=seed + 1,
                    compound_index=index,
                )[0]

                rows.append(
                    {
                        "season": season,
                        "round": round_no,
                        "event": field_data.event,
                        "driver": field_data.drivers[car],
                        "grid": field_data.grid[car],
                        "actual_strategy": actual.label(),
                        "actual_stops": actual.n_stops,
                        "recommended_strategy": result.best.strategy.label(),
                        "recommended_stops": result.best.strategy.n_stops,
                        "stop_count_agrees": actual.n_stops == result.best.strategy.n_stops,
                        "actual_expected_points": actual_eval.expected_points,
                        "recommended_expected_points": result.best.expected_points,
                        "points_gain": result.best.expected_points - actual_eval.expected_points,
                        "actual_finish": field_data.actual_finish[car],
                        "actual_points": float(
                            points_for(
                                np.array([int(field_data.actual_finish[car])])
                                if np.isfinite(field_data.actual_finish[car])
                                else np.array([25])
                            )[0]
                        )
                        if np.isfinite(field_data.actual_finish[car])
                        else 0.0,
                    }
                )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        target = paths.artifacts / "backtest"
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / "strategy_comparison.csv", index=False)
    return frame
