"""Everything the engine needs, resolved once per race.

Pulls together three sources: the Hydra config (behavioural switches and the
parameters that are not identifiable from data), the per-circuit table
estimated in :mod:`pitwall.ingest.circuits`, and the fitted degradation
posterior. Resolving them here keeps the engine free of config lookups in its
inner loop and makes an ablation a matter of handing it a different
:class:`SimParams`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pitwall.degradation.model import DegradationPosterior
from pitwall.ingest.circuits import build_circuit_profiles
from pitwall.ingest.fetch import load_seasons
from pitwall.paths import Paths

log = logging.getLogger(__name__)

__all__ = ["SimParams", "circuit_profiles", "load_sim_params"]


def circuit_profiles(cfg: DictConfig, paths: Paths, refresh: bool = False) -> pd.DataFrame:
    """Per-circuit table, cached to parquet after the first build.

    Built from the training seasons only. Using the held-out season here would
    leak information about how often the safety car came out in the races the
    backtest is about to score.
    """
    target = paths.artifacts / "circuit_profiles.parquet"
    if target.is_file() and not refresh:
        return pd.read_parquet(target)

    seasons = list(cfg.data.train_seasons)
    log.info("building circuit profiles from seasons %s", seasons)
    tables = load_seasons(seasons, paths)
    profiles = build_circuit_profiles(tables)
    paths.ensure()
    profiles.to_parquet(target, index=False)
    return profiles


@dataclass(frozen=True)
class SimParams:
    """Resolved simulation parameters for one circuit."""

    circuit: str
    race_laps: int
    n_cars: int

    # Degradation
    posterior: DegradationPosterior
    deg_stochastic: bool
    deg_use_posterior_mean: bool

    # Pace
    base_lap_s: float
    field_spread_s: float
    driver_noise_sd_s: float
    start_position_sd: float
    fuel_s_per_kg: float
    fuel_start_mass_kg: float

    # Pit
    pit_loss_s: float
    pit_loss_sd_s: float
    stop_time_sd_s: float
    botch_prob: float
    botch_extra_mean_s: float
    sc_loss_multiplier: float
    vsc_loss_multiplier: float

    # Traffic
    dirty_air_threshold_s: float
    dirty_air_max_loss_s: float
    emergence_penalty_s: float
    min_following_gap_s: float

    # Overtaking
    overtake_intercept: float
    overtake_pace_coef: float
    overtake_difficulty: float
    failed_attempt_cost_s: float
    min_gap_to_attempt_s: float

    # Neutralisations
    sc_enabled: bool
    sc_per_race: float
    vsc_per_race: float
    sc_lap1_multiplier: float
    sc_decay: float
    sc_duration_mean: float
    sc_duration_sd: float
    vsc_duration_mean: float
    vsc_duration_sd: float
    sc_lap_time_multiplier: float
    vsc_lap_time_multiplier: float
    sc_bunch_gap_s: float

    # Reliability
    reliability_enabled: bool
    mechanical_per_lap: float
    incident_per_lap: float
    lap1_incident_prob: float

    def without(self, **overrides: object) -> SimParams:
        """A copy with fields replaced. Used by the ablation study."""
        return replace(self, **overrides)  # type: ignore[arg-type]


def load_sim_params(
    cfg: DictConfig,
    paths: Paths,
    posterior: DegradationPosterior,
    circuit: str,
    race_laps: int | None = None,
    profiles: pd.DataFrame | None = None,
) -> SimParams:
    """Resolve config plus circuit estimates into a :class:`SimParams`."""
    sim = cfg.simulator
    if profiles is None:
        profiles = circuit_profiles(cfg, paths)

    row = profiles.loc[profiles["circuit"] == circuit]
    if row.empty:
        # An unseen circuit falls back to the field-wide medians. The
        # degradation posterior handles the same case by drawing from the
        # population, so both halves of the model degrade the same way.
        log.warning("no circuit profile for %r; using median circuit values", circuit)
        record = profiles.median(numeric_only=True).to_dict()
    else:
        record = row.iloc[0].to_dict()

    laps = int(race_laps if race_laps is not None else record.get("typical_laps", 57))

    return SimParams(
        circuit=circuit,
        race_laps=laps,
        n_cars=int(sim.n_cars),
        posterior=posterior,
        deg_stochastic=bool(sim.degradation.stochastic),
        deg_use_posterior_mean=str(sim.degradation.source) == "posterior_mean",
        base_lap_s=float(record.get("median_green_lap_s", 90.0)),
        field_spread_s=float(sim.pace.field_spread_s),
        driver_noise_sd_s=float(sim.pace.driver_noise_sd_s),
        start_position_sd=float(sim.pace.start_position_sd),
        fuel_s_per_kg=float(np.median(posterior.phi)),
        fuel_start_mass_kg=float(cfg.degradation.fuel.start_mass_kg),
        pit_loss_s=float(record.get("pit_loss_s", 22.0)),
        pit_loss_sd_s=float(record.get("pit_loss_sd_s", 2.0)),
        stop_time_sd_s=float(sim.pit.stop_time_sd_s),
        botch_prob=float(sim.pit.botch_prob),
        botch_extra_mean_s=float(sim.pit.botch_extra_mean_s),
        sc_loss_multiplier=float(sim.pit.sc_loss_multiplier),
        vsc_loss_multiplier=float(sim.pit.vsc_loss_multiplier),
        dirty_air_threshold_s=float(sim.traffic.dirty_air_threshold_s),
        dirty_air_max_loss_s=float(sim.traffic.max_loss_s),
        emergence_penalty_s=float(sim.traffic.emergence_penalty_s),
        min_following_gap_s=0.35,
        overtake_intercept=float(sim.overtake.intercept),
        overtake_pace_coef=float(sim.overtake.pace_delta_coef),
        overtake_difficulty=float(record.get("overtake_difficulty", 1.0)),
        failed_attempt_cost_s=float(sim.overtake.failed_attempt_cost_s),
        min_gap_to_attempt_s=float(sim.overtake.min_gap_to_attempt_s),
        sc_enabled=bool(sim.safety_car.enabled),
        sc_per_race=float(record.get("sc_per_race", 0.4)),
        vsc_per_race=float(record.get("vsc_per_race", 0.4)),
        sc_lap1_multiplier=float(sim.safety_car.lap1_multiplier),
        sc_decay=float(sim.safety_car.decay_per_race_fraction),
        sc_duration_mean=float(
            record.get("sc_duration_laps", sim.safety_car.sc_duration_laps_mean)
        ),
        sc_duration_sd=float(sim.safety_car.sc_duration_laps_sd),
        vsc_duration_mean=float(
            record.get("vsc_duration_laps", sim.safety_car.vsc_duration_laps_mean)
        ),
        vsc_duration_sd=float(sim.safety_car.vsc_duration_laps_sd),
        sc_lap_time_multiplier=float(sim.safety_car.sc_lap_time_multiplier),
        vsc_lap_time_multiplier=float(sim.safety_car.vsc_lap_time_multiplier),
        sc_bunch_gap_s=float(sim.safety_car.sc_bunch_gap_s),
        reliability_enabled=bool(sim.reliability.enabled),
        mechanical_per_lap=float(sim.reliability.mechanical_per_lap),
        incident_per_lap=float(sim.reliability.incident_per_lap),
        lap1_incident_prob=float(sim.reliability.lap1_incident_prob),
    )


def posterior_for(paths: Paths, seasons: list[int]) -> DegradationPosterior:
    from pitwall.degradation.fit import load_posterior

    return load_posterior(paths, seasons)


def profile_path(paths: Paths) -> Path:
    return paths.artifacts / "circuit_profiles.parquet"
