"""Shared fixtures.

Everything here is synthetic. The tests must run in CI with no FastF1 cache and
no network, so nothing in this file touches ``data/``. Tests that genuinely
need real session data are marked ``network`` and excluded from the CI run.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pitwall.degradation.gibbs import PosteriorDraws
from pitwall.degradation.model import DegradationPosterior
from pitwall.sim.params import SimParams

CIRCUITS = ["Alpha", "Beta"]
COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
DRIVERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def make_posterior(
    n_samples: int = 64,
    circuits: list[str] | None = None,
    compounds: list[str] | None = None,
    drivers: list[str] | None = None,
    seed: int = 7,
) -> DegradationPosterior:
    """A small, well-behaved posterior with physically sensible coefficients."""
    circuits = circuits or CIRCUITS
    compounds = compounds or COMPOUNDS
    drivers = drivers or DRIVERS
    generator = np.random.default_rng(seed)

    n_c, n_k, n_d = len(circuits), len(compounds), len(drivers)
    beta = np.zeros((n_samples, n_c, n_k, 3))
    # Wear rises from hard to soft; offsets make the softer tyre quicker fresh.
    linear = {"HARD": 0.55, "MEDIUM": 0.95, "SOFT": 1.40}
    offset = {"HARD": 0.45, "MEDIUM": 0.0, "SOFT": -0.35}
    for ki, compound in enumerate(compounds):
        beta[:, :, ki, 0] = offset[compound] + 0.02 * generator.standard_normal((n_samples, n_c))
        beta[:, :, ki, 1] = linear[compound] + 0.05 * generator.standard_normal((n_samples, n_c))
        beta[:, :, ki, 2] = 0.10 + 0.02 * generator.standard_normal((n_samples, n_c))

    active = np.ones((n_c, n_k, 3), dtype=bool)
    active[:, compounds.index("MEDIUM"), 0] = False

    return DegradationPosterior(
        beta=beta,
        mu=beta.mean(axis=1),
        tau2=np.full((n_samples, n_k, 3), 0.04),
        driver=0.05 * generator.standard_normal((n_samples, n_d, n_k)),
        phi=np.full(n_samples, 0.029),
        theta=np.zeros((n_samples, n_c)),
        sigma=np.full(n_samples, 0.25),
        active=active,
        max_age=np.full((n_c, n_k), 40.0),
        circuits=list(circuits),
        compounds=list(compounds),
        drivers=list(drivers),
        age_scale_laps=20.0,
        age_center_laps=15.0,
        fuel_mean_kg=55.0,
        circuit_scale=np.ones(n_c),
        quadratic=True,
    )


@pytest.fixture
def posterior() -> DegradationPosterior:
    return make_posterior()


def make_params(posterior: DegradationPosterior, **overrides: object) -> SimParams:
    """Simulation parameters with every realism feature switched on."""
    base = SimParams(
        circuit="Alpha",
        race_laps=40,
        n_cars=8,
        posterior=posterior,
        deg_stochastic=True,
        deg_use_posterior_mean=False,
        base_lap_s=90.0,
        field_spread_s=1.3,
        driver_noise_sd_s=0.16,
        start_position_sd=1.0,
        fuel_s_per_kg=0.029,
        fuel_start_mass_kg=110.0,
        pit_loss_s=22.0,
        pit_loss_sd_s=2.0,
        stop_time_sd_s=0.35,
        botch_prob=0.02,
        botch_extra_mean_s=6.0,
        sc_loss_multiplier=0.42,
        vsc_loss_multiplier=0.58,
        dirty_air_threshold_s=1.6,
        dirty_air_max_loss_s=0.45,
        emergence_penalty_s=0.30,
        min_following_gap_s=0.35,
        overtake_intercept=-2.2,
        overtake_pace_coef=2.35,
        overtake_difficulty=1.0,
        failed_attempt_cost_s=0.15,
        min_gap_to_attempt_s=1.2,
        sc_enabled=True,
        sc_per_race=0.5,
        vsc_per_race=0.4,
        sc_lap1_multiplier=3.4,
        sc_decay=0.55,
        sc_duration_mean=3.6,
        sc_duration_sd=1.4,
        vsc_duration_mean=2.1,
        vsc_duration_sd=0.9,
        sc_lap_time_multiplier=1.42,
        vsc_lap_time_multiplier=1.33,
        sc_bunch_gap_s=1.1,
        reliability_enabled=True,
        mechanical_per_lap=0.00075,
        incident_per_lap=0.0006,
        lap1_incident_prob=0.012,
    )
    return base.without(**overrides) if overrides else base


@pytest.fixture
def params(posterior: DegradationPosterior) -> SimParams:
    return make_params(posterior)


@pytest.fixture
def cleaning_config() -> object:
    """Stand-in for the ``data.cleaning`` config node."""

    class Cleaning:
        compounds: ClassVar[list[str]] = ["SOFT", "MEDIUM", "HARD"]
        exclude_events: ClassVar[list[str]] = []
        drop_out_laps = True
        drop_in_laps = True
        drop_lap_one = True
        require_is_accurate = True
        drop_deleted = True
        green_flag_only = True
        max_lap_ratio_to_stint_median = 1.07
        min_stint_laps = 5
        min_gap_ahead_s = 1.5
        max_rain_lap_share = 0.15
        drop_rain_laps = True
        max_wet_tyre_lap_share = 0.05

    return Cleaning()


def synthetic_laps(n_drivers: int = 6, race_laps: int = 40, seed: int = 3) -> pd.DataFrame:
    """A raw-shaped lap table with known structure, for the cleaning tests."""
    generator = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        driver = f"D{d:02d}"
        stop = race_laps // 2
        for lap in range(1, race_laps + 1):
            stint = 1 if lap <= stop else 2
            age = lap if stint == 1 else lap - stop
            compound = "MEDIUM" if stint == 1 else "HARD"
            rows.append(
                {
                    "season": 2024,
                    "round": 1,
                    "event": "Test Grand Prix",
                    "circuit": "Alpha",
                    "driver": driver,
                    "team": f"T{d // 2}",
                    "lap_number": lap,
                    "lap_time_s": 90.0 + 0.05 * age + generator.normal(0, 0.15),
                    "stint": stint,
                    "compound": compound,
                    "tyre_life": float(age),
                    "fresh_tyre": age == 1,
                    "track_status": "1",
                    "position": float(d + 1),
                    "is_accurate": True,
                    "deleted": False,
                    "is_in_lap": lap == stop,
                    "is_out_lap": lap == stop + 1,
                    "lap_start_s": 1000.0 + (lap - 1) * 90.0 + d * 4.0,
                    "air_temp_c": 25.0,
                    "track_temp_c": 35.0,
                    "rainfall": False,
                }
            )
    return pd.DataFrame(rows)


def synthetic_race_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 1,
                "event": "Test Grand Prix",
                "circuit": "Alpha",
                "country": "Testland",
                "date": "2024-03-01",
                "scheduled_laps": 40,
                "completed_laps": 40,
                "n_drivers": 6,
                "any_rainfall": False,
                "mean_track_temp_c": 35.0,
                "mean_air_temp_c": 25.0,
                "red_flagged": False,
            }
        ]
    )


def make_draws(chains: int = 2, draws: int = 40, seed: int = 1) -> PosteriorDraws:
    generator = np.random.default_rng(seed)
    n_c, n_k, n_d = 2, 3, 4
    return PosteriorDraws(
        beta=generator.standard_normal((chains, draws, n_c, n_k, 3)),
        mu=generator.standard_normal((chains, draws, n_k, 3)),
        tau2=np.abs(generator.standard_normal((chains, draws, n_k, 3))) + 0.1,
        phi=generator.normal(0.03, 0.001, (chains, draws)),
        theta=generator.standard_normal((chains, draws, n_c)),
        driver=generator.standard_normal((chains, draws, n_d, n_k)),
        sigma2=np.abs(generator.standard_normal((chains, draws))) + 0.1,
        sigma_u2=np.abs(generator.standard_normal((chains, draws))) + 0.1,
        active=np.ones((n_c, n_k, 3), dtype=bool),
        max_age=np.full((n_c, n_k), 35.0),
        circuits=CIRCUITS,
        compounds=COMPOUNDS,
        drivers=DRIVERS,
        age_scale_laps=20.0,
        age_center_laps=15.0,
        fuel_mean_kg=55.0,
        circuit_scale=np.ones(n_c),
        quadratic=True,
    )
