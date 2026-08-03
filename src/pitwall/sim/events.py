"""Safety car and virtual safety car deployment as a fitted hazard process.

A neutralisation is the single largest source of variance in a race result and
by far the most important thing a strategy model has to get right. It changes
the cost of a pit stop by roughly a factor of two, and it arrives without
warning, which is exactly why a *distribution* over outcomes is the useful
output and a point estimate is not.

The process
-----------

Deployments are modelled as an inhomogeneous Bernoulli process over laps. The
per-lap hazard at circuit ``c`` is

    h_c(l) proportional to  m_1 if l == 1 else exp(-k * l / L)

normalised so that the expected number of deployments over the race equals the
circuit's observed rate. Two features of the data drive the shape:

*Lap one is different.* Standing starts, a full field in close company and
cold tyres put a disproportionate share of first-lap incidents into the record.
It gets its own multiplier rather than being smoothed into the trend.

*Hazard decays through the race.* Early-race incidents dominate: the field is
bunched, drivers are racing hard on similar strategies, and there are simply
more cars running. The exponential decay is the simplest form that captures it
without pretending to more structure than four seasons can identify.

Duration is drawn from a truncated normal fitted to observed deployment
lengths. A full safety car and a VSC have quite different durations and quite
different consequences, so they are drawn separately, with the split between
them set by the observed share.

What this deliberately does not model
-------------------------------------

Deployments are independent of the race state. In reality a safety car is
*caused* by an incident, and incidents correlate with close racing, first-lap
chaos and rain. Modelling that endogeneity properly needs incident data this
project does not have, and the honest consequence is that the simulator cannot
represent "a safety car is more likely because the field is bunched". The
lap-one multiplier is a crude stand-in for the largest part of that effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NeutralisationSchedule", "sample_neutralisations"]

# Regime codes used throughout the engine. Ordered by severity so that
# ``maximum`` composes correctly when two overlap.
GREEN = 0
VSC = 1
SAFETY_CAR = 2


@dataclass(frozen=True)
class NeutralisationSchedule:
    """Per-race, per-lap neutralisation regime for a whole ensemble.

    ``regime`` is ``(n_races, race_laps)`` holding GREEN / VSC / SAFETY_CAR.
    ``deploy_lap`` marks the lap a deployment *began*, which is what the
    reactive policy triggers on: a car reacts to the moment the boards come
    out, not to being three laps into a neutralisation.
    """

    regime: np.ndarray
    deploy_lap: np.ndarray

    @property
    def n_races(self) -> int:
        return int(self.regime.shape[0])

    @property
    def race_laps(self) -> int:
        return int(self.regime.shape[1])

    def any_deployment(self) -> np.ndarray:
        # asarray because ndarray.any() is typed as returning a scalar-or-array
        # union: with axis= it is always an array, but the stubs cannot know it.
        return np.asarray(self.deploy_lap.any(axis=1))

    def subset(self, mask: np.ndarray) -> NeutralisationSchedule:
        return NeutralisationSchedule(regime=self.regime[mask], deploy_lap=self.deploy_lap[mask])

    def summary(self) -> dict[str, float]:
        sc_laps = float((self.regime == SAFETY_CAR).mean())
        vsc_laps = float((self.regime == VSC).mean())
        return {
            "p_any_neutralisation": float(self.any_deployment().mean()),
            "mean_deployments_per_race": float(self.deploy_lap.sum(axis=1).mean()),
            "share_laps_under_sc": sc_laps,
            "share_laps_under_vsc": vsc_laps,
        }


def hazard_profile(race_laps: int, lap1_multiplier: float, decay: float) -> np.ndarray:
    """Unnormalised per-lap deployment weight."""
    laps = np.arange(1, race_laps + 1, dtype=float)
    weight = np.exp(-decay * laps / race_laps)
    weight[0] *= lap1_multiplier
    return weight


def sample_neutralisations(
    rng: np.random.Generator,
    n_races: int,
    race_laps: int,
    sc_per_race: float,
    vsc_per_race: float,
    lap1_multiplier: float,
    decay: float,
    sc_duration_mean: float,
    sc_duration_sd: float,
    vsc_duration_mean: float,
    vsc_duration_sd: float,
    enabled: bool = True,
) -> NeutralisationSchedule:
    """Draw a neutralisation schedule for every race in an ensemble.

    Deployments are drawn lap by lap so that a race already under a
    neutralisation cannot start another one, which is what makes the expected
    count match the fitted rate rather than overshooting it.
    """
    regime = np.zeros((n_races, race_laps), dtype=np.int8)
    deploy = np.zeros((n_races, race_laps), dtype=bool)
    if not enabled or race_laps <= 0:
        return NeutralisationSchedule(regime, deploy)

    weight = hazard_profile(race_laps, lap1_multiplier, decay)
    total_rate = max(float(sc_per_race + vsc_per_race), 0.0)
    if total_rate <= 0:
        return NeutralisationSchedule(regime, deploy)

    # Scale the weights so the expected number of deployments matches the
    # circuit's observed rate. Clipped below 1 because a hazard cannot be a
    # probability greater than one at a very incident-prone circuit.
    per_lap = np.clip(weight * total_rate / weight.sum(), 0.0, 0.95)
    full_sc_share = float(sc_per_race / total_rate) if total_rate > 0 else 0.0

    # Laps remaining under an active neutralisation, per race.
    busy = np.zeros(n_races, dtype=np.int64)

    for lap in range(race_laps):
        available = busy <= 0
        triggered = available & (rng.random(n_races) < per_lap[lap])
        if triggered.any():
            is_full_sc = rng.random(n_races) < full_sc_share
            mean = np.where(is_full_sc, sc_duration_mean, vsc_duration_mean)
            sd = np.where(is_full_sc, sc_duration_sd, vsc_duration_sd)
            duration = np.rint(rng.normal(mean, sd)).astype(np.int64)
            duration = np.clip(duration, 1, max(1, race_laps - lap))

            kind = np.where(is_full_sc, SAFETY_CAR, VSC).astype(np.int8)
            rows = np.nonzero(triggered)[0]
            deploy[rows, lap] = True
            for row in rows:
                end = min(race_laps, lap + int(duration[row]))
                regime[row, lap:end] = kind[row]
                busy[row] = end - lap

        busy = np.maximum(busy - 1, 0)

    return NeutralisationSchedule(regime, deploy)
