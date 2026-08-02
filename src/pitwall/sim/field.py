"""Reconstructing a real race's starting conditions from the ingested tables.

The optimiser needs a grid, a pace for every car and a plan for the cars that
are not being optimised. The backtest needs all of that plus what the teams
actually did. Both come from here so the two cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.degradation.model import DegradationPosterior
from pitwall.ingest.fetch import RaceTables, load_race
from pitwall.paths import Paths
from pitwall.sim.strategy import Strategy

log = logging.getLogger(__name__)

__all__ = ["RaceField", "actual_strategies", "build_field"]

_GREEN = "1"


@dataclass(frozen=True)
class RaceField:
    """The grid, the pace and the strategies actually run."""

    season: int
    round: int
    circuit: str
    event: str
    race_laps: int
    drivers: list[str]
    teams: list[str]
    grid: np.ndarray  # (n_cars,) starting positions, 1-based
    pace_offsets: np.ndarray  # (n_cars,) seconds/lap relative to the quickest car
    strategies: list[Strategy]
    actual_finish: np.ndarray  # (n_cars,) classified position, NaN if unclassified
    finished: np.ndarray  # (n_cars,) bool
    # Share of laps run on inters or full wets. A dry-tyre model cannot
    # represent these races at all, so the backtest reports them separately
    # rather than letting them sink into an average.
    wet_lap_share: float = 0.0

    @property
    def n_cars(self) -> int:
        return len(self.drivers)

    def index_of(self, driver: str) -> int:
        try:
            return self.drivers.index(driver.upper())
        except ValueError as exc:
            raise KeyError(
                f"{driver!r} did not start this race; drivers are {self.drivers}"
            ) from exc

    def summary(self) -> str:
        return (
            f"{self.season} r{self.round:02d} {self.event} ({self.circuit}), "
            f"{self.race_laps} laps, {self.n_cars} cars"
        )


def actual_strategies(tables: RaceTables, drivers: list[str], race_laps: int) -> list[Strategy]:
    """The stint plan each driver actually ran.

    Stint boundaries come from FastF1's stint numbering rather than from
    counting pit laps, because a car that pits under a red flag or serves a
    penalty does not always produce a clean in-lap/out-lap pair.
    """
    laps = tables.laps
    plans: list[Strategy] = []

    for driver in drivers:
        own = laps.loc[laps["driver"] == driver].sort_values("lap_number")
        if own.empty:
            plans.append(Strategy(compounds=("MEDIUM", "HARD"), stops=(race_laps // 2,)))
            continue

        stints = (
            own.groupby("stint")
            .agg(
                compound=("compound", "first"),
                last_lap=("lap_number", "max"),
                n_laps=("lap_number", "size"),
            )
            .sort_index()
            .reset_index()
        )
        # Wet compounds have no dry-tyre equivalent, so they are mapped to the
        # hard to keep the plan well formed.
        #
        # That mapping alone produces nonsense on a wet race. Australia 2025
        # came out as "5stop H-H-H-H-H-H @2,3,4,34,44": the inter/wet/inter
        # shuffle in the opening laps all became HARD, so consecutive stints on
        # the *same* mapped compound were read as pit stops between them, and
        # one-lap stints appeared that no team ever ran.
        #
        # Collapsing runs of the same compound fixes it and is correct
        # regardless of weather -- a car that changes to the same compound has
        # taken a stop, but for strategy purposes the stint structure is what
        # matters and the simulator charges the pit loss from the plan.
        raw = [
            c if c in ("SOFT", "MEDIUM", "HARD") else "HARD" for c in stints["compound"].tolist()
        ]
        raw_ends = [int(v) for v in stints["last_lap"].tolist()]

        compounds: list[str] = []
        ends: list[int] = []
        for compound, end in zip(raw, raw_ends, strict=True):
            if compounds and compounds[-1] == compound:
                ends[-1] = end
                continue
            compounds.append(compound)
            ends.append(end)

        stops = [s for s in ends[:-1] if 0 < s < race_laps]
        compounds = compounds[: len(stops) + 1]
        while len(compounds) < len(stops) + 1:
            compounds.append("HARD")
        plans.append(Strategy(compounds=tuple(compounds), stops=tuple(stops)))

    return plans


def estimate_pace(
    tables: RaceTables,
    drivers: list[str],
    posterior: DegradationPosterior,
    circuit: str,
) -> np.ndarray:
    """Per-driver clean-air race pace, in seconds relative to the quickest car.

    A driver's raw median lap time is not their pace: it depends on how long
    they ran on which compound and how much fuel they were carrying when they
    did it. So each clean green-flag lap has the fitted degradation and fuel
    terms subtracted first, and the median of what is left is the car's
    underlying pace.

    This still confounds car pace with how much of the race the driver spent in
    traffic, which is why the laps are filtered to clean air the same way the
    training data was. What remains is the deficit that a strategy model wants:
    what this car would lap at, alone, on a given tyre.
    """
    laps = tables.laps
    clean = laps.loc[
        (laps["track_status"].fillna("") == _GREEN)
        & (~laps["is_in_lap"])
        & (~laps["is_out_lap"])
        & laps["is_accurate"]
        & laps["lap_time_s"].notna()
        & laps["compound"].isin(["SOFT", "MEDIUM", "HARD"])
        & (laps["lap_number"] > 1)
    ].copy()

    race_laps = max(int(laps["lap_number"].max()), 1)
    coefs = posterior.coefficients(circuit, slice(None), rng=np.random.default_rng(0)).mean(axis=0)
    phi = float(np.median(posterior.phi))
    burn_per_lap = 110.0 / race_laps

    pace = np.full(len(drivers), np.nan)
    for i, driver in enumerate(drivers):
        own = clean.loc[clean["driver"] == driver]
        if own.empty:
            continue
        compound_idx = np.array(
            [posterior.compound_index(c) for c in own["compound"].astype(str)], dtype=np.int64
        )
        deg = posterior.degradation_seconds(coefs, compound_idx, own["tyre_life"].to_numpy(float))
        fuel = phi * burn_per_lap * (race_laps - own["lap_number"].to_numpy(float))
        pace[i] = float(np.median(own["lap_time_s"].to_numpy(float) - deg - fuel))

    if np.all(np.isnan(pace)):
        return np.zeros(len(drivers))

    reference = np.nanmin(pace)
    offsets = pace - reference
    # A driver with no usable clean laps is put at the back of the pace order
    # rather than assumed average, which would flatter them.
    offsets[np.isnan(offsets)] = np.nanmax(offsets)
    return offsets


def build_field(
    season: int,
    round_no: int,
    paths: Paths,
    posterior: DegradationPosterior,
) -> RaceField:
    """Assemble everything about one real race."""
    tables = load_race(season, round_no, paths)
    race = tables.race.iloc[0]
    circuit = str(race["circuit"])
    race_laps = int(race["completed_laps"])

    results = tables.results.copy()
    results = results.loc[results["driver"].notna()].reset_index(drop=True)
    drivers = [str(d) for d in results["driver"]]
    teams = [str(t) for t in results["team"]]

    grid = results["grid_position"].to_numpy(dtype=float)
    # A pit-lane start is recorded as grid 0; treat it as the back of the grid.
    grid = np.where((grid <= 0) | np.isnan(grid), float(len(drivers)), grid)

    pace = estimate_pace(tables, drivers, posterior, circuit)
    plans = actual_strategies(tables, drivers, race_laps)
    wet_share = float(tables.laps["compound"].isin(["INTERMEDIATE", "WET"]).mean())

    return RaceField(
        season=season,
        round=round_no,
        circuit=circuit,
        event=str(race["event"]),
        race_laps=race_laps,
        drivers=drivers,
        teams=teams,
        grid=grid,
        pace_offsets=pace,
        strategies=plans,
        actual_finish=results["finish_position"].to_numpy(dtype=float),
        finished=results["finished"].to_numpy(dtype=bool),
        wet_lap_share=wet_share,
    )


def field_frame(field: RaceField) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver": field.drivers,
            "team": field.teams,
            "grid": field.grid,
            "pace_offset_s": field.pace_offsets,
            "strategy": [s.label() for s in field.strategies],
            "actual_finish": field.actual_finish,
            "finished": field.finished,
        }
    )
