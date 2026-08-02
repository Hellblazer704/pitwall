"""Per-circuit constants, estimated from the ingested races.

Almost everything a strategy model needs to know about a circuit is a number
that varies enormously between venues and barely at all between years: how much
a pit stop costs, how often a safety car comes out, how hard it is to pass. All
of them are estimated here from the raw tables rather than hardcoded, so
adding a season updates them and there is no reference table to drift out of
date.

The estimates are deliberately simple and each one is stated in DESIGN.md with
the identification argument behind it. Where a circuit has too few races to
estimate something stably, the estimate is shrunk towards the global mean by an
explicit empirical-Bayes weight rather than being left noisy.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pitwall.ingest.fetch import RaceTables

log = logging.getLogger(__name__)

__all__ = ["CircuitProfile", "build_circuit_profiles", "shrink"]

# Shrinkage strength, in "pseudo-races". A circuit with this many races gets
# half its own estimate and half the global mean. Four seasons gives at most
# four observations per circuit, so this matters.
_PRIOR_RACES = 3.0

# A pit stop under a neutralisation costs far less relative to the field, so
# those stops are excluded when estimating the green-flag pit loss.
_GREEN = "1"


@dataclass(frozen=True)
class CircuitProfile:
    """Everything the simulator needs to know about one venue."""

    circuit: str
    n_races: int
    typical_laps: int
    median_green_lap_s: float
    # Total time lost pitting under green, relative to staying out: pit-lane
    # transit plus the stationary time.
    pit_loss_s: float
    pit_loss_sd_s: float
    # Neutralisation frequency, per race.
    sc_per_race: float
    vsc_per_race: float
    # Mean duration of one deployment, in laps.
    sc_duration_laps: float
    vsc_duration_laps: float
    # On-track passes per racing lap, across the whole field. Higher means
    # easier to pass; feeds the overtaking model's per-circuit multiplier.
    passes_per_lap: float
    # Multiplier on the baseline overtake probability, 1.0 at an average track.
    overtake_difficulty: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def shrink(estimate: pd.Series, weight: pd.Series, prior: float) -> pd.Series:
    """Empirical-Bayes shrinkage of per-circuit estimates towards the pooled mean.

    ``weight`` is the number of observations behind each estimate. A circuit
    seen once contributes 1/(1+_PRIOR_RACES) of its own value; the rest is the
    global mean. This is the same partial-pooling logic as the degradation
    model, applied to quantities that do not justify a full hierarchical fit.
    """
    grand_mean = float(np.average(estimate, weights=weight)) if weight.sum() > 0 else float("nan")
    alpha = weight / (weight + prior)
    return alpha * estimate + (1.0 - alpha) * grand_mean


def _driver_race_baseline(laps: pd.DataFrame) -> pd.DataFrame:
    """Median green, non-pit, accurate lap time per driver per race.

    This is the reference a pit stop is measured against. Using the driver's
    own median rather than the field's removes car pace from the comparison,
    which matters because slow cars pit at different times to fast ones.
    """
    green = laps.loc[
        (laps["track_status"].fillna("") == _GREEN)
        & (~laps["is_in_lap"])
        & (~laps["is_out_lap"])
        & laps["is_accurate"]
        & laps["lap_time_s"].notna()
    ]
    return (
        green.groupby(["season", "round", "circuit", "driver"], as_index=False)["lap_time_s"]
        .median()
        .rename(columns={"lap_time_s": "baseline_s"})
    )


def estimate_pit_loss(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag pit loss per circuit.

    A stop shows up as an in-lap (ends in the pit lane) followed by an out-lap
    (starts there). Both are slower than a normal lap, and the excess over two
    baseline laps is the total time the stop cost:

        loss = (t_in + t_out) - 2 * baseline

    Stops taken under any neutralisation are excluded, because the field is
    slowed and the relative loss is much smaller: that discount is modelled
    separately in the simulator, and mixing the two here would bias the
    green-flag number down at exactly the circuits with the most safety cars.
    """
    baseline = _driver_race_baseline(laps)

    pit = laps.loc[laps["is_in_lap"] | laps["is_out_lap"]].copy()
    pit = pit.loc[pit["lap_time_s"].notna()]
    pit = pit.merge(baseline, on=["season", "round", "circuit", "driver"], how="inner")

    in_laps = pit.loc[
        pit["is_in_lap"],
        [
            "season",
            "round",
            "circuit",
            "driver",
            "lap_number",
            "lap_time_s",
            "baseline_s",
            "track_status",
        ],
    ]
    out_laps = pit.loc[
        pit["is_out_lap"], ["season", "round", "driver", "lap_number", "lap_time_s", "track_status"]
    ]
    out_laps = out_laps.rename(columns={"lap_time_s": "out_time_s", "track_status": "out_status"})
    # The out-lap is the lap after the in-lap for the same driver.
    out_laps["lap_number"] = out_laps["lap_number"] - 1

    stops = in_laps.merge(out_laps, on=["season", "round", "driver", "lap_number"], how="inner")
    green_stop = (stops["track_status"].fillna("") == _GREEN) & (
        stops["out_status"].fillna("") == _GREEN
    )
    stops = stops.loc[green_stop]

    stops["pit_loss_s"] = stops["lap_time_s"] + stops["out_time_s"] - 2.0 * stops["baseline_s"]
    # A handful of stops land outside any plausible range: red-flag stops the
    # status did not catch, drive-through penalties, and cars that stopped on
    # track and were recovered through the pit lane.
    stops = stops.loc[stops["pit_loss_s"].between(10.0, 45.0)]

    return (
        stops.groupby("circuit")
        .agg(
            pit_loss_s=("pit_loss_s", "median"),
            pit_loss_sd_s=("pit_loss_s", "std"),
            n_stops=("pit_loss_s", "size"),
        )
        .reset_index()
    )


def estimate_neutralisation_rates(
    neutralisations: pd.DataFrame, races: pd.DataFrame
) -> pd.DataFrame:
    """Safety car and VSC frequency and duration per circuit."""
    race_counts = races.groupby("circuit", as_index=False).agg(
        n_races=("round", "size"), typical_laps=("completed_laps", "median")
    )

    if neutralisations.empty:
        out = race_counts.copy()
        for column in ("sc_per_race", "vsc_per_race", "sc_duration_laps", "vsc_duration_laps"):
            out[column] = 0.0
        return out

    counts = (
        neutralisations.groupby(["circuit", "kind"])
        .agg(n=("kind", "size"), duration=("duration_laps", "mean"))
        .reset_index()
    )
    wide_n = counts.pivot(index="circuit", columns="kind", values="n").fillna(0.0)
    wide_d = counts.pivot(index="circuit", columns="kind", values="duration")

    out = race_counts.merge(wide_n, on="circuit", how="left").fillna(0.0)
    for kind in ("safety_car", "vsc"):
        if kind not in out.columns:
            out[kind] = 0.0
    out["sc_per_race"] = out["safety_car"] / out["n_races"].clip(lower=1)
    out["vsc_per_race"] = out["vsc"] / out["n_races"].clip(lower=1)

    durations = wide_d.reindex(out["circuit"]).reset_index(drop=True)
    out["sc_duration_laps"] = durations.get("safety_car", pd.Series(np.nan, index=out.index))
    out["vsc_duration_laps"] = durations.get("vsc", pd.Series(np.nan, index=out.index))
    return out.drop(columns=[c for c in ("safety_car", "vsc", "red_flag") if c in out.columns])


def estimate_overtaking(laps: pd.DataFrame) -> pd.DataFrame:
    """On-track passes per racing lap, per circuit.

    A pass is a driver improving position between consecutive green laps
    without either lap being a pit lap. Excluding pit laps is what separates
    genuine overtakes from the position churn of a pit cycle, and restricting
    to green laps removes the shuffle when a safety car catches the field.

    This counts net position gains rather than wheel-to-wheel events, so it
    undercounts a pass that is immediately re-passed. That is acceptable: it is
    used as a relative index across circuits, not an absolute count.
    """
    frame = laps.loc[
        (laps["track_status"].fillna("") == _GREEN)
        & laps["position"].notna()
        & (~laps["is_in_lap"])
        & (~laps["is_out_lap"])
    ].copy()
    frame = frame.sort_values(["season", "round", "driver", "lap_number"])

    group = frame.groupby(["season", "round", "driver"], sort=False)
    frame["prev_position"] = group["position"].shift(1)
    frame["prev_lap"] = group["lap_number"].shift(1)

    # Only consecutive laps: a gap means a pit lap or a neutralisation in
    # between, and any position change across it is not an overtake.
    consecutive = frame["lap_number"] - frame["prev_lap"] == 1
    frame = frame.loc[consecutive & frame["prev_position"].notna()]
    frame["gained"] = (frame["prev_position"] - frame["position"]).clip(lower=0)

    per_race = frame.groupby(["season", "round", "circuit"], as_index=False).agg(
        passes=("gained", "sum"), racing_laps=("lap_number", "nunique")
    )
    per_race["passes_per_lap"] = per_race["passes"] / per_race["racing_laps"].clip(lower=1)

    return per_race.groupby("circuit", as_index=False).agg(
        passes_per_lap=("passes_per_lap", "mean"), n_races_ot=("passes_per_lap", "size")
    )


def build_circuit_profiles(tables: RaceTables) -> pd.DataFrame:
    """Assemble the per-circuit table from raw ingest output."""
    laps = tables.laps
    pit = estimate_pit_loss(laps)
    neut = estimate_neutralisation_rates(tables.neutralisations, tables.race)
    over = estimate_overtaking(laps)

    green = laps.loc[
        (laps["track_status"].fillna("") == _GREEN)
        & (~laps["is_in_lap"])
        & (~laps["is_out_lap"])
        & laps["is_accurate"]
    ]
    pace = (
        green.groupby("circuit", as_index=False)["lap_time_s"]
        .median()
        .rename(columns={"lap_time_s": "median_green_lap_s"})
    )

    profiles = (
        neut.merge(pit, on="circuit", how="left")
        .merge(over, on="circuit", how="left")
        .merge(pace, on="circuit", how="left")
    )

    # Shrink the noisy estimates towards their pooled means.
    races = profiles["n_races"].astype(float).clip(lower=1)
    for column in ("pit_loss_s", "passes_per_lap", "sc_per_race", "vsc_per_race"):
        valid = profiles[column].notna()
        if valid.sum() == 0:
            continue
        pooled = float(profiles.loc[valid, column].mean())
        profiles[column] = profiles[column].fillna(pooled)
        profiles[column] = shrink(profiles[column], races, _PRIOR_RACES)

    for column, fallback in (("sc_duration_laps", 3.6), ("vsc_duration_laps", 2.1)):
        profiles[column] = profiles[column].fillna(profiles[column].mean()).fillna(fallback)
    profiles["pit_loss_sd_s"] = profiles["pit_loss_sd_s"].fillna(profiles["pit_loss_sd_s"].median())

    # Overtaking difficulty is the reciprocal of the pass rate, normalised so
    # the median circuit sits at 1.0. Clipped because Monaco would otherwise
    # produce a multiplier that makes passing literally impossible, and cars do
    # very occasionally pass there.
    median_rate = float(profiles["passes_per_lap"].median())
    profiles["overtake_difficulty"] = (profiles["passes_per_lap"] / median_rate).clip(0.15, 3.0)

    profiles["typical_laps"] = profiles["typical_laps"].round().astype(int)
    profiles = profiles.sort_values("circuit").reset_index(drop=True)
    log.info("built profiles for %d circuits", len(profiles))
    return profiles


def profiles_to_records(profiles: pd.DataFrame) -> dict[str, CircuitProfile]:
    out: dict[str, CircuitProfile] = {}
    for _, row in profiles.iterrows():
        out[str(row["circuit"])] = CircuitProfile(
            circuit=str(row["circuit"]),
            n_races=int(row["n_races"]),
            typical_laps=int(row["typical_laps"]),
            median_green_lap_s=float(row["median_green_lap_s"]),
            pit_loss_s=float(row["pit_loss_s"]),
            pit_loss_sd_s=float(row["pit_loss_sd_s"]),
            sc_per_race=float(row["sc_per_race"]),
            vsc_per_race=float(row["vsc_per_race"]),
            sc_duration_laps=float(row["sc_duration_laps"]),
            vsc_duration_laps=float(row["vsc_duration_laps"]),
            passes_per_lap=float(row["passes_per_lap"]),
            overtake_difficulty=float(row["overtake_difficulty"]),
        )
    return out
