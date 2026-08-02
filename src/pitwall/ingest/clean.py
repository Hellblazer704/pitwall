"""Raw lap tables -> the frame the degradation model is fitted on.

The filters here are the difference between a degradation model and a model of
pit-lane geometry plus safety-car deployments. Each one removes laps whose time
is dominated by something other than tyre condition, and each one is counted, so
the attrition is visible rather than implicit. See DATA.md for the full
discussion of what each quirk does to a naive fit.

Two derived quantities matter downstream:

``fuel_mass_kg``
    Estimated fuel on board at the start of the lap. This is the term that is
    confounded with tyre age, and the whole point of carrying it explicitly.

``gap_ahead_s``
    Gap to the car in front when the lap started, reconstructed from lap start
    timestamps. Used to drop traffic-affected laps here, and reused in
    Component 2 to calibrate the dirty-air loss against real data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pitwall.ingest.fetch import RaceTables, load_seasons
from pitwall.paths import Paths

log = logging.getLogger(__name__)

__all__ = ["CleaningReport", "build_modelling_frame", "write_modelling_frame"]


@dataclass
class CleaningReport:
    """Rows surviving after each filter, in the order they were applied."""

    stages: list[tuple[str, int, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, name: str, remaining: int, why: str) -> None:
        self.stages.append((name, remaining, why))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.stages, columns=["stage", "laps_remaining", "rationale"])
        frame["removed"] = frame["laps_remaining"].shift(1).sub(frame["laps_remaining"]).fillna(0)
        frame["removed"] = frame["removed"].astype(int)
        frame["pct_of_start"] = (
            100.0 * frame["laps_remaining"] / max(frame["laps_remaining"].iloc[0], 1)
        )
        return frame.loc[:, ["stage", "removed", "laps_remaining", "pct_of_start", "rationale"]]

    def render(self) -> str:
        frame = self.to_frame()
        lines = ["Cleaning attrition", "=" * 92]
        for _, row in frame.iterrows():
            lines.append(
                f"{row['stage']:<34} -{row['removed']:>7,d}  "
                f"remaining {row['laps_remaining']:>8,d}  "
                f"({row['pct_of_start']:5.1f}%)  {row['rationale']}"
            )
        if self.notes:
            lines.append("")
            lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


def add_gap_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Gap to the car ahead on the road at the moment each lap started.

    Reconstructed from ``lap_start_s``: within one race and lap number, sorting
    drivers by when they started the lap gives the running order, and adjacent
    differences give the gaps. This is the on-track gap, which is what dirty air
    depends on, and it is not the same as the classification gap when cars are
    lapped.

    Lapped traffic is the one thing this misses: a leader starting a lap two
    seconds behind a car a lap down sees that gap as clear air in the timing
    but not on the road. Those laps are rare enough not to distort the fit and
    are partly caught by the stint-median filter anyway.
    """
    out = laps.copy()
    out["gap_ahead_s"] = np.nan
    key = ["season", "round", "lap_number"]

    ordered = out.dropna(subset=["lap_start_s"]).sort_values([*key, "lap_start_s"])
    gaps = ordered.groupby(key, sort=False)["lap_start_s"].diff()
    out.loc[gaps.index, "gap_ahead_s"] = gaps.to_numpy()
    # The leader of each lap has no car ahead; infinite gap is the honest value.
    leader = out["gap_ahead_s"].isna() & out["lap_start_s"].notna()
    out.loc[leader, "gap_ahead_s"] = np.inf
    return out


def add_fuel_mass(laps: pd.DataFrame, races: pd.DataFrame, start_mass_kg: float) -> pd.DataFrame:
    """Estimated fuel load at the start of each lap.

    Cars start with the regulation maximum (110 kg since 2019) and finish close
    to empty, and consumption per lap is near enough constant that a linear
    burn-down is within a kilogram or two of the truth for most of the race.

    The approximation is worst under a safety car, where the field burns much
    less fuel per lap than under green. That biases the fuel term slightly on
    the laps after a long neutralisation. It is a second-order effect next to
    the confounding it is there to resolve, and it is listed in DATA.md.
    """
    distance = races.loc[:, ["season", "round", "completed_laps"]].drop_duplicates()
    out = laps.merge(distance, on=["season", "round"], how="left")

    # Laps already completed when this lap started.
    completed = (out["lap_number"] - 1).clip(lower=0)
    total = out["completed_laps"].clip(lower=1)
    burned_fraction = (completed / total).clip(0.0, 1.0)
    out["fuel_mass_kg"] = start_mass_kg * (1.0 - burned_fraction)
    out["race_fraction"] = (out["lap_number"] / total).clip(0.0, 1.0)
    return out


def _stint_uid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["season"].astype(str)
        + "_"
        + frame["round"].astype(str).str.zfill(2)
        + "_"
        + frame["driver"].astype(str)
        + "_s"
        + frame["stint"].astype(str)
    )


def clean_laps(
    tables: RaceTables,
    cleaning: object,
    start_mass_kg: float = 110.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply every filter in order, reporting the attrition."""
    cfg = cleaning
    report = CleaningReport()

    laps = tables.laps.copy()
    report.record("raw laps", len(laps), "everything FastF1 returned")

    # Gap to the car ahead has to be computed before anything is filtered out.
    # It is a difference between adjacent cars on the road, so removing a
    # driver's lap first would silently make the next car back appear to be
    # chasing whoever is now adjacent in the sorted order.
    laps = add_gap_ahead(laps)

    # -- Races we cannot model at all -------------------------------------
    rain_share = float(getattr(cfg, "max_rain_lap_share", 0.15))
    wet_share = laps.groupby(["season", "round"])["rainfall"].mean()
    wet_races = wet_share.loc[wet_share > rain_share].index
    if len(wet_races):
        mask = ~laps.set_index(["season", "round"]).index.isin(set(wet_races))
        laps = laps.loc[mask]
        report.note(
            f"dropped {len(wet_races)} race(s) with more than {rain_share:.0%} of laps in the wet"
        )
    report.record("mostly-dry races", len(laps), f"race-level rain share above {rain_share:.0%}")

    wet_tyre_share = float(getattr(cfg, "max_wet_tyre_lap_share", 0.05))
    on_wets = laps["compound"].isin(["INTERMEDIATE", "WET"])
    wet_tyre_by_race = on_wets.groupby([laps["season"], laps["round"]]).mean()
    drying = wet_tyre_by_race.loc[wet_tyre_by_race > wet_tyre_share].index
    if len(drying):
        mask = ~laps.set_index(["season", "round"]).index.isin(set(drying))
        laps = laps.loc[mask]
        report.note(
            f"dropped {len(drying)} race(s) where wet tyres ran for more than "
            f"{wet_tyre_share:.0%} of laps (drying track)"
        )
    report.record(
        "no drying tracks", len(laps), f"wet-tyre usage above {wet_tyre_share:.0%} of race laps"
    )

    if bool(getattr(cfg, "drop_rain_laps", True)):
        laps = laps.loc[~laps["rainfall"].astype(bool)]
        report.record("dry laps", len(laps), "individual laps with a rainfall reading")

    exclude = list(getattr(cfg, "exclude_events", []) or [])
    if exclude:
        laps = laps.loc[~laps["event"].isin(exclude)]
        report.record("event exclusions", len(laps), f"config exclude_events={exclude}")

    # -- Laps that are not a measurement of tyre pace ----------------------
    compounds = list(getattr(cfg, "compounds", ["SOFT", "MEDIUM", "HARD"]))
    laps = laps.loc[laps["compound"].isin(compounds)]
    report.record("slick compounds", len(laps), f"kept {compounds}, dropped inters/wets/unknown")

    laps = laps.loc[laps["lap_time_s"].notna()]
    report.record("timed laps", len(laps), "LapTime is NaT when the timing feed dropped the lap")

    if bool(getattr(cfg, "drop_out_laps", True)):
        laps = laps.loc[~laps["is_out_lap"]]
        report.record("no out-laps", len(laps), "carries pit-exit deficit plus a cold tyre")

    if bool(getattr(cfg, "drop_in_laps", True)):
        laps = laps.loc[~laps["is_in_lap"]]
        report.record("no in-laps", len(laps), "carries pit-entry deficit, not tyre pace")

    if bool(getattr(cfg, "drop_lap_one", True)):
        laps = laps.loc[laps["lap_number"] > 1]
        report.record("no lap 1", len(laps), "standing start and first-lap scrap")

    if bool(getattr(cfg, "require_is_accurate", True)):
        laps = laps.loc[laps["is_accurate"]]
        report.record("IsAccurate", len(laps), "FastF1's own timing-integrity flag")

    if bool(getattr(cfg, "drop_deleted", True)):
        laps = laps.loc[~laps["deleted"]]
        report.record("not deleted", len(laps), "track-limits deletion implies an off-track lap")

    if bool(getattr(cfg, "green_flag_only", True)):
        # Pure green only. Any other code in the concatenated status means the
        # lap saw a yellow, an SC, a VSC or a red flag at some point.
        laps = laps.loc[laps["track_status"].fillna("") == "1"]
        report.record("green flag only", len(laps), "status 1 throughout; excludes SC/VSC/yellow")

    if laps.empty:
        return laps, report

    # -- Derived quantities, then the filters that need them ---------------
    min_gap = float(getattr(cfg, "min_gap_ahead_s", 1.5))
    if min_gap > 0:
        laps = laps.loc[laps["gap_ahead_s"] >= min_gap]
        report.record(
            "clean air only",
            len(laps),
            f"started the lap within {min_gap:.1f}s of the car ahead",
        )

    laps = add_fuel_mass(laps, tables.race, start_mass_kg)
    laps["stint_uid"] = _stint_uid(laps)
    laps["tyre_age"] = laps["tyre_life"].astype(float)
    laps = laps.loc[laps["tyre_age"].notna()]
    report.record("known tyre age", len(laps), "TyreLife missing when the stint start was not seen")

    threshold = float(getattr(cfg, "max_lap_ratio_to_stint_median", 1.07))
    stint_median = laps.groupby("stint_uid")["lap_time_s"].transform("median")
    laps["stint_median_s"] = stint_median
    laps = laps.loc[laps["lap_time_s"] <= threshold * stint_median]
    report.record(
        "traffic / mistakes",
        len(laps),
        f"lap time above {threshold:.2f}x its stint median",
    )

    min_stint = int(getattr(cfg, "min_stint_laps", 5))
    counts = laps.groupby("stint_uid")["lap_time_s"].transform("size")
    laps = laps.loc[counts >= min_stint]
    report.record(
        "usable stints",
        len(laps),
        f"stints with < {min_stint} surviving laps cannot identify a slope",
    )

    laps = laps.reset_index(drop=True)
    report.note(
        f"{laps['stint_uid'].nunique():,} stints across {laps['circuit'].nunique()} circuits"
    )
    report.note(
        f"{laps['driver'].nunique()} drivers, {len(laps.groupby(['season', 'round']))} races"
    )
    return laps, report


def build_modelling_frame(
    seasons: list[int],
    paths: Paths,
    cleaning: object,
    start_mass_kg: float = 110.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    tables = load_seasons(seasons, paths)
    log.info("loaded %d raw laps from %d races", len(tables.laps), len(tables.race))
    return clean_laps(tables, cleaning, start_mass_kg)


def modelling_frame_path(paths: Paths, seasons: list[int]) -> Path:
    tag = "-".join(str(season) for season in sorted(seasons))
    return paths.processed / f"laps_{tag}.parquet"


def write_modelling_frame(
    frame: pd.DataFrame,
    report: CleaningReport,
    paths: Paths,
    seasons: list[int],
) -> Path:
    paths.ensure()
    target = modelling_frame_path(paths, seasons)
    # gap_ahead_s carries +inf for race leaders, which parquet round-trips fine
    # but several readers do not, so it is stored as NaN and restored on load.
    out = frame.copy()
    out["gap_ahead_s"] = out["gap_ahead_s"].replace(np.inf, np.nan)
    out.to_parquet(target, index=False)
    report.to_frame().to_csv(target.with_suffix(".attrition.csv"), index=False)
    log.info("wrote %s (%d laps)", target, len(out))
    return target


def read_modelling_frame(paths: Paths, seasons: list[int]) -> pd.DataFrame:
    target = modelling_frame_path(paths, seasons)
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} not found; run `pitwall clean --seasons {' '.join(map(str, seasons))}`"
        )
    frame = pd.read_parquet(target)
    frame["gap_ahead_s"] = frame["gap_ahead_s"].fillna(np.inf)
    return frame
