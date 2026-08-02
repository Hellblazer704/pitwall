"""FastF1 session -> flat per-race parquet.

Deliberately lossless. Every lap FastF1 reports is written out, including the
in-laps, the safety-car laps and the ones flagged inaccurate, with the flags
preserved so the cleaning stage can make its own decisions. Re-downloading
four seasons costs about an hour, so this stage is run once and treated as
immutable input afterwards.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitwall.ingest.schema import (
    LAP_COLUMNS,
    NEUTRALISATION_COLUMNS,
    RACE_COLUMNS,
    RESULT_COLUMNS,
    coerce,
)
from pitwall.paths import Paths

log = logging.getLogger(__name__)

__all__ = ["RaceTables", "fetch_race", "fetch_seasons", "load_race", "load_seasons"]

# A lap counts as neutralised when at least this share of the cars running it
# reported the code. Individual cars can miss a code entirely if they pit
# across the boundary, so requiring unanimity would drop real deployments.
_NEUTRALISATION_QUORUM = 0.5


@dataclass(frozen=True)
class RaceTables:
    """The four tables one race produces."""

    laps: pd.DataFrame
    race: pd.DataFrame
    results: pd.DataFrame
    neutralisations: pd.DataFrame

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.laps.to_parquet(directory / "laps.parquet", index=False)
        self.race.to_parquet(directory / "race.parquet", index=False)
        self.results.to_parquet(directory / "results.parquet", index=False)
        self.neutralisations.to_parquet(directory / "neutralisations.parquet", index=False)

    @classmethod
    def read(cls, directory: Path) -> RaceTables:
        return cls(
            laps=pd.read_parquet(directory / "laps.parquet"),
            race=pd.read_parquet(directory / "race.parquet"),
            results=pd.read_parquet(directory / "results.parquet"),
            neutralisations=pd.read_parquet(directory / "neutralisations.parquet"),
        )

    @classmethod
    def complete(cls, directory: Path) -> bool:
        names = ("laps.parquet", "race.parquet", "results.parquet", "neutralisations.parquet")
        return all((directory / name).is_file() for name in names)


def _enable_cache(paths: Paths) -> None:
    import fastf1

    paths.fastf1_cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(paths.fastf1_cache))


def _seconds(series: pd.Series) -> pd.Series:
    """Timedelta column -> float seconds, keeping NaT as NaN."""
    return pd.to_timedelta(series).dt.total_seconds()


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the weather reading in force when each lap started.

    The weather feed samples about once a minute and the laps are irregular, so
    this is an as-of join backwards in time rather than an interpolation. Track
    temperature moves slowly enough that the sub-minute error is irrelevant next
    to the ~1 degree sensor resolution.
    """
    if weather.empty:
        for column in ("air_temp_c", "track_temp_c"):
            laps[column] = np.nan
        laps["rainfall"] = False
        return laps

    wx = weather.loc[:, ["Time", "AirTemp", "TrackTemp", "Rainfall"]].copy()
    wx["_t"] = _seconds(wx["Time"])
    wx = wx.dropna(subset=["_t"]).sort_values("_t")

    left = laps.copy()
    left["_t"] = left["lap_start_s"]
    # merge_asof cannot take NaN keys, so laps with no start time (usually the
    # first lap of a driver whose timing feed dropped) are handled separately.
    known = left["_t"].notna()
    merged = pd.merge_asof(
        left.loc[known].sort_values("_t"),
        wx.loc[:, ["_t", "AirTemp", "TrackTemp", "Rainfall"]],
        on="_t",
        direction="backward",
    )
    unknown = left.loc[~known].copy()
    for column in ("AirTemp", "TrackTemp", "Rainfall"):
        unknown[column] = np.nan
    out = pd.concat([merged, unknown], ignore_index=True)

    out = out.rename(columns={"AirTemp": "air_temp_c", "TrackTemp": "track_temp_c"})
    out["rainfall"] = out["Rainfall"].astype("boolean").fillna(False).astype(bool)
    return out.drop(columns=["Rainfall", "_t"])


def _neutralisation_flags(laps: pd.DataFrame) -> pd.DataFrame:
    """Per lap number, whether the field was under each neutralisation regime.

    Built from the per-lap TrackStatus strings rather than the raw timing
    stream. FastF1 concatenates every status code seen during a lap, so a
    substring test is the whole test; the quorum across drivers is what makes
    it robust to a single car's feed missing a code.
    """
    if laps.empty:
        return pd.DataFrame(columns=["lap_number", "safety_car", "vsc", "red_flag"])

    status = laps["track_status"].fillna("")
    frame = pd.DataFrame(
        {
            "lap_number": laps["lap_number"].to_numpy(),
            "safety_car": status.str.contains("4").to_numpy(),
            # 6 is deployed and 7 is ending; both are laps run under VSC pace.
            "vsc": (status.str.contains("6") | status.str.contains("7")).to_numpy(),
            "red_flag": status.str.contains("5").to_numpy(),
        }
    )
    share = frame.groupby("lap_number", as_index=False).mean(numeric_only=True)
    for column in ("safety_car", "vsc", "red_flag"):
        share[column] = share[column] >= _NEUTRALISATION_QUORUM
    # A lap that saw a full safety car is not also counted as a VSC lap, even
    # though a VSC often precedes the SC on the same lap.
    share.loc[share["safety_car"], "vsc"] = False
    return share.sort_values("lap_number").reset_index(drop=True)


def _runs(flags: pd.Series, laps: pd.Series) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of ``flags``, as (start_lap, end_lap) pairs."""
    values = flags.to_numpy(dtype=bool)
    lap_numbers = laps.to_numpy()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, active in enumerate(values):
        if active and start is None:
            start = int(lap_numbers[i])
        elif not active and start is not None:
            spans.append((start, int(lap_numbers[i - 1])))
            start = None
    if start is not None:
        spans.append((start, int(lap_numbers[-1])))
    return spans


def _neutralisations(laps: pd.DataFrame, season: int, round_no: int, circuit: str) -> pd.DataFrame:
    flags = _neutralisation_flags(laps)
    rows: list[dict[str, Any]] = []
    for kind, column in (("safety_car", "safety_car"), ("vsc", "vsc"), ("red_flag", "red_flag")):
        if column not in flags:
            continue
        for start, end in _runs(flags[column], flags["lap_number"]):
            rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "circuit": circuit,
                    "kind": kind,
                    "start_lap": start,
                    "end_lap": end,
                    "duration_laps": end - start + 1,
                }
            )
    if not rows:
        empty = pd.DataFrame(
            {name: pd.Series(dtype=dt) for name, dt in NEUTRALISATION_COLUMNS.items()}
        )
        return empty
    return coerce(pd.DataFrame(rows), NEUTRALISATION_COLUMNS)


def _build_laps(session: Any, season: int, round_no: int, event: str, circuit: str) -> pd.DataFrame:
    raw = session.laps.copy()
    if raw.empty:
        return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in LAP_COLUMNS.items()})

    frame = pd.DataFrame(
        {
            "season": season,
            "round": round_no,
            "event": event,
            "circuit": circuit,
            "driver": raw["Driver"].astype("string"),
            "team": raw["Team"].astype("string"),
            "lap_number": raw["LapNumber"].astype("float").fillna(-1).astype("int16"),
            "lap_time_s": _seconds(raw["LapTime"]),
            "stint": raw["Stint"].astype("float").fillna(-1).astype("int16"),
            "compound": raw["Compound"].astype("string"),
            "tyre_life": raw["TyreLife"].astype("float32"),
            "fresh_tyre": raw["FreshTyre"],
            "track_status": raw["TrackStatus"].astype("string"),
            "position": raw["Position"].astype("float32"),
            "is_accurate": raw["IsAccurate"],
            "deleted": raw["Deleted"],
            # A lap with a PitInTime ends in the pit lane; one with a PitOutTime
            # begins there. FastF1 records both on the lap they belong to, so
            # this is a direct read rather than an inference from lap times.
            "is_in_lap": raw["PitInTime"].notna(),
            "is_out_lap": raw["PitOutTime"].notna(),
            "lap_start_s": _seconds(raw["LapStartTime"]),
        }
    )
    frame["fresh_tyre"] = frame["fresh_tyre"].fillna(False).astype(bool)
    frame["is_accurate"] = frame["is_accurate"].fillna(False).astype(bool)
    frame["deleted"] = frame["deleted"].fillna(False).astype(bool)

    frame = _merge_weather(frame, session.weather_data)
    return coerce(frame, LAP_COLUMNS)


def _build_results(session: Any, season: int, round_no: int) -> pd.DataFrame:
    res = session.results.copy()
    if res.empty:
        return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in RESULT_COLUMNS.items()})
    frame = pd.DataFrame(
        {
            "season": season,
            "round": round_no,
            "driver": res["Abbreviation"].astype("string"),
            "team": res["TeamName"].astype("string"),
            "grid_position": res["GridPosition"].astype("float32"),
            "finish_position": res["Position"].astype("float32"),
            "classified_position": res["ClassifiedPosition"].astype("string"),
            "status": res["Status"].astype("string"),
            "points": res["Points"].astype("float32"),
            "laps_completed": res["Laps"].astype("float32"),
        }
    )
    # Classification comes from ClassifiedPosition, not Status. A car two laps
    # down is classified but FastF1 reports its status as "Lapped", and older
    # seasons use "+1 Lap" instead, so keying on Status gets both wrong.
    # ClassifiedPosition is numeric for a classified finisher and a letter
    # otherwise: R retired, D disqualified, W withdrawn, N not classified.
    frame["finished"] = frame["classified_position"].fillna("").str.fullmatch(r"\d+").fillna(False)
    return coerce(frame, RESULT_COLUMNS)


def _build_race(
    session: Any,
    laps: pd.DataFrame,
    neutralisations: pd.DataFrame,
    season: int,
    round_no: int,
    event: str,
    circuit: str,
) -> pd.DataFrame:
    weather = session.weather_data
    completed = int(laps["lap_number"].max()) if not laps.empty else 0
    row = {
        "season": season,
        "round": round_no,
        "event": event,
        "circuit": circuit,
        "country": str(session.event.get("Country", "")),
        "date": str(session.event.get("EventDate", ""))[:10],
        # FastF1 does not expose the scheduled distance, and a red-flagged or
        # timed-out race stops short. The max lap actually completed is the
        # honest number for anything downstream.
        "scheduled_laps": completed,
        "completed_laps": completed,
        "n_drivers": int(laps["driver"].nunique()) if not laps.empty else 0,
        "any_rainfall": bool(weather["Rainfall"].any()) if len(weather) else False,
        "mean_track_temp_c": float(weather["TrackTemp"].mean()) if len(weather) else np.nan,
        "mean_air_temp_c": float(weather["AirTemp"].mean()) if len(weather) else np.nan,
        "red_flagged": bool((neutralisations["kind"] == "red_flag").any())
        if not neutralisations.empty
        else False,
    }
    return coerce(pd.DataFrame([row]), RACE_COLUMNS)


def fetch_race(season: int, round_no: int, paths: Paths, session_name: str = "R") -> RaceTables:
    """Download and flatten one race session."""
    import fastf1

    _enable_cache(paths)
    with warnings.catch_warnings():
        # FastF1 warns about drivers with partial timing data on older seasons.
        # It is expected and the cleaning stage removes the affected laps.
        warnings.simplefilter("ignore", FutureWarning)
        session = fastf1.get_session(season, round_no, session_name)
        session.load(laps=True, telemetry=False, weather=True, messages=True)

    event = str(session.event["EventName"])
    circuit = str(session.event["Location"])

    laps = _build_laps(session, season, round_no, event, circuit)
    neutralisations = _neutralisations(laps, season, round_no, circuit)
    results = _build_results(session, season, round_no)
    race = _build_race(session, laps, neutralisations, season, round_no, event, circuit)
    return RaceTables(laps=laps, race=race, results=results, neutralisations=neutralisations)


def fetch_seasons(
    seasons: list[int],
    paths: Paths,
    session_name: str = "R",
    force: bool = False,
) -> list[tuple[int, int]]:
    """Fetch every race in ``seasons``, writing one directory per round.

    Returns the (season, round) pairs that failed. A failure is normally a
    session FastF1 has no timing data for, which is not worth aborting a
    multi-hour ingest over.
    """
    import fastf1

    _enable_cache(paths)
    paths.ensure()
    failures: list[tuple[int, int]] = []

    for season in seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        for _, event in schedule.iterrows():
            round_no = int(event["RoundNumber"])
            if round_no == 0:
                continue
            directory = paths.race_raw(season, round_no)
            if RaceTables.complete(directory) and not force:
                log.info("skip %s r%02d (already fetched)", season, round_no)
                continue
            try:
                tables = fetch_race(season, round_no, paths, session_name)
            except Exception as exc:
                log.warning("failed %s r%02d (%s): %s", season, round_no, event["EventName"], exc)
                failures.append((season, round_no))
                continue
            if tables.laps.empty:
                log.warning("empty %s r%02d (%s)", season, round_no, event["EventName"])
                failures.append((season, round_no))
                continue
            tables.write(directory)
            log.info(
                "wrote %s r%02d %s: %d laps, %d neutralisations",
                season,
                round_no,
                event["EventName"],
                len(tables.laps),
                len(tables.neutralisations),
            )
    return failures


def load_race(season: int, round_no: int, paths: Paths) -> RaceTables:
    return RaceTables.read(paths.race_raw(season, round_no))


def load_seasons(seasons: list[int], paths: Paths) -> RaceTables:
    """Concatenate every fetched race in ``seasons`` into one set of tables."""
    laps, races, results, neutralisations = [], [], [], []
    for season in seasons:
        season_dir = paths.raw / str(season)
        if not season_dir.is_dir():
            continue
        for directory in sorted(season_dir.iterdir()):
            if not RaceTables.complete(directory):
                continue
            tables = RaceTables.read(directory)
            laps.append(tables.laps)
            races.append(tables.race)
            results.append(tables.results)
            neutralisations.append(tables.neutralisations)

    if not laps:
        raise FileNotFoundError(
            f"no fetched races under {paths.raw} for seasons {seasons}; "
            "run `python -m pitwall.cli ingest` first"
        )

    def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return frames[0]
        return pd.concat(non_empty, ignore_index=True)

    return RaceTables(
        laps=_cat(laps),
        race=_cat(races),
        results=_cat(results),
        neutralisations=_cat(neutralisations),
    )
