"""Column contracts for the ingest tables.

These exist so a downstream module can assert what it is getting instead of
discovering a renamed column three layers into a Gibbs sampler. FastF1 does
change column names between releases; when it does, exactly one file needs
editing and the failure is loud.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "LAP_COLUMNS",
    "NEUTRALISATION_COLUMNS",
    "RACE_COLUMNS",
    "RESULT_COLUMNS",
    "TRACK_STATUS",
    "require_columns",
]

# FastF1 track status codes. A lap's TrackStatus is the concatenation of every
# code that was active at any point during that lap, so "26" means the lap saw
# both a yellow and a VSC deployment.
TRACK_STATUS: dict[str, str] = {
    "1": "green",
    "2": "yellow",
    "3": "unknown",  # appears rarely and undocumented; treated as non-green
    "4": "safety_car",
    "5": "red_flag",
    "6": "vsc_deployed",
    "7": "vsc_ending",
}

LAP_COLUMNS: dict[str, str] = {
    "season": "int16",
    "round": "int16",
    "event": "string",
    "circuit": "string",
    "driver": "string",
    "team": "string",
    "lap_number": "int16",
    "lap_time_s": "float64",
    "stint": "int16",
    "compound": "string",
    "tyre_life": "float32",
    "fresh_tyre": "boolean",
    "track_status": "string",
    "position": "float32",
    "is_accurate": "boolean",
    "deleted": "boolean",
    "is_in_lap": "boolean",
    "is_out_lap": "boolean",
    "lap_start_s": "float64",
    "air_temp_c": "float32",
    "track_temp_c": "float32",
    "rainfall": "boolean",
}

RACE_COLUMNS: dict[str, str] = {
    "season": "int16",
    "round": "int16",
    "event": "string",
    "circuit": "string",
    "country": "string",
    "date": "string",
    "scheduled_laps": "int16",
    "completed_laps": "int16",
    "n_drivers": "int16",
    "any_rainfall": "boolean",
    "mean_track_temp_c": "float32",
    "mean_air_temp_c": "float32",
    "red_flagged": "boolean",
}

RESULT_COLUMNS: dict[str, str] = {
    "season": "int16",
    "round": "int16",
    "driver": "string",
    "team": "string",
    "grid_position": "float32",
    "finish_position": "float32",
    "classified_position": "string",
    "status": "string",
    "points": "float32",
    "laps_completed": "float32",
    "finished": "boolean",
}

# One row per contiguous neutralisation, resolved to lap numbers.
NEUTRALISATION_COLUMNS: dict[str, str] = {
    "season": "int16",
    "round": "int16",
    "circuit": "string",
    "kind": "string",  # safety_car | vsc | red_flag
    "start_lap": "int16",
    "end_lap": "int16",
    "duration_laps": "int16",
}


def require_columns(frame: pd.DataFrame, contract: dict[str, str], name: str) -> None:
    """Raise if ``frame`` is missing any column in ``contract``."""
    missing = [column for column in contract if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; got {list(frame.columns)}")


def coerce(frame: pd.DataFrame, contract: dict[str, str]) -> pd.DataFrame:
    """Project onto the contract's columns and cast to its dtypes."""
    out = frame.loc[:, list(contract)].copy()
    for column, dtype in contract.items():
        # pandas-stubs only accepts dtype literals here, not a str variable.
        out[column] = out[column].astype(dtype)  # type: ignore[call-overload]
    return out
