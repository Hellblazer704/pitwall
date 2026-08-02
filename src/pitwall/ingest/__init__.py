"""Turning FastF1 sessions into modelling tables.

Three stages, deliberately kept apart:

``fetch``
    FastF1 session -> flat per-race parquet. Lossless. Nothing is dropped
    here, because deciding what counts as a bad lap is a modelling choice and
    re-downloading four seasons to revisit it is an hour of wall clock.

``clean``
    Raw parquet -> the frame the degradation model is fitted on. This is where
    in-laps, out-laps, safety-car laps and the rest get filtered, and every
    filter is counted so the attrition is auditable.

``circuits``
    Per-circuit constants, part static reference and part estimated from the
    raw tables (pit-lane loss, neutralisation rates).
"""

from pitwall.ingest.schema import (
    LAP_COLUMNS,
    NEUTRALISATION_COLUMNS,
    RACE_COLUMNS,
    RESULT_COLUMNS,
)

__all__ = [
    "LAP_COLUMNS",
    "NEUTRALISATION_COLUMNS",
    "RACE_COLUMNS",
    "RESULT_COLUMNS",
]
