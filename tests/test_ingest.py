"""Cleaning and per-circuit estimation, on synthetic raw tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_laps, synthetic_race_table
from pitwall.ingest.circuits import estimate_pit_loss, shrink
from pitwall.ingest.clean import add_fuel_mass, add_gap_ahead, clean_laps
from pitwall.ingest.fetch import RaceTables, _neutralisation_flags, _runs
from pitwall.ingest.schema import RESULT_COLUMNS, coerce


def _tables(laps: pd.DataFrame) -> RaceTables:
    return RaceTables(
        laps=laps,
        race=synthetic_race_table(),
        results=pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in RESULT_COLUMNS.items()}),
        neutralisations=pd.DataFrame(
            columns=["season", "round", "circuit", "kind", "start_lap", "end_lap", "duration_laps"]
        ),
    )


def test_in_and_out_laps_are_removed(cleaning_config) -> None:
    laps = synthetic_laps()
    cleaned, report = clean_laps(_tables(laps), cleaning_config)
    assert not cleaned["is_in_lap"].any()
    assert not cleaned["is_out_lap"].any()
    assert (cleaned["lap_number"] > 1).all()
    stages = [name for name, _, _ in report.stages]
    assert "no in-laps" in stages and "no out-laps" in stages


def test_non_green_laps_are_removed(cleaning_config) -> None:
    laps = synthetic_laps()
    laps.loc[laps["lap_number"] == 10, "track_status"] = "4"
    cleaned, _ = clean_laps(_tables(laps), cleaning_config)
    assert 10 not in set(cleaned["lap_number"])


def test_inaccurate_and_deleted_laps_are_removed(cleaning_config) -> None:
    laps = synthetic_laps()
    laps.loc[laps["lap_number"] == 12, "is_accurate"] = False
    laps.loc[laps["lap_number"] == 14, "deleted"] = True
    cleaned, _ = clean_laps(_tables(laps), cleaning_config)
    assert 12 not in set(cleaned["lap_number"])
    assert 14 not in set(cleaned["lap_number"])


def test_wet_tyre_races_are_dropped_entirely(cleaning_config) -> None:
    """A drying track poisons the whole race, not just the wet laps."""
    laps = synthetic_laps()
    wet = laps["lap_number"] <= 8
    laps.loc[wet, "compound"] = "INTERMEDIATE"
    cleaned, report = clean_laps(_tables(laps), cleaning_config)
    assert cleaned.empty
    assert any("drying track" in note for note in report.notes)


def test_attrition_report_is_monotone(cleaning_config) -> None:
    laps = synthetic_laps()
    _, report = clean_laps(_tables(laps), cleaning_config)
    remaining = [n for _, n, _ in report.stages]
    assert remaining == sorted(remaining, reverse=True)
    assert "Cleaning attrition" in report.render()


def test_gap_ahead_is_computed_before_filtering() -> None:
    """Gaps are between adjacent cars on the road, so order must be complete."""
    laps = synthetic_laps(n_drivers=4, race_laps=3)
    with_gaps = add_gap_ahead(laps)
    lap_two = with_gaps[with_gaps["lap_number"] == 2].sort_values("lap_start_s")
    # Drivers are 4s apart by construction; the leader has no car ahead.
    assert np.isinf(lap_two["gap_ahead_s"].iloc[0])
    assert lap_two["gap_ahead_s"].iloc[1:].to_numpy() == pytest.approx(4.0)


def test_close_following_laps_are_dropped(cleaning_config) -> None:
    laps = synthetic_laps(n_drivers=4, race_laps=30)
    # Put every car within a second of the one ahead.
    laps["lap_start_s"] = (
        1000.0 + (laps["lap_number"] - 1) * 90.0 + laps["driver"].str.slice(1).astype(int) * 0.5
    )
    cleaned, _ = clean_laps(_tables(laps), cleaning_config)
    # Only the leader of each lap survives the clean-air filter.
    assert cleaned["driver"].nunique() <= 2


def test_fuel_mass_falls_linearly_to_empty() -> None:
    laps = synthetic_laps(n_drivers=1, race_laps=50)
    with_fuel = add_fuel_mass(laps, synthetic_race_table().assign(completed_laps=50), 110.0)
    first = with_fuel[with_fuel["lap_number"] == 1]["fuel_mass_kg"].iloc[0]
    last = with_fuel[with_fuel["lap_number"] == 50]["fuel_mass_kg"].iloc[0]
    assert first == pytest.approx(110.0)
    assert last == pytest.approx(110.0 * (1 - 49 / 50))
    assert with_fuel["fuel_mass_kg"].is_monotonic_decreasing


def test_neutralisation_flags_need_a_quorum() -> None:
    """One car's feed missing a code should not lose a real deployment."""
    laps = synthetic_laps(n_drivers=10, race_laps=5)
    on_lap_three = laps["lap_number"] == 3
    drivers = sorted(laps["driver"].unique())
    laps.loc[on_lap_three & laps["driver"].isin(drivers[:8]), "track_status"] = "4"
    flags = _neutralisation_flags(laps)
    assert bool(flags.loc[flags["lap_number"] == 3, "safety_car"].iloc[0])

    # A single car reporting it is not a deployment.
    laps2 = synthetic_laps(n_drivers=10, race_laps=5)
    laps2.loc[(laps2["lap_number"] == 3) & (laps2["driver"] == drivers[0]), "track_status"] = "4"
    flags2 = _neutralisation_flags(laps2)
    assert not bool(flags2.loc[flags2["lap_number"] == 3, "safety_car"].iloc[0])


def test_runs_finds_contiguous_spans() -> None:
    laps = pd.Series([1, 2, 3, 4, 5, 6])
    flags = pd.Series([False, True, True, False, True, False])
    assert _runs(flags, laps) == [(2, 3), (5, 5)]


def test_pit_loss_estimated_from_in_out_lap_pairs() -> None:
    laps = synthetic_laps(n_drivers=6, race_laps=40)
    stop = 20
    # Make the stop cost a known 24 seconds, split across the in and out laps.
    laps.loc[laps["lap_number"] == stop, "lap_time_s"] += 10.0
    laps.loc[laps["lap_number"] == stop + 1, "lap_time_s"] += 14.0

    estimate = estimate_pit_loss(laps)
    assert len(estimate) == 1
    assert float(estimate["pit_loss_s"].iloc[0]) == pytest.approx(24.0, abs=1.0)


def test_pit_loss_ignores_stops_under_neutralisation() -> None:
    laps = synthetic_laps(n_drivers=6, race_laps=40)
    laps.loc[laps["lap_number"].isin([20, 21]), "track_status"] = "4"
    laps.loc[laps["lap_number"] == 20, "lap_time_s"] += 10.0
    laps.loc[laps["lap_number"] == 21, "lap_time_s"] += 14.0
    assert estimate_pit_loss(laps).empty


def test_shrinkage_pulls_sparse_estimates_towards_the_mean() -> None:
    estimate = pd.Series([10.0, 30.0])
    weight = pd.Series([1.0, 100.0])
    shrunk = shrink(estimate, weight, prior=3.0)
    # The one-observation circuit moves a long way; the well-observed one barely.
    assert shrunk.iloc[0] > estimate.iloc[0]
    assert abs(shrunk.iloc[1] - estimate.iloc[1]) < 1.0


def test_coerce_enforces_the_column_contract() -> None:
    frame = pd.DataFrame({name: [1] for name in RESULT_COLUMNS})
    frame["driver"] = "VER"
    frame["team"] = "Team"
    frame["status"] = "Finished"
    frame["classified_position"] = "1"
    frame["finished"] = True
    out = coerce(frame, RESULT_COLUMNS)
    assert list(out.columns) == list(RESULT_COLUMNS)
