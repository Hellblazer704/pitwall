from __future__ import annotations

import numpy as np
import pytest

from pitwall.optimize.candidates import compound_shortlist, enumerate_candidates, refine_around
from pitwall.sim.strategy import Strategy, plan_to_matrix


def test_compounds_must_be_one_longer_than_stops() -> None:
    with pytest.raises(ValueError, match="compounds needs"):
        Strategy(compounds=("SOFT", "HARD"), stops=(10, 20))


def test_stops_must_increase() -> None:
    with pytest.raises(ValueError, match="increasing"):
        Strategy(compounds=("SOFT", "MEDIUM", "HARD"), stops=(20, 10))


def test_duplicate_stops_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Strategy(compounds=("SOFT", "MEDIUM", "HARD"), stops=(20, 20))


def test_stint_lengths_sum_to_race_distance() -> None:
    strategy = Strategy(compounds=("SOFT", "MEDIUM", "HARD"), stops=(15, 35))
    assert strategy.stint_lengths(57) == [15, 20, 22]
    assert sum(strategy.stint_lengths(57)) == 57


def test_two_compound_rule_is_enforced() -> None:
    """A dry race requires two different slick specifications."""
    single = Strategy(compounds=("SOFT", "SOFT"), stops=(20,))
    assert not single.is_legal(57, two_compounds=True)
    assert single.is_legal(57, two_compounds=False)


def test_strategy_running_past_the_flag_is_illegal() -> None:
    assert not Strategy(compounds=("SOFT", "HARD"), stops=(60,)).is_legal(57)


def test_plan_to_matrix_marks_the_right_laps() -> None:
    index = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    strategies = [Strategy(compounds=("SOFT", "HARD"), stops=(20,))]
    pit, compound = plan_to_matrix(strategies, 40, index)

    assert pit.shape == (1, 40)
    # Pitting "on lap 20" means at the end of lap 20, so index 19.
    assert pit[0, 19]
    assert pit[0].sum() == 1
    # Laps 1-20 on the soft, 21-40 on the hard.
    assert np.all(compound[0, :20] == 0)
    assert np.all(compound[0, 20:] == 2)


def test_enumerated_candidates_are_all_legal() -> None:
    candidates = enumerate_candidates(
        race_laps=57,
        stop_counts=[1, 2],
        compounds=["SOFT", "MEDIUM", "HARD"],
        lap_grid_step=6,
        min_stint_laps=8,
    )
    assert candidates
    for candidate in candidates:
        assert candidate.is_legal(57, min_stint=8, two_compounds=True)


def test_lap_grid_coarsens_with_stop_count() -> None:
    """Otherwise three-stop combinations swamp the screening budget."""
    one = enumerate_candidates(57, [1], ["SOFT", "HARD"], lap_grid_step=3, min_stint_laps=6)
    three = enumerate_candidates(57, [3], ["SOFT", "HARD"], lap_grid_step=3, min_stint_laps=6)
    one_laps = {c.stops for c in one}
    three_laps = {c.stops for c in three}
    assert len(one_laps) > 10
    # Without coarsening this would be C(16,3) = 560.
    assert len(three_laps) < 60


def test_compound_shortlist_covers_every_sequence_once() -> None:
    shortlist = compound_shortlist(57, [1], ["SOFT", "MEDIUM", "HARD"])
    sequences = [s.compounds for s in shortlist]
    assert len(sequences) == len(set(sequences))
    # 3^2 orderings minus the three single-compound ones.
    assert len(sequences) == 6


def test_refine_keeps_compounds_and_varies_laps() -> None:
    base = Strategy(compounds=("SOFT", "HARD"), stops=(20,))
    refined = refine_around(base, 57, radius=2, min_stint_laps=6)
    assert all(c.compounds == base.compounds for c in refined)
    assert {c.stops[0] for c in refined} == {18, 19, 20, 21, 22}
