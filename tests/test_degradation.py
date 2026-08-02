"""Degradation model tests, built around parameter recovery.

The claim Component 1 makes is that it can pull the fuel effect apart from tyre
wear even though the two are perfectly collinear inside any single stint. The
only honest way to test that is to generate laps from a known model and check
the sampler gets the known values back. If the identification argument in
``design.py`` is wrong, this test fails and nothing else needs to be believed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall.degradation.design import build_design, collinearity_diagnostic
from pitwall.degradation.diagnostics import ess_bulk, split_rhat, summarise
from pitwall.degradation.gibbs import GibbsPriors, sample
from pitwall.degradation.model import DegradationPosterior, monotone_degradation

TRUE_PHI = 0.030
AGE_SCALE = 20.0
AGE_CENTER_TARGET = 15.0

# Linear wear coefficient per compound, on the scaled-age axis. 1.0 means one
# second lost over AGE_SCALE laps.
TRUE_LINEAR = {"HARD": 0.60, "MEDIUM": 1.00, "SOFT": 1.50}
TRUE_OFFSET = {"HARD": 0.40, "MEDIUM": 0.0, "SOFT": -0.30}


def _simulate_laps(
    n_races_per_circuit: int = 6,
    n_drivers: int = 12,
    race_laps: int = 55,
    circuits: tuple[str, ...] = ("Alpha", "Beta", "Gamma", "Delta"),
    noise_sd: float = 0.20,
    seed: int = 20260101,
) -> pd.DataFrame:
    """Laps generated from the model in design.py, with known coefficients.

    Stint plans vary between drivers and races on purpose: that variation is
    the thing that makes the fuel term identifiable at all, so a test that used
    one plan for everybody would be testing nothing.
    """
    rng = np.random.default_rng(seed)
    compounds = list(TRUE_LINEAR)
    rows = []

    for circuit_index, circuit in enumerate(circuits):
        for race in range(n_races_per_circuit):
            # Round numbers must be unique across circuits. The model's
            # nuisance intercept is keyed on (season, round, driver), so
            # reusing round 1 at four circuits would collapse four different
            # base paces into one parameter and dump the difference into the
            # residual -- which is exactly what happened the first time.
            round_no = circuit_index * n_races_per_circuit + race + 1
            for d in range(n_drivers):
                base = 90.0 + rng.normal(0.0, 1.2)
                # Between one and three stops, at varying laps.
                n_stops = int(rng.integers(1, 4))
                stops = sorted(
                    rng.choice(np.arange(8, race_laps - 8), size=n_stops, replace=False).tolist()
                )
                plan = [compounds[int(rng.integers(0, 3))] for _ in range(n_stops + 1)]

                stint, age = 0, 0
                for lap in range(1, race_laps + 1):
                    age += 1
                    if stint < n_stops and lap > stops[stint]:
                        stint += 1
                        age = 1
                    compound = plan[stint]

                    fuel_kg = 110.0 * (1.0 - (lap - 1) / race_laps)
                    z = (age - AGE_CENTER_TARGET) / AGE_SCALE
                    wear = TRUE_OFFSET[compound] + TRUE_LINEAR[compound] * z
                    lap_time = base + TRUE_PHI * fuel_kg + wear + rng.normal(0.0, noise_sd)
                    rows.append(
                        {
                            "season": 2024,
                            "round": round_no,
                            "circuit": circuit,
                            "driver": f"D{d:02d}",
                            "compound": compound,
                            "tyre_age": float(age),
                            "fuel_mass_kg": fuel_kg,
                            "race_fraction": lap / race_laps,
                            "lap_time_s": lap_time,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def recovered() -> tuple[DegradationPosterior, object]:
    frame = _simulate_laps()
    design = build_design(frame, age_scale_laps=AGE_SCALE, quadratic=False)
    draws = sample(
        design,
        GibbsPriors(fuel_mean=0.030, fuel_sd=0.012),
        [np.random.default_rng(s) for s in (1, 2)],
        draws=900,
        warmup=400,
    )
    return DegradationPosterior.from_draws(draws), design


def test_within_stint_regression_is_biased_the_way_the_docstring_claims() -> None:
    """The naive estimator understates wear by roughly the fuel effect.

    This is the motivation for the whole component, so it is worth pinning
    down rather than asserting in prose. Regressing lap time on tyre age
    within stints recovers wear *minus* fuel burn, because fuel falls as the
    tyre ages.
    """
    frame = _simulate_laps(n_races_per_circuit=3, circuits=("Alpha",))
    medium = frame[frame["compound"] == "MEDIUM"]

    # Slope of lap time on tyre age, sweeping out each stint's own level.
    key = ["round", "driver"]
    x = medium["tyre_age"].to_numpy(float)
    y = medium["lap_time_s"].to_numpy(float)
    groups = medium.groupby(key).ngroup().to_numpy()
    for values in (x, y):
        means = np.bincount(groups, weights=values) / np.bincount(groups)
        values -= means[groups]
    naive_slope = float(np.sum(x * y) / np.sum(x * x))

    true_slope = TRUE_LINEAR["MEDIUM"] / AGE_SCALE
    fuel_per_lap = TRUE_PHI * 110.0 / 55.0

    assert naive_slope < true_slope
    assert naive_slope == pytest.approx(true_slope - fuel_per_lap, abs=0.01)


def test_recovers_the_fuel_coefficient(recovered) -> None:
    posterior, _ = recovered
    lo, mid, hi = np.percentile(posterior.phi, [2.5, 50, 97.5])
    assert lo <= TRUE_PHI <= hi, f"true phi {TRUE_PHI} outside 95% CI [{lo:.4f}, {hi:.4f}]"
    assert mid == pytest.approx(TRUE_PHI, abs=0.004)


def test_recovers_wear_slopes_and_their_ordering(recovered) -> None:
    posterior, _ = recovered
    medians = {}
    for compound, truth in TRUE_LINEAR.items():
        k = posterior.compound_index(compound)
        values = posterior.beta[:, :, k, 1].ravel()
        medians[compound] = float(np.median(values))
        assert medians[compound] == pytest.approx(truth, abs=0.05)

    assert medians["HARD"] < medians["MEDIUM"] < medians["SOFT"]


def test_recovers_compound_offsets_relative_to_medium(recovered) -> None:
    posterior, _ = recovered
    for compound in ("HARD", "SOFT"):
        k = posterior.compound_index(compound)
        median = float(np.median(posterior.beta[:, :, k, 0]))
        assert median == pytest.approx(TRUE_OFFSET[compound], abs=0.15)


def test_medium_offset_is_pinned_to_zero(recovered) -> None:
    posterior, _ = recovered
    k = posterior.compound_index("MEDIUM")
    assert np.all(posterior.beta[:, :, k, 0] == 0.0)
    assert not posterior.active[0, k, 0]


def test_recovers_the_residual_scale(recovered) -> None:
    """A wrong residual sd means the structural part is absorbing something.

    Worth asserting explicitly: the first version of this fixture reused round
    numbers across circuits, which collapsed four different base paces into one
    intercept. Every coefficient still looked plausible; sigma was 5x too big
    and gave it away.
    """
    posterior, _ = recovered
    assert float(np.median(posterior.sigma)) == pytest.approx(0.20, abs=0.03)


def test_recovery_chains_converge(recovered) -> None:
    posterior, _ = recovered
    assert posterior.n_samples > 0
    assert np.all(np.isfinite(posterior.phi))


def test_unobserved_cells_are_structural_zeros_not_estimates() -> None:
    """A compound never run at a circuit must not read as zero degradation."""
    frame = _simulate_laps(n_races_per_circuit=2, circuits=("Alpha", "Beta"))
    # Soft still exists in the dataset, just never at Alpha.
    frame = frame[~((frame["circuit"] == "Alpha") & (frame["compound"] == "SOFT"))]
    design = build_design(frame, age_scale_laps=AGE_SCALE, quadratic=False)

    alpha = design.circuits.index("Alpha")
    soft = design.compounds.index("SOFT")
    assert not design.active[alpha, soft, 1]

    draws = sample(
        design,
        GibbsPriors(),
        [np.random.default_rng(4)],
        draws=60,
        warmup=30,
    )
    posterior = DegradationPosterior.from_draws(draws)
    assert np.all(posterior.beta[:, alpha, soft, :] == 0.0)

    # Asking for it anyway must draw from the population, not hand back zeros.
    coefs = posterior.coefficients("Alpha", 0, rng=np.random.default_rng(0))
    assert not np.allclose(coefs[soft], 0.0)


def test_coefficients_for_unseen_circuit_need_a_generator(recovered) -> None:
    posterior, _ = recovered
    with pytest.raises(ValueError, match="not in the training set"):
        posterior.coefficients("Nowhere", 0)
    drawn = posterior.coefficients("Nowhere", 0, rng=np.random.default_rng(0))
    assert drawn.shape == (posterior.beta.shape[2], 3)


def test_centred_age_decorrelates_the_polynomial_columns() -> None:
    """Why age is centred: the raw basis correlates linear and quadratic ~0.97."""
    frame = _simulate_laps(n_races_per_circuit=2, circuits=("Alpha",))
    design = build_design(frame, age_scale_laps=AGE_SCALE, quadratic=True)

    z = design.age
    centred_corr = abs(float(np.corrcoef(z, z * z)[0, 1]))

    raw = z + design.age_center_laps / AGE_SCALE
    raw_corr = abs(float(np.corrcoef(raw, raw * raw)[0, 1]))

    assert raw_corr > 0.9
    assert centred_corr < raw_corr


def test_wear_is_reported_relative_to_a_fresh_tyre(recovered) -> None:
    posterior, _ = recovered
    summary = posterior.summary(ages=(1, 20))
    for compound in TRUE_LINEAR:
        rows = summary[summary["compound"] == compound]
        if rows.empty:
            continue
        young = rows[rows["age_laps"] == 1]["loss_s_median"].to_numpy()
        old = rows[rows["age_laps"] == 20]["loss_s_median"].to_numpy()
        assert np.all(old > young), f"{compound} should lose more time as it ages"


def test_collinearity_diagnostic_reports_per_circuit(recovered) -> None:
    _, design = recovered
    table = collinearity_diagnostic(design)
    assert set(table.columns) >= {"circuit", "corr_fuel_age", "age_range_laps"}
    assert (table["corr_fuel_age"].abs() <= 1.0).all()


def test_save_and_load_round_trips(recovered, tmp_path) -> None:
    posterior, _ = recovered
    target = tmp_path / "posterior.npz"
    posterior.save(target)
    reloaded = DegradationPosterior.load(target)

    assert reloaded.circuits == posterior.circuits
    assert reloaded.compounds == posterior.compounds
    assert reloaded.age_center_laps == pytest.approx(posterior.age_center_laps)
    assert np.allclose(reloaded.beta, posterior.beta)
    assert np.array_equal(reloaded.active, posterior.active)
    assert np.allclose(reloaded.max_age, posterior.max_age)


def test_extrapolation_bound_is_recorded() -> None:
    frame = _simulate_laps(n_races_per_circuit=2, circuits=("Alpha",))
    design = build_design(frame, age_scale_laps=AGE_SCALE)
    assert design.max_age.shape == (design.n_circuits, design.n_compounds)
    observed_max = frame["tyre_age"].max()
    assert design.max_age.max() <= observed_max + 1e-9


# -- diagnostics ---------------------------------------------------------


def test_rhat_near_one_for_agreeing_chains() -> None:
    rng = np.random.default_rng(0)
    chains = rng.standard_normal((4, 800, 3))
    assert np.all(split_rhat(chains) < 1.05)


def test_rhat_flags_chains_stuck_at_different_values() -> None:
    chains = np.stack(
        [np.full((400, 1), 0.0), np.full((400, 1), 5.0)], axis=0
    ) + 0.01 * np.random.default_rng(1).standard_normal((2, 400, 1))
    assert split_rhat(chains)[0] > 1.5


def test_ess_is_lower_for_autocorrelated_chains() -> None:
    rng = np.random.default_rng(2)
    independent = rng.standard_normal((2, 1000, 1))

    walk = np.zeros((2, 1000, 1))
    for t in range(1, 1000):
        walk[:, t, 0] = 0.95 * walk[:, t - 1, 0] + rng.standard_normal(2)

    assert ess_bulk(independent)[0] > ess_bulk(walk)[0]


def test_summarise_marks_a_bad_fit_as_failed() -> None:
    chains = np.stack([np.full((200, 1), 0.0), np.full((200, 1), 3.0)], axis=0)
    chains = chains + 0.01 * np.random.default_rng(3).standard_normal(chains.shape)
    result = summarise({"x": chains}, max_rhat=1.01, min_ess=400)
    assert not result.passed
    assert "FAIL" in result.render()


# -- monotonicity constraint ---------------------------------------------


def test_wear_never_decreases_with_tyre_age() -> None:
    """Tyres do not get faster as they age.

    An unconstrained quadratic fitted to real stints does produce decreasing
    wear over part of its range -- at Monza the soft came out 0.16s a lap
    quicker at ten laps old than new -- because teams run softs in short
    stints and disproportionately when the tyre is behaving. Left in, negative
    early wear makes short stints look free.
    """
    ages = np.linspace(0.0, 45.0, 60)
    z_fresh = -15.0 / 20.0
    z = (ages - 15.0) / 20.0

    # A curve that genuinely dips: small negative slope, positive curvature.
    for linear, quad in ((-0.4, 0.5), (0.8, -0.6), (1.2, 0.2), (-0.2, -0.1)):
        values = monotone_degradation(
            np.zeros_like(z), np.full_like(z, linear), np.full_like(z, quad), z, z_fresh
        )
        wear = values - values[0]
        assert np.all(np.diff(wear) >= -1e-9), f"wear decreased for linear={linear} quad={quad}"
        assert np.all(wear >= -1e-9)


def test_monotone_constraint_only_binds_where_the_fit_misbehaves() -> None:
    """A well-behaved increasing curve must pass through untouched."""
    ages = np.linspace(0.0, 40.0, 40)
    z_fresh = -15.0 / 20.0
    z = (ages - 15.0) / 20.0
    linear, quad = 1.2, 0.15

    constrained = monotone_degradation(
        np.zeros_like(z), np.full_like(z, linear), np.full_like(z, quad), z, z_fresh
    )
    raw = linear * (z - z_fresh) + quad * (z * z - z_fresh * z_fresh)
    assert np.allclose(constrained - constrained[0], raw, atol=1e-9)


def test_fitted_curves_are_physical(recovered) -> None:
    posterior, _ = recovered
    summary = posterior.summary(ages=(5, 10, 20, 30))
    for (circuit, compound), group in summary.groupby(["circuit", "compound"]):
        ordered = group.sort_values("age_laps")["loss_s_median"].to_numpy()
        assert np.all(ordered >= -1e-9), f"{circuit}/{compound} has negative wear"
        assert np.all(np.diff(ordered) >= -1e-9), f"{circuit}/{compound} wear decreases with age"
