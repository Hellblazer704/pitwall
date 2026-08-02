"""Design matrices, and the fuel/tyre-age identification argument.

The hard part of this component is not the sampler, it is that fuel burn and
tyre wear are close to unidentifiable from raw lap times. Both make lap time
change monotonically with laps completed, in opposite directions, and inside a
single stint they are perfectly collinear: on lap 12 of a stint the tyre is 12
laps old and the car is 12 laps lighter, always. A regression of lap time on
tyre age within a stint estimates the *sum* of the two effects and calls it
degradation, which understates real degradation by the whole size of the fuel
effect. This model puts that at about 0.056 s/lap of burn-off, so a 20-lap
stint is mis-costed by more than a second, and a 30-lap stint by nearly two.
That error sits directly on top of the stint-length decision the entire
strategy turns on, and it is signed the wrong way: it makes long stints look
cheaper than they are, which is exactly the bias that makes a naive simulator
recommend one-stopping.

What breaks the collinearity
----------------------------

Three features of race data, in decreasing order of how much work they do.

1. **Stints reset tyre age but not fuel load.** Lap 2 of the race and lap 2 of
   a stint that began on lap 30 both have a tyre roughly two laps old, but the
   second is carrying ~50 kg less fuel. Comparing equal-age laps at different
   fuel loads identifies the fuel coefficient directly. This is the main
   source of identification and it is why the race-driver intercept below is
   the right nuisance parameter: it absorbs everything constant within a
   driver's race, leaving exactly these within-race contrasts.

2. **Stint lengths and pit laps vary across drivers and races.** Two cars at
   the same circuit on the same lap can be 25 laps apart in tyre age. The
   design is therefore not collinear once a whole field is pooled, even though
   each individual car's stint is.

3. **The fuel coefficient is shared across every circuit and season.** One
   number is estimated from every lap in the dataset, while degradation is
   free to vary by compound and circuit. Pooling the quantity that is
   physically common and freeing the quantity that genuinely varies is what
   makes the separation stable rather than merely possible.

What remains confounded, honestly
---------------------------------

Track evolution. A circuit rubbers in over a race distance and lap times fall
for reasons that have nothing to do with fuel. Within a race, "laps completed"
is the only regressor either effect can load onto, so evolution and fuel burn
are not separately identified from race data alone. This model attributes both
to the fuel term, which biases the fuel coefficient upwards.

Two things keep that from contaminating the degradation estimates. The prior
on the fuel coefficient is informative and physically motivated (roughly
0.03 s per lap per kg), so the posterior cannot drift far to soak up
evolution. And because evolution is common to the whole field while the
contrasts that identify degradation are *between* cars at different tyre ages
on the same lap, evolution largely cancels out of the degradation terms. The
residual bias is discussed in DESIGN.md and quantified in the backtest.

Model
-----

For lap ``i`` driven by driver ``d`` at circuit ``c`` on compound ``k``, in
race-driver group ``g``::

    t_i = m_g                                  race-driver base pace
        + phi * L_c * (F_i - Fbar)             fuel
        + gamma_{c,k}                          compound offset at this circuit
        + beta1_{c,k} * a_i                    linear degradation
        + beta2_{c,k} * a_i^2                  convex degradation / cliff
        + u_{d,k} * a_i                        driver tyre management
        + eps_i

with ``a_i`` the tyre age in laps divided by ``age_scale_laps``, ``F_i`` the
estimated fuel mass in kg, and ``L_c`` a per-circuit scale (see below).

``m_g`` is a free intercept per race-driver combination. It absorbs car pace,
driver pace, circuit base pace, season-to-season development, fuel-corrected
track temperature and anything else constant across one driver's race. That is
a lot of nuisance parameters, ~1800 of them, but they are conditionally
independent given everything else so they cost one vectorised update per sweep,
and they buy an enormous amount: no part of the degradation estimate can be
contaminated by which teams happened to run which compound.

``gamma_{c,MEDIUM}`` is pinned to zero. Compound offsets are only ever used as
differences, and without a reference the offsets and ``m_g`` share a flat
direction that mixes badly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["REFERENCE_COMPOUND", "DesignData", "build_design"]

# Compound whose offset is fixed at zero, making the others interpretable as
# "seconds per lap slower than the medium at this circuit".
REFERENCE_COMPOUND = "MEDIUM"

# Coefficient slots in the per-(circuit, compound) block.
COEF_OFFSET = 0
COEF_LINEAR = 1
COEF_QUADRATIC = 2
COEF_NAMES = ("offset", "linear", "quadratic")


@dataclass(frozen=True)
class DesignData:
    """Everything the sampler needs, as flat integer-indexed arrays."""

    y: np.ndarray  # (N,) lap time in seconds
    age: np.ndarray  # (N,) tyre age, scaled
    fuel: np.ndarray  # (N,) circuit-scaled, centred fuel mass
    group_idx: np.ndarray  # (N,) race-driver
    circuit_idx: np.ndarray  # (N,)
    compound_idx: np.ndarray  # (N,)
    driver_idx: np.ndarray  # (N,)

    groups: list[str]
    circuits: list[str]
    compounds: list[str]
    drivers: list[str]

    # Column mask over the 3 coefficient slots per (circuit, compound):
    # False means the coefficient is held at zero rather than sampled.
    active: np.ndarray  # (n_circuits, n_compounds, 3) bool

    age_scale_laps: float
    age_center_laps: float
    fuel_mean_kg: float
    circuit_scale: np.ndarray  # (n_circuits,) L_c
    quadratic: bool

    @property
    def n_obs(self) -> int:
        return int(self.y.size)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_circuits(self) -> int:
        return len(self.circuits)

    @property
    def n_compounds(self) -> int:
        return len(self.compounds)

    @property
    def n_drivers(self) -> int:
        return len(self.drivers)

    def design_columns(self) -> np.ndarray:
        """(N, 3) per-observation design for the (circuit, compound) block."""
        return np.column_stack([np.ones_like(self.age), self.age, self.age**2])

    def summary(self) -> str:
        return (
            f"{self.n_obs:,} laps | {self.n_groups:,} race-driver groups | "
            f"{self.n_circuits} circuits | {self.n_compounds} compounds | "
            f"{self.n_drivers} drivers"
        )


def _index(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Factorise into contiguous indices with a sorted, stable label order."""
    labels = sorted(values.dropna().unique().tolist())
    lookup = {label: i for i, label in enumerate(labels)}
    return values.map(lookup).to_numpy(dtype=np.int64), [str(label) for label in labels]


def build_design(
    frame: pd.DataFrame,
    age_scale_laps: float = 20.0,
    quadratic: bool = True,
    reference_lap_s: float | None = None,
) -> DesignData:
    """Turn the cleaned modelling frame into arrays for the Gibbs sampler.

    ``reference_lap_s`` normalises the per-circuit fuel scale. A kilogram of
    fuel costs more time at a circuit with more accelerations and a longer lap,
    and median green-flag lap time is the observable that tracks this without
    needing a hardcoded table of lap distances. Scaling by it lets a single
    pooled fuel coefficient apply everywhere, which is what point 3 above
    depends on.
    """
    required = {
        "lap_time_s",
        "tyre_age",
        "fuel_mass_kg",
        "circuit",
        "compound",
        "driver",
        "season",
        "round",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"modelling frame is missing {sorted(missing)}")

    work = frame.dropna(subset=["lap_time_s", "tyre_age", "fuel_mass_kg"]).copy()
    work["group"] = (
        work["season"].astype(str)
        + "_"
        + work["round"].astype(str).str.zfill(2)
        + "_"
        + work["driver"].astype(str)
    )

    group_idx, groups = _index(work["group"])
    circuit_idx, circuits = _index(work["circuit"])
    compound_idx, compounds = _index(work["compound"])
    driver_idx, drivers = _index(work["driver"])

    # Per-circuit fuel scale, normalised so the median circuit sits at 1.0.
    median_lap = work.groupby("circuit")["lap_time_s"].median()
    reference = float(reference_lap_s if reference_lap_s is not None else median_lap.median())
    circuit_scale = np.array([float(median_lap[c]) / reference for c in circuits])

    # Age is centred before scaling. With an all-positive regressor the linear
    # and quadratic columns are correlated at about 0.97, and the sampler pays
    # for it directly: the two coefficients trade off along a ridge and the
    # chain crawls along it (effective sample sizes in the tens, not hundreds).
    # Centring makes the two columns near-orthogonal because the age
    # distribution is roughly symmetric, and costs nothing but bookkeeping.
    #
    # The consequence is that the per-cell offset now means "pace at the mean
    # tyre age" rather than "pace on a brand new tyre". Nothing downstream
    # cares, because wear is always reported as a difference between two ages,
    # which cancels the offset.
    age_laps = work["tyre_age"].to_numpy(dtype=np.float64)
    age_center = float(np.mean(age_laps))
    age = (age_laps - age_center) / float(age_scale_laps)
    fuel_kg = work["fuel_mass_kg"].to_numpy(dtype=np.float64)
    fuel_mean = float(fuel_kg.mean())
    fuel = (fuel_kg - fuel_mean) * circuit_scale[circuit_idx]

    # Which coefficients are sampled. A (circuit, compound) cell with no laps
    # is left inactive so it contributes nothing to the hierarchy's variance
    # estimate; it is predicted from the population mean instead.
    n_c, n_k = len(circuits), len(compounds)
    observed = np.zeros((n_c, n_k), dtype=bool)
    observed[circuit_idx, compound_idx] = True

    active = np.zeros((n_c, n_k, 3), dtype=bool)
    active[:, :, COEF_OFFSET] = observed
    active[:, :, COEF_LINEAR] = observed
    active[:, :, COEF_QUADRATIC] = observed if quadratic else False
    if REFERENCE_COMPOUND in compounds:
        active[:, compounds.index(REFERENCE_COMPOUND), COEF_OFFSET] = False
    else:  # pragma: no cover - only if a dataset has no medium-tyre laps at all
        log.warning("no %s laps; compound offsets are only weakly identified", REFERENCE_COMPOUND)

    design = DesignData(
        y=work["lap_time_s"].to_numpy(dtype=np.float64),
        age=age,
        fuel=fuel,
        group_idx=group_idx,
        circuit_idx=circuit_idx,
        compound_idx=compound_idx,
        driver_idx=driver_idx,
        groups=groups,
        circuits=circuits,
        compounds=compounds,
        drivers=drivers,
        active=active,
        age_scale_laps=float(age_scale_laps),
        age_center_laps=age_center,
        fuel_mean_kg=fuel_mean,
        circuit_scale=circuit_scale,
        quadratic=bool(quadratic),
    )
    log.info("design: %s", design.summary())
    return design


def collinearity_diagnostic(design: DesignData) -> pd.DataFrame:
    """How well separated fuel and tyre age actually are, per circuit.

    Reports the within-race-driver correlation between the fuel regressor and
    tyre age. Near -1 means that circuit's data cannot separate the two on its
    own and the estimate there is leaning on the pooled fuel coefficient; the
    number is worth looking at before believing any single circuit's curve.

    Correlations are computed after sweeping out the race-driver means, because
    that is the variation the sampler actually sees.
    """
    fuel = design.fuel.copy()
    age = design.age.copy()
    counts = np.bincount(design.group_idx, minlength=design.n_groups).astype(float)
    counts[counts == 0] = 1.0
    for values in (fuel, age):
        means = np.bincount(design.group_idx, weights=values, minlength=design.n_groups) / counts
        values -= means[design.group_idx]

    rows = []
    for i, circuit in enumerate(design.circuits):
        mask = design.circuit_idx == i
        if mask.sum() < 10:
            continue
        f, a = fuel[mask], age[mask]
        if f.std() < 1e-9 or a.std() < 1e-9:
            continue
        rows.append(
            {
                "circuit": circuit,
                "n_laps": int(mask.sum()),
                "corr_fuel_age": float(np.corrcoef(f, a)[0, 1]),
                "age_range_laps": float((a.max() - a.min()) * design.age_scale_laps),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_fuel_age").reset_index(drop=True)
