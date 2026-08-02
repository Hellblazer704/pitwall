"""The fitted posterior, in the form the rest of the project consumes.

:class:`DegradationPosterior` is the boundary between Component 1 and
Component 2. The simulator never sees a point estimate: it draws a whole
parameter vector per simulated race, so a Monte Carlo ensemble carries
degradation *model* uncertainty alongside race-to-race randomness. Drawing one
index per race rather than one per car also preserves the posterior
correlations between compounds, which matters because the medium and hard
curves at a circuit are estimated from overlapping data and their errors move
together. Treating them as independent would make a compound switch look less
risky than it is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pitwall.degradation.gibbs import PosteriorDraws

log = logging.getLogger(__name__)

__all__ = ["DegradationPosterior"]


@dataclass(frozen=True)
class DegradationPosterior:
    """Posterior draws, indexed for fast lookup during simulation."""

    beta: np.ndarray  # (n_samples, n_circuits, n_compounds, 3)
    mu: np.ndarray  # (n_samples, n_compounds, 3)
    tau2: np.ndarray  # (n_samples, n_compounds, 3)
    driver: np.ndarray  # (n_samples, n_drivers, n_compounds)
    phi: np.ndarray  # (n_samples,)
    theta: np.ndarray  # (n_samples, n_circuits) track evolution, s per race fraction
    sigma: np.ndarray  # (n_samples,)
    active: np.ndarray  # (n_circuits, n_compounds, 3) bool
    max_age: np.ndarray  # (n_circuits, n_compounds) oldest observed tyre age
    circuits: list[str]
    compounds: list[str]
    drivers: list[str]
    age_scale_laps: float
    age_center_laps: float
    fuel_mean_kg: float
    circuit_scale: np.ndarray
    quadratic: bool

    # ------------------------------------------------------------- basis

    def basis(self, age_laps: float | np.ndarray) -> np.ndarray:
        """Design row ``[1, z, z^2]`` for a tyre of ``age_laps``.

        ``z`` is the centred, scaled age used when fitting. Every prediction
        goes through here so the centring can never drift out of sync between
        the fit and the simulator.
        """
        z = (np.asarray(age_laps, dtype=float) - self.age_center_laps) / self.age_scale_laps
        quad = z * z if self.quadratic else np.zeros_like(z)
        return np.stack([np.ones_like(z), z, quad], axis=-1)

    # ---------------------------------------------------------------- build

    @classmethod
    def from_draws(cls, draws: PosteriorDraws) -> DegradationPosterior:
        return cls(
            beta=draws.flat("beta"),
            mu=draws.flat("mu"),
            tau2=draws.flat("tau2"),
            driver=draws.flat("driver"),
            phi=draws.flat("phi"),
            theta=draws.flat("theta"),
            sigma=np.sqrt(draws.flat("sigma2")),
            active=np.asarray(draws.active, dtype=bool),
            max_age=np.asarray(draws.max_age, dtype=float),
            circuits=list(draws.circuits),
            compounds=list(draws.compounds),
            drivers=list(draws.drivers),
            age_scale_laps=float(draws.age_scale_laps),
            age_center_laps=float(draws.age_center_laps),
            fuel_mean_kg=float(draws.fuel_mean_kg),
            circuit_scale=np.asarray(draws.circuit_scale, dtype=float),
            quadratic=bool(draws.quadratic),
        )

    @property
    def n_samples(self) -> int:
        return int(self.beta.shape[0])

    def compound_index(self, compound: str) -> int:
        try:
            return self.compounds.index(compound.upper())
        except ValueError as exc:
            raise KeyError(f"compound {compound!r} not in {self.compounds}") from exc

    def driver_index(self, driver: str | None) -> int | None:
        if driver is None:
            return None
        try:
            return self.drivers.index(driver.upper())
        except ValueError:
            return None

    # ----------------------------------------------------------- prediction

    def coefficients(
        self, circuit: str, sample: int | slice | np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Degradation coefficients ``(..., n_compounds, 3)`` at ``circuit``.

        A circuit that was never fitted (a new venue, or one that only ever ran
        wet) is predicted from the population: coefficients are drawn from
        ``N(mu, tau^2)`` using the same posterior sample, so the prediction
        carries the full between-circuit spread rather than pretending the
        average circuit is a safe guess. This is the payoff of the hierarchy
        and it is why a new track gets wide intervals instead of a wrong
        confident answer.
        """
        if circuit in self.circuits:
            index = self.circuits.index(circuit)
            coefs = self.beta[sample, index]
            missing = ~self.active[index]
            # A compound nobody ran at this circuit is a structural zero, not an
            # estimate of zero degradation. Fill those from the population so a
            # strategy that reaches for an unused compound gets the pooled curve
            # with its full between-circuit spread instead of a free tyre.
            if missing[:, 1].any():
                if rng is None:
                    raise ValueError(
                        f"{circuit!r} has no data for compound(s) "
                        f"{[self.compounds[k] for k in np.nonzero(missing[:, 1])[0]]}; "
                        "pass rng to draw them from the population distribution"
                    )
                coefs = np.array(coefs, copy=True)
                drawn = self._population_draw(sample, rng)
                coefs[..., missing] = drawn[..., missing]
            return coefs

        if rng is None:
            raise ValueError(
                f"circuit {circuit!r} was not in the training set; pass rng to draw it "
                "from the population distribution"
            )
        log.info("circuit %r unseen in training; drawing coefficients from the population", circuit)
        return self._population_draw(sample, rng)

    def _population_draw(
        self, sample: int | slice | np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw coefficients from ``N(mu, tau^2)`` for the given posterior sample."""
        mu = self.mu[sample]
        tau = np.sqrt(self.tau2[sample])
        return mu + tau * rng.standard_normal(np.shape(mu))

    def degradation_seconds(
        self,
        coefficients: np.ndarray,
        compound_idx: int | np.ndarray,
        age_laps: float | np.ndarray,
        driver_slope: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Lap-time loss in seconds for a tyre of ``age_laps``.

        Measured relative to the reference compound at the mean tyre age, so
        the return value bundles the compound offset with the wear. Positive
        means slower. Only differences between two calls are meaningful in
        absolute terms; the simulator uses it that way.
        """
        row = self.basis(age_laps)
        coefs = np.asarray(coefficients)
        selected = coefs[..., compound_idx, :]
        base = np.einsum("...j,...j->...", selected, row)
        return base + np.asarray(driver_slope) * row[..., 1]

    def driver_slope(
        self, sample: int | slice | np.ndarray, driver: str | None, compound_idx: int
    ) -> float:
        index = self.driver_index(driver)
        if index is None:
            return 0.0
        return float(np.asarray(self.driver[sample, index, compound_idx]))

    def marginal_slope_per_lap(
        self, circuit: str, compound: str, age_laps: float, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Instantaneous degradation rate, seconds per lap, across all draws.

        The derivative of the curve rather than its level, which is the
        quantity a strategist actually reasons with ("we're losing three tenths
        a lap on these").
        """
        k = self.compound_index(compound)
        coefs = self.coefficients(circuit, slice(None), rng=rng)
        linear = coefs[:, k, 1]
        quad = coefs[:, k, 2] if self.quadratic else 0.0
        scale = self.age_scale_laps
        z = (age_laps - self.age_center_laps) / scale
        return (linear + 2.0 * quad * z) / scale

    # -------------------------------------------------------------- reports

    def summary(self, ages: tuple[int, ...] = (5, 15, 25)) -> pd.DataFrame:
        """Per circuit and compound, degradation with 90% credible intervals.

        Reports the *increment* from a fresh tyre: the basis row at age zero is
        subtracted, which cancels the compound offset and leaves pure wear,
        comparable across compounds and across circuits.
        """
        rows = []
        fresh = self.basis(0.0)
        for ci, circuit in enumerate(self.circuits):
            for ki, compound in enumerate(self.compounds):
                if not self.active[ci, ki, 1]:
                    continue
                coefs = self.beta[:, ci, ki, :]
                for age in ages:
                    delta = self.basis(float(age)) - fresh
                    loss = coefs @ delta
                    lower, median, upper = np.percentile(loss, [5, 50, 95])
                    rows.append(
                        {
                            "circuit": circuit,
                            "compound": compound,
                            "age_laps": age,
                            "loss_s_median": float(median),
                            "loss_s_q05": float(lower),
                            "loss_s_q95": float(upper),
                            "loss_s_per_lap": float(median / age),
                            "ci_width_s": float(upper - lower),
                        }
                    )
        return pd.DataFrame(rows)

    def compound_offsets(self) -> pd.DataFrame:
        """Pace of each compound relative to the medium, per circuit.

        Two different numbers, and confusing them is easy. Because tyre age is
        centred before fitting, the raw coefficient is pace at the *mean* tyre
        age in the dataset, not on a new tyre. At mean age a hard can genuinely
        be quicker than a medium, because the medium has worn more by then;
        that is not a broken fit, it is the whole point of the wear model.

        ``at_mean_age_s`` is the raw coefficient. ``fresh_s`` evaluates the
        curve at zero age, which is the number people mean by "compound
        offset", and there the softer tyre should be the quicker one.
        """
        rows = []
        fresh_row = self.basis(0.0)
        for ci, circuit in enumerate(self.circuits):
            reference = None
            for ki in range(len(self.compounds)):
                if not self.active[ci, ki, 1]:
                    continue
                fresh = self.beta[:, ci, ki, :] @ fresh_row
                if not self.active[ci, ki, 0]:
                    reference = fresh
            for ki, compound in enumerate(self.compounds):
                if not self.active[ci, ki, 1]:
                    continue
                fresh = self.beta[:, ci, ki, :] @ fresh_row
                if reference is not None:
                    fresh = fresh - reference
                raw = self.beta[:, ci, ki, 0]
                lo_r, mid_r, hi_r = np.percentile(raw, [5, 50, 95])
                lo_f, mid_f, hi_f = np.percentile(fresh, [5, 50, 95])
                rows.append(
                    {
                        "circuit": circuit,
                        "compound": compound,
                        "at_mean_age_s": float(mid_r),
                        "at_mean_age_q05": float(lo_r),
                        "at_mean_age_q95": float(hi_r),
                        "fresh_s": float(mid_f),
                        "fresh_q05": float(lo_f),
                        "fresh_q95": float(hi_f),
                        "max_observed_age": float(self.max_age[ci, ki]),
                    }
                )
        return pd.DataFrame(rows)

    def fuel_effect(self) -> dict[str, float]:
        """The pooled fuel coefficient, and what it means per lap.

        ``s_per_kg`` is the estimated coefficient. ``s_per_lap_at_reference`` is
        it multiplied by the fuel burned in one lap of a race of typical length,
        which is the form the number is usually quoted in.
        """
        burn_per_lap_kg = 110.0 / 57.0  # a typical modern race distance
        lower, median, upper = np.percentile(self.phi, [5, 50, 95])
        return {
            "s_per_kg_median": float(median),
            "s_per_kg_q05": float(lower),
            "s_per_kg_q95": float(upper),
            "s_per_lap_median": float(median * burn_per_lap_kg),
            "s_per_lap_q05": float(lower * burn_per_lap_kg),
            "s_per_lap_q95": float(upper * burn_per_lap_kg),
        }

    def track_evolution(self) -> pd.DataFrame:
        """Per-circuit lap-time change from the start of the race to the end.

        Negative means the track gets faster as it rubbers in, which is the
        usual direction. A positive value is a circuit where rising track
        temperature outweighs the rubber, which does happen at hot venues.
        """
        rows = []
        for ci, circuit in enumerate(self.circuits):
            lower, median, upper = np.percentile(self.theta[:, ci], [5, 50, 95])
            rows.append(
                {
                    "circuit": circuit,
                    "evolution_s_median": float(median),
                    "evolution_s_q05": float(lower),
                    "evolution_s_q95": float(upper),
                }
            )
        return pd.DataFrame(rows).sort_values("evolution_s_median").reset_index(drop=True)

    def driver_ranking(self, compound: str | None = None) -> pd.DataFrame:
        """Driver tyre-management effect, negative meaning kinder on the tyre.

        Units are seconds per ``age_scale_laps`` laps of tyre age, i.e. the
        driver's private adjustment to the degradation slope.
        """
        rows = []
        indices = (
            range(len(self.compounds)) if compound is None else [self.compound_index(compound)]
        )
        for di, driver in enumerate(self.drivers):
            values = np.concatenate([self.driver[:, di, ki] for ki in indices])
            if np.allclose(values, 0.0):
                continue
            lower, median, upper = np.percentile(values, [5, 50, 95])
            rows.append(
                {
                    "driver": driver,
                    "slope_adjust_median": float(median),
                    "slope_adjust_q05": float(lower),
                    "slope_adjust_q95": float(upper),
                    # Does the 90% interval exclude zero?
                    "credible": bool(lower > 0 or upper < 0),
                }
            )
        return pd.DataFrame(rows).sort_values("slope_adjust_median").reset_index(drop=True)

    # ------------------------------------------------------------- storage

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            beta=self.beta,
            mu=self.mu,
            tau2=self.tau2,
            driver=self.driver,
            phi=self.phi,
            theta=self.theta,
            sigma=self.sigma,
            circuit_scale=self.circuit_scale,
            active=self.active,
            max_age=self.max_age,
        )
        meta = {
            "circuits": self.circuits,
            "compounds": self.compounds,
            "drivers": self.drivers,
            "age_scale_laps": self.age_scale_laps,
            "age_center_laps": self.age_center_laps,
            "fuel_mean_kg": self.fuel_mean_kg,
            "quadratic": self.quadratic,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("saved posterior (%d draws) to %s", self.n_samples, path)
        return path

    @classmethod
    def load(cls, path: Path) -> DegradationPosterior:
        path = Path(path)
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return cls(
            beta=arrays["beta"],
            mu=arrays["mu"],
            tau2=arrays["tau2"],
            driver=arrays["driver"],
            phi=arrays["phi"],
            theta=arrays["theta"],
            sigma=arrays["sigma"],
            active=arrays["active"].astype(bool),
            max_age=arrays["max_age"],
            circuits=list(meta["circuits"]),
            compounds=list(meta["compounds"]),
            drivers=list(meta["drivers"]),
            age_scale_laps=float(meta["age_scale_laps"]),
            age_center_laps=float(meta["age_center_laps"]),
            fuel_mean_kg=float(meta["fuel_mean_kg"]),
            circuit_scale=arrays["circuit_scale"],
            quadratic=bool(meta["quadratic"]),
        )
