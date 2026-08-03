"""Gibbs sampler for the hierarchical degradation model.

Why hand-rolled rather than PyMC or Stan. The model in
:mod:`pitwall.degradation.design` is a Gaussian hierarchical linear model, and
every full conditional is available in closed form, including the half-Cauchy
scale priors via the standard inverse-gamma scale mixture

    sigma^2 | a ~ InvGamma(1/2, 1/a),   a ~ InvGamma(1/2, 1/A^2)
      =>  sigma ~ HalfCauchy(0, A)

so there is nothing for a gradient-based sampler to buy. A Gibbs sweep here is
a handful of vectorised numpy passes, it needs no tuning, no warmup adaptation
and no C compiler, and it is exactly reproducible from an integer seed. The
cost is that highly correlated blocks mix slowly, which is why the compound
offsets are anchored to a reference compound in the design rather than left to
float against the race-driver intercepts.

Blocks per sweep:

1. ``m``      race-driver intercepts, conditionally independent
2. ``phi``    the pooled fuel coefficient
3. ``B``      per (circuit, compound) offset / linear / quadratic, batched
4. ``mu``     population mean per compound and coefficient
5. ``tau2``   between-circuit variance, with its auxiliary variable
6. ``U``      driver tyre-management slopes
7. ``sigma_u2``, ``sigma2``, with theirs
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pitwall.degradation.design import COEF_LINEAR, DesignData

log = logging.getLogger(__name__)

__all__ = ["GibbsPriors", "PosteriorDraws", "run_chain", "sample"]


@dataclass(frozen=True)
class GibbsPriors:
    """Hyperparameters. Defaults mirror ``conf/degradation/default.yaml``."""

    # Fuel coefficient, seconds per lap per kg. Informative on purpose: this is
    # the prior that stops the fuel term drifting to absorb track evolution.
    fuel_mean: float = 0.030
    fuel_sd: float = 0.012

    # Population mean of each degradation coefficient.
    mu_mean: float = 0.0
    mu_sd: float = 2.0

    # Track evolution, per circuit, in seconds per unit of race fraction.
    # Weakly informative and centred on zero: the sign is not assumed, because
    # a circuit can get slower over a race (rising track temperature) as well
    # as faster (rubber).
    theta_sd: float = 3.0

    # Half-Cauchy scale on the between-circuit sd, one per coefficient slot
    # (offset, linear, quadratic).
    #
    # The quadratic gets a much tighter scale than the other two. It is the
    # weakest-identified coefficient in the model -- a cell whose stints all sit
    # in a narrow age band cannot pin curvature at all -- and because the scale
    # is shared across circuits, a single runaway cell inflates tau for
    # everybody and switches off the shrinkage that was supposed to catch it.
    # Real degradation curvature is mild, so a tight scale is also the honest
    # prior belief.
    tau_scale: tuple[float, float, float] = (0.5, 0.5, 0.15)
    driver_sd_scale: float = 0.4  # driver tyre-management spread
    sigma_scale: float = 0.5  # residual lap-time sd

    # Race-driver intercepts get a flat prior; every group has at least
    # ``min_stint_laps`` observations so the posterior is proper.
    intercept_sd: float = 1e3


@dataclass
class PosteriorDraws:
    """Stacked draws from every chain.

    Leading axes are ``(chain, draw, ...)`` so the diagnostics can compute
    between- and within-chain variance without reshaping.
    """

    beta: np.ndarray  # (chain, draw, n_circuits, n_compounds, 3)
    mu: np.ndarray  # (chain, draw, n_compounds, 3)
    tau2: np.ndarray  # (chain, draw, n_compounds, 3)
    phi: np.ndarray  # (chain, draw)
    theta: np.ndarray  # (chain, draw, n_circuits) track evolution
    driver: np.ndarray  # (chain, draw, n_drivers, n_compounds)
    sigma2: np.ndarray  # (chain, draw)
    sigma_u2: np.ndarray  # (chain, draw)
    # (n_circuits, n_compounds, 3): which coefficients were actually sampled.
    # Everything else is a structural zero, not an estimate.
    active: np.ndarray = field(default_factory=lambda: np.ones((1, 1, 3), dtype=bool))
    # (n_circuits, n_compounds) oldest tyre age observed, bounding extrapolation.
    max_age: np.ndarray = field(default_factory=lambda: np.full((1, 1), 30.0))
    circuits: list[str] = field(default_factory=list)
    compounds: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    age_scale_laps: float = 20.0
    age_center_laps: float = 0.0
    fuel_mean_kg: float = 0.0
    circuit_scale: np.ndarray = field(default_factory=lambda: np.ones(1))
    quadratic: bool = True
    runtime_s: float = 0.0

    @property
    def n_chains(self) -> int:
        return int(self.phi.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.phi.shape[1])

    def flat(self, name: str) -> np.ndarray:
        """Draws for ``name`` with chains merged into one axis."""
        values = getattr(self, name)
        return values.reshape((-1, *values.shape[2:]))


def _inv_gamma(
    rng: np.random.Generator, shape: float | np.ndarray, scale: float | np.ndarray
) -> np.ndarray:
    """Draw from InvGamma(shape, scale), i.e. 1 / Gamma(shape, rate=scale)."""
    scale_arr = np.asarray(scale, dtype=np.float64)
    gamma = rng.gamma(shape, 1.0 / np.maximum(scale_arr, 1e-300))
    return 1.0 / np.maximum(gamma, 1e-300)


def _half_cauchy_variance(
    rng: np.random.Generator,
    sum_squares: np.ndarray | float,
    n: np.ndarray | float,
    aux: np.ndarray | float,
    scale: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One update of a variance with a half-Cauchy prior on its sd.

    Returns ``(variance, auxiliary)``. ``sum_squares`` and ``n`` describe the
    zero-mean normal deviates the variance governs.
    """
    variance = _inv_gamma(
        rng,
        0.5 * (np.asarray(n, dtype=float) + 1.0),
        1.0 / np.asarray(aux) + 0.5 * np.asarray(sum_squares),
    )
    new_aux = _inv_gamma(rng, 1.0, 1.0 / np.asarray(scale, dtype=float) ** 2 + 1.0 / variance)
    return variance, new_aux


def _group_sums(values: np.ndarray, index: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(index, weights=values, minlength=size)


class _ActivePattern:
    """A set of (circuit, compound) cells sharing the same active columns.

    Cells are grouped so their 3x3 (or smaller) normal equations can be solved
    as one batched linear algebra call instead of a Python loop over ~90 cells
    per sweep.
    """

    __slots__ = ("cells", "columns", "n_cells", "n_cols")

    def __init__(self, cells: np.ndarray, columns: np.ndarray) -> None:
        self.cells = cells
        self.columns = columns
        self.n_cells = int(cells.size)
        self.n_cols = int(columns.size)


def _build_patterns(design: DesignData) -> list[_ActivePattern]:
    flat_active = design.active.reshape(-1, 3)
    observed = flat_active.any(axis=1)

    patterns: dict[tuple[int, ...], list[int]] = {}
    for cell in np.nonzero(observed)[0]:
        key = tuple(np.nonzero(flat_active[cell])[0].tolist())
        patterns.setdefault(key, []).append(int(cell))

    return [
        _ActivePattern(np.array(cells, dtype=np.int64), np.array(key, dtype=np.int64))
        for key, cells in patterns.items()
        if key
    ]


def _batched_normal(rng: np.random.Generator, precision: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sample from N(P^-1 b, P^-1) for a stack of small precision matrices."""
    cov = np.linalg.inv(precision)
    # Symmetrise: inv() of a symmetric matrix can drift by a few ulp, and
    # cholesky rejects a matrix that is not exactly symmetric.
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    mean = np.einsum("nij,nj->ni", cov, b)
    chol = np.linalg.cholesky(cov)
    noise = rng.standard_normal(mean.shape)
    return mean + np.einsum("nij,nj->ni", chol, noise)


def run_chain(
    design: DesignData,
    priors: GibbsPriors,
    rng: np.random.Generator,
    draws: int,
    warmup: int,
    thin: int = 1,
) -> dict[str, np.ndarray]:
    """Run one chain, returning post-warmup draws keyed by parameter name."""
    n_c, n_k = design.n_circuits, design.n_compounds
    n_d, n_g = design.n_drivers, design.n_groups
    n_obs = design.n_obs
    n_cells = n_c * n_k

    y = design.y
    age = design.age
    fuel = design.fuel
    progress = design.progress
    gi = design.group_idx
    ci = design.circuit_idx
    cell_idx = design.circuit_idx * n_k + design.compound_idx
    du_idx = design.driver_idx * n_k + design.compound_idx

    # Per-observation design columns for the (circuit, compound) block.
    x0 = np.ones(n_obs)
    x1 = age
    x2 = age * age
    columns = (x0, x1, x2)

    patterns = _build_patterns(design)

    # Sufficient statistics that never change: X'X per cell, and the
    # normal-equation pieces for the scalar blocks.
    xtx = np.zeros((n_cells, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            s = _group_sums(columns[i] * columns[j], cell_idx, n_cells)
            xtx[:, i, j] = s
            xtx[:, j, i] = s

    group_counts = np.bincount(gi, minlength=n_g).astype(float)
    group_counts[group_counts == 0] = 1.0
    # Sufficient statistics for the joint (phi, theta) block. The two are
    # exactly collinear within any one race -- fuel mass is an affine function
    # of laps completed, so is race fraction -- which means updating them in
    # separate Gibbs blocks makes the chain random-walk along a ridge instead
    # of sampling it. Measured: Rhat 2.24 and ESS 19 with separate blocks. As
    # one block it is a single 26-dimensional normal draw and mixes fine.
    fuel_ss = float(np.sum(fuel * fuel))
    progress_ss = _group_sums(progress * progress, ci, n_c)
    fuel_progress_cross = _group_sums(fuel * progress, ci, n_c)
    driver_age_ss = _group_sums(age * age, du_idx, n_d * n_k)

    # Arrow-shaped prior precision: phi first, then one theta per circuit.
    prior_prec_joint = np.concatenate(
        [[1.0 / priors.fuel_sd**2], np.full(n_c, 1.0 / priors.theta_sd**2)]
    )
    prior_mean_joint = np.concatenate([[priors.fuel_mean], np.zeros(n_c)])

    # -- initial values, dispersed so R-hat is meaningful ------------------
    m = np.full(n_g, float(np.mean(y)))
    phi = float(priors.fuel_mean + priors.fuel_sd * rng.standard_normal())
    theta = rng.normal(0.0, 0.5, size=n_c)

    # Only cells with data are ever updated, so an inactive cell keeps whatever
    # it was initialised to for the whole run. Dispersing those would leave
    # each chain frozen at a different constant, which is indistinguishable
    # from catastrophic non-convergence in the Rhat table and poisons any
    # summary that averages over cells. They are pinned at zero instead, and
    # excluded from the diagnostics; prediction for an unobserved compound at a
    # circuit draws from the population distribution rather than reading them.
    B = np.zeros((n_cells, 3))
    active_flat_init = design.active.reshape(-1, 3)
    dispersed = rng.normal(0.5, 0.5, size=n_cells)
    B[:, COEF_LINEAR] = np.where(active_flat_init[:, COEF_LINEAR], dispersed, 0.0)
    mu = np.zeros((n_k, 3))
    tau2 = np.full((n_k, 3), 0.1)
    aux_tau = np.ones((n_k, 3))
    U = np.zeros(n_d * n_k)
    sigma_u2 = 0.05
    aux_u = 1.0
    sigma2 = float(np.var(y)) * 0.05
    aux_sigma = 1.0

    active_flat = design.active.reshape(-1, 3)
    driver_active = np.zeros(n_d * n_k, dtype=bool)
    driver_active[np.unique(du_idx)] = True

    keep = max(1, thin)
    n_kept = draws // keep
    store = {
        "beta": np.empty((n_kept, n_c, n_k, 3)),
        "mu": np.empty((n_kept, n_k, 3)),
        "tau2": np.empty((n_kept, n_k, 3)),
        "phi": np.empty(n_kept),
        "theta": np.empty((n_kept, n_c)),
        "driver": np.empty((n_kept, n_d, n_k)),
        "sigma2": np.empty(n_kept),
        "sigma_u2": np.empty(n_kept),
    }

    total = warmup + draws
    kept = 0
    for sweep in range(total):
        beta_fit = B[cell_idx, 0] + B[cell_idx, 1] * x1 + B[cell_idx, 2] * x2
        driver_fit = U[du_idx] * age
        progress_fit = theta[ci] * progress

        # 1. race-driver intercepts -----------------------------------------
        resid = y - phi * fuel - progress_fit - beta_fit - driver_fit
        group_mean = _group_sums(resid, gi, n_g) / group_counts
        post_var = sigma2 / group_counts
        m = group_mean + np.sqrt(post_var) * rng.standard_normal(n_g)

        # 2. fuel coefficient and per-circuit track evolution, jointly --------
        #
        # These two are what the identification argument in design.py turns on.
        # Only their combination is pinned by the data within a race; the split
        # between them comes from phi being one number shared by every circuit
        # and carrying a physical prior, while theta is free per circuit. That
        # makes the posterior a narrow ridge, which is exactly why they have to
        # be drawn together.
        resid = y - m[gi] - beta_fit - driver_fit

        precision = np.zeros((n_c + 1, n_c + 1))
        precision[0, 0] = fuel_ss / sigma2
        precision[0, 1:] = fuel_progress_cross / sigma2
        precision[1:, 0] = fuel_progress_cross / sigma2
        precision[np.arange(1, n_c + 1), np.arange(1, n_c + 1)] = progress_ss / sigma2
        precision[np.diag_indices(n_c + 1)] += prior_prec_joint

        rhs = np.empty(n_c + 1)
        rhs[0] = float(np.dot(fuel, resid)) / sigma2
        rhs[1:] = _group_sums(progress * resid, ci, n_c) / sigma2
        rhs += prior_prec_joint * prior_mean_joint

        covariance = np.linalg.inv(precision)
        covariance = 0.5 * (covariance + covariance.T)
        joint = covariance @ rhs + np.linalg.cholesky(covariance) @ rng.standard_normal(n_c + 1)
        phi = float(joint[0])
        theta = joint[1:]
        progress_fit = theta[ci] * progress

        # 3. per (circuit, compound) coefficients ---------------------------
        resid = y - m[gi] - phi * fuel - progress_fit - driver_fit
        xtr = np.column_stack(
            [_group_sums(columns[a] * resid, cell_idx, n_cells) for a in range(3)]
        )
        prior_prec_cell = np.repeat((1.0 / tau2)[None, :, :], n_c, axis=0).reshape(n_cells, 3)
        prior_mean_cell = np.repeat(mu[None, :, :], n_c, axis=0).reshape(n_cells, 3)

        for pattern in patterns:
            cols = pattern.columns
            cells = pattern.cells
            precision = xtx[np.ix_(cells, cols, cols)] / sigma2
            precision[:, np.arange(pattern.n_cols), np.arange(pattern.n_cols)] += prior_prec_cell[
                np.ix_(cells, cols)
            ]
            rhs_cell = xtr[np.ix_(cells, cols)] / sigma2 + (
                prior_prec_cell[np.ix_(cells, cols)] * prior_mean_cell[np.ix_(cells, cols)]
            )
            B[np.ix_(cells, cols)] = _batched_normal(rng, precision, rhs_cell)

        # 4. population means -----------------------------------------------
        B_grid = B.reshape(n_c, n_k, 3)
        counts = active_flat.reshape(n_c, n_k, 3).sum(axis=0).astype(float)
        sums = np.where(active_flat.reshape(n_c, n_k, 3), B_grid, 0.0).sum(axis=0)
        prior_prec_mu = 1.0 / priors.mu_sd**2
        prec_mu = counts / tau2 + prior_prec_mu
        mean_mu = (sums / tau2 + prior_prec_mu * priors.mu_mean) / prec_mu
        mu = mean_mu + rng.standard_normal((n_k, 3)) / np.sqrt(prec_mu)

        # 5. between-circuit variances --------------------------------------
        centred = np.where(active_flat.reshape(n_c, n_k, 3), B_grid - mu[None, :, :], 0.0)
        ss_tau = (centred**2).sum(axis=0)
        tau2, aux_tau = _half_cauchy_variance(rng, ss_tau, counts, aux_tau, priors.tau_scale)
        tau2 = np.maximum(tau2, 1e-10)

        # 6. driver tyre-management slopes ----------------------------------
        beta_fit = B[cell_idx, 0] + B[cell_idx, 1] * x1 + B[cell_idx, 2] * x2
        resid = y - m[gi] - phi * fuel - progress_fit - beta_fit
        xtr_u = _group_sums(age * resid, du_idx, n_d * n_k)
        prec_u = driver_age_ss / sigma2 + 1.0 / sigma_u2
        mean_u = (xtr_u / sigma2) / prec_u
        U = mean_u + rng.standard_normal(n_d * n_k) / np.sqrt(prec_u)
        U[~driver_active] = 0.0

        # 7. variance components --------------------------------------------
        n_active_u = float(driver_active.sum())
        sigma_u2_arr, aux_u_arr = _half_cauchy_variance(
            rng, float(np.sum(U[driver_active] ** 2)), n_active_u, aux_u, priors.driver_sd_scale
        )
        sigma_u2 = float(np.maximum(sigma_u2_arr, 1e-10))
        aux_u = float(aux_u_arr)

        driver_fit = U[du_idx] * age
        full_resid = y - m[gi] - phi * fuel - progress_fit - beta_fit - driver_fit
        sigma2_arr, aux_sigma_arr = _half_cauchy_variance(
            rng, float(np.dot(full_resid, full_resid)), float(n_obs), aux_sigma, priors.sigma_scale
        )
        sigma2 = float(np.maximum(sigma2_arr, 1e-12))
        aux_sigma = float(aux_sigma_arr)

        if sweep >= warmup:
            index = sweep - warmup
            if index % keep == 0 and kept < n_kept:
                store["beta"][kept] = B.reshape(n_c, n_k, 3)
                store["mu"][kept] = mu
                store["tau2"][kept] = tau2
                store["phi"][kept] = phi
                store["theta"][kept] = theta
                store["driver"][kept] = U.reshape(n_d, n_k)
                store["sigma2"][kept] = sigma2
                store["sigma_u2"][kept] = sigma_u2
                kept += 1

    return store


def sample(
    design: DesignData,
    priors: GibbsPriors,
    generators: list[np.random.Generator],
    draws: int = 2000,
    warmup: int = 1000,
    thin: int = 1,
) -> PosteriorDraws:
    """Run every chain and stack the draws."""
    started = time.perf_counter()
    chains = []
    for i, rng in enumerate(generators):
        chain_started = time.perf_counter()
        chains.append(run_chain(design, priors, rng, draws=draws, warmup=warmup, thin=thin))
        log.info(
            "chain %d/%d done in %.1fs", i + 1, len(generators), time.perf_counter() - chain_started
        )

    def stack(name: str) -> np.ndarray:
        return np.stack([chain[name] for chain in chains], axis=0)

    return PosteriorDraws(
        beta=stack("beta"),
        mu=stack("mu"),
        tau2=stack("tau2"),
        phi=stack("phi"),
        theta=stack("theta"),
        driver=stack("driver"),
        sigma2=stack("sigma2"),
        sigma_u2=stack("sigma_u2"),
        active=design.active.copy(),
        max_age=design.max_age.copy(),
        circuits=list(design.circuits),
        compounds=list(design.compounds),
        drivers=list(design.drivers),
        age_scale_laps=design.age_scale_laps,
        age_center_laps=design.age_center_laps,
        fuel_mean_kg=design.fuel_mean_kg,
        circuit_scale=design.circuit_scale.copy(),
        quadratic=design.quadratic,
        runtime_s=time.perf_counter() - started,
    )
