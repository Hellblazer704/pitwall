"""Convergence diagnostics.

Split-Rhat and effective sample size, following Vehtari et al. (2021),
"Rank-normalization, folding, and localization: an improved Rhat for assessing
convergence of MCMC". Implemented here rather than pulled from ArviZ because
the whole sampler is hand-rolled and a diagnostic you cannot read is not much
of a check.

Split-Rhat splits each chain in half before comparing between- and
within-chain variance, which catches a chain that is drifting steadily but has
not yet visited a second mode: unsplit Rhat is blind to that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["DiagnosticResult", "ess_bulk", "split_rhat", "summarise"]


@dataclass(frozen=True)
class DiagnosticResult:
    table: pd.DataFrame
    max_rhat: float
    min_ess: float
    passed: bool

    def render(self) -> str:
        worst = self.table.sort_values("rhat", ascending=False).head(10)
        lines = [
            f"Convergence: max Rhat {self.max_rhat:.4f}, min ESS {self.min_ess:.0f} "
            f"-> {'PASS' if self.passed else 'FAIL'}",
            "",
            "Worst 10 by Rhat:",
            worst.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)


def _as_chain_draw(values: np.ndarray) -> np.ndarray:
    """Coerce to (chain, draw), flattening any trailing parameter axes to one."""
    if values.ndim < 2:
        raise ValueError("expected at least (chain, draw)")
    return values.reshape(values.shape[0], values.shape[1], -1)


def split_rhat(values: np.ndarray) -> np.ndarray:
    """Split-Rhat per parameter. ``values`` is (chain, draw, ...)."""
    x = _as_chain_draw(values)
    n_draws, n_params = x.shape[1], x.shape[2]
    half = n_draws // 2
    if half < 2:
        return np.full(n_params, np.nan)

    # Split each chain in two, giving 2 * n_chains sequences of length `half`.
    split = np.concatenate([x[:, :half, :], x[:, half : 2 * half, :]], axis=0)
    n = split.shape[1]

    chain_means = split.mean(axis=1)  # (m, n_params)
    chain_vars = split.var(axis=1, ddof=1)  # (m, n_params)

    within = chain_vars.mean(axis=0)
    between = n * chain_means.var(axis=0, ddof=1)

    var_hat = ((n - 1) / n) * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / within)
    # A parameter that is constant in every chain (an inactive coefficient held
    # at zero) has zero variance and no meaningful Rhat.
    rhat[~np.isfinite(rhat)] = np.nan
    rhat[within <= 0] = np.nan
    return rhat


def _autocorr(chain: np.ndarray) -> np.ndarray:
    """Autocorrelation of a 1-d sequence via FFT."""
    n = chain.size
    centred = chain - chain.mean()
    size = int(2 ** np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centred, n=size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n]
    acov /= n
    if acov[0] <= 0:
        return np.zeros(n)
    return acov / acov[0]


def ess_bulk(values: np.ndarray) -> np.ndarray:
    """Effective sample size per parameter, using Geyer's initial positive sequence.

    Sums the autocorrelations pairwise and truncates at the first negative
    pair, which is the standard way of keeping the estimator from being wrecked
    by the noisy tail of the empirical autocorrelation function.
    """
    x = _as_chain_draw(values)
    n_chains, n_draws, n_params = x.shape
    out = np.empty(n_params)

    for p in range(n_params):
        series = x[:, :, p]
        if np.allclose(series, series.flat[0]):
            out[p] = np.nan
            continue

        rho = np.mean([_autocorr(series[c]) for c in range(n_chains)], axis=0)
        # Pairwise sums, truncated at the first non-positive pair.
        total = 0.0
        t = 1
        while t + 1 < n_draws:
            pair = rho[t] + rho[t + 1]
            if pair <= 0:
                break
            total += pair
            t += 2
        tau = 1.0 + 2.0 * total
        out[p] = n_chains * n_draws / max(tau, 1e-6)

    return out


def summarise(
    named: dict[str, np.ndarray],
    max_rhat: float = 1.01,
    min_ess: float = 400.0,
) -> DiagnosticResult:
    """Diagnostics for a dict of ``name -> (chain, draw, ...)`` arrays."""
    rows = []
    for name, values in named.items():
        rhat = split_rhat(values)
        ess = ess_bulk(values)
        flat = _as_chain_draw(values)
        merged = flat.reshape(-1, flat.shape[-1])
        for i in range(rhat.size):
            if not np.isfinite(rhat[i]):
                continue
            rows.append(
                {
                    "parameter": name if rhat.size == 1 else f"{name}[{i}]",
                    "mean": float(merged[:, i].mean()),
                    "sd": float(merged[:, i].std(ddof=1)),
                    "rhat": float(rhat[i]),
                    "ess": float(ess[i]),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return DiagnosticResult(table, np.nan, np.nan, False)

    worst_rhat = float(table["rhat"].max())
    worst_ess = float(table["ess"].min())
    passed = bool(worst_rhat <= max_rhat and worst_ess >= min_ess)
    if not passed:
        log.warning(
            "convergence check failed: max Rhat %.4f (limit %.3f), min ESS %.0f (limit %.0f)",
            worst_rhat,
            max_rhat,
            worst_ess,
            min_ess,
        )
    return DiagnosticResult(table, worst_rhat, worst_ess, passed)
