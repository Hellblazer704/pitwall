"""Fit orchestration: cleaned laps in, saved posterior and a model card out."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from pitwall.degradation import diagnostics
from pitwall.degradation.design import build_design, collinearity_diagnostic
from pitwall.degradation.gibbs import GibbsPriors, sample
from pitwall.degradation.model import DegradationPosterior
from pitwall.ingest.clean import build_modelling_frame, modelling_frame_path, read_modelling_frame
from pitwall.paths import Paths
from pitwall.rng import SeedBank

log = logging.getLogger(__name__)

__all__ = ["fit_from_config", "load_posterior", "posterior_path"]


def posterior_path(paths: Paths, seasons: list[int]) -> Path:
    tag = "-".join(str(season) for season in sorted(seasons))
    return paths.artifacts / f"degradation_{tag}.npz"


def load_posterior(paths: Paths, seasons: list[int]) -> DegradationPosterior:
    target = posterior_path(paths, seasons)
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} not found; run `pitwall fit --seasons {' '.join(map(str, seasons))}`"
        )
    return DegradationPosterior.load(target)


def _priors_from_config(cfg: DictConfig) -> GibbsPriors:
    deg = cfg.degradation
    return GibbsPriors(
        fuel_mean=float(deg.fuel.prior_mean_s_per_kg),
        fuel_sd=float(deg.fuel.prior_sd_s_per_kg),
        mu_mean=float(deg.priors.mu_deg_mean),
        mu_sd=float(deg.priors.mu_deg_sd),
        tau_scale=tuple(float(v) for v in deg.priors.tau_deg_scale),  # type: ignore[arg-type]
        driver_sd_scale=float(deg.priors.driver_sd_scale),
        sigma_scale=float(deg.priors.sigma_scale),
    )


def _model_card(
    posterior: DegradationPosterior,
    result: diagnostics.DiagnosticResult,
    collinearity: pd.DataFrame,
    seasons: list[int],
    n_laps: int,
    runtime_s: float,
) -> str:
    fuel = posterior.fuel_effect()
    summary = posterior.summary(ages=(5, 15, 25))
    at_15 = summary.loc[summary["age_laps"] == 15]

    lines = [
        "# Degradation model card",
        "",
        f"Fitted on seasons {seasons} from {n_laps:,} clean green-flag laps.",
        f"{posterior.n_samples:,} posterior draws, sampled in {runtime_s:.1f}s.",
        "",
        "## Convergence",
        "",
        "```",
        result.render(),
        "```",
        "",
        "## Fuel effect (the separated term)",
        "",
        f"- {fuel['s_per_kg_median']:.4f} s/lap/kg "
        f"(90% CI {fuel['s_per_kg_q05']:.4f} to {fuel['s_per_kg_q95']:.4f})",
        f"- equivalently {fuel['s_per_lap_median']:.3f} s/lap of burn-off "
        f"(90% CI {fuel['s_per_lap_q05']:.3f} to {fuel['s_per_lap_q95']:.3f})",
        "",
        "Any strategy model that skips this term folds it into the degradation",
        "slope with the wrong sign, understating true wear by roughly this much",
        "per lap of stint age.",
        "",
        "## Degradation at 15 laps of age, by compound",
        "",
    ]

    by_compound = (
        at_15.groupby("compound")
        .agg(
            median_loss_s=("loss_s_median", "median"),
            min_loss_s=("loss_s_median", "min"),
            max_loss_s=("loss_s_median", "max"),
            median_ci_width_s=("ci_width_s", "median"),
        )
        .reset_index()
    )
    lines += [
        "```",
        by_compound.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "```",
        "",
    ]

    lines += [
        "The spread between the softest and hardest circuit for a given compound",
        "is the reason the model is hierarchical rather than pooled: a single",
        "global curve would be wrong at both ends of that range.",
        "",
        "## Fuel / tyre-age separability, worst circuits",
        "",
        "Within-race-driver correlation between the fuel regressor and tyre age.",
        "Close to -1 means that circuit cannot separate the two from its own data",
        "and is leaning on the pooled fuel coefficient.",
        "",
        "```",
        collinearity.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "```",
        "",
        "## Driver tyre management",
        "",
        "Adjustment to the degradation slope, in seconds per "
        f"{posterior.age_scale_laps:.0f} laps of tyre age. Negative is kinder on the tyre.",
        "",
        "```",
        posterior.driver_ranking()
        .head(10)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "```",
        "",
    ]
    return "\n".join(lines)


def fit_from_config(cfg: DictConfig, seasons: list[int] | None = None) -> DegradationPosterior:
    paths = Paths.from_config(cfg).ensure()
    seasons = list(seasons or cfg.data.train_seasons)

    frame_path = modelling_frame_path(paths, seasons)
    if frame_path.is_file():
        frame = read_modelling_frame(paths, seasons)
        log.info("loaded %d clean laps from %s", len(frame), frame_path)
    else:
        log.info("no cached modelling frame; building it now")
        frame, report = build_modelling_frame(
            seasons, paths, cfg.data.cleaning, float(cfg.degradation.fuel.start_mass_kg)
        )
        print(report.render())

    design = build_design(
        frame,
        age_scale_laps=float(cfg.degradation.degradation.age_scale_laps),
        quadratic=str(cfg.degradation.degradation.shape) == "quadratic",
    )
    collinearity = collinearity_diagnostic(design)

    bank = SeedBank(int(cfg.seed))
    chains = int(cfg.degradation.sampler.chains)
    generators = bank.chain_generators("degradation", chains)

    draws = sample(
        design,
        _priors_from_config(cfg),
        generators,
        draws=int(cfg.degradation.sampler.draws),
        warmup=int(cfg.degradation.sampler.warmup),
        thin=int(cfg.degradation.sampler.thin),
    )

    result = diagnostics.summarise(
        {
            "phi": draws.phi,
            "sigma2": draws.sigma2,
            "sigma_u2": draws.sigma_u2,
            "mu": draws.mu,
            "tau2": draws.tau2,
            # Only the coefficients the sampler actually updated. Structural
            # zeros have no posterior and would show up as degenerate chains.
            "beta": draws.beta[:, :, draws.active],
        },
        max_rhat=float(cfg.degradation.diagnostics.max_rhat),
        min_ess=float(cfg.degradation.diagnostics.min_ess),
    )
    print(result.render())

    posterior = DegradationPosterior.from_draws(draws)
    target = posterior_path(paths, seasons)
    posterior.save(target)

    posterior.summary().to_csv(target.with_name(f"{target.stem}_summary.csv"), index=False)
    posterior.compound_offsets().to_csv(target.with_name(f"{target.stem}_offsets.csv"), index=False)
    result.table.to_csv(target.with_name(f"{target.stem}_diagnostics.csv"), index=False)
    collinearity.to_csv(target.with_name(f"{target.stem}_collinearity.csv"), index=False)

    card = _model_card(posterior, result, collinearity, seasons, design.n_obs, draws.runtime_s)
    card_path = target.with_name(f"{target.stem}_card.md")
    card_path.write_text(card, encoding="utf-8")
    log.info("wrote model card to %s", card_path)
    print()
    print(card)
    return posterior
