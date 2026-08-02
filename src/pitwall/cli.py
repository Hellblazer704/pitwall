"""Command line entry points.

    pitwall ingest      download and flatten race sessions
    pitwall clean       build the modelling frame from the raw tables
    pitwall fit         fit the hierarchical degradation model
    pitwall optimize    Monte Carlo strategy search for one race
    pitwall backtest    validate against a held-out season
    pitwall ablate      run the realism ablation study

Every command takes trailing Hydra-style overrides, so

    pitwall optimize --race 2025:1 simulator.safety_car.enabled=false

is the same code path as the default run with one switch flipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from pitwall.config import load_config
from pitwall.paths import Paths

log = logging.getLogger("pitwall")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # FastF1 narrates every cache hit at INFO, which drowns out our own logs
    # during a multi-hour ingest.
    for noisy in ("fastf1", "fastf1.core", "fastf1.req", "fastf1._api", "fastf1.events"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_race(text: str) -> tuple[int, int]:
    """``"2025:14"`` -> ``(2025, 14)``."""
    season, _, round_no = text.partition(":")
    if not round_no:
        raise argparse.ArgumentTypeError(f"expected SEASON:ROUND, got {text!r}")
    return int(season), int(round_no)


def cmd_ingest(args: argparse.Namespace) -> int:
    from pitwall.ingest.fetch import fetch_seasons

    cfg = load_config(args.overrides)
    paths = Paths.from_config(cfg).ensure()

    seasons = args.seasons or list(cfg.data.train_seasons) + list(cfg.data.holdout_seasons)
    log.info("ingesting seasons %s into %s", seasons, paths.raw)

    failures = fetch_seasons(seasons, paths, session_name=str(cfg.data.session), force=args.force)
    if failures:
        log.warning("%d sessions failed: %s", len(failures), failures)
    log.info("ingest complete")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    from pitwall.ingest.clean import build_modelling_frame, write_modelling_frame

    cfg = load_config(args.overrides)
    paths = Paths.from_config(cfg).ensure()
    seasons = args.seasons or list(cfg.data.train_seasons)

    frame, report = build_modelling_frame(seasons, paths, cfg.data.cleaning)
    write_modelling_frame(frame, report, paths, seasons)
    print(report.render())
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from pitwall.degradation.fit import fit_from_config

    cfg = load_config(args.overrides)
    fit_from_config(cfg, seasons=args.seasons)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from pitwall.optimize.run import optimize_race

    cfg = load_config(args.overrides)
    season, round_no = args.race
    optimize_race(cfg, season, round_no, driver=args.driver, grid=args.grid)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from pitwall.validate.backtest import run_backtest

    cfg = load_config(args.overrides)
    seasons = args.seasons or list(cfg.data.holdout_seasons)
    run_backtest(cfg, seasons, limit=args.limit)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from pitwall.ablation.study import run_ablation

    cfg = load_config(args.overrides)
    run_ablation(cfg, races=args.race, n_races=args.n_races)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitwall", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_overrides(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "overrides",
            nargs="*",
            help="Hydra overrides, e.g. seed=7 simulator.traffic.max_loss_s=0",
        )

    p_ingest = sub.add_parser("ingest", help="download and flatten race sessions")
    p_ingest.add_argument("--seasons", type=int, nargs="+")
    p_ingest.add_argument("--force", action="store_true", help="refetch races already on disk")
    add_overrides(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_clean = sub.add_parser("clean", help="build the modelling frame")
    p_clean.add_argument("--seasons", type=int, nargs="+")
    add_overrides(p_clean)
    p_clean.set_defaults(func=cmd_clean)

    p_fit = sub.add_parser("fit", help="fit the hierarchical degradation model")
    p_fit.add_argument("--seasons", type=int, nargs="+")
    add_overrides(p_fit)
    p_fit.set_defaults(func=cmd_fit)

    p_opt = sub.add_parser("optimize", help="Monte Carlo strategy search for one race")
    p_opt.add_argument("--race", type=_parse_race, required=True, metavar="SEASON:ROUND")
    p_opt.add_argument("--driver", type=str, default=None)
    p_opt.add_argument("--grid", type=int, default=None, help="override starting position")
    add_overrides(p_opt)
    p_opt.set_defaults(func=cmd_optimize)

    p_back = sub.add_parser("backtest", help="validate against a held-out season")
    p_back.add_argument("--seasons", type=int, nargs="+")
    p_back.add_argument("--limit", type=int, default=None, help="only the first N races")
    add_overrides(p_back)
    p_back.set_defaults(func=cmd_backtest)

    p_abl = sub.add_parser("ablate", help="run the realism ablation study")
    p_abl.add_argument("--race", type=_parse_race, nargs="+", metavar="SEASON:ROUND")
    p_abl.add_argument("--n-races", type=int, default=None)
    add_overrides(p_abl)
    p_abl.set_defaults(func=cmd_ablate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
