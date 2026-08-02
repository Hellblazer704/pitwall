"""Hydra config loading.

pitwall has several entry points that all want the same config tree, so rather
than decorating each one with ``@hydra.main`` we compose the config explicitly.
That also means tests and notebooks can get a real config object without
Hydra taking over argv or the working directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from pitwall.paths import repo_root

__all__ = ["conf_dir", "load_config"]


def conf_dir() -> Path:
    return repo_root() / "conf"


def load_config(overrides: Sequence[str] | None = None, config_name: str = "config") -> DictConfig:
    """Compose the config tree, applying dotlist ``overrides``.

    Overrides use ordinary Hydra syntax, e.g.
    ``["simulator.safety_car.enabled=false", "seed=7"]``. This is how the
    ablation study switches realism features off: the code path is identical,
    only the config differs.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(conf_dir()), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    # ``paths.root`` interpolates hydra.runtime.cwd, which is not resolvable
    # outside a Hydra job, so it is pinned to the repo root here instead.
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root = str(repo_root())
    OmegaConf.set_struct(cfg, True)
    return cfg
