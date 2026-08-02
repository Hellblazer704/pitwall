"""Filesystem layout.

Resolved once, from the Hydra config when running under Hydra and from the
repo root otherwise, so notebooks and tests see the same directories as the
CLI does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Paths", "default_paths", "repo_root"]


def repo_root() -> Path:
    """Repo root, found by walking up for the pyproject."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Paths:
    root: Path
    fastf1_cache: Path
    raw: Path
    processed: Path
    artifacts: Path

    def ensure(self) -> Paths:
        for directory in (self.fastf1_cache, self.raw, self.processed, self.artifacts):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def race_raw(self, season: int, round_no: int) -> Path:
        return self.raw / f"{season}" / f"round_{round_no:02d}"

    @classmethod
    def from_config(cls, cfg: Any) -> Paths:
        """Build from the ``paths`` node of a Hydra config.

        Takes ``Any`` rather than ``DictConfig`` so tests can pass a plain
        object with the same attributes.
        """
        node = getattr(cfg, "paths", cfg)
        return cls(
            root=Path(str(node.root)),
            fastf1_cache=Path(str(node.fastf1_cache)),
            raw=Path(str(node.raw)),
            processed=Path(str(node.processed)),
            artifacts=Path(str(node.artifacts)),
        )


def default_paths() -> Paths:
    root = repo_root()
    return Paths(
        root=root,
        fastf1_cache=root / "data" / "fastf1_cache",
        raw=root / "data" / "raw",
        processed=root / "data" / "processed",
        artifacts=root / "artifacts",
    )
