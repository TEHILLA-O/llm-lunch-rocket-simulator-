"""Shared path helpers. Directories are created on demand, not at import."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    path = Path(os.environ.get("LUNARMIND_DATA", ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = Path(os.environ.get("LUNARMIND_MODELS", ROOT / "models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(experiment_id: str | None = None) -> Path:
    path = models_dir() / "checkpoints"
    if experiment_id:
        path = path / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def trajectories_dir() -> Path:
    path = data_dir() / "trajectories"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensorboard_dir(experiment_id: str) -> Path:
    path = data_dir() / "tensorboard" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
