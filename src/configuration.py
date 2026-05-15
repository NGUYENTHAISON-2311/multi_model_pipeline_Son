"""Configuration helpers for the reproduced prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default_config.json"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else default_config_path()
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    pipeline_root = path.parents[1]
    workspace_root = pipeline_root.parent
    config["_config_path"] = str(path)
    config["_pipeline_root"] = str(pipeline_root)
    config["_workspace_root"] = str(workspace_root)
    return config


def pipeline_root(config: dict[str, Any]) -> Path:
    return Path(config["_pipeline_root"])


def workspace_root(config: dict[str, Any]) -> Path:
    return Path(config["_workspace_root"])


def resolve_workspace_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace_root(config) / path


def resolve_pipeline_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return pipeline_root(config) / path


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)