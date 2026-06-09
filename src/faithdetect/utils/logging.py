"""Lightweight, reproducible result logging.

Every experiment writes a single JSON file capturing config, environment, per-seed metrics
and aggregated CIs. No result is ever hard-coded into a figure: figures are rendered strictly
from these files.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_serializable)
    return path


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        pass
    return info


class ResultLogger:
    """Accumulate a result record and write it atomically."""

    def __init__(self, name: str, config: dict, out_dir: str | Path = "results"):
        self.record: dict = {
            "name": name,
            "config": config,
            "env": env_info(),
            "per_seed": [],
            "aggregated": {},
            "extra": {},
        }
        self.out_dir = Path(out_dir)

    def add_seed(self, seed: int, metrics: dict) -> None:
        self.record["per_seed"].append({"seed": seed, **metrics})

    def set_aggregated(self, aggregated: dict) -> None:
        self.record["aggregated"] = aggregated

    def set_extra(self, key: str, value: Any) -> None:
        self.record["extra"][key] = value

    def write(self, filename: str | None = None) -> Path:
        fname = filename or f"{self.record['name']}.json"
        return save_json(self.out_dir / fname, self.record)
