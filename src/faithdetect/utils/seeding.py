"""Deterministic seeding and device selection.

Addresses flaw F9 (no multiple seeds / no reproducibility) by giving every experiment a
single entry point for full reproducibility across python / numpy / torch, including the
DataLoader worker streams.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed:
        The integer seed.
    deterministic:
        If True, request deterministic cuDNN/algorithms where supported. On MPS some ops
        are non-deterministic regardless; we still fix the seeds so runs are as repeatable
        as the backend allows.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN determinism (no-op on MPS/CPU but harmless).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order of preference: explicit `prefer` -> CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn` for reproducible multi-worker shuffling."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
