from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy, and torch. Returns a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def get_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Iterable[int] = (128, 128),
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end
    if t >= duration:
        return end
    return start + (end - start) * (t / duration)
