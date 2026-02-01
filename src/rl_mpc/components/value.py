from __future__ import annotations

import chz
from torch import nn

from rl_mpc.utils import build_mlp


@chz.chz
class ValueConfig:
    hidden_sizes: tuple[int, int] = (128, 128)


def build_value(obs_dim: int, cfg: ValueConfig) -> nn.Module:
    return build_mlp(obs_dim, 1, cfg.hidden_sizes)
