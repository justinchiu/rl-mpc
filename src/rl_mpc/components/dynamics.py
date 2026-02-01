from __future__ import annotations

import chz
from torch import nn

from rl_mpc.utils import build_mlp


@chz.chz
class DynamicsConfig:
    hidden_sizes: tuple[int, int] = (128, 128)


def build_dynamics(obs_dim: int, num_actions: int, cfg: DynamicsConfig) -> nn.Module:
    return build_mlp(obs_dim + num_actions, obs_dim, cfg.hidden_sizes)
