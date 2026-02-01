from __future__ import annotations

import chz

from rl_mpc.envs import make_cartpole_vec


@chz.chz
class EnvConfig:
    num_envs: int = 1
    vec_backend: str = "multiprocessing"
    num_workers: int | None = None
    vec_batch_size: int | None = None
    render: bool = False


def make_envs(cfg: EnvConfig, render_mode: str | None = None) -> object:
    return make_cartpole_vec(
        num_envs=cfg.num_envs,
        backend=cfg.vec_backend,
        render_mode=render_mode,
        num_workers=cfg.num_workers,
        batch_size=cfg.vec_batch_size,
    )
