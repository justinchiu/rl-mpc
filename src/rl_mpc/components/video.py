from __future__ import annotations

from collections.abc import Callable

import chz
import gymnasium as gym
import numpy as np


@chz.chz
class VideoConfig:
    dir: str | None = None
    every_steps: int = 0
    episodes: int = 1
    deterministic: bool = True
    seed: int | None = None
    name_prefix: str | None = None


class VideoLogger:
    def __init__(self, env_id: str, cfg: VideoConfig) -> None:
        self.env_id = env_id
        self.cfg = cfg
        self._next_step = cfg.every_steps if self.enabled else 0

    @property
    def enabled(self) -> bool:
        return (
            self.cfg.dir is not None
            and self.cfg.every_steps > 0
            and self.cfg.episodes > 0
        )

    def maybe_record(self, step: int, act_fn: Callable[[np.ndarray], int]) -> None:
        if not self.enabled:
            return
        while step >= self._next_step and self._next_step > 0:
            self._record(self._next_step, act_fn)
            self._next_step += self.cfg.every_steps

    def _record(self, step: int, act_fn: Callable[[np.ndarray], int]) -> None:
        env = gym.make(self.env_id, render_mode="rgb_array")
        prefix = self.cfg.name_prefix or f"{self.env_id}_step{step}"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=self.cfg.dir,
            episode_trigger=lambda _episode: True,
            name_prefix=prefix,
        )
        try:
            for ep in range(self.cfg.episodes):
                seed = None if self.cfg.seed is None else self.cfg.seed + ep
                obs, _ = env.reset(seed=seed)
                done = False
                truncated = False
                while not (done or truncated):
                    action = act_fn(obs)
                    obs, _reward, done, truncated, _info = env.step(action)
        finally:
            env.close()
