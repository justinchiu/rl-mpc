from __future__ import annotations

from typing import Literal

import chz
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env


@chz.chz
class SB3Config:
    algo: Literal["dqn", "ppo"] = "dqn"
    env_id: str = "CartPole-v1"
    total_timesteps: int = 200_000
    seed: int = 0
    num_envs: int = 1
    device: str = "auto"
    log_dir: str | None = None
    save_path: str | None = None
    render_episodes: int = 0
    render_deterministic: bool = True
    render_seed: int | None = None
    render_delay_ms: int = 0


def train(cfg: SB3Config) -> None:
    env = make_vec_env(cfg.env_id, n_envs=cfg.num_envs, seed=cfg.seed)
    model_cls = DQN if cfg.algo == "dqn" else PPO
    model = model_cls(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
        tensorboard_log=cfg.log_dir,
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    if cfg.save_path:
        model.save(cfg.save_path)
    env.close()

    if cfg.render_episodes > 0:
        render_env = gym.make(cfg.env_id, render_mode="human")
        try:
            for ep in range(cfg.render_episodes):
                seed = cfg.render_seed if cfg.render_seed is not None else cfg.seed + ep
                obs, _ = render_env.reset(seed=seed)
                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=cfg.render_deterministic)
                    if isinstance(action, np.ndarray):
                        action = int(action.squeeze())
                    obs, _reward, done, truncated, _info = render_env.step(action)
                    render_env.render()
                    if cfg.render_delay_ms > 0:
                        time.sleep(cfg.render_delay_ms / 1000.0)
        finally:
            render_env.close()


if __name__ == "__main__":
    train(chz.entrypoint(SB3Config))
