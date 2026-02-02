from __future__ import annotations

from typing import Literal

import chz
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


if __name__ == "__main__":
    train(chz.entrypoint(SB3Config))
