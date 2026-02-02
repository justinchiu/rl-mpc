from __future__ import annotations

from typing import Literal

import chz
import gymnasium as gym
import numpy as np
import sys
import time
sys.modules.setdefault("gym", gym)

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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
    wandb: bool = False
    wandb_project: str = "rl-mpc"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_sync_tensorboard: bool = True
    wandb_save_code: bool = True
    render_episodes: int = 0
    render_deterministic: bool = True
    render_seed: int | None = None
    render_delay_ms: int = 0
    video_dir: str | None = None
    video_every_steps: int = 0
    video_episodes: int = 1


class VideoEvalCallback(BaseCallback):
    def __init__(
        self,
        env_id: str,
        video_dir: str,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        deterministic: bool,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.video_dir = video_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq == 0:
            self._record(self.num_timesteps)
        return True

    def _record(self, step: int) -> None:
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=self.video_dir,
            episode_trigger=lambda _episode: True,
            name_prefix=f"{self.env_id}_step{step}",
        )
        try:
            for ep in range(self.n_eval_episodes):
                obs, _ = env.reset(seed=self.seed + ep)
                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    if isinstance(action, np.ndarray):
                        action = int(action.squeeze())
                    obs, _reward, done, truncated, _info = env.step(action)
        finally:
            env.close()


def train(cfg: SB3Config) -> None:
    log_dir = cfg.log_dir
    if cfg.wandb and log_dir is None:
        log_dir = "runs/sb3"
    env = make_vec_env(cfg.env_id, n_envs=cfg.num_envs, seed=cfg.seed)
    model_cls = DQN if cfg.algo == "dqn" else PPO
    model = model_cls(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
        tensorboard_log=log_dir,
    )
    callbacks: list[BaseCallback] = []
    if cfg.video_dir and cfg.video_every_steps > 0:
        callbacks.append(
            VideoEvalCallback(
                env_id=cfg.env_id,
                video_dir=cfg.video_dir,
                eval_freq=cfg.video_every_steps,
                n_eval_episodes=cfg.video_episodes,
                seed=cfg.seed,
                deterministic=cfg.render_deterministic,
            )
        )

    if cfg.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
        except ImportError as exc:
            raise RuntimeError(
                "wandb is enabled but not installed. Add it to dependencies or run "
                "`uv add wandb`."
            ) from exc
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            sync_tensorboard=cfg.wandb_sync_tensorboard,
            save_code=cfg.wandb_save_code,
            config=chz.asdict(cfg),
            mode=cfg.wandb_mode,
        )
        callbacks.append(WandbCallback())

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    if cfg.save_path:
        model.save(cfg.save_path)
    env.close()

    if cfg.wandb:
        wandb.finish()

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
