from __future__ import annotations

import chz
import numpy as np
from tqdm import trange

from rl_mpc.cartpole_dynamics import step_cartpole
from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.planner import RandomShootingConfig, RandomShootingPlanner
from rl_mpc.components.video import VideoConfig, VideoLogger
from rl_mpc.utils import set_seed


@chz.chz
class MPCConfig:
    episodes: int = 10
    gamma: float = 0.99
    seed: int = 0
    env: EnvConfig = EnvConfig()
    planner: RandomShootingConfig = RandomShootingConfig()
    video: VideoConfig = VideoConfig()


def run(cfg: MPCConfig) -> None:
    rng = set_seed(cfg.seed)
    if cfg.env.render and cfg.env.num_envs != 1:
        raise ValueError("Render mode only supports num_envs=1")
    render_mode = "human" if cfg.env.render else None
    envs = make_envs(cfg.env, render_mode=render_mode)

    obs, _ = envs.reset(seed=cfg.seed)
    obs_array = np.asarray(obs)
    if obs_array.ndim == 1:
        obs_array = obs_array[None, :]
    num_envs = obs_array.shape[0]

    driver_env = getattr(envs, "driver_env", envs)
    action_space = getattr(driver_env, "single_action_space", getattr(driver_env, "action_space", None))
    if action_space is None or not hasattr(action_space, "n"):
        raise RuntimeError("Unable to determine discrete action space size")
    n_actions = int(action_space.n)

    planner = RandomShootingPlanner(cfg.planner, step_cartpole, n_actions, cfg.gamma)
    video_rng = np.random.default_rng(cfg.video.seed) if cfg.video.deterministic else rng
    video_logger = VideoLogger(cfg.env.env_id, cfg.video)

    def act_fn(obs: np.ndarray) -> int:
        return planner.act(obs, video_rng)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lens = np.zeros(num_envs, dtype=np.int32)
    episode = 0
    global_step = 0
    pbar = trange(cfg.episodes, desc="MPC")
    while episode < cfg.episodes:
        actions = np.zeros(num_envs, dtype=np.int64)
        for i in range(num_envs):
            actions[i] = planner.act(obs_array[i], rng)

        next_obs, rewards, terminals, truncateds, _ = envs.step(actions)
        next_obs_array = np.asarray(next_obs)
        if next_obs_array.ndim == 1:
            next_obs_array = next_obs_array[None, :]
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

        episode_returns += rewards_array
        episode_lens += 1

        for i in range(num_envs):
            if dones[i]:
                episode += 1
                pbar.update(1)
                print(f"episode={episode} return={episode_returns[i]:.1f} len={episode_lens[i]}")
                episode_returns[i] = 0.0
                episode_lens[i] = 0

        obs_array = next_obs_array
        global_step += num_envs
        video_logger.maybe_record(global_step, act_fn)
    pbar.close()
    envs.close()
