from __future__ import annotations

import chz
import numpy as np
import torch
from tqdm import trange

from rl_mpc.cartpole_dynamics import is_done
from rl_mpc.components.dynamics import DynamicsConfig, build_dynamics
from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.planner import PolicyGuidedConfig, PolicyGuidedPlanner
from rl_mpc.components.policy import PolicyConfig, build_policy
from rl_mpc.components.trainer import MBMPCTrainer, MBMPCTrainerConfig
from rl_mpc.components.value import ValueConfig, build_value
from rl_mpc.components.video import VideoConfig, VideoLogger
from rl_mpc.replay_buffer import ReplayBuffer
from rl_mpc.utils import get_device, set_seed


@chz.chz
class MBMPCConfig:
    total_steps: int = 100_000
    warmup_steps: int = 2_000
    buffer_size: int = 100_000
    gamma: float = 0.99
    seed: int = 0
    device: str = "auto"
    env: EnvConfig = EnvConfig()
    planner: PolicyGuidedConfig = PolicyGuidedConfig()
    trainer: MBMPCTrainerConfig = MBMPCTrainerConfig()
    policy: PolicyConfig = PolicyConfig()
    value: ValueConfig = ValueConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    video: VideoConfig = VideoConfig()


def train(cfg: MBMPCConfig) -> None:
    device = get_device(cfg.device)
    rng = set_seed(cfg.seed)
    torch_rng = torch.Generator(device=device).manual_seed(cfg.seed)

    envs = make_envs(cfg.env)
    obs, _ = envs.reset(seed=cfg.seed)
    obs_array = np.asarray(obs)
    if obs_array.ndim == 1:
        obs_array = obs_array[None, :]
    num_envs = obs_array.shape[0]
    obs_dim = int(np.prod(obs_array.shape[1:]))

    driver_env = getattr(envs, "driver_env", envs)
    action_space = getattr(driver_env, "single_action_space", getattr(driver_env, "action_space", None))
    if action_space is None or not hasattr(action_space, "n"):
        raise RuntimeError("Unable to determine discrete action space size")
    n_actions = int(action_space.n)

    policy = build_policy(obs_dim, n_actions, cfg.policy).to(device)
    value = build_value(obs_dim, cfg.value).to(device)
    dynamics = build_dynamics(obs_dim, n_actions, cfg.dynamics).to(device)

    planner = PolicyGuidedPlanner(
        cfg.planner,
        n_actions,
        device,
        cfg.gamma,
        is_done,
    )
    trainer = MBMPCTrainer(policy, value, dynamics, n_actions, cfg.trainer, cfg.gamma)
    video_logger = VideoLogger(cfg.env.env_id, cfg.video)
    video_torch_rng = torch.Generator(device=device)
    video_seed = cfg.video.seed if cfg.video.seed is not None else cfg.seed + 123
    video_torch_rng.manual_seed(video_seed)

    def act_fn(obs: np.ndarray) -> int:
        obs_array = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            return planner.act(
                obs_array,
                policy,
                dynamics,
                value,
                video_torch_rng,
            )

    buffer = ReplayBuffer(obs_dim, cfg.buffer_size, device)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lens = np.zeros(num_envs, dtype=np.int32)
    episode = 0
    global_step = 0

    for step in trange(cfg.total_steps, desc="MBMPC"):
        if step < cfg.warmup_steps:
            actions = rng.integers(0, n_actions, size=num_envs)
        else:
            actions = np.zeros(num_envs, dtype=np.int64)
            with torch.no_grad():
                for i in range(num_envs):
                    actions[i] = planner.act(
                        obs_array[i],
                        policy,
                        dynamics,
                        value,
                        torch_rng,
                    )

        next_obs, rewards, terminals, truncateds, _ = envs.step(actions)
        next_obs_array = np.asarray(next_obs)
        if next_obs_array.ndim == 1:
            next_obs_array = next_obs_array[None, :]
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

        for i in range(num_envs):
            buffer.add(
                obs_array[i],
                int(actions[i]),
                float(rewards_array[i]),
                next_obs_array[i],
                bool(dones[i]),
            )

        episode_returns += rewards_array
        episode_lens += 1

        for i in range(num_envs):
            if dones[i]:
                episode += 1
                print(
                    f"episode={episode} step={step} return={episode_returns[i]:.1f} "
                    f"len={episode_lens[i]}"
                )
                episode_returns[i] = 0.0
                episode_lens[i] = 0

        obs_array = next_obs_array
        global_step += num_envs
        video_logger.maybe_record(global_step, act_fn)

        if (
            step >= cfg.warmup_steps
            and len(buffer) >= cfg.trainer.batch_size
            and step % cfg.trainer.train_freq == 0
        ):
            for _ in range(cfg.trainer.updates_per_step):
                batch = buffer.sample(cfg.trainer.batch_size)
                trainer.update(batch)

    envs.close()
