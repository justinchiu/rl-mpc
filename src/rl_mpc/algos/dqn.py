from __future__ import annotations

import chz
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.policy import PolicyConfig, build_policy
from rl_mpc.components.video import VideoConfig, VideoLogger
from rl_mpc.replay_buffer import ReplayBuffer
from rl_mpc.utils import get_device, linear_schedule, set_seed


@chz.chz
class DQNConfig:
    total_steps: int = 200_000
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    target_update: int = 1_000
    learning_starts: int = 1_000
    train_freq: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    max_grad_norm: float = 10.0
    seed: int = 0
    device: str = "auto"
    env: EnvConfig = EnvConfig()
    policy: PolicyConfig = PolicyConfig()
    video: VideoConfig = VideoConfig()


def train(cfg: DQNConfig) -> None:
    device = get_device(cfg.device)
    rng = set_seed(cfg.seed)

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

    q_net = build_policy(obs_dim, n_actions, cfg.policy).to(device)
    target_net = build_policy(obs_dim, n_actions, cfg.policy).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(obs_dim, cfg.buffer_size, device)

    def act_fn(obs: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    video_logger = VideoLogger(cfg.env.env_id, cfg.video)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lens = np.zeros(num_envs, dtype=np.int32)
    episode = 0
    global_step = 0

    for step in trange(cfg.total_steps, desc="DQN"):
        eps = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
        obs_batch = obs_array if obs_array.ndim == 2 else obs_array[None, :]

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            q_values = q_net(obs_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        random_mask = rng.random(num_envs) < eps
        actions = greedy_actions.astype(np.int64)
        if random_mask.any():
            actions[random_mask] = rng.integers(0, n_actions, size=int(random_mask.sum()))

        next_obs, rewards, terminals, truncateds, _ = envs.step(actions)
        next_obs_array = np.asarray(next_obs)
        if next_obs_array.ndim == 1:
            next_obs_array = next_obs_array[None, :]
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

        for i in range(num_envs):
            buffer.add(
                obs_batch[i],
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
                    f"len={episode_lens[i]} eps={eps:.3f}"
                )
                episode_returns[i] = 0.0
                episode_lens[i] = 0

        obs_array = next_obs_array
        global_step += num_envs
        video_logger.maybe_record(global_step, act_fn)

        if step >= cfg.learning_starts and len(buffer) >= cfg.batch_size and step % cfg.train_freq == 0:
            batch = buffer.sample(cfg.batch_size)
            q_values = q_net(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(batch["next_obs"]).max(dim=1).values
                target = batch["rewards"] + cfg.gamma * (1.0 - batch["dones"]) * next_q
            loss = F.smooth_l1_loss(q_values, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if step % cfg.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    envs.close()
