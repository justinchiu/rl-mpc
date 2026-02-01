from __future__ import annotations

import chz
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import trange

from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.policy import PolicyConfig, build_policy
from rl_mpc.utils import get_device, set_seed


@chz.chz
class PGConfig:
    total_episodes: int = 1_000
    gamma: float = 0.99
    lr: float = 1e-3
    normalize_returns: bool = True
    max_grad_norm: float = 10.0
    seed: int = 0
    device: str = "auto"
    env: EnvConfig = EnvConfig()
    policy: PolicyConfig = PolicyConfig()


def train(cfg: PGConfig) -> None:
    device = get_device(cfg.device)
    set_seed(cfg.seed)

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
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    log_probs_per_env: list[list[torch.Tensor]] = [[] for _ in range(num_envs)]
    rewards_per_env: list[list[float]] = [[] for _ in range(num_envs)]

    episode = 0
    pbar = trange(cfg.total_episodes, desc="PolicyGrad")
    while episode < cfg.total_episodes:
        obs_batch = obs_array if obs_array.ndim == 2 else obs_array[None, :]
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        logits = policy(obs_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        next_obs, rewards, terminals, truncateds, _ = envs.step(actions.cpu().numpy())
        next_obs_array = np.asarray(next_obs)
        if next_obs_array.ndim == 1:
            next_obs_array = next_obs_array[None, :]
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

        for i in range(num_envs):
            log_probs_per_env[i].append(log_probs[i])
            rewards_per_env[i].append(float(rewards_array[i]))

        losses: list[torch.Tensor] = []
        for i in range(num_envs):
            if dones[i]:
                episode += 1
                pbar.update(1)
                rewards_i = rewards_per_env[i]
                returns: list[float] = []
                G = 0.0
                for r in reversed(rewards_i):
                    G = r + cfg.gamma * G
                    returns.append(G)
                returns.reverse()
                returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
                if cfg.normalize_returns and returns_tensor.numel() > 1:
                    returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                        returns_tensor.std() + 1e-8
                    )
                loss = -(torch.stack(log_probs_per_env[i]) * returns_tensor).sum()
                losses.append(loss)

                ep_return = float(sum(rewards_i))
                ep_len = len(rewards_i)
                print(f"episode={episode} return={ep_return:.1f} len={ep_len}")

                log_probs_per_env[i].clear()
                rewards_per_env[i].clear()

        if losses:
            total_loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

        obs_array = next_obs_array

    pbar.close()
    envs.close()
