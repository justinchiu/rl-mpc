from __future__ import annotations

import numpy as np
import torch

from rl_mpc.components.types import TransitionBatch


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, device: torch.device) -> None:
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int64)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)
        self.size = size
        self.device = device
        self.ptr = 0
        self.count = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.count < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        idx = np.random.randint(0, self.count, size=batch_size)
        return TransitionBatch(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            actions=torch.as_tensor(self.actions[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self.count
