from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from rl_mpc.components.types import StepBatch


class StepStream:
    def __init__(
        self,
        envs: object,
        policy: torch.nn.Module,
        value: torch.nn.Module,
        device: torch.device,
        obs_array: np.ndarray,
        n_steps: int,
    ) -> None:
        self._envs = envs
        self._policy = policy
        self._value = value
        self._device = device
        self._obs_array = obs_array
        self._n_steps = n_steps
        self.last_obs: np.ndarray | None = None

    def __iter__(self):
        obs_array = np.asarray(self._obs_array)
        if obs_array.ndim == 1:
            obs_array = obs_array[None, :]
        num_envs = obs_array.shape[0]
        obs_dim = int(np.prod(obs_array.shape[1:]))

        for _ in range(self._n_steps):
            obs_batch = obs_array.reshape(num_envs, obs_dim)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self._device)
                logits = self._policy(obs_tensor)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self._value(obs_tensor).squeeze(-1)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminals, truncateds, _ = self._envs.step(actions_np)
            next_obs_array = np.asarray(next_obs)
            if next_obs_array.ndim == 1:
                next_obs_array = next_obs_array[None, :]
            rewards_array = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

            yield StepBatch(
                obs=obs_batch,
                actions=actions_np,
                logp=log_probs.cpu().numpy(),
                rewards=rewards_array,
                dones=dones.astype(np.float32),
                values=values.cpu().numpy(),
                next_obs=next_obs_array.reshape(num_envs, obs_dim),
            )

            obs_array = next_obs_array

        self.last_obs = obs_array.reshape(num_envs, obs_dim)


class OnPolicyCollector:
    def __init__(
        self,
        envs: object,
        policy: torch.nn.Module,
        value: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.envs = envs
        self.policy = policy
        self.value = value
        self.device = device

    def stream(self, obs_array: np.ndarray, n_steps: int) -> StepStream:
        return StepStream(
            envs=self.envs,
            policy=self.policy,
            value=self.value,
            device=self.device,
            obs_array=obs_array,
            n_steps=n_steps,
        )
