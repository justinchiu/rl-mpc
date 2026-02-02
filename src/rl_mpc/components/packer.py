from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from rl_mpc.components.types import RolloutChunk, StepBatch


class RolloutPacker:
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim

    def pack(self, stream: Iterable[StepBatch]) -> tuple[RolloutChunk, np.ndarray]:
        obs = np.zeros((self.rollout_steps, self.num_envs, self.obs_dim), dtype=np.float32)
        actions = np.zeros((self.rollout_steps, self.num_envs), dtype=np.int64)
        logp = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        rewards = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        dones = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        next_obs = np.zeros((self.rollout_steps, self.num_envs, self.obs_dim), dtype=np.float32)
        last_obs: np.ndarray | None = None
        step_count = 0
        for t, step in enumerate(stream):
            if t >= self.rollout_steps:
                raise ValueError("Stream produced more steps than rollout_steps")
            obs[t] = step.obs
            actions[t] = step.actions
            logp[t] = step.logp
            rewards[t] = step.rewards
            dones[t] = step.dones
            values[t] = step.values
            next_obs[t] = step.next_obs
            last_obs = step.next_obs
            step_count += 1
        if step_count != self.rollout_steps:
            raise ValueError(
                f"Stream ended early (got {step_count} steps, expected {self.rollout_steps})"
            )
        if last_obs is None:
            raise ValueError("No steps collected; cannot determine last observation")
        return RolloutChunk(
            obs=obs,
            actions=actions,
            logp=logp,
            rewards=rewards,
            dones=dones,
            values=values,
            next_obs=next_obs,
        ), last_obs
