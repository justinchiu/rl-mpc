from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from rl_mpc.components.rollout import RolloutBuffer
from rl_mpc.components.types import StepBatch


class RolloutPacker:
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim

    def pack(self, stream: Iterable[StepBatch]) -> tuple[RolloutBuffer, np.ndarray]:
        buffer = RolloutBuffer(self.rollout_steps, self.num_envs, self.obs_dim)
        last_obs: np.ndarray | None = None
        step_count = 0
        for t, step in enumerate(stream):
            if t >= self.rollout_steps:
                raise ValueError("Stream produced more steps than rollout_steps")
            buffer.store(
                t,
                step.obs,
                step.actions,
                step.logp,
                step.rewards,
                step.dones,
                step.values,
            )
            last_obs = step.next_obs
            step_count += 1
        if step_count != self.rollout_steps:
            raise ValueError(
                f"Stream ended early (got {step_count} steps, expected {self.rollout_steps})"
            )
        if last_obs is None:
            raise ValueError("No steps collected; cannot determine last observation")
        return buffer, last_obs
