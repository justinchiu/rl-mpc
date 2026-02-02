from __future__ import annotations

import numpy as np

from rl_mpc.components.types import RolloutBatch, RolloutChunk, StepBatch


class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.logp = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    def store(
        self,
        t: int,
        obs: np.ndarray,
        actions: np.ndarray,
        logp: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.obs[t] = obs
        self.actions[t] = actions
        self.logp[t] = logp
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values

    def store_step(self, t: int, step: StepBatch) -> None:
        self.store(
            t,
            step.obs,
            step.actions,
            step.logp,
            step.rewards,
            step.dones,
            step.values,
        )

    def compute_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        last_adv = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_values = last_values if t == self.rollout_steps - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + self.values
        return advantages, returns

    def flatten(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> RolloutBatch:
        return RolloutBatch(
            obs=self.obs.reshape(-1, self.obs_dim),
            actions=self.actions.reshape(-1),
            logp=self.logp.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=self.values.reshape(-1),
        )


def explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
    var_target = float(np.var(targets))
    if var_target <= 1e-8:
        return float("nan")
    return 1.0 - float(np.var(targets - predictions)) / var_target


def chunk_to_batch(
    chunk: RolloutChunk,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    values_override: np.ndarray | None = None,
) -> RolloutBatch:
    rewards = chunk.rewards
    dones = chunk.dones
    values = values_override if values_override is not None else chunk.values

    rollout_steps, num_envs = rewards.shape
    advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    last_adv = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(rollout_steps)):
        next_values = last_values if t == rollout_steps - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values

    obs_dim = int(np.prod(chunk.obs.shape[2:]))
    return RolloutBatch(
        obs=chunk.obs.reshape(-1, obs_dim),
        actions=chunk.actions.reshape(-1),
        logp=chunk.logp.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=values.reshape(-1),
    )
