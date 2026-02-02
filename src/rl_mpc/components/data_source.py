from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from rl_mpc.components.collector import OnPolicyCollector
from rl_mpc.components.packer import RolloutPacker
from rl_mpc.components.rollout import chunk_to_batch
from rl_mpc.components.types import RolloutBatch, RolloutChunk, RolloutSample


class RolloutReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._chunks: list[RolloutChunk] = []

    def add(self, chunk: RolloutChunk) -> None:
        if len(self._chunks) >= self.capacity:
            self._chunks.pop(0)
        self._chunks.append(chunk)

    def sample(self, num_rollouts: int, rng: np.random.Generator) -> list[RolloutChunk]:
        if num_rollouts <= 0:
            return []
        if not self._chunks:
            return []
        idx = rng.integers(0, len(self._chunks), size=num_rollouts)
        return [self._chunks[i] for i in idx]

    def __len__(self) -> int:
        return len(self._chunks)


class StreamRolloutSource:
    def __init__(
        self,
        collector: OnPolicyCollector,
        packer: RolloutPacker,
        value: torch.nn.Module,
        gamma: float,
        gae_lambda: float,
        obs_array: np.ndarray,
        device: torch.device,
    ) -> None:
        self.collector = collector
        self.packer = packer
        self.value = value
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_array = obs_array
        self.last_sample: RolloutSample | None = None

    def next(self) -> RolloutSample:
        stream = self.collector.stream(self.obs_array, self.packer.rollout_steps)
        chunk, last_obs = self.packer.pack(stream)
        self.obs_array = stream.last_obs if stream.last_obs is not None else last_obs
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.obs_array, dtype=torch.float32, device=self.device)
            last_values = self.value(obs_tensor).squeeze(-1).cpu().numpy()
        batch = chunk_to_batch(chunk, last_values, self.gamma, self.gae_lambda)
        sample = RolloutSample(batch=batch, chunk=chunk)
        self.last_sample = sample
        return sample


class ReplayRolloutSource:
    def __init__(
        self,
        buffer: RolloutReplayBuffer,
        value: torch.nn.Module,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        self.buffer = buffer
        self.value = value
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def _compute_values(self, chunk: RolloutChunk) -> tuple[np.ndarray, np.ndarray]:
        obs = chunk.obs
        rollout_steps, num_envs, obs_dim = obs.shape
        obs_flat = obs.reshape(-1, obs_dim)
        with torch.no_grad():
            values_flat = self.value(
                torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)
            ).squeeze(-1).cpu().numpy()
            last_obs = chunk.next_obs[-1].reshape(num_envs, obs_dim)
            last_values = self.value(
                torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
            ).squeeze(-1).cpu().numpy()
        return values_flat.reshape(rollout_steps, num_envs), last_values

    def sample(self, num_rollouts: int, rng: np.random.Generator) -> list[RolloutBatch]:
        chunks = self.buffer.sample(num_rollouts, rng)
        batches: list[RolloutBatch] = []
        for chunk in chunks:
            values, last_values = self._compute_values(chunk)
            batches.append(
                chunk_to_batch(
                    chunk,
                    last_values,
                    self.gamma,
                    self.gae_lambda,
                    values_override=values,
                )
            )
        return batches


def concat_rollout_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    if not batches:
        raise ValueError("No batches to concatenate")
    obs = np.concatenate([b.obs for b in batches], axis=0)
    actions = np.concatenate([b.actions for b in batches], axis=0)
    logp = np.concatenate([b.logp for b in batches], axis=0)
    advantages = np.concatenate([b.advantages for b in batches], axis=0)
    returns = np.concatenate([b.returns for b in batches], axis=0)
    values = np.concatenate([b.values for b in batches], axis=0)
    return RolloutBatch(
        obs=obs,
        actions=actions,
        logp=logp,
        advantages=advantages,
        returns=returns,
        values=values,
    )


class MixedRolloutSource:
    def __init__(
        self,
        stream_source: StreamRolloutSource,
        replay_source: ReplayRolloutSource,
        replay_buffer: RolloutReplayBuffer,
        off_policy_fraction: float,
        min_replay: int,
        rng: np.random.Generator,
    ) -> None:
        if off_policy_fraction < 0.0 or off_policy_fraction >= 1.0:
            raise ValueError("off_policy_fraction must be in [0.0, 1.0)")
        self.stream_source = stream_source
        self.replay_source = replay_source
        self.replay_buffer = replay_buffer
        self.off_policy_fraction = off_policy_fraction
        self.min_replay = min_replay
        self.rng = rng
        self.last_on_policy: RolloutSample | None = None

    def next(self) -> RolloutBatch:
        on_policy = self.stream_source.next()
        self.last_on_policy = on_policy
        self.replay_buffer.add(on_policy.chunk)

        batches = [on_policy.batch]
        if self.off_policy_fraction > 0.0 and len(self.replay_buffer) >= self.min_replay:
            ratio = self.off_policy_fraction / max(1e-8, 1.0 - self.off_policy_fraction)
            num_extra = max(0, int(math.ceil(ratio)))
            replay_batches = self.replay_source.sample(num_extra, self.rng)
            batches.extend(replay_batches)
        return concat_rollout_batches(batches)
