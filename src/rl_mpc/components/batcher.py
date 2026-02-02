from __future__ import annotations

import numpy as np

from rl_mpc.components.types import RolloutBatch


class RolloutBatcher:
    def __init__(self, batch_size: int, rng: np.random.Generator) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = batch_size
        self.rng = rng

    def iter(self, batch: RolloutBatch, n_epochs: int):
        total = batch.obs.shape[0]
        for _ in range(n_epochs):
            indices = self.rng.permutation(total)
            for start in range(0, total, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]
                if mb_idx.size == 0:
                    continue
                yield RolloutBatch(
                    obs=batch.obs[mb_idx],
                    actions=batch.actions[mb_idx],
                    logp=batch.logp[mb_idx],
                    advantages=batch.advantages[mb_idx],
                    returns=batch.returns[mb_idx],
                    values=batch.values[mb_idx],
                )
