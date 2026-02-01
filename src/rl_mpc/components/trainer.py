from __future__ import annotations

from typing import Literal

import chz
import torch
from torch import nn
from torch.nn import functional as F


@chz.chz
class MBMPCTrainerConfig:
    batch_size: int = 256
    train_freq: int = 1
    updates_per_step: int = 1
    lr_policy: float = 1e-3
    lr_value: float = 1e-3
    lr_dynamics: float = 1e-3
    max_grad_norm: float = 10.0
    policy_loss: Literal["bc", "ac"] = "bc"


def _one_hot(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=num_actions).float()


class MBMPCTrainer:
    def __init__(
        self,
        policy: nn.Module,
        value: nn.Module,
        dynamics: nn.Module,
        num_actions: int,
        cfg: MBMPCTrainerConfig,
        gamma: float,
    ) -> None:
        self.policy = policy
        self.value = value
        self.dynamics = dynamics
        self.num_actions = num_actions
        self.cfg = cfg
        self.gamma = gamma

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.lr_value)
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=cfg.lr_dynamics)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs_batch = batch["obs"]
        actions_batch = batch["actions"]
        next_obs_batch = batch["next_obs"]
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]

        dyn_in = torch.cat([obs_batch, _one_hot(actions_batch, self.num_actions)], dim=-1)
        pred_next = self.dynamics(dyn_in)
        dyn_loss = F.mse_loss(pred_next, next_obs_batch)

        value_pred = self.value(obs_batch).squeeze(-1)
        with torch.no_grad():
            target = rewards_batch + self.gamma * (1.0 - dones_batch) * self.value(
                next_obs_batch
            ).squeeze(-1)
        value_loss = F.mse_loss(value_pred, target)

        logits = self.policy(obs_batch)
        if self.cfg.policy_loss == "ac":
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_logp = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
            advantage = (target - value_pred).detach()
            policy_loss = -(chosen_logp * advantage).mean()
        else:
            policy_loss = F.cross_entropy(logits, actions_batch)

        self.dynamics_opt.zero_grad(set_to_none=True)
        dyn_loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.cfg.max_grad_norm)
        self.dynamics_opt.step()

        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
        self.value_opt.step()

        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.policy_opt.step()

        return {
            "dynamics_loss": float(dyn_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }
