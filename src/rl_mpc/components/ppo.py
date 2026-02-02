from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PPOTrainer:
    def __init__(
        self,
        policy: nn.Module,
        value: nn.Module,
        lr: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.value = value
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        params: Iterable[nn.Parameter] = list(policy.parameters()) + list(value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.device = device

    def update(
        self,
        batch: dict[str, np.ndarray],
        n_epochs: int,
        batch_size: int,
        normalize_advantage: bool,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_logp = batch["logp"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        if normalize_advantage and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        batch_size_total = obs.shape[0]
        for _epoch in range(n_epochs):
            indices = rng.permutation(batch_size_total)
            for start in range(0, batch_size_total, batch_size):
                mb_idx = indices[start:start + batch_size]
                if mb_idx.size == 0:
                    continue

                obs_tensor = torch.as_tensor(obs[mb_idx], dtype=torch.float32, device=self.device)
                actions_tensor = torch.as_tensor(actions[mb_idx], dtype=torch.int64, device=self.device)
                old_logp_tensor = torch.as_tensor(old_logp[mb_idx], dtype=torch.float32, device=self.device)
                adv_tensor = torch.as_tensor(advantages[mb_idx], dtype=torch.float32, device=self.device)
                returns_tensor = torch.as_tensor(returns[mb_idx], dtype=torch.float32, device=self.device)

                logits = self.policy(obs_tensor)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_tensor)
                pg_loss_1 = ratio * adv_tensor
                pg_loss_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_tensor
                policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                values_pred = self.value(obs_tensor).squeeze(-1)
                value_loss = 0.5 * (returns_tensor - values_pred).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kls.append((old_logp_tensor - logp).mean().item())
                    clip_fracs.append((torch.abs(ratio - 1.0) > self.clip_range).float().mean().item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_values.append(entropy.item())
                total_losses.append(loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }
