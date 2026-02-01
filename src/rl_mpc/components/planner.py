from __future__ import annotations

from typing import Callable

import chz
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@chz.chz
class RandomShootingConfig:
    horizon: int = 15
    num_sequences: int = 1_000


@chz.chz
class PolicyGuidedConfig:
    horizon: int = 15
    num_sequences: int = 256


def _rollout_return(
    state: np.ndarray,
    actions: np.ndarray,
    gamma: float,
    step_fn: Callable[[np.ndarray, int], tuple[np.ndarray, bool]],
) -> float:
    total = 0.0
    discount = 1.0
    current = state
    for action in actions:
        current, done = step_fn(current, int(action))
        total += discount
        discount *= gamma
        if done:
            break
    return total


class RandomShootingPlanner:
    def __init__(
        self,
        cfg: RandomShootingConfig,
        step_fn: Callable[[np.ndarray, int], tuple[np.ndarray, bool]],
        num_actions: int,
        gamma: float,
    ) -> None:
        self.cfg = cfg
        self.step_fn = step_fn
        self.num_actions = num_actions
        self.gamma = gamma

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        best_return = -float("inf")
        best_action = 0
        for _ in range(self.cfg.num_sequences):
            actions = rng.integers(0, self.num_actions, size=self.cfg.horizon)
            score = _rollout_return(state, actions, self.gamma, self.step_fn)
            if score > best_return:
                best_return = score
                best_action = int(actions[0])
        return best_action


def _one_hot(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=num_actions).float()


def _rollout_score(
    start_state: torch.Tensor,
    policy: nn.Module,
    dynamics: nn.Module,
    value: nn.Module | None,
    horizon: int,
    num_actions: int,
    gamma: float,
    rng: torch.Generator,
    done_fn: Callable[[np.ndarray], bool],
) -> tuple[float, int]:
    state = start_state
    total = 0.0
    discount = 1.0
    first_action = 0

    for t in range(horizon):
        logits = policy(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(0)
        if t == 0:
            first_action = int(action.item())

        dyn_in = torch.cat([state, _one_hot(action, num_actions)], dim=-1)
        next_state = dynamics(dyn_in)
        done = done_fn(next_state.detach().cpu().numpy())
        total += discount
        discount *= gamma
        if done:
            return total, first_action
        state = next_state

    if value is not None:
        with torch.no_grad():
            total += discount * float(value(state).item())
    return total, first_action


class PolicyGuidedPlanner:
    def __init__(
        self,
        cfg: PolicyGuidedConfig,
        num_actions: int,
        device: torch.device,
        gamma: float,
        done_fn: Callable[[np.ndarray], bool],
    ) -> None:
        self.cfg = cfg
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.done_fn = done_fn

    def act(
        self,
        obs: np.ndarray,
        policy: nn.Module,
        dynamics: nn.Module,
        value: nn.Module | None,
        rng: torch.Generator,
    ) -> int:
        best_return = -float("inf")
        best_action = 0

        start_state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        for _ in range(self.cfg.num_sequences):
            score, action = _rollout_score(
                start_state,
                policy,
                dynamics,
                value,
                self.cfg.horizon,
                self.num_actions,
                self.gamma,
                rng,
                self.done_fn,
            )
            if score > best_return:
                best_return = score
                best_action = action
        return best_action
