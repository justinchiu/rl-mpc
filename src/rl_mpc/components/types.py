from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class RolloutBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray


class StepBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_obs: np.ndarray


class RolloutChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_obs: np.ndarray


class TransitionBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
