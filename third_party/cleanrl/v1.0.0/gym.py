from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as _gym
import numpy as np


def _seed(self: Any, seed: int | None = None):
    self.reset(seed=seed)
    return [seed] if seed is not None else []


if not hasattr(_gym.Env, "seed"):
    _gym.Env.seed = _seed  # type: ignore[attr-defined]
if not hasattr(_gym.Wrapper, "seed"):
    _gym.Wrapper.seed = _seed  # type: ignore[attr-defined]


def make(*args: Any, **kwargs: Any):
    return _gym.make(*args, **kwargs)


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).squeeze())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _coerce_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return {k: _coerce_scalar(v) for k, v in episode.items()}


def _convert_infos(
    infos: dict[str, Any], dones: np.ndarray, obs: np.ndarray
) -> list[dict[str, Any]]:
    num_envs = int(dones.shape[0])
    info_list: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    if not infos:
        return info_list

    final_obs = infos.get("final_observation")
    final_info = infos.get("final_info")

    if final_obs is not None:
        for i in range(num_envs):
            if i < len(final_obs) and final_obs[i] is not None:
                info_list[i]["terminal_observation"] = final_obs[i]

    if final_info is not None:
        for i in range(num_envs):
            if i < len(final_info) and final_info[i] is not None:
                fi = final_info[i]
                if "episode" in fi:
                    info_list[i]["episode"] = _coerce_episode(fi["episode"])
                for key, value in fi.items():
                    if key != "episode":
                        info_list[i][key] = value

    for key, value in infos.items():
        if key in ("final_observation", "final_info"):
            continue
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == num_envs:
            for i in range(num_envs):
                info_list[i][key] = value[i]

    return info_list


class SyncVectorEnv:
    def __init__(self, env_fns):
        self._env = _gym.vector.SyncVectorEnv(env_fns)
        self.single_action_space = self._env.single_action_space
        self.single_observation_space = self._env.single_observation_space
        self.num_envs = self._env.num_envs

    def reset(self):
        obs, _info = self._env.reset()
        return obs

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self._env.step(actions)
        dones = np.logical_or(terms, truncs)
        info_list = _convert_infos(infos, dones, obs)
        return obs, rewards, dones, info_list

    def close(self):
        return self._env.close()


spaces = _gym.spaces
wrappers = _gym.wrappers
vector = SimpleNamespace(SyncVectorEnv=SyncVectorEnv)
