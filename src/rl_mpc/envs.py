from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import pufferlib.emulation
try:
    import pufferlib.vector as pvector
except ImportError:  # pragma: no cover - fallback for older pufferlib
    import pufferlib.vectorization as pvector


def _make_puffer_env(
    env_id: str,
    render_mode: str | None,
    buf: object | None = None,
    seed: int = 0,
) -> gym.Env:
    def creator() -> gym.Env:
        return gym.make(env_id, render_mode=render_mode)

    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=creator, buf=buf, seed=seed)


def make_cartpole_env(render_mode: str | None = None, env_id: str = "CartPole-v1") -> gym.Env:
    return _make_puffer_env(env_id=env_id, render_mode=render_mode)


def make_cartpole_vec(
    num_envs: int,
    backend: str = "multiprocessing",
    render_mode: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
    env_id: str = "CartPole-v1",
) -> object:
    def creator(*_args: object, buf: object | None = None, seed: int = 0, **_kwargs: object) -> gym.Env:
        return _make_puffer_env(env_id=env_id, render_mode=render_mode, buf=buf, seed=seed)

    backend_map: dict[str, Callable[..., object]] = {
        "serial": pvector.Serial,
        "multiprocessing": pvector.Multiprocessing,
    }
    if hasattr(pvector, "Ray"):
        backend_map["ray"] = pvector.Ray
    backend_cls = backend_map.get(backend.lower())
    if backend_cls is None:
        raise ValueError(f"Unknown backend: {backend}")

    kwargs: dict[str, object] = {}
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    return pvector.make(creator, backend=backend_cls, num_envs=num_envs, **kwargs)
