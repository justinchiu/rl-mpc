from __future__ import annotations

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import pufferlib.emulation
try:
    import pufferlib.vector as pvector
except ImportError:  # pragma: no cover - fallback for older pufferlib
    import pufferlib.vectorization as pvector


def make_cartpole_env(render_mode: str | None = None) -> gym.Env:
    def creator() -> gym.Env:
        return gym.make("CartPole-v1", render_mode=render_mode)

    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=creator)


def make_cartpole_vec(
    num_envs: int,
    backend: str = "multiprocessing",
    render_mode: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> object:
    def creator() -> gym.Env:
        return pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: gym.make("CartPole-v1", render_mode=render_mode)
        )

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
