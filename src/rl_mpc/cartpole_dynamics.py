from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # actually half the pole's length
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * math.pi / 360
    x_threshold: float = 2.4

    @property
    def total_mass(self) -> float:
        return self.masspole + self.masscart

    @property
    def polemass_length(self) -> float:
        return self.masspole * self.length


DEFAULT_PARAMS = CartPoleParams()


def is_done(state: np.ndarray, params: CartPoleParams = DEFAULT_PARAMS) -> bool:
    x, _, theta, _ = state
    return bool(
        x < -params.x_threshold
        or x > params.x_threshold
        or theta < -params.theta_threshold_radians
        or theta > params.theta_threshold_radians
    )


def step_cartpole(
    state: np.ndarray,
    action: int,
    params: CartPoleParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, bool]:
    """One-step transition using CartPole physics from Gymnasium.

    Returns: (next_state, done)
    """
    x, x_dot, theta, theta_dot = state
    force = params.force_mag if action == 1 else -params.force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + params.polemass_length * theta_dot**2 * sintheta) / params.total_mass
    thetaacc = (
        params.gravity * sintheta - costheta * temp
    ) / (params.length * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass))
    xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

    x = x + params.tau * x_dot
    x_dot = x_dot + params.tau * xacc
    theta = theta + params.tau * theta_dot
    theta_dot = theta_dot + params.tau * thetaacc

    next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
    done = is_done(next_state, params)
    return next_state, done
