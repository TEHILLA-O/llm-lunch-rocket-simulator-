"""Observation vector for the flight controller.

Eight values, in order:

0. Horizontal position of the centre of mass relative to the pad, metres / 20.
1. Altitude of the centre of mass, metres / 20.
2. Horizontal velocity, (m/s) / 10.
3. Vertical velocity, (m/s) / 10. Positive is upward.
4. Wrapped tilt angle / pi. 0 is upright, +1 is nose-left (counter-clockwise).
5. Angular velocity, (rad/s) / 2. Positive is counter-clockwise.
6. Left landing-foot contact, 0 or 1.
7. Right landing-foot contact, 0 or 1.

The scales are ``PhysicsConfig.norm_pos``, ``norm_vel`` and ``norm_omega``.
Leg contacts are the only binary sensors. The vector does not include the
reward, the time limit, the terrain map, the wind (there is none) or the
controller identity.

Raw SI values are returned alongside the vector in ``info["telemetry"]`` and
in recorded trajectories. The normalised vector is what the policy sees.
"""

from __future__ import annotations

import math

import numpy as np

from simulation.physics.config import PhysicsConfig
from simulation.physics.dynamics import ACTION_NAMES
from simulation.physics.geometry import Contact, RocketState, tilt_magnitude, wrap_angle

OBS_DIM = 8


def observation_vector(
    state: RocketState, contact: Contact, config: PhysicsConfig | None = None
) -> np.ndarray:
    cfg = config or PhysicsConfig()
    angle = float(np.clip(wrap_angle(state.angle) / math.pi, -1.0, 1.0))
    obs = np.array(
        [
            state.x / cfg.norm_pos,
            state.y / cfg.norm_pos,
            state.vx / cfg.norm_vel,
            state.vy / cfg.norm_vel,
            angle,
            state.omega / cfg.norm_omega,
            1.0 if contact.left else 0.0,
            1.0 if contact.right else 0.0,
        ],
        dtype=np.float32,
    )
    return obs


def telemetry(
    state: RocketState,
    contact: Contact,
    config: PhysicsConfig,
    *,
    action: int | None,
    step_reward_value: float,
    episode_reward: float,
    step_index: int,
    outcome: str | None,
    terminated: bool,
    truncated: bool,
    controller: str,
    distance: float,
    speed: float,
) -> dict:
    if terminated and outcome == "landed":
        status = "landed"
    elif terminated and outcome == "crashed":
        status = "crashed"
    elif truncated:
        status = "truncated"
    else:
        status = "flying"
    return {
        "x_m": state.x,
        "altitude_m": state.y,
        "vx_mps": state.vx,
        "vy_mps": state.vy,
        "speed_mps": speed,
        "tilt_rad": tilt_magnitude(state.angle),
        "tilt_deg": math.degrees(tilt_magnitude(state.angle)),
        "signed_angle_rad": wrap_angle(state.angle),
        "omega_radps": state.omega,
        "action": None if action is None else int(action),
        "action_name": "none" if action is None else ACTION_NAMES[int(action)],
        "step_reward": step_reward_value,
        "episode_reward": episode_reward,
        "distance_m": distance,
        "landing_status": status,
        "controller": controller,
        "left_contact": bool(contact.left),
        "right_contact": bool(contact.right),
        "step": step_index,
    }
