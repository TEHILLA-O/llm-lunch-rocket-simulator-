"""Fixed-step rigid-body dynamics.

The discrete action is converted into an actuator command and then integrated.
A future continuous controller can build the same command with fractional
main-engine and torque values; the environment currently exposes only the
four discrete actions.

Equations, integrated with semi-implicit Euler at a fixed ``dt``:

    nose = (-sin(angle), cos(angle))
    horizontal acceleration = main * main_accel * nose_x
    vertical acceleration   = main * main_accel * nose_y - gravity
    angular acceleration    = torque_command * attitude_torque / inertia

There is no atmospheric drag. The Moon has no atmosphere in this model, and
the brief asks for damping only when the model justifies it. Attitude
thrusters are a pure couple, so they change angular velocity without
changing linear velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulation.physics.config import PhysicsConfig
from simulation.physics.geometry import RocketState, wrap_angle

ACTION_COAST = 0
ACTION_MAIN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = {
    ACTION_COAST: "coast",
    ACTION_MAIN: "main",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

# Action 2, the left attitude thruster, applies positive (counter-clockwise)
# torque: the nose moves toward world -x. Action 3 applies the opposite torque.
LEFT_TORQUE_SIGN = 1.0
RIGHT_TORQUE_SIGN = -1.0


@dataclass(frozen=True)
class ActuatorCommand:
    """Normalised thruster command.

    ``main`` is the main-engine throttle in ``[0, 1]``. ``torque`` is the
    signed attitude command in ``[-1, 1]``: positive is counter-clockwise.
    """

    main: float = 0.0
    torque: float = 0.0


def discrete_to_command(action: int) -> ActuatorCommand:
    value = int(action)
    if value == ACTION_COAST:
        return ActuatorCommand(0.0, 0.0)
    if value == ACTION_MAIN:
        return ActuatorCommand(1.0, 0.0)
    if value == ACTION_LEFT:
        return ActuatorCommand(0.0, LEFT_TORQUE_SIGN)
    if value == ACTION_RIGHT:
        return ActuatorCommand(0.0, RIGHT_TORQUE_SIGN)
    raise ValueError(f"action must be 0, 1, 2 or 3, got {action}")


def accelerations(
    state: RocketState, command: ActuatorCommand, config: PhysicsConfig | None = None
) -> tuple[float, float, float]:
    """Return ``(ax, ay, angular_acceleration)`` in m/s^2 and rad/s^2."""

    cfg = config or PhysicsConfig()
    nose_x = -math.sin(state.angle)
    nose_y = math.cos(state.angle)
    ax = command.main * cfg.main_accel * nose_x
    ay = command.main * cfg.main_accel * nose_y - cfg.gravity
    alpha = command.torque * cfg.attitude_torque / cfg.inertia
    return ax, ay, alpha


def integrate(
    state: RocketState,
    command: ActuatorCommand,
    config: PhysicsConfig | None = None,
    dt: float | None = None,
) -> RocketState:
    """Advance the rocket by one fixed timestep. Does not resolve collisions."""

    cfg = config or PhysicsConfig()
    step = cfg.dt if dt is None else dt
    if step <= 0:
        raise ValueError("dt must be positive")
    ax, ay, alpha = accelerations(state, command, cfg)
    vx = state.vx + ax * step
    vy = state.vy + ay * step
    omega = state.omega + alpha * step
    return RocketState(
        x=state.x + vx * step,
        y=state.y + vy * step,
        vx=vx,
        vy=vy,
        angle=wrap_angle(state.angle + omega * step),
        omega=omega,
    )
