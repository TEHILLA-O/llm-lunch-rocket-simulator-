"""Physical parameters for the LunarMind rocket.

The integrator always advances by ``dt`` seconds. Wall-clock time and the
dashboard playback speed never enter these equations, so a faster training
loop produces the same trajectories as real-time playback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    """Tunable vehicle and landing-site parameters.

    Attitude thrusters are a pure couple: they apply torque and no net force.
    The main engine thrust line passes through the centre of mass, so it
    accelerates the vehicle without adding torque.

    Angles are radians. ``angle = 0`` is upright. Positive angle is
    counter-clockwise, which points the nose toward negative world x
    (to the left when +x is right and +y is up).
    """

    gravity: float = 1.62
    main_accel: float = 4.0
    attitude_torque: float = 0.8
    inertia: float = 1.0
    dt: float = 0.02
    pad_x: float = 0.0
    pad_half_width: float = 4.0
    target_y: float = 0.0
    max_horizontal_speed: float = 1.0
    max_descent_speed: float = 1.5
    max_ascent_speed: float = 0.5
    max_tilt_rad: float = 0.30
    max_angular_speed: float = 0.40
    contact_eps: float = 0.02
    ceiling: float = 80.0
    x_limit: float = 100.0
    norm_pos: float = 20.0
    norm_vel: float = 10.0
    norm_omega: float = 2.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("physics timestep dt must be positive")
        if self.inertia <= 0:
            raise ValueError("inertia must be positive")
        if self.main_accel <= 0 or self.attitude_torque <= 0:
            raise ValueError("thruster magnitudes must be positive")
        if self.pad_half_width <= 0:
            raise ValueError("pad half-width must be positive")
        if self.norm_pos <= 0 or self.norm_vel <= 0 or self.norm_omega <= 0:
            raise ValueError("observation scales must be positive")
