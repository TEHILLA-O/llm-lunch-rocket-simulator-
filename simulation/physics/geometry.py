"""Geometry, terrain and deterministic contact rules.

World frame: +x right, +y up, origin on the landing-pad centre at ground
level. Body frame: +local y along the nose, +local x toward the rocket's
right. The rotation from body to world is

    [wx]   [ cos(a)  -sin(a) ] [lx]
    [wy] = [ sin(a)   cos(a) ] [ly]

which is the standard counter-clockwise rotation. At angle 0 the nose
points up. A positive angle points the nose left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulation.physics.config import PhysicsConfig

# Landing feet and the rigid hull, in metres, body frame.
LEFT_FOOT = (-0.72, -0.95)
RIGHT_FOOT = (0.72, -0.95)
LEFT_HIP = (-0.34, -0.48)
RIGHT_HIP = (0.34, -0.48)
HULL = (
    (0.0, 1.15),
    (0.46, 0.38),
    (0.38, -0.42),
    (-0.38, -0.42),
    (-0.46, 0.38),
)
MAIN_FLAME_LOCAL = (0.0, -0.50)
LEFT_FLAME_LOCAL = (-0.50, -0.10)
RIGHT_FLAME_LOCAL = (0.50, -0.10)


@dataclass(frozen=True)
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    omega: float


@dataclass(frozen=True)
class Contact:
    left: bool
    right: bool
    hull: bool
    feet_inside_pad: bool
    left_xy: tuple[float, float]
    right_xy: tuple[float, float]

    @property
    def any_touch(self) -> bool:
        return self.left or self.right or self.hull


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into the interval ``[-pi, pi]``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def tilt_magnitude(angle: float) -> float:
    """Absolute deviation from upright, using a wrapped angle."""

    return abs(wrap_angle(angle))


def to_world(x: float, y: float, angle: float, lx: float, ly: float) -> tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (x + lx * c - ly * s, y + lx * s + ly * c)


def terrain_height(x: float, config: PhysicsConfig | None = None) -> float:
    """Ground elevation in metres.

    The landing pad is perfectly flat. Outside the pad the surface rises
    into low ridges. Height is continuous at the pad edge, so a foot that
    slides off the pad meets a slope rather than a step.
    """

    cfg = config or PhysicsConfig()
    d = abs(x - cfg.pad_x) - cfg.pad_half_width
    if d <= 0.0:
        return 0.0
    ridges = 1.15 * math.sin(d * 0.55) ** 2
    bumps = 0.25 * math.sin(x * 0.35) ** 2 * (d / (d + 2.0))
    return ridges + bumps


def terrain_profile(
    config: PhysicsConfig | None = None,
    x0: float = -80.0,
    x1: float = 80.0,
    samples: int = 321,
) -> list[dict[str, float]]:
    cfg = config or PhysicsConfig()
    if samples < 2:
        raise ValueError("terrain profile needs at least two samples")
    span = x1 - x0
    return [
        {"x": x0 + span * i / (samples - 1), "y": terrain_height(x0 + span * i / (samples - 1), cfg)}
        for i in range(samples)
    ]


def _touching(px: float, py: float, config: PhysicsConfig) -> bool:
    return py <= terrain_height(px, config) + config.contact_eps


def probe_contact(state: RocketState, config: PhysicsConfig | None = None) -> Contact:
    cfg = config or PhysicsConfig()
    left = to_world(state.x, state.y, state.angle, *LEFT_FOOT)
    right = to_world(state.x, state.y, state.angle, *RIGHT_FOOT)
    hull_hit = False
    for lx, ly in HULL:
        hx, hy = to_world(state.x, state.y, state.angle, lx, ly)
        if _touching(hx, hy, cfg):
            hull_hit = True
            break
    inside = abs(left[0] - cfg.pad_x) <= cfg.pad_half_width and abs(right[0] - cfg.pad_x) <= cfg.pad_half_width
    return Contact(
        left=_touching(*left, cfg),
        right=_touching(*right, cfg),
        hull=hull_hit,
        feet_inside_pad=inside,
        left_xy=left,
        right_xy=right,
    )


def is_safe_landing(state: RocketState, contact: Contact, config: PhysicsConfig | None = None) -> bool:
    """Return True only when every published landing rule passes.

    A near miss on any single rule is a crash. Landing and crashing are
    therefore mutually exclusive: this predicate is the only path to a
    landing, and every other contact is a crash.
    """

    cfg = config or PhysicsConfig()
    if not (contact.left and contact.right):
        return False
    if contact.hull or not contact.feet_inside_pad:
        return False
    if abs(state.vx) > cfg.max_horizontal_speed:
        return False
    descent = -state.vy if state.vy < 0.0 else 0.0
    ascent = state.vy if state.vy > 0.0 else 0.0
    if descent > cfg.max_descent_speed or ascent > cfg.max_ascent_speed:
        return False
    if tilt_magnitude(state.angle) > cfg.max_tilt_rad:
        return False
    if abs(state.omega) > cfg.max_angular_speed:
        return False
    return True


def evaluate_contact(
    state: RocketState, config: PhysicsConfig | None = None
) -> tuple[Contact, str | None]:
    """Classify a state as flying, landed or crashed.

    Any foot or hull contact ends the episode. Safe two-foot contact on
    the pad is ``landed``. Every other contact, including a single foot,
    a hull strike, an off-pad touchdown or an unsafe speed or angle, is
    ``crashed``. No contact leaves the outcome as ``None``.
    """

    cfg = config or PhysicsConfig()
    contact = probe_contact(state, cfg)
    if state.y < -2.0 and not contact.any_touch:
        # The hull solver should already have touched terrain. Falling
        # through the surface is still a crash, never a landing.
        return contact, "crashed"
    if not contact.any_touch:
        return contact, None
    if is_safe_landing(state, contact, cfg):
        return contact, "landed"
    return contact, "crashed"


def out_of_bounds(state: RocketState, config: PhysicsConfig | None = None) -> bool:
    """True when the vehicle has left the operational volume while still flying."""

    cfg = config or PhysicsConfig()
    return state.y > cfg.ceiling or abs(state.x) > cfg.x_limit


def sample_initial_state(rng, config: PhysicsConfig | None = None) -> RocketState:
    """Sample a start that is clear of the terrain.

    ``rng`` is a ``numpy.random.Generator`` (Gymnasium's ``np_random``).
    """

    cfg = config or PhysicsConfig()
    for _ in range(80):
        state = RocketState(
            x=float(rng.uniform(-18.0, 18.0)),
            y=float(rng.uniform(10.0, 22.0)),
            vx=float(rng.uniform(-1.5, 1.5)),
            vy=float(rng.uniform(-2.0, 0.5)),
            angle=float(rng.uniform(-0.6, 0.6)),
            omega=float(rng.uniform(-0.4, 0.4)),
        )
        contact, outcome = evaluate_contact(state, cfg)
        if outcome is not None:
            continue
        clearance = min(contact.left_xy[1], contact.right_xy[1]) - terrain_height(contact.left_xy[0], cfg)
        if clearance > 2.0:
            return state
    return RocketState(0.0, 18.0, 0.0, -0.4, 0.0, 0.0)
