"""Render geometry derived from the physics state.

Flame anchors describe where pixels may be drawn. They are not forces and
are not read by the integrator.
"""

from __future__ import annotations

import math

from simulation.physics.config import PhysicsConfig
from simulation.physics.dynamics import ACTION_LEFT, ACTION_MAIN, ACTION_RIGHT
from simulation.physics.geometry import (
    HULL,
    LEFT_FLAME_LOCAL,
    LEFT_FOOT,
    LEFT_HIP,
    MAIN_FLAME_LOCAL,
    RIGHT_FLAME_LOCAL,
    RIGHT_FOOT,
    RIGHT_HIP,
    Contact,
    RocketState,
    to_world,
)


def _unit_from_local(angle: float, lx: float, ly: float) -> list[float]:
    ox, oy = to_world(0.0, 0.0, angle, lx, ly)
    norm = math.hypot(ox, oy) or 1.0
    return [ox / norm, oy / norm]


def build_scene(
    state: RocketState,
    contact: Contact,
    action: int | None,
    outcome: str | None,
    config: PhysicsConfig | None = None,
) -> dict:
    cfg = config or PhysicsConfig()
    body = [list(to_world(state.x, state.y, state.angle, lx, ly)) for lx, ly in HULL]
    legs = []
    for hip, foot, touched in (
        (LEFT_HIP, LEFT_FOOT, contact.left),
        (RIGHT_HIP, RIGHT_FOOT, contact.right),
    ):
        legs.append(
            {
                "hip": list(to_world(state.x, state.y, state.angle, *hip)),
                "foot": list(to_world(state.x, state.y, state.angle, *foot)),
                "contact": bool(touched),
            }
        )

    def anchor(local, direction_local, active: bool) -> dict:
        return {
            "active": active,
            "origin": list(to_world(state.x, state.y, state.angle, *local)),
            "direction": _unit_from_local(state.angle, *direction_local),
        }

    flames = {
        "main": anchor(MAIN_FLAME_LOCAL, (0.0, -1.0), action == ACTION_MAIN),
        "left": anchor(LEFT_FLAME_LOCAL, (-1.0, 0.0), action == ACTION_LEFT),
        "right": anchor(RIGHT_FLAME_LOCAL, (1.0, 0.0), action == ACTION_RIGHT),
    }
    return {
        "x": state.x,
        "y": state.y,
        "angle": state.angle,
        "body": body,
        "legs": legs,
        "flames": flames,
        "outcome": outcome,
        "pad": {"x": cfg.pad_x, "half_width": cfg.pad_half_width, "y": 0.0},
    }
