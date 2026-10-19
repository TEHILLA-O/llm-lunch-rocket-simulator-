"""Step rewards for LunarMind.

Default mode ``delta`` is the requested baseline and is not replaced at
runtime:

    distance_reward = distance_weight * (previous_distance - current_distance)
    speed_reward    = speed_weight    * (previous_speed - current_speed)
    upright_reward  = upright_weight  * (previous_tilt - current_tilt)
    total           = distance_reward + speed_reward + upright_reward

with ``distance_weight = 5``, ``speed_weight = 3`` and ``upright_weight = 2``.
A landing adds exactly ``+100``. A crash adds exactly ``-1000``. There is no
fuel term and no time term.

Because each component is a difference, the sum over an episode telescopes:

    sum_t dense_t = 5*(d0 - dT) + 3*(s0 - sT) + 2*(tilt0 - tiltT)

Oscillating out and back therefore cancels. It does not farm reward. The
same identity creates a real local optimum: a vehicle that reaches the pad
slowly and upright, then hovers until truncation, banks the shaping without
risking the crash penalty. That property is reported, not patched, in the
default mode.

Optional mode ``potential`` is potential-based shaping (Ng, Harada, Russell
1999) with the same weights and a discount ``gamma``:

    component = weight * (previous - gamma * current)

When ``gamma`` equals the reinforcement-learning discount, this form is
policy-invariant relative to the terminal rewards. The default ``delta``
mode is the ``gamma = 1`` special case. It is not policy-invariant under a
discounted PPO objective. ``potential`` is opt-in and is stored on the
experiment so a run never changes formula silently.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from simulation.physics.config import PhysicsConfig
from simulation.physics.geometry import RocketState, tilt_magnitude

LANDING_BONUS = 100.0
CRASH_PENALTY = -1000.0

DISTANCE_WEIGHT = 5.0
SPEED_WEIGHT = 3.0
UPRIGHT_WEIGHT = 2.0


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "delta"
    distance_weight: float = DISTANCE_WEIGHT
    speed_weight: float = SPEED_WEIGHT
    upright_weight: float = UPRIGHT_WEIGHT
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.mode not in ("delta", "potential"):
            raise ValueError("reward mode must be 'delta' or 'potential'")
        if self.distance_weight < 0 or self.speed_weight < 0 or self.upright_weight < 0:
            raise ValueError("dense reward weights must be non-negative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")

    @property
    def label(self) -> str:
        if self.mode == "delta":
            return "delta (requested baseline)"
        return f"potential-based shaping (gamma={self.gamma}, optional)"


@dataclass(frozen=True)
class Metrics:
    distance: float
    speed: float
    tilt: float


@dataclass(frozen=True)
class RewardParts:
    distance: float
    speed: float
    upright: float
    landing_bonus: float
    crash_penalty: float

    @property
    def total(self) -> float:
        return self.distance + self.speed + self.upright + self.landing_bonus + self.crash_penalty

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def metrics_from_state(state: RocketState, config: PhysicsConfig | None = None) -> Metrics:
    cfg = config or PhysicsConfig()
    dx = state.x - cfg.pad_x
    dy = state.y - cfg.target_y
    return Metrics(
        distance=math.hypot(dx, dy),
        speed=math.hypot(state.vx, state.vy),
        tilt=tilt_magnitude(state.angle),
    )


def step_reward(
    previous: Metrics,
    current: Metrics,
    outcome: str | None,
    config: RewardConfig | None = None,
) -> RewardParts:
    """Score one physics step.

    ``outcome`` is ``None`` while flying and also when an episode truncates.
    Truncation is neither a landing nor a crash, so it does not add a
    terminal bonus or penalty. The function has no action, fuel or time
    argument: those quantities cannot affect the score.
    """

    cfg = config or RewardConfig()
    if outcome not in (None, "landed", "crashed"):
        raise ValueError(f"unknown outcome {outcome!r}")
    scale_current = 1.0 if cfg.mode == "delta" else cfg.gamma
    parts = RewardParts(
        distance=cfg.distance_weight * (previous.distance - scale_current * current.distance),
        speed=cfg.speed_weight * (previous.speed - scale_current * current.speed),
        upright=cfg.upright_weight * (previous.tilt - scale_current * current.tilt),
        landing_bonus=LANDING_BONUS if outcome == "landed" else 0.0,
        crash_penalty=CRASH_PENALTY if outcome == "crashed" else 0.0,
    )
    return parts


def dense_closed_form(initial: Metrics, final: Metrics, config: RewardConfig | None = None) -> float:
    """Closed form of the summed dense reward from the first state to the last.

    For ``delta`` mode this is an exact telescope. For ``potential`` mode the
    intermediate samples matter, so this helper only applies to ``delta``.
    """

    cfg = config or RewardConfig()
    if cfg.mode != "delta":
        raise ValueError("closed form is for delta mode; potential mode is not a pure telescope")
    return (
        cfg.distance_weight * (initial.distance - final.distance)
        + cfg.speed_weight * (initial.speed - final.speed)
        + cfg.upright_weight * (initial.tilt - final.tilt)
    )


def public_reward_spec() -> dict:
    return {
        "default_mode": "delta",
        "optional_mode": "potential",
        "distance": f"{DISTANCE_WEIGHT:g} * (previous_distance - current_distance)",
        "speed": f"{SPEED_WEIGHT:g} * (previous_speed - current_speed)",
        "upright": f"{UPRIGHT_WEIGHT:g} * (previous_tilt - current_tilt)",
        "landing_bonus": LANDING_BONUS,
        "crash_penalty": CRASH_PENALTY,
        "fuel_penalty": None,
        "time_penalty": None,
        "distance_definition": "Euclidean distance in metres from the centre of mass to the pad centre (0, 0).",
        "speed_definition": "Euclidean magnitude of horizontal and vertical velocity, m/s.",
        "tilt_definition": "Absolute wrapped angle from upright, radians, in [0, pi].",
        "notes": [
            "Dense rewards telescope, so oscillating back to the same distance, speed and tilt cancels.",
            "Truncation at the step limit adds neither the landing bonus nor the crash penalty.",
            "Potential-based shaping is optional and must be selected in the experiment configuration.",
        ],
    }
