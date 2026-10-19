"""LunarMind Gymnasium environment.

Every transition is one call to the fixed-step integrator followed by the
contact classifier and the reward function. Rendering, when enabled, reads
the resulting state and does not feed back into it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation import ENV_VERSION
from simulation.physics.config import PhysicsConfig
from simulation.physics.dynamics import ACTION_NAMES, discrete_to_command, integrate
from simulation.physics.geometry import (
    Contact,
    RocketState,
    evaluate_contact,
    out_of_bounds,
    sample_initial_state,
)
from simulation.environment.observations import OBS_DIM, observation_vector, telemetry
from simulation.rendering.scene import build_scene
from simulation.rewards.shaping import RewardConfig, metrics_from_state, step_reward

_OBS_HIGH = np.array([50, 50, 50, 50, 1, 50, 1, 1], dtype=np.float32)
_OBS_LOW = np.array([-50, -5, -50, -50, -1, -50, 0, 0], dtype=np.float32)


class LunarMindEnv(gym.Env):
    """2D rocket landing task with the LunarMind reward.

    Parameters
    ----------
    physics:
        Frozen physical parameters. The running episode never mutates them.
    reward:
        Frozen reward parameters. Terminal bonus and penalty are constants
        inside the reward function and are not fields on this object.
    max_steps:
        Flying longer than this truncates the episode. Truncation is not a
        landing and not a crash.
    record:
        When true, every step is appended to ``trajectory`` for export.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        physics: PhysicsConfig | None = None,
        reward: RewardConfig | None = None,
        max_steps: int = 1000,
        render_mode: str | None = None,
        record: bool = False,
        controller_name: str = "unassigned",
    ):
        super().__init__()
        self.physics = physics or PhysicsConfig()
        self.reward = reward or RewardConfig()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.record = record
        self.controller_name = controller_name
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(_OBS_LOW, _OBS_HIGH, dtype=np.float32)
        self.state: RocketState | None = None
        self.contact = Contact(False, False, False, False, (0.0, 0.0), (0.0, 0.0))
        self.steps = 0
        self.episode_reward = 0.0
        self.outcome: str | None = None
        self.last_action: int | None = None
        self.last_parts: dict[str, float] | None = None
        self.trajectory: list[dict] = []
        self._renderer = None
        self.episode_over = False
        self.env_version = ENV_VERSION

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode_over = False
        override = None if options is None else options.get("state")
        if override is not None:
            self.state = RocketState(
                x=float(override["x"]),
                y=float(override["y"]),
                vx=float(override["vx"]),
                vy=float(override["vy"]),
                angle=float(override["angle"]),
                omega=float(override["omega"]),
            )
        else:
            self.state = sample_initial_state(self.np_random, self.physics)
        self.contact, self.outcome = evaluate_contact(self.state, self.physics)
        if self.outcome is not None and override is None:
            self.state = RocketState(0.0, 18.0, 0.0, -0.2, 0.0, 0.0)
            self.contact, self.outcome = evaluate_contact(self.state, self.physics)
        self.steps = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.last_parts = None
        self.trajectory = []
        obs = self._observe()
        info = self._info(0.0, False, False)
        if self.record:
            self.trajectory.append(self._snapshot(obs, info, action=None, reward=0.0, terminated=False, truncated=False))
        return obs, info

    def step(self, action):
        if self.state is None or self.episode_over:
            raise RuntimeError("reset the environment before stepping")
        try:
            command = discrete_to_command(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid action {action!r}") from exc
        action_i = int(action)
        previous = metrics_from_state(self.state, self.physics)
        self.state = integrate(self.state, command, self.physics)
        self.contact, self.outcome = evaluate_contact(self.state, self.physics)
        bounded = out_of_bounds(self.state, self.physics)
        current = metrics_from_state(self.state, self.physics)
        parts = step_reward(previous, current, self.outcome, self.reward)
        reward = float(parts.total)
        self.episode_reward += reward
        self.steps += 1
        self.last_action = action_i
        self.last_parts = parts.as_dict()
        terminated = self.outcome is not None
        truncated = (not terminated) and (self.steps >= self.max_steps or bounded)
        self.episode_over = bool(terminated or truncated)
        obs = self._observe()
        info = self._info(reward, terminated, truncated)
        info["left_operational_volume"] = bool(bounded and not terminated)
        if self.record:
            self.trajectory.append(
                self._snapshot(obs, info, action=action_i, reward=reward, terminated=terminated, truncated=truncated)
            )
        return obs, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        assert self.state is not None
        obs = observation_vector(self.state, self.contact, self.physics)
        np.clip(obs, self.observation_space.low, self.observation_space.high, out=obs)
        return obs

    def _info(self, reward: float, terminated: bool, truncated: bool) -> dict[str, Any]:
        assert self.state is not None
        metrics = metrics_from_state(self.state, self.physics)
        parts = self.last_parts or {
            "distance": 0.0,
            "speed": 0.0,
            "upright": 0.0,
            "landing_bonus": 0.0,
            "crash_penalty": 0.0,
        }
        info = {
            "outcome": self.outcome if terminated else None,
            "reward_parts": parts,
            "reward_mode": self.reward.mode,
            "reward_label": self.reward.label,
            "env_version": self.env_version,
            "telemetry": telemetry(
                self.state,
                self.contact,
                self.physics,
                action=self.last_action,
                step_reward_value=reward,
                episode_reward=self.episode_reward,
                step_index=self.steps,
                outcome=self.outcome if terminated else None,
                terminated=terminated,
                truncated=truncated,
                controller=self.controller_name,
                distance=metrics.distance,
                speed=metrics.speed,
            ),
            "scene": build_scene(self.state, self.contact, self.last_action, self.outcome if terminated else None, self.physics),
        }
        return info

    def _snapshot(self, obs, info, *, action, reward, terminated, truncated) -> dict:
        assert self.state is not None
        tel = info["telemetry"]
        return {
            "step": self.steps,
            "observation": [float(v) for v in obs],
            "raw": {
                "x": self.state.x,
                "y": self.state.y,
                "vx": self.state.vx,
                "vy": self.state.vy,
                "angle": self.state.angle,
                "omega": self.state.omega,
                "tilt_rad": tel["tilt_rad"],
                "left_contact": tel["left_contact"],
                "right_contact": tel["right_contact"],
            },
            "action": action,
            "action_name": None if action is None else ACTION_NAMES[action],
            "reward": reward,
            "reward_parts": info["reward_parts"],
            "cumulative_reward": self.episode_reward,
            "distance_m": tel["distance_m"],
            "speed_mps": tel["speed_mps"],
            "terminated": terminated,
            "truncated": truncated,
            "outcome": info["outcome"],
            "scene": info["scene"],
        }

    def render(self):
        if self.render_mode is None:
            return None
        from simulation.rendering.pygame_view import PygameView

        if self._renderer is None:
            self._renderer = PygameView(self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
