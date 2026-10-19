"""Request models for the versioned HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    seed: int = 0
    timesteps: int = Field(default=100_000, ge=256, le=5_000_000)
    gravity: float = Field(default=1.62, gt=0, le=30)
    main_accel: float = Field(default=4.0, gt=0, le=50)
    attitude_torque: float = Field(default=0.8, gt=0, le=20)
    inertia: float = Field(default=1.0, gt=0, le=50)
    max_steps: int = Field(default=1000, ge=50, le=5000)
    reward_mode: Literal["delta", "potential"] = "delta"
    distance_weight: float = Field(default=5.0, ge=0, le=100)
    speed_weight: float = Field(default=3.0, ge=0, le=100)
    upright_weight: float = Field(default=2.0, ge=0, le=100)
    gamma: float = Field(default=0.99, gt=0, le=1)
    n_envs: int = Field(default=1, ge=1, le=8)
    n_steps: int = Field(default=2048, ge=64, le=8192)
    batch_size: int = Field(default=256, ge=16, le=2048)
    learning_rate: float = Field(default=3e-4, gt=0, le=1)
    auto_eval_episodes: int = Field(default=3, ge=0, le=20)

    @model_validator(mode="before")
    @classmethod
    def reject_terminal_overrides(cls, data):
        if isinstance(data, dict):
            forbidden = {"landing_bonus", "crash_penalty", "fuel_penalty", "time_penalty"}
            present = forbidden.intersection(data)
            if present:
                raise ValueError(
                    "Landing bonus (+100), crash penalty (-1000), fuel and time penalties are fixed. "
                    f"Remove {sorted(present)}."
                )
        return data


class SessionCreate(BaseModel):
    controller: Literal["human", "random", "ppo", "llm", "demo"]
    seed: int = 0
    checkpoint_id: str | None = None
    demo: Literal["safe", "crash", "tilt"] | None = None
    speed: float = Field(default=1.0, gt=0, le=8)


class ControllerSpec(BaseModel):
    type: Literal["random", "ppo", "llm"]
    checkpoint_id: str | None = None


class CompareRequest(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=20)
    episodes_per_seed: int = Field(default=1, ge=1, le=10)
    controllers: list[ControllerSpec] = Field(min_length=1, max_length=6)
    gravity: float = Field(default=1.62, gt=0, le=30)
    main_accel: float = Field(default=4.0, gt=0, le=50)
    attitude_torque: float = Field(default=0.8, gt=0, le=20)
    inertia: float = Field(default=1.0, gt=0, le=50)
    max_steps: int = Field(default=800, ge=50, le=2000)
    reward_mode: Literal["delta", "potential"] = "delta"
    distance_weight: float = Field(default=5.0, ge=0, le=100)
    speed_weight: float = Field(default=3.0, ge=0, le=100)
    upright_weight: float = Field(default=2.0, ge=0, le=100)
    gamma: float = Field(default=0.99, gt=0, le=1)
