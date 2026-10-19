"""Trajectory records for replay and export.

A trajectory stores the observation that was acted on, the action, the
reward and its breakdown, and the resulting physical state. Controller
diagnostics such as language-model token counts live under
``controller_trace`` and are not part of the reward.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def summarise(steps: list[dict], outcome: str) -> dict:
    acted = [step for step in steps if step.get("action") is not None]
    total = float(sum(step["reward"] for step in acted))
    final = steps[-1] if steps else {}
    raw = final.get("raw") or {}
    return {
        "outcome": outcome,
        "episode_reward": total,
        "physics_steps": len(acted),
        "final_distance_m": final.get("distance_m"),
        "final_speed_mps": final.get("speed_mps"),
        "final_tilt_rad": raw.get("tilt_rad"),
        "final_x_m": raw.get("x"),
        "final_altitude_m": raw.get("y"),
    }


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_csv(payload: dict) -> str:
    buffer = io.StringIO()
    fields = [
        "step",
        "x_m",
        "altitude_m",
        "vx_mps",
        "vy_mps",
        "angle_rad",
        "omega_radps",
        "left_contact",
        "right_contact",
        "action",
        "action_name",
        "reward",
        "cumulative_reward",
        "distance_reward",
        "speed_reward",
        "upright_reward",
        "landing_bonus",
        "crash_penalty",
        "distance_m",
        "speed_mps",
        "tilt_rad",
        "terminated",
        "truncated",
        "outcome",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for step in payload.get("steps", []):
        raw = step.get("raw") or {}
        parts = step.get("reward_parts") or {}
        writer.writerow(
            {
                "step": step.get("step"),
                "x_m": raw.get("x"),
                "altitude_m": raw.get("y"),
                "vx_mps": raw.get("vx"),
                "vy_mps": raw.get("vy"),
                "angle_rad": raw.get("angle"),
                "omega_radps": raw.get("omega"),
                "left_contact": raw.get("left_contact"),
                "right_contact": raw.get("right_contact"),
                "action": step.get("action"),
                "action_name": step.get("action_name"),
                "reward": step.get("reward"),
                "cumulative_reward": step.get("cumulative_reward"),
                "distance_reward": parts.get("distance"),
                "speed_reward": parts.get("speed"),
                "upright_reward": parts.get("upright"),
                "landing_bonus": parts.get("landing_bonus"),
                "crash_penalty": parts.get("crash_penalty"),
                "distance_m": step.get("distance_m"),
                "speed_mps": step.get("speed_mps"),
                "tilt_rad": raw.get("tilt_rad"),
                "terminated": step.get("terminated"),
                "truncated": step.get("truncated"),
                "outcome": step.get("outcome"),
            }
        )
    return buffer.getvalue()
