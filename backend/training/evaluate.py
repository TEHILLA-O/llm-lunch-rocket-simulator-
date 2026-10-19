"""Evaluation episodes are separate from the training rollouts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.controllers.agents import PPOController, RandomController
from backend.training.config import make_env
from simulation import ENV_VERSION
from simulation.physics.geometry import terrain_profile
from simulation.recording import dump_json, summarise


def aggregate(episodes: list[dict]) -> dict:
    rewards = np.array([ep["reward"] for ep in episodes], dtype=np.float64)
    lengths = np.array([ep["length"] for ep in episodes], dtype=np.float64)
    distances = np.array([ep["final_distance_m"] for ep in episodes], dtype=np.float64)
    speeds = np.array([ep["final_speed_mps"] for ep in episodes], dtype=np.float64)

    def rate(name: str) -> float:
        if not episodes:
            return 0.0
        return float(sum(ep["outcome"] == name for ep in episodes) / len(episodes))

    near_pad = sum(
        1
        for ep in episodes
        if ep["outcome"] == "truncated" and ep["final_distance_m"] < 5.0 and ep["final_speed_mps"] < 2.0
    )
    return {
        "episodes": len(episodes),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "success_rate": rate("landed"),
        "crash_rate": rate("crashed"),
        "truncation_rate": rate("truncated"),
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "std_length": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "mean_final_distance_m": float(distances.mean()) if len(distances) else 0.0,
        "mean_final_speed_mps": float(speeds.mean()) if len(speeds) else 0.0,
        "timeout_near_pad": near_pad,
        "per_episode": episodes,
        "note": "These are evaluation episodes. A high or low success rate is a measurement, not a guarantee.",
    }


def rollout(controller, config: dict, seeds: list[int], record_path: Path | None = None) -> tuple[dict, dict | None]:
    episodes = []
    recorded = None
    for index, seed in enumerate(seeds):
        controller.reset(seed)
        env = make_env(config, record=record_path is not None and index == 0, controller_name=controller.name)
        observation, _info = env.reset(seed=seed)
        terminated = truncated = False
        info = _info
        while not (terminated or truncated):
            action = controller.act(observation, info.get("telemetry"))
            trace = controller.consume_trace()
            observation, _reward, terminated, truncated, info = env.step(action)
            if env.trajectory:
                env.trajectory[-1]["controller_trace"] = trace
        telemetry = info["telemetry"]
        outcome = "truncated" if truncated and not terminated else info["outcome"]
        episodes.append(
            {
                "seed": seed,
                "reward": env.episode_reward,
                "length": env.steps,
                "outcome": outcome,
                "final_distance_m": telemetry["distance_m"],
                "final_speed_mps": telemetry["speed_mps"],
                "final_tilt_rad": telemetry["tilt_rad"],
            }
        )
        if record_path is not None and index == 0:
            payload = {
                "controller": controller.name,
                "label": controller.label,
                "seed": seed,
                "env_version": ENV_VERSION,
                "config": config,
                "outcome": outcome or "incomplete",
                "steps": env.trajectory,
                "terrain": terrain_profile(env.physics),
                "pad": {"x": env.physics.pad_x, "half_width": env.physics.pad_half_width},
            }
            payload["summary"] = summarise(env.trajectory, payload["outcome"])
            dump_json(payload, record_path)
            recorded = payload
        env.close()
    stats = aggregate(episodes)
    stats["seeds"] = seeds
    return stats, recorded


def finite(value: float) -> bool:
    return math.isfinite(value)
