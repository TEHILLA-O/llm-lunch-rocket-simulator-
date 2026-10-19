"""Fair comparisons on one shared environment configuration."""

from __future__ import annotations

import logging

from backend.controllers.agents import LLMController, PPOController, RandomController, llm_configured
from backend.storage import repository as repo
from backend.training.config import same_task
from backend.training.evaluate import rollout
from simulation import ENV_VERSION

logger = logging.getLogger("lunarmind.compare")


def comparison_config(body) -> dict:
    return {
        "env_version": ENV_VERSION,
        "physics": {
            "gravity": body.gravity,
            "main_accel": body.main_accel,
            "attitude_torque": body.attitude_torque,
            "inertia": body.inertia,
            "dt": 0.02,
        },
        "reward": {
            "mode": body.reward_mode,
            "distance_weight": body.distance_weight,
            "speed_weight": body.speed_weight,
            "upright_weight": body.upright_weight,
            "gamma": body.gamma,
            "landing_bonus": 100,
            "crash_penalty": -1000,
            "fuel_penalty": None,
            "time_penalty": None,
        },
        "max_steps": body.max_steps,
    }


def begin_comparison(group_id: str, body) -> list:
    """Insert one running row per controller before any rollout starts."""

    seeds = [seed + rep for seed in body.seeds for rep in range(body.episodes_per_seed)]
    pending = []
    for spec in body.controllers:
        row = repo.insert_evaluation(
            {
                "controller": spec.type,
                "label": spec.type,
                "checkpoint_id": spec.checkpoint_id,
                "seed": seeds[0] if seeds else 0,
                "episodes": 0,
                "metrics": {},
                "group_id": group_id,
                "status": "running",
            }
        )
        pending.append((spec, row["id"]))
    return pending


def finish_comparison(group_id: str, body, pending) -> None:
    config = comparison_config(body)
    seeds = [seed + rep for seed in body.seeds for rep in range(body.episodes_per_seed)]
    try:
        _roll_out_controllers(config, seeds, pending)
    except Exception as exc:  # noqa: BLE001 - a setup error must not leave rows running
        logger.exception("comparison group %s failed", group_id)
        for _spec, evaluation_id in pending:
            current = repo.get_evaluation(evaluation_id)
            if current["status"] == "running":
                repo.update_evaluation(
                    evaluation_id,
                    episodes=0,
                    metrics={"error": str(exc)},
                    status="failed",
                )


def _roll_out_controllers(config: dict, seeds: list[int], pending) -> None:
    for spec, evaluation_id in pending:
        label = spec.type
        try:
            warning = None
            experiment_id = None
            checkpoint_id = spec.checkpoint_id
            if spec.type == "random":
                controller = RandomController(seeds[0])
                label = "Random"
            elif spec.type == "ppo":
                if not spec.checkpoint_id:
                    raise ValueError("PPO comparison needs a checkpoint.")
                checkpoint = repo.get_checkpoint(spec.checkpoint_id)
                experiment = repo.get_experiment(checkpoint["experiment_id"])
                experiment_id = experiment["id"]
                checkpoint_id = checkpoint["id"]
                label = f"PPO {experiment['name']} @ {checkpoint['timesteps']}"
                if not same_task(experiment["config"], config):
                    warning = (
                        "This checkpoint was trained with a different physics or reward configuration. "
                        "It is still evaluated on the shared comparison environment."
                    )
                controller = PPOController(checkpoint["path"], label)
            elif spec.type == "llm":
                if not llm_configured():
                    raise RuntimeError(
                        "No OPENAI_API_KEY is configured. The language-model controller was not run and no score was invented."
                    )
                controller = LLMController()
                label = "LLM baseline (not RL)"
            else:
                raise ValueError(f"unsupported controller {spec.type}")
            metrics, _payload = rollout(controller, config, seeds)
            if warning:
                metrics["warning"] = warning
            repo.update_evaluation(
                evaluation_id,
                experiment_id=experiment_id,
                checkpoint_id=checkpoint_id,
                label=label,
                episodes=len(seeds),
                metrics=metrics,
                status="completed",
            )
        except Exception as exc:  # noqa: BLE001 - one controller must not sink the group
            logger.exception("comparison controller failed")
            repo.update_evaluation(
                evaluation_id,
                label=label,
                episodes=0,
                metrics={"error": str(exc)},
                status="failed",
            )


def run_comparison(group_id: str, body) -> None:
    pending = begin_comparison(group_id, body)
    finish_comparison(group_id, body, pending)
