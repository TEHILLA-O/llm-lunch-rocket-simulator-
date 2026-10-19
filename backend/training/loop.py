"""PPO training loop.

Training runs in this function, which the API starts in a separate process.
``model.learn`` is never called from a request handler.

Stable-Baselines3 treats ``total_timesteps`` as additional steps when
``reset_num_timesteps`` is false (it adds the current counter inside
``_setup_learn``). The loop below passes the size of the next chunk and the
test suite checks that the counter actually advances.
"""

from __future__ import annotations

import logging
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from backend.paths import checkpoints_dir, tensorboard_dir, trajectories_dir
from backend.storage import repository as repo
from backend.training.config import make_vec_env
from backend.training.evaluate import rollout
from backend.controllers.agents import PPOController

logger = logging.getLogger("lunarmind.train")


class ProgressCallback(BaseCallback):
    """Persist completed training episodes and honour pause/stop between rollouts."""

    def __init__(self, experiment_id: str):
        super().__init__()
        self.experiment_id = experiment_id

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            episode = info.get("episode")
            if not episode:
                continue
            telemetry = info.get("telemetry") or {}
            status = telemetry.get("landing_status")
            if status not in ("landed", "crashed", "truncated"):
                status = info.get("outcome") or "unknown"
            repo.insert_training_log(
                self.experiment_id,
                int(self.num_timesteps),
                float(episode["r"]),
                int(episode["l"]),
                str(status),
                telemetry.get("distance_m"),
                telemetry.get("speed_mps"),
            )
        if self.n_calls % 512 == 0:
            status = repo.get_experiment(self.experiment_id)["status"]
            if status in ("pause_requested", "paused", "stop_requested", "stopped"):
                return False
        return True


def _new_model(vec, config: dict, seed: int) -> PPO:
    ppo = config["ppo"]
    set_random_seed(seed)
    return PPO(
        "MlpPolicy",
        vec,
        learning_rate=ppo["learning_rate"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        ent_coef=ppo["ent_coef"],
        policy_kwargs={"net_arch": ppo["net_arch"]},
        seed=seed,
        device="cpu",
        verbose=0,
        tensorboard_log=str(tensorboard_dir(config.get("_experiment_id", "run"))),
    )


def _save(experiment_id: str, model: PPO, steps: int, milestone: int | None, config: dict) -> None:
    path = checkpoints_dir(experiment_id) / f"ppo_{steps}_steps"
    model.save(str(path))
    metrics = {"milestone": milestone, "timesteps": steps}
    eval_episodes = int(config.get("auto_eval_episodes") or 0)
    if eval_episodes > 0:
        controller = PPOController(str(path), f"PPO step {steps}")
        seeds = [int(config.get("_seed", 0)) + 100_000 + index for index in range(eval_episodes)]
        record = trajectories_dir() / f"eval_{experiment_id}_{steps}.json"
        stats, _payload = rollout(controller, config, seeds, record_path=record)
        metrics["evaluation"] = {key: value for key, value in stats.items() if key != "per_episode"}
        from simulation.recording import summarise
        import json

        payload = json.loads(record.read_text(encoding="utf-8"))
        repo.insert_trajectory(
            {
                "experiment_id": experiment_id,
                "controller": "ppo",
                "label": f"PPO evaluation at {steps} steps",
                "outcome": payload["outcome"],
                "seed": seeds[0],
                "path": str(record),
                "summary": payload["summary"] if "summary" in payload else summarise(payload["steps"], payload["outcome"]),
            }
        )
        checkpoint = repo.insert_checkpoint(experiment_id, steps, str(path) + ".zip", metrics)
        repo.insert_evaluation(
            {
                "experiment_id": experiment_id,
                "checkpoint_id": checkpoint["id"],
                "controller": "ppo",
                "label": f"PPO @ {steps}",
                "seed": seeds[0],
                "episodes": eval_episodes,
                "metrics": metrics["evaluation"],
                "status": "completed",
            }
        )
        return
    repo.insert_checkpoint(experiment_id, steps, str(path) + ".zip", metrics)


def train_experiment(experiment_id: str) -> None:
    experiment = repo.get_experiment(experiment_id)
    config = dict(experiment["config"])
    config["_experiment_id"] = experiment_id
    config["_seed"] = experiment["seed"]
    repo.update_experiment(experiment_id, status="running", error=None)
    vector = make_vec_env(config, experiment["seed"])
    try:
        latest = repo.latest_checkpoint(experiment_id)
        if latest:
            model = PPO.load(latest["path"], env=vector, device="cpu")
            logger.info("resumed %s from %s steps", experiment_id, latest["timesteps"])
        else:
            model = _new_model(vector, config, experiment["seed"])
            _save(experiment_id, model, 0, 0, config)
        callback = ProgressCallback(experiment_id)
        target = int(experiment["timesteps_target"])
        milestones = list(config.get("milestones") or [])
        while int(model.num_timesteps) < target:
            status = repo.get_experiment(experiment_id)["status"]
            if status in ("stop_requested", "stopped"):
                repo.update_experiment(experiment_id, status="stopped", timesteps_completed=int(model.num_timesteps))
                return
            if status in ("pause_requested", "paused"):
                repo.update_experiment(experiment_id, status="paused", timesteps_completed=int(model.num_timesteps))
                while True:
                    time.sleep(0.2)
                    status = repo.get_experiment(experiment_id)["status"]
                    if status in ("running", "resume_requested"):
                        repo.update_experiment(experiment_id, status="running")
                        break
                    if status in ("stop_requested", "stopped"):
                        repo.update_experiment(
                            experiment_id, status="stopped", timesteps_completed=int(model.num_timesteps)
                        )
                        return
            before = int(model.num_timesteps)
            chunk = min(int(config["ppo"]["n_steps"]) * int(config["n_envs"]), target - before)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=callback,
                progress_bar=False,
            )
            after = int(model.num_timesteps)
            status = repo.get_experiment(experiment_id)["status"]
            if after <= before and status == "running" and before < target:
                # Unexpected SB3 builds treat total_timesteps as an absolute target.
                model.learn(
                    total_timesteps=before + chunk,
                    reset_num_timesteps=False,
                    callback=callback,
                    progress_bar=False,
                )
                after = int(model.num_timesteps)
                status = repo.get_experiment(experiment_id)["status"]
            if after <= before and status == "running" and before < target:
                repo.update_experiment(
                    experiment_id,
                    status="failed",
                    timesteps_completed=after,
                    error="The training step counter did not advance, so the run was stopped.",
                )
                return
            repo.update_experiment(experiment_id, timesteps_completed=after)
            saved = {item["timesteps"] for item in repo.list_checkpoints(experiment_id)}
            crossed = [
                milestone
                for milestone in milestones
                if before < milestone <= after and milestone not in saved
            ]
            if crossed:
                _save(experiment_id, model, after, crossed[-1], config)
            logger.info("experiment %s reached %s/%s steps", experiment_id, after, target)
        final = int(model.num_timesteps)
        saved = {item["timesteps"] for item in repo.list_checkpoints(experiment_id)}
        if final not in saved:
            _save(experiment_id, model, final, None, config)
        repo.update_experiment(experiment_id, status="completed", timesteps_completed=final, error=None)
    finally:
        vector.close()
