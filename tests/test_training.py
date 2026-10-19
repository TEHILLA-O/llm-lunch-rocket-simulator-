"""API validation and a short PPO train/load/evaluate smoke test."""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from backend.api.app import app
from backend.storage.db import init_db
from backend.storage.repository import get_experiment, update_experiment
from backend.training.evaluate import aggregate, rollout
from backend.training.loop import train_experiment
from backend.controllers.agents import PPOController, RandomController


def test_api_rejects_terminal_reward_overrides_and_creates_an_experiment(monkeypatch):
    init_db()
    monkeypatch.setattr("backend.training.service.spawn_worker", lambda experiment_id: 4242)
    with TestClient(app) as client:
        rejected = client.post("/api/v1/experiments", json={"name": "bad", "crash_penalty": -1})
        assert rejected.status_code == 422
        created = client.post("/api/v1/experiments", json={"name": "baseline", "seed": 3, "timesteps": 2048})
        assert created.status_code == 200
        body = created.json()
        assert body["config"]["reward"]["landing_bonus"] == 100
        assert body["config"]["reward"]["crash_penalty"] == -1000
        assert body["config"]["reward"]["fuel_penalty"] is None
        assert body["config"]["reward"]["mode"] == "delta"
        assert body["status"] == "created"
        started = client.post(f"/api/v1/experiments/{body['id']}/start")
        assert started.status_code == 200
        assert started.json()["status"] == "running"
        assert started.json()["pid"] == 4242
        paused = client.post(f"/api/v1/experiments/{body['id']}/pause")
        assert paused.status_code == 409
        meta = client.get("/api/v1/meta")
        assert meta.json()["reward"]["landing_bonus"] == 100
        assert meta.json()["reward"]["crash_penalty"] == -1000


def test_comparison_stays_running_until_every_controller_finishes(monkeypatch):
    init_db()
    entered = threading.Event()
    release = threading.Event()

    def fake_rollout(*_args, **_kwargs):
        entered.set()
        assert release.wait(3)
        return (
            {
                "episodes": 1,
                "mean_reward": 1.0,
                "std_reward": 0.0,
                "success_rate": 0.0,
                "crash_rate": 0.0,
                "truncation_rate": 1.0,
                "mean_length": 1.0,
                "std_length": 0.0,
                "mean_final_distance_m": 1.0,
                "mean_final_speed_mps": 1.0,
                "timeout_near_pad": 0,
                "per_episode": [],
            },
            None,
        )

    monkeypatch.setattr("backend.training.compare.rollout", fake_rollout)
    from backend.api.schemas import CompareRequest
    from backend.training.compare import run_comparison

    body = CompareRequest(seeds=[1], controllers=[{"type": "random"}, {"type": "random"}])
    thread = threading.Thread(target=run_comparison, args=("group-test", body))
    thread.start()
    assert entered.wait(3)
    with TestClient(app) as client:
        running = client.get("/api/v1/comparisons/group-test").json()
    assert running["status"] == "running"
    assert len(running["results"]) == 2
    release.set()
    thread.join(3)
    with TestClient(app) as client:
        finished = client.get("/api/v1/comparisons/group-test").json()
    assert finished["status"] == "completed"
    assert {row["status"] for row in finished["results"]} == {"completed"}


def test_unknown_comparison_is_not_reported_as_running():
    init_db()
    with TestClient(app) as client:
        missing = client.get("/api/v1/comparisons/does-not-exist")
        assert missing.status_code == 404


def test_a_dead_worker_is_marked_failed_when_the_dashboard_reads_it():
    init_db()
    with TestClient(app) as client:
        created = client.post("/api/v1/experiments", json={"name": "dead", "timesteps": 2048}).json()
        update_experiment(created["id"], status="running", pid=999999)
        listed = client.get("/api/v1/experiments").json()
        row = next(item for item in listed if item["id"] == created["id"])
        assert row["status"] == "failed"
        assert "no longer running" in row["error"]


def test_pause_is_refused_when_the_worker_is_not_alive():
    init_db()
    with TestClient(app) as client:
        created = client.post("/api/v1/experiments", json={"name": "idle", "timesteps": 2048}).json()
        update_experiment(created["id"], status="running", pid=999999)
        response = client.post(f"/api/v1/experiments/{created['id']}/pause")
        assert response.status_code == 409


def test_smoke_ppo_trains_saves_loads_and_evaluates():
    """A tiny PPO run proves the pipeline. It is not evidence of landing skill."""

    init_db()
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/experiments",
            json={
                "name": "smoke",
                "seed": 7,
                "timesteps": 256,
                "n_steps": 128,
                "batch_size": 64,
                "auto_eval_episodes": 0,
                "max_steps": 80,
            },
        )
        assert created.status_code == 200
        experiment_id = created.json()["id"]
    experiment = get_experiment(experiment_id)
    config = experiment["config"]
    config["milestones"] = [0, 128, 256]
    config["ppo"]["n_epochs"] = 2
    config["auto_eval_episodes"] = 0
    update_experiment(experiment_id, config_json=json.dumps(config), timesteps_target=128)
    train_experiment(experiment_id)
    first = get_experiment(experiment_id)
    assert first["status"] == "completed"
    assert first["timesteps_completed"] >= 128
    update_experiment(experiment_id, timesteps_target=256, status="stopped")
    train_experiment(experiment_id)
    second = get_experiment(experiment_id)
    assert second["status"] == "completed"
    assert second["timesteps_completed"] >= 256
    with TestClient(app) as client:
        checkpoints = client.get(f"/api/v1/experiments/{experiment_id}/checkpoints").json()
        analytics = client.get(f"/api/v1/experiments/{experiment_id}/analytics").json()
    steps = [item["timesteps"] for item in checkpoints]
    assert 0 in steps
    assert second["timesteps_completed"] in steps
    assert analytics["episodes"], "training episodes should be stored from the real rollout"
    assert all(row["episode_reward"] is not None for row in analytics["episodes"])
    checkpoint = checkpoints[-1]
    controller = PPOController(checkpoint["path"], "smoke")
    stats, _payload = rollout(controller, second["config"], seeds=[1001])
    assert stats["episodes"] == 1
    assert stats["success_rate"] in (0.0, 1.0)
    action = controller.act(__import__("numpy").zeros(8, dtype="float32"))
    assert action in (0, 1, 2, 3)
    random_stats, _payload = rollout(RandomController(1), second["config"], seeds=[1001])
    summary = aggregate(random_stats["per_episode"])
    assert summary["crash_rate"] + summary["success_rate"] + summary["truncation_rate"] == 1
