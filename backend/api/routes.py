"""Versioned HTTP and websocket routes."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response

from backend.controllers.agents import llm_configured
from backend.paths import trajectories_dir
from backend.sessions import SESSIONS, create_session
from backend.storage import repository as repo
from backend.storage.db import reconcile_interrupted_jobs
from backend.training.compare import begin_comparison, finish_comparison
from backend.training.config import build_experiment_config
from backend.training.service import pause_job, resume_job, start_job, stop_job
from backend.api.schemas import CompareRequest, ExperimentCreate, SessionCreate
from simulation import ENV_VERSION
from simulation.environment.baseline import lunarlander_available
from simulation.physics.dynamics import ACTION_NAMES
from simulation.recording import to_csv
from simulation.rewards.shaping import public_reward_spec

router = APIRouter(prefix="/api/v1")


def _public_experiment(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "seed": row["seed"],
        "status": row["status"],
        "env_version": row["env_version"],
        "config": row["config"],
        "timesteps_target": row["timesteps_target"],
        "timesteps_completed": row["timesteps_completed"],
        "pid": row["pid"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.get("/health")
def health():
    return {"status": "ok", "env_version": ENV_VERSION}


@router.get("/meta")
def meta():
    available, reason = lunarlander_available()
    return {
        "env_version": ENV_VERSION,
        "reward": public_reward_spec(),
        "actions": [
            {"id": 0, "name": ACTION_NAMES[0], "effect": "No thrust. Gravity and existing momentum act alone."},
            {"id": 1, "name": ACTION_NAMES[1], "effect": "Main engine accelerates the rocket along its nose."},
            {
                "id": 2,
                "name": ACTION_NAMES[2],
                "effect": "Left attitude thruster applies positive counter-clockwise torque. The nose moves toward -x.",
            },
            {
                "id": 3,
                "name": ACTION_NAMES[3],
                "effect": "Right attitude thruster applies negative clockwise torque. The nose moves toward +x.",
            },
        ],
        "observation": [
            "x / 20 m, horizontal position relative to the pad",
            "y / 20 m, altitude",
            "vx / 10 m/s, horizontal velocity",
            "vy / 10 m/s, vertical velocity, positive upward",
            "wrapped angle / pi, 0 upright, positive nose-left",
            "angular velocity / 2 rad/s, positive counter-clockwise",
            "left foot contact, 0 or 1",
            "right foot contact, 0 or 1",
        ],
        "collision": (
            "Any foot or hull contact ends the episode. A landing requires both feet on the pad, "
            "hull clear of the terrain, horizontal speed <= 1 m/s, descent speed <= 1.5 m/s, "
            "ascent speed <= 0.5 m/s, tilt <= 0.30 rad and angular speed <= 0.40 rad/s. "
            "Every other contact is a crash. Leaving the operational volume or reaching the step limit truncates the episode."
        ),
        "llm_configured": llm_configured(),
        "lunarlander_baseline": {"available": available, "detail": reason},
    }


@router.get("/experiments")
def experiments():
    reconcile_interrupted_jobs()
    return [_public_experiment(row) for row in repo.list_experiments()]


@router.post("/experiments")
def create_experiment(body: ExperimentCreate):
    try:
        config = build_experiment_config(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = repo.insert_experiment(body.name, body.seed, body.timesteps, ENV_VERSION, config)
    return _public_experiment(row)


@router.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str):
    reconcile_interrupted_jobs()
    try:
        return _public_experiment(repo.get_experiment(experiment_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None


@router.post("/experiments/{experiment_id}/start")
def experiment_start(experiment_id: str):
    try:
        return _public_experiment(start_job(experiment_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/pause")
def experiment_pause(experiment_id: str):
    try:
        return _public_experiment(pause_job(experiment_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/resume")
def experiment_resume(experiment_id: str):
    try:
        return _public_experiment(resume_job(experiment_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/stop")
def experiment_stop(experiment_id: str):
    try:
        return _public_experiment(stop_job(experiment_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/experiments/{experiment_id}/checkpoints")
def experiment_checkpoints(experiment_id: str):
    try:
        repo.get_experiment(experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    return repo.list_checkpoints(experiment_id)


@router.get("/experiments/{experiment_id}/analytics")
def experiment_analytics(experiment_id: str):
    reconcile_interrupted_jobs()
    try:
        experiment = repo.get_experiment(experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Experiment not found.") from None
    logs = repo.training_logs(experiment_id)
    window = 20

    def series(predicate) -> list[dict]:
        points = []
        for index, row in enumerate(logs):
            chunk = logs[max(0, index - window + 1) : index + 1]
            points.append(
                {
                    "timestep": row["timestep"],
                    "value": sum(1 for item in chunk if predicate(item)) / len(chunk),
                }
            )
        return points

    rolling = []
    for index, row in enumerate(logs):
        chunk = logs[max(0, index - window + 1) : index + 1]
        rolling.append(
            {
                "timestep": row["timestep"],
                "value": sum(item["episode_reward"] or 0.0 for item in chunk) / len(chunk),
            }
        )
    landed = [row for row in logs if row["outcome"] == "landed" and row["final_speed_mps"] is not None]
    checkpoints = {item["id"]: item for item in repo.list_checkpoints(experiment_id)}
    evaluations = []
    for evaluation in repo.list_evaluations(experiment_id):
        checkpoint = checkpoints.get(evaluation["checkpoint_id"])
        evaluations.append(
            {
                **evaluation,
                "timesteps": None if checkpoint is None else checkpoint["timesteps"],
            }
        )
    return {
        "experiment": _public_experiment(experiment),
        "episodes": logs,
        "rolling_reward": rolling,
        "success_rate": series(lambda item: item["outcome"] == "landed"),
        "crash_rate": series(lambda item: item["outcome"] == "crashed"),
        "mean_episode_length": _mean(logs, "episode_length"),
        "mean_landing_speed_mps": _mean(landed, "final_speed_mps"),
        "mean_final_distance_m": _mean(logs, "final_distance_m"),
        "timesteps_completed": experiment["timesteps_completed"],
        "evaluations": evaluations,
        "landing_count": len(landed),
    }


def _mean(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


@router.get("/checkpoints")
def all_checkpoints():
    rows = []
    for experiment in repo.list_experiments():
        for checkpoint in repo.list_checkpoints(experiment["id"]):
            rows.append({**checkpoint, "experiment_name": experiment["name"], "experiment_status": experiment["status"]})
    return rows


@router.post("/sessions")
def open_session(body: SessionCreate):
    try:
        session = create_session(body.controller, body.seed, body.checkpoint_id, body.demo, body.speed)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": session.id, "websocket": f"/api/v1/ws/sessions/{session.id}", "model_label": session.label}


@router.get("/trajectories")
def trajectories():
    return repo.list_trajectories()


def _trajectory_file(trajectory_id: str) -> tuple[dict, dict]:
    try:
        row = repo.get_trajectory_row(trajectory_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Trajectory not found.") from None
    path = Path(row["path"]).resolve()
    root = trajectories_dir().resolve()
    if root != path.parent:
        raise HTTPException(status_code=400, detail="Trajectory path is outside the data directory.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Trajectory file is missing.")
    return row, json.loads(path.read_text(encoding="utf-8"))


@router.get("/trajectories/{trajectory_id}")
def trajectory_detail(trajectory_id: str):
    _row, payload = _trajectory_file(trajectory_id)
    return payload


@router.get("/trajectories/{trajectory_id}/export.json")
def trajectory_json(trajectory_id: str):
    _row, payload = _trajectory_file(trajectory_id)
    return Response(content=json.dumps(payload), media_type="application/json")


@router.get("/trajectories/{trajectory_id}/export.csv")
def trajectory_csv(trajectory_id: str):
    _row, payload = _trajectory_file(trajectory_id)
    return PlainTextResponse(to_csv(payload), media_type="text/csv")


def _run_comparison(group_id: str, body: CompareRequest, pending) -> None:
    finish_comparison(group_id, body, pending)


@router.post("/comparisons")
def start_comparison(body: CompareRequest):
    group_id = repo.new_id()
    pending = begin_comparison(group_id, body)
    thread = threading.Thread(
        target=_run_comparison,
        args=(group_id, body, pending),
        daemon=True,
        name=f"compare-{group_id}",
    )
    thread.start()
    return {"group_id": group_id, "status": "running"}


@router.get("/comparisons/{group_id}")
def comparison_status(group_id: str):
    rows = repo.list_evaluations(group_id=group_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    status = "completed" if all(row["status"] in ("completed", "failed") for row in rows) else "running"
    return {"group_id": group_id, "status": status, "results": rows}


@router.get("/overview")
def overview():
    reconcile_interrupted_jobs()
    rows = [_public_experiment(row) for row in repo.list_experiments()]
    active = next((row for row in rows if row["status"] in ("running", "paused", "pause_requested")), rows[0] if rows else None)
    evaluations = repo.list_evaluations()
    latest_eval = evaluations[-1] if evaluations else None
    return {
        "experiments": rows,
        "active_experiment": active,
        "latest_evaluation": latest_eval,
        "trajectory_count": len(repo.list_trajectories()),
        "llm_configured": llm_configured(),
    }


@router.websocket("/ws/sessions/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    session.connected = True
    await websocket.send_json(session.hello())

    async def reader():
        while session.alive:
            message = await websocket.receive_json()
            session.handle(message)

    async def writer():
        while session.alive:
            await asyncio.to_thread(session.advance)
            await websocket.send_json(session.frame())
            await asyncio.sleep(0.05)

    try:
        await asyncio.gather(reader(), writer())
    except WebSocketDisconnect:
        pass
    finally:
        session.alive = False
        session.persist()
        SESSIONS.pop(session_id, None)


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_socket(websocket: WebSocket, experiment_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                experiment = _public_experiment(repo.get_experiment(experiment_id))
            except KeyError:
                await websocket.send_json({"type": "error", "message": "Experiment not found."})
                return
            logs = repo.training_logs(experiment_id)[-30:]
            await websocket.send_json({"type": "progress", "experiment": experiment, "recent_episodes": logs})
            if experiment["status"] in ("completed", "failed", "stopped"):
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
