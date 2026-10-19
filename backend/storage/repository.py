"""Experiment queries."""

from __future__ import annotations

import json
import uuid
from typing import Any

from backend.storage.db import connect, utcnow


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def insert_experiment(name: str, seed: int, timesteps: int, env_version: str, config: dict) -> dict:
    experiment_id = new_id()
    now = utcnow()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                id, name, seed, status, env_version, config_json,
                timesteps_target, timesteps_completed, created_at, updated_at
            ) VALUES (?, ?, ?, 'created', ?, ?, ?, 0, ?, ?)
            """,
            (experiment_id, name, seed, env_version, json.dumps(config), timesteps, now, now),
        )
    return get_experiment(experiment_id)


def get_experiment(experiment_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError(experiment_id)
    return _experiment(row)


def list_experiments() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [_experiment(row) for row in rows]


def update_experiment(experiment_id: str, **fields: Any) -> dict:
    if not fields:
        return get_experiment(experiment_id)
    fields["updated_at"] = utcnow()
    keys = list(fields)
    assignments = ", ".join(f"{key} = ?" for key in keys)
    with connect() as conn:
        conn.execute(
            f"UPDATE experiments SET {assignments} WHERE id = ?",
            [fields[key] for key in keys] + [experiment_id],
        )
    return get_experiment(experiment_id)


def insert_checkpoint(experiment_id: str, timesteps: int, path: str, metrics: dict | None = None) -> dict:
    checkpoint_id = new_id()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO checkpoints (id, experiment_id, timesteps, path, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (checkpoint_id, experiment_id, timesteps, path, json.dumps(metrics) if metrics else None, utcnow()),
        )
    return get_checkpoint(checkpoint_id)


def get_checkpoint(checkpoint_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
    if row is None:
        raise KeyError(checkpoint_id)
    return _checkpoint(row)


def list_checkpoints(experiment_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE experiment_id = ? ORDER BY timesteps ASC",
            (experiment_id,),
        ).fetchall()
    return [_checkpoint(row) for row in rows]


def latest_checkpoint(experiment_id: str) -> dict | None:
    rows = list_checkpoints(experiment_id)
    return rows[-1] if rows else None


def insert_training_log(experiment_id: str, timestep: int, reward: float, length: int, outcome: str, distance: float | None, speed: float | None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO training_logs (
                experiment_id, timestep, episode_reward, episode_length, outcome,
                final_distance_m, final_speed_mps, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, timestep, reward, length, outcome, distance, speed, utcnow()),
        )


def training_logs(experiment_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT timestep, episode_reward, episode_length, outcome, final_distance_m, final_speed_mps
            FROM training_logs WHERE experiment_id = ? ORDER BY timestep ASC, id ASC
            """,
            (experiment_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_evaluation(payload: dict) -> dict:
    evaluation_id = payload.get("id") or new_id()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO evaluations (
                id, experiment_id, checkpoint_id, controller, label, seed, episodes,
                metrics_json, group_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                payload.get("experiment_id"),
                payload.get("checkpoint_id"),
                payload["controller"],
                payload["label"],
                payload.get("seed", 0),
                payload.get("episodes", 0),
                json.dumps(payload.get("metrics", {})),
                payload.get("group_id"),
                payload.get("status", "completed"),
                utcnow(),
            ),
        )
    return get_evaluation(evaluation_id)


def update_evaluation(evaluation_id: str, **fields: Any) -> dict:
    if "metrics" in fields:
        fields["metrics_json"] = json.dumps(fields.pop("metrics"))
    keys = list(fields)
    assignments = ", ".join(f"{key} = ?" for key in keys)
    with connect() as conn:
        conn.execute(
            f"UPDATE evaluations SET {assignments} WHERE id = ?",
            [fields[key] for key in keys] + [evaluation_id],
        )
    return get_evaluation(evaluation_id)


def get_evaluation(evaluation_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    if row is None:
        raise KeyError(evaluation_id)
    return _evaluation(row)


def list_evaluations(experiment_id: str | None = None, group_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM evaluations"
    clauses = []
    args: list[Any] = []
    if experiment_id:
        clauses.append("experiment_id = ?")
        args.append(experiment_id)
    if group_id:
        clauses.append("group_id = ?")
        args.append(group_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at ASC"
    with connect() as conn:
        rows = conn.execute(query, args).fetchall()
    return [_evaluation(row) for row in rows]


def insert_trajectory(payload: dict) -> dict:
    trajectory_id = payload.get("id") or new_id()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO trajectories (
                id, experiment_id, controller, label, outcome, seed, path, summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory_id,
                payload.get("experiment_id"),
                payload["controller"],
                payload["label"],
                payload["outcome"],
                payload.get("seed"),
                payload["path"],
                json.dumps(payload["summary"]),
                utcnow(),
            ),
        )
    return get_trajectory_row(trajectory_id)


def get_trajectory_row(trajectory_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM trajectories WHERE id = ?", (trajectory_id,)).fetchone()
    if row is None:
        raise KeyError(trajectory_id)
    return _trajectory(row)


def list_trajectories() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM trajectories ORDER BY created_at DESC").fetchall()
    return [_trajectory(row) for row in rows]


def _loads(value: str | None, fallback):
    if not value:
        return fallback
    return json.loads(value)


def _experiment(row: sqlite3_row) -> dict:  # type: ignore[name-defined]
    item = dict(row)
    item["config"] = _loads(item.pop("config_json"), {})
    return item


def _checkpoint(row) -> dict:
    item = dict(row)
    item["metrics"] = _loads(item.pop("metrics_json"), None)
    return item


def _evaluation(row) -> dict:
    item = dict(row)
    item["metrics"] = _loads(item.pop("metrics_json"), {})
    return item


def _trajectory(row) -> dict:
    item = dict(row)
    item["summary"] = _loads(item.pop("summary_json"), {})
    return item
