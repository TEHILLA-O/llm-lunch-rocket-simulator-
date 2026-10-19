"""SQLite storage for experiments, checkpoints, evaluations and flights.

The database is the source of truth for dashboard charts. Rows are written
by the training worker and by live sessions. Restarting the API process
reopens the same file; it does not drop tables.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from backend.paths import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    env_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    timesteps_target INTEGER NOT NULL,
    timesteps_completed INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    timesteps INTEGER NOT NULL,
    path TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    checkpoint_id TEXT,
    controller TEXT NOT NULL,
    label TEXT NOT NULL,
    seed INTEGER NOT NULL,
    episodes INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    group_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    controller TEXT NOT NULL,
    label TEXT NOT NULL,
    outcome TEXT NOT NULL,
    seed INTEGER,
    path TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    timestep INTEGER NOT NULL,
    episode_reward REAL,
    episode_length INTEGER,
    outcome TEXT,
    final_distance_m REAL,
    final_speed_mps REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_experiment ON training_logs(experiment_id, timestep);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id, timesteps);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    path = data_dir() / "lunarmind.sqlite"
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def reconcile_interrupted_jobs() -> list[str]:
    """Mark dead workers as failed without deleting their checkpoints or logs."""

    changed: list[str] = []
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, pid, status FROM experiments
            WHERE status IN ('running', 'paused', 'pause_requested', 'stop_requested')
            """
        ).fetchall()
        for row in rows:
            if pid_alive(row["pid"]):
                continue
            status = "stopped" if row["status"] == "stop_requested" else "failed"
            message = (
                "Stop was requested and the training process is no longer running. Checkpoints already saved were kept."
                if status == "stopped"
                else "The training process is no longer running. Checkpoints already saved were kept."
            )
            conn.execute(
                "UPDATE experiments SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, message, utcnow(), row["id"]),
            )
            changed.append(row["id"])
    return changed
