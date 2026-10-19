"""Start, pause, resume and stop training jobs."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from backend.paths import ROOT, logs_dir
from backend.storage.db import pid_alive
from backend.storage.repository import get_experiment, latest_checkpoint, update_experiment

logger = logging.getLogger("lunarmind.jobs")
_WORKERS: dict[int, tuple[subprocess.Popen, object]] = {}


def _reap_finished_workers() -> None:
    finished = [pid for pid, (process, _handle) in _WORKERS.items() if process.poll() is not None]
    for pid in finished:
        _process, handle = _WORKERS.pop(pid)
        handle.close()


def spawn_worker(experiment_id: str) -> int:
    _reap_finished_workers()
    log_path = logs_dir() / f"worker_{experiment_id}.log"
    handle = open(log_path, "a", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "backend.training.worker", "--experiment-id", experiment_id],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    except Exception:
        handle.close()
        raise
    # Keep the Popen and the log handle alive. Dropping them lets the log
    # file close while the worker is still writing to it.
    _WORKERS[process.pid] = (process, handle)
    logger.info("spawned worker pid %s for %s", process.pid, experiment_id)
    return process.pid


def start_job(experiment_id: str) -> dict:
    experiment = get_experiment(experiment_id)
    if experiment["status"] in ("running", "pause_requested") and pid_alive(experiment["pid"]):
        raise RuntimeError("Training is already running.")
    if experiment["status"] not in ("created", "failed", "stopped"):
        raise RuntimeError(f"Cannot start training from status {experiment['status']}.")
    pid = spawn_worker(experiment_id)
    return update_experiment(experiment_id, status="running", pid=pid, error=None)


def pause_job(experiment_id: str) -> dict:
    experiment = get_experiment(experiment_id)
    if experiment["status"] != "running" or not pid_alive(experiment["pid"]):
        raise RuntimeError("Training is not running, so there is nothing to pause.")
    return update_experiment(experiment_id, status="pause_requested")


def resume_job(experiment_id: str) -> dict:
    experiment = get_experiment(experiment_id)
    if experiment["status"] in ("paused", "pause_requested") and pid_alive(experiment["pid"]):
        return update_experiment(experiment_id, status="running")
    if experiment["status"] in ("paused", "stopped", "failed") and latest_checkpoint(experiment_id):
        pid = spawn_worker(experiment_id)
        return update_experiment(experiment_id, status="running", pid=pid, error=None)
    raise RuntimeError("There is no paused run or checkpoint to resume.")


def stop_job(experiment_id: str) -> dict:
    experiment = get_experiment(experiment_id)
    if experiment["status"] in ("running", "paused", "pause_requested") and pid_alive(experiment["pid"]):
        return update_experiment(experiment_id, status="stop_requested")
    if experiment["status"] in ("running", "paused", "pause_requested"):
        return update_experiment(experiment_id, status="stopped")
    raise RuntimeError("Training is not active.")
