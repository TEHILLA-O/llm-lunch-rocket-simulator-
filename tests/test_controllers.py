"""Controller and session tests that do not train a policy."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.controllers.agents import HumanController, RandomController, parse_llm_action
from backend.sessions import create_session
from backend.storage.db import init_db
from simulation.recording import to_csv


def test_human_uses_the_most_recent_key_and_holds_it():
    human = HumanController()
    obs = np.zeros(8, dtype=np.float32)
    assert human.act(obs) == 0
    human.set_keys(["ArrowLeft"])
    assert human.act(obs) == 2
    human.set_keys(["ArrowLeft", "Space"])
    assert human.act(obs) == 1
    human.set_keys(["ArrowLeft"])
    assert human.act(obs) == 2
    human.set_keys(["ArrowLeft", "ArrowRight"])
    assert human.act(obs) == 3
    human.set_keys([])
    assert human.act(obs) == 0


def test_random_controller_is_seeded():
    obs = np.zeros(8, dtype=np.float32)
    first = [RandomController(4).act(obs) for _ in range(6)]
    second = [RandomController(4).act(obs) for _ in range(6)]
    assert first == second
    assert set(first).issubset({0, 1, 2, 3})


def test_llm_parser_rejects_anything_except_a_thruster_choice():
    assert parse_llm_action(json.dumps({"action": 2, "reason": "nose is right"})) == 2
    assert parse_llm_action(json.dumps({"action": 0})) == 0
    with pytest.raises(ValueError):
        parse_llm_action('{"action": 1, "reason": "ok", "code": "import os"}')
    with pytest.raises(ValueError):
        parse_llm_action('{"action": true}')
    with pytest.raises(ValueError):
        parse_llm_action('{"action": 4}')
    with pytest.raises(ValueError):
        parse_llm_action("print('hello')")


def test_live_session_records_a_real_physics_step_and_exports():
    init_db()
    session = create_session("demo", seed=1, checkpoint_id=None, demo="safe", speed=1)
    session.last_tick = 0.0
    session.advance()
    assert session.env.steps > 0
    telemetry = session.frame()["telemetry"]
    for key in ("x_m", "altitude_m", "vx_mps", "vy_mps", "speed_mps", "tilt_deg", "omega_radps", "action_name", "step_reward", "episode_reward", "distance_m", "landing_status", "controller"):
        assert key in telemetry
    parts = session.frame()["reward_parts"]
    assert set(parts) == {"distance", "speed", "upright", "landing_bonus", "crash_penalty"}
    assert "fuel" not in parts
    session.done = True
    session.persist()
    assert session.trajectory_id
    from backend.storage.repository import get_trajectory_row
    import json
    from pathlib import Path

    row = get_trajectory_row(session.trajectory_id)
    payload = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
    csv_text = to_csv(payload)
    assert "crash_penalty" in csv_text
    assert "landing_bonus" in csv_text
    acted = [step for step in payload["steps"] if step["action"] is not None]
    assert acted[-1]["cumulative_reward"] == pytest.approx(sum(step["reward"] for step in acted))
