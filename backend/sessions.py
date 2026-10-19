"""In-memory live flights. Completed trajectories are written to disk and SQLite."""

from __future__ import annotations

import threading
import time

from backend.controllers.agents import CoastController, HumanController, LLMController, PPOController, RandomController
from backend.paths import trajectories_dir
from backend.storage import repository as repo
from backend.training.config import make_env, physics_from_config, reward_from_config
from simulation import ENV_VERSION
from simulation.physics.config import PhysicsConfig
from simulation.physics.geometry import terrain_profile
from simulation.recording import summarise
from simulation.rewards.shaping import RewardConfig

DEMOS = {
    "safe": {"x": 0.0, "y": 1.15, "vx": 0.0, "vy": -0.05, "angle": 0.0, "omega": 0.0},
    "crash": {"x": 0.0, "y": 6.0, "vx": 0.2, "vy": -6.0, "angle": 0.0, "omega": 0.0},
    "tilt": {"x": 0.0, "y": 3.0, "vx": 0.0, "vy": -1.0, "angle": 1.1, "omega": 0.2},
}

SESSIONS: dict[str, "FlightSession"] = {}
MAX_SESSIONS = 4


class FlightSession:
    def __init__(self, session_id: str, controller, config: dict, seed: int, label: str, demo: str | None, speed: float):
        self.id = session_id
        self.controller = controller
        self.config = config
        self.seed = seed
        self.label = label
        self.demo = demo
        self.speed = speed
        self.env = make_env(config, record=True, controller_name=controller.name)
        self.physics = self.env.physics
        self.obs, self.info = self.env.reset(seed=seed, options=self._options())
        self.paused = False
        self.alive = True
        self.connected = False
        self.created_at = time.monotonic()
        self.done = False
        self.saved = False
        self.trail = [[self.env.state.x, self.env.state.y]]
        self.lock = threading.Lock()
        self.sim_debt = 0.0
        self.last_tick: float | None = None
        self.trajectory_id: str | None = None

    def _options(self):
        if self.demo and self.demo in DEMOS:
            return {"state": DEMOS[self.demo]}
        return None

    def hello(self) -> dict:
        return {
            "type": "hello",
            "session_id": self.id,
            "controller": self.controller.name,
            "model_label": self.label,
            "reward_mode": self.env.reward.mode,
            "reward_label": self.env.reward.label,
            "demo": self.demo,
            "dt": self.physics.dt,
            "terrain": terrain_profile(self.physics),
            "pad": {"x": self.physics.pad_x, "half_width": self.physics.pad_half_width},
            "env_version": ENV_VERSION,
        }

    def frame(self) -> dict:
        return {
            "type": "frame",
            "telemetry": self.info["telemetry"],
            "reward_parts": self.info["reward_parts"],
            "geometry": self.info["scene"],
            "cumulative_reward": self.env.episode_reward,
            "trail": self.trail,
            "done": self.done,
            "paused": self.paused,
            "speed": self.speed,
            "model_label": self.label,
        }

    def handle(self, message: dict) -> None:
        kind = message.get("type")
        with self.lock:
            if kind == "keys" and hasattr(self.controller, "set_keys"):
                self.controller.set_keys(list(message.get("pressed") or []))
            elif kind == "speed":
                self.speed = float(min(8.0, max(0.25, float(message["value"]))))
            elif kind == "pause":
                self.paused = bool(message.get("paused"))
                self.last_tick = None
                self.sim_debt = 0.0
            elif kind == "reset":
                self._reset_locked()

    def _reset_locked(self) -> None:
        if self.done and not self.saved:
            self.persist()
        self.controller.reset(self.seed)
        self.obs, self.info = self.env.reset(seed=self.seed, options=self._options())
        self.done = False
        self.saved = False
        self.trail = [[self.env.state.x, self.env.state.y]]
        self.sim_debt = 0.0
        self.last_tick = None

    def advance(self) -> None:
        with self.lock:
            if not self.alive or self.paused or self.done:
                self.last_tick = None
                self.sim_debt = 0.0
                return
            now = time.perf_counter()
            if self.last_tick is None:
                self.last_tick = now
                return
            elapsed = min(0.25, now - self.last_tick)
            self.last_tick = now
            self.sim_debt += elapsed * self.speed
            steps = 0
            while self.sim_debt >= self.physics.dt and steps < 20 and not self.done:
                self.sim_debt -= self.physics.dt
                self._step_locked()
                steps += 1
            if steps >= 20:
                self.sim_debt = 0.0
            if self.done:
                self.persist()

    def _step_locked(self) -> None:
        action = self.controller.act(self.obs, self.info.get("telemetry"))
        trace = self.controller.consume_trace()
        self.obs, _reward, terminated, truncated, self.info = self.env.step(action)
        if self.env.trajectory:
            self.env.trajectory[-1]["controller_trace"] = trace
        self.trail.append([self.env.state.x, self.env.state.y])
        self.trail = self.trail[-300:]
        self.done = bool(terminated or truncated)

    def persist(self) -> None:
        if self.saved:
            return
        acted = any(step.get("action") is not None for step in self.env.trajectory)
        if not acted:
            self.saved = True
            return
        outcome = "incomplete"
        if self.env.trajectory:
            last = self.env.trajectory[-1]
            if last.get("terminated"):
                outcome = last.get("outcome") or "crashed"
            elif last.get("truncated"):
                outcome = "truncated"
        trajectory_id = repo.new_id()
        path = trajectories_dir() / f"{trajectory_id}.json"
        payload = {
            "id": trajectory_id,
            "controller": self.controller.name,
            "label": self.label,
            "seed": self.seed,
            "env_version": ENV_VERSION,
            "demo": self.demo,
            "config": self.config,
            "outcome": outcome,
            "steps": self.env.trajectory,
            "terrain": terrain_profile(self.physics),
            "pad": {"x": self.physics.pad_x, "half_width": self.physics.pad_half_width},
        }
        payload["summary"] = summarise(self.env.trajectory, outcome)
        from simulation.recording import dump_json

        dump_json(payload, path)
        repo.insert_trajectory(
            {
                "id": trajectory_id,
                "controller": self.controller.name,
                "label": self.label,
                "outcome": outcome,
                "seed": self.seed,
                "path": str(path),
                "summary": payload["summary"],
            }
        )
        self.trajectory_id = trajectory_id
        self.saved = True


def default_config() -> dict:
    physics = PhysicsConfig()
    reward = RewardConfig()
    return {
        "env_version": ENV_VERSION,
        "physics": {
            "gravity": physics.gravity,
            "main_accel": physics.main_accel,
            "attitude_torque": physics.attitude_torque,
            "inertia": physics.inertia,
            "dt": physics.dt,
        },
        "reward": {
            "mode": reward.mode,
            "distance_weight": reward.distance_weight,
            "speed_weight": reward.speed_weight,
            "upright_weight": reward.upright_weight,
            "gamma": reward.gamma,
            "landing_bonus": 100,
            "crash_penalty": -1000,
            "fuel_penalty": None,
            "time_penalty": None,
        },
        "max_steps": 1000,
        "n_envs": 1,
        "ppo": {"n_steps": 2048, "batch_size": 256, "n_epochs": 10, "gamma": reward.gamma, "learning_rate": 3e-4, "ent_coef": 0.01, "net_arch": [64, 64], "device": "cpu"},
    }


def _drop_stale_sessions() -> None:
    now = time.monotonic()
    stale = [
        key
        for key, session in SESSIONS.items()
        if not session.alive or (not session.connected and now - session.created_at > 15)
    ]
    for key in stale:
        session = SESSIONS.pop(key, None)
        if session is not None:
            session.alive = False


def create_session(controller_name: str, seed: int, checkpoint_id: str | None, demo: str | None, speed: float) -> FlightSession:
    _drop_stale_sessions()
    if len(SESSIONS) >= MAX_SESSIONS:
        raise RuntimeError("Too many live flights. Close one before starting another.")
    config = default_config()
    label = controller_name
    if controller_name == "human":
        controller = HumanController()
        label = "Human"
    elif controller_name == "random":
        controller = RandomController(seed)
        label = "Random"
    elif controller_name == "ppo":
        if not checkpoint_id:
            raise ValueError("A PPO flight needs a checkpoint.")
        checkpoint = repo.get_checkpoint(checkpoint_id)
        experiment = repo.get_experiment(checkpoint["experiment_id"])
        config = experiment["config"]
        label = f"PPO · {experiment['name']} · {checkpoint['timesteps']} steps"
        controller = PPOController(checkpoint["path"], label)
    elif controller_name == "llm":
        controller = LLMController()
        label = "LLM baseline (not reinforcement learning)"
    elif controller_name == "demo":
        if demo not in DEMOS:
            raise ValueError("demo must be safe, crash or tilt")
        label = f"Physics demonstration ({demo}, not an AI)"
        controller = CoastController("The rocket coasts under gravity so the contact rules can be seen. This is not a learned policy.")
    else:
        raise ValueError(f"unknown controller {controller_name}")
    # Touch builders so a bad stored config fails before the socket opens.
    physics_from_config(config)
    reward_from_config(config)
    session = FlightSession(repo.new_id(), controller, config, seed, label, demo if controller_name == "demo" else None, speed)
    SESSIONS[session.id] = session
    return session
