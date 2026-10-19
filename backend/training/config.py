"""Build environments and experiment configurations from stored JSON."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from simulation import ENV_VERSION
from simulation.environment.lunar_env import LunarMindEnv
from simulation.physics.config import PhysicsConfig
from simulation.rewards.shaping import RewardConfig

MILESTONES = [0, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def build_experiment_config(data) -> dict:
    """Capture every value that changes the learning problem.

    Terminal rewards are intentionally absent: they are constants in the
    reward function. A request that tries to set them is rejected by the API.
    """

    n_envs = int(data.n_envs)
    n_steps = int(data.n_steps)
    batch_size = int(data.batch_size)
    if batch_size > n_steps * n_envs:
        raise ValueError("batch_size cannot exceed n_steps * n_envs")
    return {
        "env_version": ENV_VERSION,
        "physics": {
            "gravity": float(data.gravity),
            "main_accel": float(data.main_accel),
            "attitude_torque": float(data.attitude_torque),
            "inertia": float(data.inertia),
            "dt": 0.02,
        },
        "reward": {
            "mode": data.reward_mode,
            "distance_weight": float(data.distance_weight),
            "speed_weight": float(data.speed_weight),
            "upright_weight": float(data.upright_weight),
            "gamma": float(data.gamma),
            "landing_bonus": 100,
            "crash_penalty": -1000,
            "fuel_penalty": None,
            "time_penalty": None,
        },
        "max_steps": int(data.max_steps),
        "n_envs": n_envs,
        "milestones": list(MILESTONES),
        "auto_eval_episodes": int(data.auto_eval_episodes),
        "ppo": {
            "learning_rate": float(data.learning_rate),
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": 10,
            "gamma": float(data.gamma),
            "ent_coef": 0.01,
            "net_arch": [64, 64],
            "device": "cpu",
        },
        "versions": {
            "lunarmind_env": ENV_VERSION,
            "python": __import__("sys").version.split()[0],
            "numpy": package_version("numpy"),
            "gymnasium": package_version("gymnasium"),
            "stable-baselines3": package_version("stable-baselines3"),
            "torch": package_version("torch"),
        },
    }


def physics_from_config(config: dict) -> PhysicsConfig:
    physics = config["physics"]
    return PhysicsConfig(
        gravity=physics["gravity"],
        main_accel=physics["main_accel"],
        attitude_torque=physics["attitude_torque"],
        inertia=physics["inertia"],
        dt=physics.get("dt", 0.02),
    )


def reward_from_config(config: dict) -> RewardConfig:
    reward = config["reward"]
    return RewardConfig(
        mode=reward["mode"],
        distance_weight=reward["distance_weight"],
        speed_weight=reward["speed_weight"],
        upright_weight=reward["upright_weight"],
        gamma=reward["gamma"],
    )


def make_env(config: dict, *, record: bool = False, controller_name: str = "unassigned") -> LunarMindEnv:
    return LunarMindEnv(
        physics=physics_from_config(config),
        reward=reward_from_config(config),
        max_steps=int(config["max_steps"]),
        record=record,
        controller_name=controller_name,
    )


def make_vec_env(config: dict, seed: int):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make(rank: int):
        def _init():
            env = Monitor(make_env(config))
            env.reset(seed=int(seed) + rank)
            return env

        return _init

    count = int(config["n_envs"])
    return DummyVecEnv([make(rank) for rank in range(count)])


def same_task(left: dict, right: dict) -> bool:
    """True when two experiment configs share physics and the reward formula."""

    physics_keys = ("gravity", "main_accel", "attitude_torque", "inertia", "dt")
    reward_keys = ("mode", "distance_weight", "speed_weight", "upright_weight", "gamma")
    return all(left["physics"][key] == right["physics"][key] for key in physics_keys) and all(
        left["reward"][key] == right["reward"][key] for key in reward_keys
    )
