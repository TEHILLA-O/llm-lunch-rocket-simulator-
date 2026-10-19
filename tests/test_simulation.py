"""Physics, reward and environment tests."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from simulation.environment.lunar_env import LunarMindEnv
from simulation.physics.config import PhysicsConfig
from simulation.physics.dynamics import (
    ActuatorCommand,
    accelerations,
    discrete_to_command,
    integrate,
)
from simulation.physics.geometry import (
    RocketState,
    evaluate_contact,
    terrain_height,
    tilt_magnitude,
    to_world,
    wrap_angle,
)
from simulation.rewards.shaping import (
    CRASH_PENALTY,
    LANDING_BONUS,
    Metrics,
    RewardConfig,
    dense_closed_form,
    metrics_from_state,
    step_reward,
)


CFG = PhysicsConfig()


def test_gravity_only_matches_the_difference_equation():
    state = RocketState(1.0, 10.0, 0.2, 0.0, 0.0, 0.0)
    ax, ay, alpha = accelerations(state, ActuatorCommand(0.0, 0.0), CFG)
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(-CFG.gravity)
    assert alpha == pytest.approx(0.0)
    nxt = integrate(state, ActuatorCommand(0.0, 0.0), CFG)
    assert nxt.vx == pytest.approx(0.2)
    assert nxt.vy == pytest.approx(-CFG.gravity * CFG.dt)
    assert nxt.x == pytest.approx(state.x + nxt.vx * CFG.dt)
    assert nxt.y == pytest.approx(state.y + nxt.vy * CFG.dt)
    assert nxt.omega == pytest.approx(0.0)
    assert nxt.angle == pytest.approx(0.0)


def test_main_engine_follows_the_nose_and_adds_no_torque():
    upright = RocketState(0, 5, 0, 0, 0, 0.3)
    ax, ay, alpha = accelerations(upright, discrete_to_command(1), CFG)
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(-CFG.gravity + CFG.main_accel)
    assert alpha == pytest.approx(0.0)

    nose_left = RocketState(0, 5, 0, 0, math.pi / 2, 0)
    ax, ay, alpha = accelerations(nose_left, discrete_to_command(1), CFG)
    assert ax == pytest.approx(-CFG.main_accel)
    assert ay == pytest.approx(-CFG.gravity)
    assert alpha == pytest.approx(0.0)


def test_attitude_thrusters_are_opposite_pure_couples():
    state = RocketState(0, 5, 1.0, -0.5, 0.2, 0.0)
    left = discrete_to_command(2)
    right = discrete_to_command(3)
    _, _, alpha_left = accelerations(state, left, CFG)
    _, _, alpha_right = accelerations(state, right, CFG)
    assert alpha_left == pytest.approx(CFG.attitude_torque / CFG.inertia)
    assert alpha_right == pytest.approx(-CFG.attitude_torque / CFG.inertia)
    moved = integrate(state, left, CFG)
    assert moved.vx == pytest.approx(state.vx)
    assert moved.omega == pytest.approx(alpha_left * CFG.dt)
    assert moved.angle > state.angle


def test_fractional_command_is_available_for_a_future_continuous_mode():
    state = RocketState(0, 5, 0, 0, 0, 0)
    half = accelerations(state, ActuatorCommand(main=0.5, torque=0.0), CFG)
    full = accelerations(state, ActuatorCommand(main=1.0, torque=0.0), CFG)
    assert half[1] + CFG.gravity == pytest.approx(0.5 * (full[1] + CFG.gravity))


def test_fixed_timestep_is_deterministic_and_independent_of_repeat_calls():
    start = RocketState(0.4, 12, -0.2, 0.3, 0.1, -0.05)
    one = integrate(start, discrete_to_command(1), CFG)
    again = integrate(start, discrete_to_command(1), CFG)
    assert one == again


def test_position_changes_only_by_velocity_times_dt():
    state = RocketState(-2, 8, 3, -4, 0.4, 0.2)
    nxt = integrate(state, discrete_to_command(0), CFG)
    assert nxt.x - state.x == pytest.approx(nxt.vx * CFG.dt)
    assert nxt.y - state.y == pytest.approx(nxt.vy * CFG.dt)


def test_angular_wrap_identifies_the_branch_cut():
    assert wrap_angle(2 * math.pi + 0.2) == pytest.approx(0.2)
    assert wrap_angle(-2 * math.pi - 0.2) == pytest.approx(-0.2)
    assert tilt_magnitude(math.pi + 0.05) == pytest.approx(tilt_magnitude(-math.pi + 0.05))
    assert tilt_magnitude(3 * math.pi) == pytest.approx(math.pi)


def test_pad_is_flat_and_terrain_is_continuous_at_the_edge():
    edge = CFG.pad_half_width
    assert terrain_height(0.0, CFG) == 0.0
    assert terrain_height(edge, CFG) == 0.0
    assert terrain_height(edge + 1e-6, CFG) == pytest.approx(0.0, abs=1e-4)
    assert terrain_height(edge + 3.0, CFG) > 0.0


def _landable(**overrides) -> RocketState:
    values = dict(x=0.0, y=0.94, vx=0.0, vy=0.0, angle=0.0, omega=0.0)
    values.update(overrides)
    return RocketState(**values)


def test_safe_contact_is_a_landing_and_not_a_crash():
    contact, outcome = evaluate_contact(_landable(), CFG)
    assert contact.left and contact.right
    assert outcome == "landed"


def test_fast_descent_cannot_land():
    _, outcome = evaluate_contact(_landable(vy=-8.0), CFG)
    assert outcome == "crashed"


def test_horizontal_speed_cannot_land():
    _, outcome = evaluate_contact(_landable(vx=3.0), CFG)
    assert outcome == "crashed"


def test_dangerous_tilt_cannot_land():
    state = RocketState(0.0, 3.0, 0.0, 0.0, 1.1, 0.0)
    outcome = None
    for _ in range(80):
        _, outcome = evaluate_contact(state, CFG)
        if outcome is not None:
            break
        state = RocketState(state.x, state.y - 0.05, 0.0, 0.0, state.angle, 0.0)
    assert outcome == "crashed"


def test_spinning_touchdown_is_not_stable():
    _, outcome = evaluate_contact(_landable(omega=2.0), CFG)
    assert outcome == "crashed"


def test_off_pad_touchdown_is_a_crash():
    x = 10.0
    state = RocketState(x, 0.0, 0, 0, 0, 0)
    # Drop the centre of mass until a foot reaches the local terrain.
    for _ in range(20):
        _, outcome = evaluate_contact(state, CFG)
        if outcome is not None:
            break
        state = RocketState(state.x, state.y - 0.05, 0, 0, 0, 0)
    assert outcome == "crashed"


def test_single_foot_contact_is_a_crash():
    angle = 0.9
    offset = to_world(0, 0, angle, -0.72, -0.95)
    other = to_world(0, 0, angle, 0.72, -0.95)
    lower = offset if offset[1] < other[1] else other
    y = -lower[1] - 0.01
    state = RocketState(0.0, y, 0, 0, angle, 0)
    contact, outcome = evaluate_contact(state, CFG)
    assert contact.any_touch
    assert not (contact.left and contact.right and not contact.hull)
    assert outcome == "crashed"


def test_landing_and_crash_are_mutually_exclusive():
    for state in (
        _landable(),
        _landable(vy=-6),
        _landable(x=12, y=2),
        RocketState(0, 15, 0, 0, 0, 0),
    ):
        _, outcome = evaluate_contact(state, CFG)
        assert outcome in (None, "landed", "crashed")
        assert not (outcome == "landed" and outcome == "crashed")


def test_reward_formula_and_exact_terminal_values():
    prev = Metrics(10.0, 4.0, 0.5)
    curr = Metrics(8.0, 3.0, 0.25)
    parts = step_reward(prev, curr, None)
    assert parts.distance == pytest.approx(5 * (10 - 8))
    assert parts.speed == pytest.approx(3 * (4 - 3))
    assert parts.upright == pytest.approx(2 * (0.5 - 0.25))
    assert parts.landing_bonus == 0
    assert parts.crash_penalty == 0
    landed = step_reward(prev, curr, "landed")
    crashed = step_reward(prev, curr, "crashed")
    assert landed.landing_bonus == LANDING_BONUS == 100
    assert crashed.crash_penalty == CRASH_PENALTY == -1000
    assert landed.crash_penalty == 0
    assert crashed.landing_bonus == 0
    assert crashed.total == pytest.approx(parts.total - 1000)
    assert landed.total == pytest.approx(parts.total + 100)


def test_moving_away_faster_or_tilting_reduces_the_step_reward():
    prev = Metrics(5, 2, 0.2)
    worse = step_reward(prev, Metrics(6, 3, 0.4), None)
    assert worse.distance < 0
    assert worse.speed < 0
    assert worse.upright < 0


def test_wrapped_tilt_is_what_the_upright_term_uses():
    near_cut = RocketState(0, 5, 0, 0, math.pi + 0.05, 0)
    wrapped = RocketState(0, 5, 0, 0, -math.pi + 0.05, 0)
    assert metrics_from_state(near_cut).tilt == pytest.approx(metrics_from_state(wrapped).tilt)


def test_no_fuel_or_time_argument_exists_on_the_reward():
    params = set(inspect.signature(step_reward).parameters)
    assert "action" not in params
    assert "fuel" not in params
    assert "timestep" not in params
    assert "time" not in params
    first = step_reward(Metrics(3, 1, 0.2), Metrics(2, 1, 0.2), None)
    second = step_reward(Metrics(3, 1, 0.2), Metrics(2, 1, 0.2), None)
    assert first == second


def test_dense_reward_telescopes_so_oscillation_cancels():
    series = [
        Metrics(10, 3, 0.4),
        Metrics(7, 1, 0.1),
        Metrics(9, 2.5, 0.3),
        Metrics(6, 1.2, 0.05),
    ]
    total = 0.0
    for prev, curr in zip(series, series[1:]):
        total += step_reward(prev, curr, None).total
    assert total == pytest.approx(dense_closed_form(series[0], series[-1]))
    cycle = [Metrics(4, 1, 0.2), Metrics(1, 0, 0), Metrics(4, 1, 0.2)]
    cyc = sum(step_reward(a, b, None).total for a, b in zip(cycle, cycle[1:]))
    assert cyc == pytest.approx(0.0)


def test_potential_mode_is_optional_and_matches_delta_only_at_gamma_one():
    prev, curr = Metrics(4, 2, 0.4), Metrics(3, 1, 0.2)
    delta = step_reward(prev, curr, None, RewardConfig(mode="delta"))
    same = step_reward(prev, curr, None, RewardConfig(mode="potential", gamma=1.0))
    other = step_reward(prev, curr, None, RewardConfig(mode="potential", gamma=0.99))
    assert same == delta
    assert other != delta
    assert RewardConfig().mode == "delta"
    assert RewardConfig(mode="potential").label.startswith("potential-based")


def test_default_weights_match_the_requested_baseline():
    cfg = RewardConfig()
    assert (cfg.distance_weight, cfg.speed_weight, cfg.upright_weight) == (5, 3, 2)


def _env_from_state(state: RocketState, **kwargs) -> LunarMindEnv:
    env = LunarMindEnv(record=True, **kwargs)
    env.reset(seed=0, options={"state": {"x": 0, "y": 15, "vx": 0, "vy": 0, "angle": 0, "omega": 0}})
    env.state = state
    return env


def test_environment_landing_bonus_is_exactly_100():
    env = _env_from_state(RocketState(0, 0.97, 0, 0, 0, 0))
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["outcome"] == "landed"
    assert info["reward_parts"]["landing_bonus"] == 100
    assert info["reward_parts"]["crash_penalty"] == 0
    assert obs.shape == (8,)
    assert obs[6] == 1 and obs[7] == 1


def test_environment_crash_penalty_is_exactly_minus_1000():
    env = _env_from_state(RocketState(0, 0.97, 0, -8, 0, 0))
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["outcome"] == "crashed"
    assert info["reward_parts"]["crash_penalty"] == -1000
    assert info["reward_parts"]["landing_bonus"] == 0
    assert reward == pytest.approx(
        info["reward_parts"]["distance"]
        + info["reward_parts"]["speed"]
        + info["reward_parts"]["upright"]
        - 1000
    )


def test_step_reward_matches_formula_for_every_action_without_a_fuel_term():
    for action in range(4):
        env = _env_from_state(RocketState(2, 12, 0.3, -0.4, 0.2, -0.1))
        prev = metrics_from_state(env.state, env.physics)
        _, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        curr = metrics_from_state(env.state, env.physics)
        expected = 5 * (prev.distance - curr.distance) + 3 * (prev.speed - curr.speed) + 2 * (prev.tilt - curr.tilt)
        assert reward == pytest.approx(expected)
        assert info["reward_parts"]["landing_bonus"] == 0
        assert info["reward_parts"]["crash_penalty"] == 0


def test_truncation_is_not_a_landing_or_a_crash():
    env = LunarMindEnv(max_steps=5, record=True)
    env.reset(seed=0, options={"state": {"x": 0, "y": 20, "vx": 0, "vy": 0, "angle": 0, "omega": 0}})
    last = None
    for _ in range(5):
        last = env.step(0)
    obs, reward, terminated, truncated, info = last
    assert truncated and not terminated
    assert info["outcome"] is None
    assert info["reward_parts"]["landing_bonus"] == 0
    assert info["reward_parts"]["crash_penalty"] == 0
    assert info["telemetry"]["landing_status"] == "truncated"


def test_observation_scales_and_action_validation():
    env = LunarMindEnv()
    state = {"x": 10.0, "y": 20.0, "vx": -5.0, "vy": 2.5, "angle": 2 * math.pi + 0.5, "omega": -1.0}
    obs, info = env.reset(seed=1, options={"state": state})
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    tel = info["telemetry"]
    assert obs[0] == pytest.approx(tel["x_m"] / 20)
    assert obs[1] == pytest.approx(tel["altitude_m"] / 20)
    assert obs[2] == pytest.approx(tel["vx_mps"] / 10)
    assert obs[3] == pytest.approx(tel["vy_mps"] / 10)
    assert obs[4] == pytest.approx(0.5 / math.pi)
    assert obs[5] == pytest.approx(-1.0 / 2)
    assert list(obs[6:]) == [0, 0]
    with pytest.raises(ValueError):
        env.step(4)
    with pytest.raises(ValueError):
        env.step(-1)


def test_step_after_the_episode_ends_requires_reset():
    env = LunarMindEnv(max_steps=5)
    env.reset(seed=0)
    ended = False
    for _ in range(5):
        _obs, _reward, terminated, truncated, _info = env.step(0)
        ended = terminated or truncated
        if ended:
            break
    assert ended
    with pytest.raises(RuntimeError, match="reset the environment"):
        env.step(0)


def test_same_seed_same_actions_match_and_invalid_ground_contact_is_not_rewarded():
    def roll(seed):
        env = LunarMindEnv()
        obs, _ = env.reset(seed=seed)
        seen = [obs.tolist()]
        for action in (0, 1, 2, 3, 1, 0):
            obs, reward, done, trunc, info = env.step(action)
            seen.append((round(reward, 6), obs.tolist(), info["outcome"]))
            if done or trunc:
                break
        return seen

    assert roll(5) == roll(5)
    assert roll(5)[0] != roll(6)[0]


def test_recorded_trajectory_reward_is_consistent():
    env = LunarMindEnv(record=True, max_steps=30)
    env.reset(seed=2)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(1)
        done = terminated or truncated
    acted = [step for step in env.trajectory if step["action"] is not None]
    total = sum(step["reward"] for step in acted)
    assert acted[-1]["cumulative_reward"] == pytest.approx(total)
    for step in acted:
        parts = step["reward_parts"]
        assert step["reward"] == pytest.approx(sum(parts.values()))
        assert len(step["observation"]) == 8
        assert step["action"] in (0, 1, 2, 3)


def test_gymnasium_checker_accepts_the_environment():
    env = LunarMindEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_gentle_drop_lands_and_a_fast_drop_crashes():
    def finish(state):
        env = LunarMindEnv(max_steps=400)
        env.reset(seed=0, options={"state": state})
        info = {"outcome": None}
        for _ in range(400):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        return info

    landed = finish({"x": 0, "y": 1.15, "vx": 0, "vy": -0.05, "angle": 0, "omega": 0})
    assert landed["outcome"] == "landed"
    assert landed["reward_parts"]["landing_bonus"] == 100
    crashed = finish({"x": 0, "y": 6, "vx": 0.2, "vy": -6, "angle": 0, "omega": 0})
    assert crashed["outcome"] == "crashed"
    assert crashed["reward_parts"]["crash_penalty"] == -1000
    assert crashed["reward_parts"]["landing_bonus"] == 0


def test_rgb_array_render_uses_the_physics_state():
    env = LunarMindEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape[2] == 3
    env.step(1)
    fired = env.render()
    assert fired.shape == frame.shape
    env.close()


def test_main_flame_points_down_when_the_rocket_is_upright():
    from simulation.rendering.scene import build_scene

    state = RocketState(0, 5, 0, 0, 0, 0)
    contact, _ = evaluate_contact(state, CFG)
    scene = build_scene(state, contact, 1, None, CFG)
    assert scene["flames"]["main"]["active"] is True
    assert scene["flames"]["main"]["direction"][1] == pytest.approx(-1.0)
    assert scene["flames"]["left"]["active"] is False
    left = build_scene(state, contact, 2, None, CFG)
    assert left["flames"]["left"]["direction"][0] == pytest.approx(-1.0)
    right = build_scene(state, contact, 3, None, CFG)
    assert right["flames"]["right"]["direction"][0] == pytest.approx(1.0)
