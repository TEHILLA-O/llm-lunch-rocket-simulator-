"""Controllers that choose thruster actions.

The physics environment never calls these objects. A session or an
evaluation loop asks a controller for one discrete action, then calls
``env.step``. Language-model output is parsed as data. It is not executed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from simulation.physics.dynamics import ACTION_NAMES


class Controller:
    name = "base"
    label = "Controller"
    blocking = False

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        raise NotImplementedError

    def consume_trace(self) -> dict:
        return {}


class HumanController(Controller):
    """Discrete human control from held keys.

    Only one thruster can fire per step. The most recently pressed key wins
    and is held until that key is released. Left and right arrows are the
    attitude thrusters; space is the main engine.
    """

    name = "human"
    label = "Human"

    def __init__(self):
        self.order: list[str] = []
        self.trace: dict = {}

    def set_keys(self, pressed: list[str]) -> None:
        allowed = ["ArrowLeft", "ArrowRight", "Space"]
        current = [key for key in pressed if key in allowed]
        self.order = [key for key in self.order if key in current]
        for key in current:
            if key not in self.order:
                self.order.append(key)

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        key = self.order[-1] if self.order else None
        action = {"ArrowLeft": 2, "ArrowRight": 3, "Space": 1}.get(key, 0)
        self.trace = {"source": "keyboard", "key": key, "action_name": ACTION_NAMES[action]}
        return action

    def consume_trace(self) -> dict:
        return self.trace


class CoastController(Controller):
    """Holds the coast action. Used for physics demonstrations, not as an AI."""

    name = "demo"
    label = "Physics demonstration"

    def __init__(self, note: str):
        self.note = note
        self.trace = {"source": "physics-demo", "note": note}

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        self.trace = {"source": "physics-demo", "note": self.note, "action_name": "coast"}
        return 0

    def consume_trace(self) -> dict:
        return self.trace


class RandomController(Controller):
    name = "random"
    label = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trace = {}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        action = int(self.rng.integers(0, 4))
        self.trace = {"source": "random", "action_name": ACTION_NAMES[action]}
        return action

    def consume_trace(self) -> dict:
        return self.trace


class PPOController(Controller):
    name = "ppo"
    label = "PPO"

    def __init__(self, path: str, label: str):
        from stable_baselines3 import PPO

        self.path = path
        self.label = label
        self.model = PPO.load(path, device="cpu")
        self.trace = {}

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        action, _state = self.model.predict(observation, deterministic=True)
        value = int(np.asarray(action).reshape(-1)[0])
        self.trace = {"source": "ppo", "checkpoint": self.label, "action_name": ACTION_NAMES[value]}
        return value

    def consume_trace(self) -> dict:
        return self.trace


def parse_llm_action(text: str) -> int:
    """Accept only a JSON object with an integer action and an optional reason."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("language-model response was not JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("language-model response must be a JSON object")
    unexpected = set(data) - {"action", "reason"}
    if unexpected:
        raise ValueError(f"unexpected keys in language-model response: {sorted(unexpected)}")
    action = data.get("action")
    if type(action) is not int or action not in (0, 1, 2, 3):
        raise ValueError("action must be the integer 0, 1, 2 or 3")
    reason = data.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 240:
        raise ValueError("reason must be a short string")
    return action


class LLMController(Controller):
    """Prompted language-model baseline.

    This controller does not learn. A reinforcement-trained language model
    would need a separate training stack. Inference is slow, so the chosen
    action is held for several physics steps. Timeouts and invalid responses
    fall back to coasting (action 0). Token counts and latency are recorded
    on the controller trace and are not added to the rocket's reward.
    """

    name = "llm"
    label = "LLM (not RL)"
    blocking = True

    def __init__(self, hold_steps: int | None = None, timeout: float | None = None):
        self.hold_steps = hold_steps or int(os.environ.get("LUNARMIND_LLM_HOLD_STEPS", "10"))
        self.timeout = timeout or float(os.environ.get("LUNARMIND_LLM_TIMEOUT", "2.5"))
        self._held_action = 0
        self._remaining = 0
        self.trace = {}

    def reset(self, seed: int | None = None) -> None:
        self._remaining = 0
        self._held_action = 0

    def act(self, observation: np.ndarray, telemetry: dict | None = None) -> int:
        if self._remaining > 0:
            self._remaining -= 1
            self.trace = {
                "source": "llm",
                "held": True,
                "action_name": ACTION_NAMES[self._held_action],
                "fallback": False,
            }
            return self._held_action
        action, trace = self._decide(telemetry or {})
        self._held_action = action
        self._remaining = max(0, self.hold_steps - 1)
        self.trace = trace
        return action

    def consume_trace(self) -> dict:
        return self.trace

    def _decide(self, telemetry: dict) -> tuple[int, dict]:
        started = time.perf_counter()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return 0, self._trace(0, True, "llm_unconfigured", started, None, "")
        payload = {
            "x_m": telemetry.get("x_m"),
            "altitude_m": telemetry.get("altitude_m"),
            "vx_mps": telemetry.get("vx_mps"),
            "vy_mps": telemetry.get("vy_mps"),
            "tilt_deg": telemetry.get("tilt_deg"),
            "omega_radps": telemetry.get("omega_radps"),
            "left_contact": telemetry.get("left_contact"),
            "right_contact": telemetry.get("right_contact"),
        }
        prompt = (
            "Select one rocket thruster. You are an experimental baseline, not a reinforcement-trained model. "
            "Actions: 0 coast, 1 fire the main engine along the nose, "
            "2 fire the left attitude thruster (positive counter-clockwise torque, nose moves toward -x), "
            "3 fire the right attitude thruster (negative clockwise torque). "
            'Reply with only JSON of the form {"action": 0, "reason": "short"}. '
            f"Telemetry: {json.dumps(payload)}"
        )
        body = {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You output only a JSON object. You do not write code."},
                {"role": "user", "content": prompt},
            ],
        }
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        try:
            import httpx

            response = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            content = data["choices"][0]["message"]["content"]
            action = parse_llm_action(content)
            usage = data.get("usage") or {}
            tokens = {
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
            }
            return action, self._trace(action, False, "", started, tokens, content[:500])
        except Exception as exc:  # noqa: BLE001 - fallback must be total
            return 0, self._trace(0, True, exc.__class__.__name__, started, None, "")

    def _trace(self, action: int, fallback: bool, error: str, started: float, tokens, raw: str) -> dict:
        return {
            "source": "llm",
            "held": False,
            "fallback": fallback,
            "error": error,
            "action_name": ACTION_NAMES[action],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "tokens": tokens,
            "raw_response": raw,
            "note": "Prompted language-model baseline. This is not reinforcement learning.",
        }


def llm_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())
