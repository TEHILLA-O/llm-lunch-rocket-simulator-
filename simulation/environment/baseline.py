"""Optional Gymnasium LunarLander-v3 baseline.

LunarMind's own environment integrates the equations requested for this
project. LunarLander-v3 remains available as a reference task when Box2D is
installed. Its reward includes fuel costs, so it is a different experiment
and must not be mixed into a LunarMind reward curve.
"""

from __future__ import annotations


def lunarlander_available() -> tuple[bool, str]:
    try:
        import gymnasium as gym

        gym.spec("LunarLander-v3")
    except Exception as exc:  # noqa: BLE001 - availability probe
        return False, str(exc)
    try:
        import Box2D  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"LunarLander-v3 is registered but Box2D failed to import: {exc}"
    return True, "ok"


def make_lunarlander():
    ok, reason = lunarlander_available()
    if not ok:
        raise RuntimeError(f"LunarLander-v3 baseline is unavailable: {reason}")
    import gymnasium as gym

    return gym.make("LunarLander-v3")
