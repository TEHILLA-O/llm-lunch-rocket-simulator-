"""Standalone pygame flight. The web dashboard is the full console.

Left and right arrows rotate. Space fires the main engine. The most recently
pressed key is the only thruster, matching the discrete action space.
R resets the episode. Escape closes the window.
"""

from __future__ import annotations

import argparse

from backend.controllers.agents import HumanController, RandomController
from simulation.environment.lunar_env import LunarMindEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["human", "random"], default="human")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    controller = HumanController() if args.controller == "human" else RandomController(args.seed)
    env = LunarMindEnv(render_mode="human", record=False, controller_name=controller.name)
    env.reset(seed=args.seed)
    env.render()
    running = True
    while running:
        keys = env._renderer.poll_keys()
        if keys["quit"]:
            break
        if args.controller == "human":
            pressed = []
            if keys["left"]:
                pressed.append("ArrowLeft")
            if keys["right"]:
                pressed.append("ArrowRight")
            if keys["main"]:
                pressed.append("Space")
            controller.set_keys(pressed)
        observation = env._observe()
        action = controller.act(observation, None)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset(seed=args.seed)
    env.close()


if __name__ == "__main__":
    main()
