"""Pygame view of a LunarMind episode.

The window is optional. Training constructs the environment with
``render_mode=None`` and never imports this module. Pixel effects are drawn
from the scene description; they do not change the rocket state.
"""

from __future__ import annotations

import math

import numpy as np

from simulation.physics.geometry import terrain_height


class PygameView:
    def __init__(self, mode: str, width: int = 960, height: int = 600):
        import os

        import pygame

        if mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.pygame = pygame
        self.mode = mode
        self.width = width
        self.height = height
        pygame.init()
        if mode == "human":
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("LunarMind")
        else:
            self.screen = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("consolas", 16)
        self.clock = pygame.time.Clock()
        star_rng = np.random.default_rng(7)
        self._stars = [
            (int(star_rng.uniform(0, width)), int(star_rng.uniform(0, height * 0.72)), 1 + (i % 2))
            for i in range(90)
        ]

    def draw(self, env):
        pg = self.pygame
        from simulation.rendering.scene import build_scene

        scene = build_scene(env.state, env.contact, env.last_action, env.outcome, env.physics)
        surface = self.screen
        surface.fill((6, 10, 16))
        for x, y, r in self._stars:
            pg.draw.circle(surface, (180, 190, 205), (x, y), r)
        self._draw_world(surface, env, scene)
        self._draw_hud(surface, env, scene)
        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        array = pg.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

    def _camera(self, env):
        scale = 22.0
        cam_x = env.state.x
        return scale, cam_x

    def world(self, env, x, y):
        scale, cam_x = self._camera(env)
        sx = self.width / 2 + (x - cam_x) * scale
        sy = self.height * 0.78 - y * scale
        return int(sx), int(sy)

    def _draw_world(self, surface, env, scene):
        pg = self.pygame
        cfg = env.physics
        xs = np.linspace(env.state.x - 30, env.state.x + 30, 120)
        points = [self.world(env, float(x), terrain_height(float(x), cfg)) for x in xs]
        pg.draw.lines(surface, (120, 110, 100), False, points, 2)
        pad_l = self.world(env, cfg.pad_x - cfg.pad_half_width, 0)
        pad_r = self.world(env, cfg.pad_x + cfg.pad_half_width, 0)
        pg.draw.line(surface, (90, 210, 255), pad_l, pad_r, 4)
        self._draw_flames(surface, env, scene)
        for leg in scene["legs"]:
            pg.draw.line(surface, (210, 220, 230), self.world(env, *leg["hip"]), self.world(env, *leg["foot"]), 2)
        body = [self.world(env, *p) for p in scene["body"]]
        pg.draw.polygon(surface, (220, 228, 236), body)
        pg.draw.polygon(surface, (40, 50, 60), body, 2)
        if scene["outcome"] == "crashed":
            pg.draw.circle(surface, (255, 90, 90), self.world(env, scene["x"], scene["y"]), 18, 2)
        elif scene["outcome"] == "landed":
            pg.draw.circle(surface, (80, 220, 160), self.world(env, cfg.pad_x, 0.2), 22, 2)

    def _draw_flames(self, surface, env, scene):
        pg = self.pygame
        flicker = 0.85 + 0.15 * math.sin(pg.time.get_ticks() / 40.0)
        for name, color, length in (
            ("main", (255, 170, 70), 1.1),
            ("left", (120, 190, 255), 0.45),
            ("right", (120, 190, 255), 0.45),
        ):
            flame = scene["flames"][name]
            if not flame["active"]:
                continue
            ox, oy = flame["origin"]
            dx, dy = flame["direction"]
            end = (ox + dx * length * flicker, oy + dy * length * flicker)
            pg.draw.line(surface, color, self.world(env, ox, oy), self.world(env, *end), 3)

    def _draw_hud(self, surface, env, scene):
        if env.state is None:
            return
        lines = [
            f"alt {env.state.y:6.2f} m   vx {env.state.vx:6.2f}   vy {env.state.vy:6.2f}",
            f"tilt {env.state.angle:6.2f} rad   omega {env.state.omega:6.2f}",
            f"action {env.last_action}   reward {env.episode_reward:8.2f}",
        ]
        y = 12
        for line in lines:
            image = self.font.render(line, True, (210, 220, 230))
            surface.blit(image, (12, y))
            y += 20

    def poll_keys(self) -> dict[str, bool]:
        pg = self.pygame
        pressed = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return {"quit": True}
        return {
            "quit": False,
            "left": bool(pressed[pg.K_LEFT]),
            "right": bool(pressed[pg.K_RIGHT]),
            "main": bool(pressed[pg.K_SPACE]),
        }

    def close(self):
        self.pygame.quit()
