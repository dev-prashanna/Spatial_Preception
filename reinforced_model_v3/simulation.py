import argparse
import json
import math
import random
import sys
import time

import numpy as np

try:
    import pygame
except ImportError:
    print("pygame is not installed. Run:  pip install pygame")
    sys.exit(1)

BG          = (245, 245, 240)
GRID_LINE   = (210, 210, 205)
WALL_FILL   = (220,  90,  90)
WALL_BORDER = (163,  45,  45)
GOAL_FILL   = (186, 117,  23)
GOAL_ZONE   = (186, 117,  23,  40)
ROBOT_FILL  = ( 59, 109,  17)
ROBOT_RING  = ( 39,  80,  10)
RAY_CENTER  = ( 24,  95, 165)
RAY_SIDE    = ( 90, 160, 220)
HIT_DOT     = ( 24,  95, 165)
ARROW       = ( 23,  52,   4)
TEXT_DARK   = ( 30,  30,  30)
TEXT_LIGHT  = (245, 245, 240)
OVERLAY_BG  = (  0,   0,   0, 160)
STATUS_BG   = ( 30,  30,  30, 200)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)


def net_forward(layers, x):
    x = np.array(x, dtype=np.float32)
    for i, layer in enumerate(layers):
        W = np.array(layer["W"], dtype=np.float32)
        b = np.array(layer["b"], dtype=np.float32)
        x = W @ x + b
        if i < len(layers) - 1:
            x = leaky_relu(x)
    return int(np.argmax(x))


def load_weights(path):
    with open(path) as f:
        return json.load(f)


class RobotEnv:

    GRID      = 10
    RAY_STEP  = 0.5
    MAX_STEPS = 200

    def __init__(self):
        self.world = np.zeros((self.GRID, self.GRID), dtype=np.int32)
        self.world[5, :] = 1
        self.world[5, 4] = 0
        self.world[5, 5] = 0
        self.goal_x = 8.0
        self.goal_y = 8.0
        self.reset()

    def reset(self):
        self.x     = 1.0
        self.y     = 1.0
        self.theta = 0.0
        self.current_steps = 0
        self._reason = None
        return self.get_state()

    def step(self, action):
        self.current_steps += 1
        old_dist = self.goal_distance()

        if action == 0:
            self.x += math.cos(self.theta) * 0.3
            self.y += math.sin(self.theta) * 0.3
        elif action == 1:
            self.theta += 0.15
        elif action == 2:
            self.theta -= 0.15

        reward = self._get_reward(old_dist)
        done   = self._is_done()
        if done:
            self._reason = self.termination_reason()
        return self.get_state(), reward, done

    def _ray_distance(self, angle):
        x, y  = self.x, self.y
        dx    = math.cos(angle)
        dy    = math.sin(angle)
        dist  = 0.0
        while True:
            x    += dx * self.RAY_STEP
            y    += dy * self.RAY_STEP
            dist += self.RAY_STEP
            ix, iy = int(x), int(y)
            if ix < 0 or iy < 0 or ix >= self.GRID or iy >= self.GRID:
                break
            if self.world[iy][ix] == 1:
                break
        return dist, x, y

    def _ray_angles(self):
        return [self.theta, self.theta + 0.5, self.theta - 0.5]

    def get_rays(self):
        return [self._ray_distance(a) for a in self._ray_angles()]

    def get_state(self):
        rays   = self.get_rays()
        dx     = self.goal_x - self.x
        dy     = self.goal_y - self.y
        theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))
        return np.array([
            rays[0][0] / 10.0,
            rays[1][0] / 10.0,
            rays[2][0] / 10.0,
            dx / 10.0,
            dy / 10.0,
            theta / math.pi,
        ], dtype=np.float32)

    def goal_distance(self):
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def is_collision(self):
        ix, iy = int(self.x), int(self.y)
        if ix < 0 or iy < 0 or ix >= self.GRID or iy >= self.GRID:
            return True
        return self.world[iy][ix] == 1

    def is_goal_reached(self):
        return self.goal_distance() < 0.8

    def _is_done(self):
        return self.is_goal_reached() or self.is_collision() or \
               self.current_steps >= self.MAX_STEPS

    def _get_reward(self, old_dist):
        if self.is_goal_reached():
            return 10.0
        if self.is_collision():
            return -10.0
        progress = old_dist - self.goal_distance()
        return 2 * progress - 0.01

    def termination_reason(self):
        if self.is_goal_reached():
            return "goal"
        if self.is_collision():
            return "collision"
        if self.current_steps >= self.MAX_STEPS:
            return "timeout"
        return None


class Visualizer:

    CELL     = 56
    SIDE_W   = 240
    PAD      = 10
    FPS_AUTO = 12

    def __init__(self, weights=None, auto=False):
        pygame.init()
        pygame.display.set_caption("Robot Environment — DQN Visualizer")

        grid_px     = self.CELL * 10
        self.W      = grid_px + self.SIDE_W
        self.H      = grid_px + 60
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock  = pygame.time.Clock()

        self.env     = RobotEnv()
        self.state   = self.env.reset()
        self.done    = False
        self.reason  = None
        self.total_r = 0.0
        self.ep      = 0

        self.reward_history = []

        self.auto_run  = auto
        self.use_net   = auto and (weights is not None)
        self.weights   = weights

        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_md = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 22, bold=True)

        r = int(0.8 * self.CELL)
        self.goal_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.goal_surf, (186, 117, 23, 45), (r, r), r)

        self.overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        self._last_step = time.time()

    def _px(self, gx, gy=None):
        if gy is None:
            gx, gy = gx
        return int(gx * self.CELL), int(gy * self.CELL)

    def _select_action(self):
        if self.use_net and self.weights:
            return net_forward(self.weights, self.state)
        return random.randint(0, 2)

    def _step_env(self, action):
        self.state, r, self.done = self.env.step(action)
        self.total_r += r
        if self.done:
            self.reason = self.env.termination_reason()
            self.reward_history.append(self.total_r)

    def _reset(self):
        self.state   = self.env.reset()
        self.done    = False
        self.reason  = None
        self.total_r = 0.0
        self.ep     += 1

    def _draw_world(self):
        surf = self.screen
        env  = self.env

        for r in range(10):
            for c in range(10):
                rx = c * self.CELL
                ry = r * self.CELL
                if env.world[r][c] == 1:
                    pygame.draw.rect(surf, WALL_FILL,   (rx, ry, self.CELL, self.CELL))
                    pygame.draw.rect(surf, WALL_BORDER, (rx, ry, self.CELL, self.CELL), 1)
                else:
                    pygame.draw.rect(surf, BG,        (rx, ry, self.CELL, self.CELL))
                    pygame.draw.rect(surf, GRID_LINE, (rx, ry, self.CELL, self.CELL), 1)

        gx, gy = env.goal_x, env.goal_y
        r_px   = int(0.8 * self.CELL)
        surf.blit(self.goal_surf,
                  (int(gx * self.CELL) - r_px, int(gy * self.CELL) - r_px))

        pygame.draw.circle(surf, GOAL_FILL,
                           (int(gx * self.CELL), int(gy * self.CELL)),
                           int(0.22 * self.CELL))
        lbl = self.font_sm.render("GOAL", True, (99, 56, 6))
        surf.blit(lbl, (int(gx * self.CELL) - lbl.get_width() // 2,
                        int(gy * self.CELL) + int(0.26 * self.CELL)))

        rays = env.get_rays()
        for i, (dist, ex, ey) in enumerate(rays):
            col   = RAY_CENTER if i == 0 else RAY_SIDE
            width = 2 if i == 0 else 1
            sx, sy = int(env.x * self.CELL), int(env.y * self.CELL)
            ex_px, ey_px = int(ex * self.CELL), int(ey * self.CELL)
            if i == 0:
                pygame.draw.line(surf, col, (sx, sy), (ex_px, ey_px), width)
            else:
                steps = max(1, int(dist / 0.5))
                for s in range(steps):
                    t0 = s / steps
                    t1 = (s + 0.5) / steps
                    p0 = (int(sx + (ex_px - sx) * t0), int(sy + (ey_px - sy) * t0))
                    p1 = (int(sx + (ex_px - sx) * t1), int(sy + (ey_px - sy) * t1))
                    pygame.draw.line(surf, col, p0, p1, width)
            pygame.draw.circle(surf, HIT_DOT, (ex_px, ey_px), 4)

        rx, ry = int(env.x * self.CELL), int(env.y * self.CELL)
        pygame.draw.circle(surf, ROBOT_FILL, (rx, ry), 10)
        pygame.draw.circle(surf, ROBOT_RING, (rx, ry), 10, 2)

        ax = rx + int(math.cos(env.theta) * 16)
        ay = ry + int(math.sin(env.theta) * 16)
        pygame.draw.line(surf, ARROW, (rx, ry), (ax, ay), 2)
        pygame.draw.circle(surf, ARROW, (ax, ay), 3)

    def _draw_panel(self):
        env  = self.env
        x0   = 10 * self.CELL + self.PAD
        y    = self.PAD
        surf = self.screen

        def text(s, color=TEXT_DARK, big=False):
            nonlocal y
            f   = self.font_md if big else self.font_sm
            img = f.render(s, True, color)
            surf.blit(img, (x0, y))
            y  += img.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(surf, GRID_LINE,
                             (x0, y), (x0 + self.SIDE_W - self.PAD * 2, y))
            y += 6

        text("DQN Robot Env", big=True)
        sep()

        if self.use_net and self.weights:
            mode_s, mode_c = "MODE: NETWORK", (24, 95, 165)
        elif self.auto_run:
            mode_s, mode_c = "MODE: RANDOM", (163, 45, 45)
        else:
            mode_s, mode_c = "MODE: MANUAL", (59, 109, 17)
        text(mode_s, color=mode_c, big=True)
        sep()

        rays = env.get_rays()
        text(f"Episode  : {self.ep}")
        text(f"Steps    : {env.current_steps} / {env.MAX_STEPS}")
        text(f"Reward   : {self.total_r:.2f}")
        sep()
        text(f"Pos  x   : {env.x:.2f}")
        text(f"Pos  y   : {env.y:.2f}")
        text(f"Angle    : {math.degrees(env.theta):.1f}°")
        text(f"Dist goal: {env.goal_distance():.2f}")
        sep()
        text(f"Ray fwd  : {rays[0][0]:.2f}")
        text(f"Ray left : {rays[1][0]:.2f}")
        text(f"Ray right: {rays[2][0]:.2f}")
        sep()

        text("Reward history:")
        if len(self.reward_history) > 1:
            hist   = self.reward_history[-40:]
            mn, mx = min(hist), max(hist)
            span   = max(mx - mn, 1)
            sw     = self.SIDE_W - self.PAD * 2
            sh     = 40
            bx, by = x0, y
            pygame.draw.rect(surf, (230, 230, 225), (bx, by, sw, sh))
            for i in range(len(hist) - 1):
                t0   = i / max(len(hist) - 1, 1)
                t1   = (i + 1) / max(len(hist) - 1, 1)
                v0   = (hist[i] - mn) / span
                v1   = (hist[i + 1] - mn) / span
                p0   = (int(bx + t0 * sw), int(by + sh - v0 * sh))
                p1   = (int(bx + t1 * sw), int(by + sh - v1 * sh))
                pygame.draw.line(surf, (24, 95, 165), p0, p1, 2)
            y += sh + 6
        else:
            text("  (no data yet)")

        sep()
        text("WASD / arrows: move")
        text("SPACE: toggle agent")
        text("N: toggle network")
        text("R: reset   ESC: quit")

    def _draw_bottom_bar(self):
        y    = 10 * self.CELL
        h    = self.H - y
        pygame.draw.rect(self.screen, (230, 230, 225), (0, y, 10 * self.CELL, h))
        lines = [
            ("W/↑ Forward", (0, 110, 0)),
            ("A/← Turn L",  (0,  80, 160)),
            ("D/→ Turn R",  (0,  80, 160)),
            ("SPACE Agent", (130, 60, 0)),
            ("N Network",   (60,  0, 130)),
            ("R Reset",     (130, 0,  0)),
        ]
        slot_w = (10 * self.CELL) // len(lines)
        for i, (s, c) in enumerate(lines):
            img = self.font_sm.render(s, True, c)
            self.screen.blit(img,
                             (i * slot_w + (slot_w - img.get_width()) // 2,
                              y + (h - img.get_height()) // 2))

    def _draw_overlay(self):
        if not self.done:
            return
        msgs = {
            "goal":      ("Goal reached!", (59, 109, 17)),
            "collision": ("Collision!",    (163, 45, 45)),
            "timeout":   ("Timeout",       (130, 80, 0)),
        }
        s, c = msgs.get(self.reason, ("Done", TEXT_DARK))

        ov = pygame.Surface((10 * self.CELL, 10 * self.CELL), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 130))
        self.screen.blit(ov, (0, 0))

        img = self.font_lg.render(s, True, (255, 255, 255))
        cx  = (10 * self.CELL) // 2
        cy  = (10 * self.CELL) // 2
        self.screen.blit(img, (cx - img.get_width() // 2, cy - img.get_height()))

        sub = self.font_sm.render("Press R to reset or SPACE to continue agent",
                                  True, (220, 220, 220))
        self.screen.blit(sub, (cx - sub.get_width() // 2, cy + 8))

    def run(self):
        while True:
            self.clock.tick(60)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if ev.type == pygame.KEYDOWN:
                    key = ev.key

                    if key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()

                    if key == pygame.K_r:
                        self._reset()

                    if key == pygame.K_SPACE:
                        self.auto_run = not self.auto_run
                        if self.auto_run and self.done:
                            self._reset()

                    if key == pygame.K_n:
                        if self.weights:
                            self.use_net = not self.use_net
                        else:
                            print("No weights loaded. Pass --weights path.json")

                    if not self.auto_run:
                        if key in (pygame.K_w, pygame.K_UP):
                            if self.done: self._reset()
                            self._step_env(0)
                        if key in (pygame.K_a, pygame.K_LEFT):
                            if self.done: self._reset()
                            self._step_env(1)
                        if key in (pygame.K_d, pygame.K_RIGHT):
                            if self.done: self._reset()
                            self._step_env(2)

            if self.auto_run:
                now = time.time()
                if now - self._last_step > 1 / self.FPS_AUTO:
                    self._last_step = now
                    if self.done:
                        self._reset()
                    else:
                        self._step_env(self._select_action())

            self.screen.fill(BG)
            self._draw_world()
            self._draw_panel()
            self._draw_bottom_bar()
            self._draw_overlay()

            pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Robot Env Visualizer")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to exported QNet weights JSON file")
    parser.add_argument("--auto", action="store_true",
                        help="Start in autonomous agent mode immediately")
    args = parser.parse_args()

    weights = None
    if args.weights:
        try:
            weights = load_weights(args.weights)
            print(f"Loaded weights from '{args.weights}' "
                  f"({len(weights)} linear layers)")
        except Exception as e:
            print(f"Could not load weights: {e}")

    Visualizer(weights=weights, auto=args.auto).run()


if __name__ == "__main__":
    main()