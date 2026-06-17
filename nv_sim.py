import json
import math
import numpy as np
import pygame
import sys
import os

# ── constants ────────────────────────────────────────────────────────────────
CELL      = 64
GRID      = 10
W         = CELL * GRID
PANEL     = 260
H         = CELL * GRID
FPS       = 30
RAY_STEP  = 0.5

# colours
BG        = (18,  18,  24)
WALL      = (60,  60,  80)
FLOOR     = (30,  30,  40)
GRID_LINE = (40,  40,  55)
ROBOT_C   = (80, 180, 255)
GOAL_C    = (80, 255, 140)
RAY_C     = (255, 220,  80, 120)
PANEL_BG  = (24,  24,  32)
TEXT_C    = (200, 200, 220)
DIM_C     = (100, 100, 130)
ACCENT    = (80, 180, 255)
TRAIL_C   = (80, 130, 200)

# ── coordinate helper ─────────────────────────────────────────────────────────
# The env uses standard math coords: x→right, y→up.
# Pygame has y→down, so we flip: screen_y = (GRID - env_y) * CELL
def to_screen(ex, ey):
    return int(ex * CELL), int((GRID - ey) * CELL)

# ── network forward pass ──────────────────────────────────────────────────────
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def layer_norm(x, eps=1e-5):
    mean = x.mean()
    std  = x.std()
    return (x - mean) / (std + eps)

class QNet:
    def __init__(self, path):
        with open(path) as f:
            layers = json.load(f)
        self.weights = []
        for l in layers:
            W = np.array(l["W"], dtype=np.float32)
            b = np.array(l["b"], dtype=np.float32)
            self.weights.append((W, b))

    def forward(self, x):
        for i, (W, b) in enumerate(self.weights):
            x = W @ x + b
            if i < len(self.weights) - 1:
                x = layer_norm(x)
                x = leaky_relu(x)
        return x

    def best_action(self, state):
        q = self.forward(np.array(state, dtype=np.float32))
        return int(np.argmax(q)), q

# ── environment (exact mirror of gym.py) ─────────────────────────────────────
class RobotEnv:
    def __init__(self):
        self.world = np.zeros((10, 10))
        self.world[5, :] = 1
        self.world[5, 4] = 0
        self.world[5, 5] = 0
        self.max_steps = 200

    def reset(self):
        self.x, self.y, self.theta = 1.0, 1.0, 0.0
        self.current_steps = 0
        self.trail = []
        while True:
            gx = np.random.randint(1, 9)
            gy = np.random.randint(1, 9)
            if self.world[gy][gx] == 0 and math.hypot(gx - 1, gy - 1) > 3:
                self.goal_x, self.goal_y = gx, gy
                break
        return self._state()

    def step(self, action):
        old_dist = self._dist()
        if action == 0:
            self.x += math.cos(self.theta) * 0.3
            self.y += math.sin(self.theta) * 0.3
        elif action == 1:
            self.theta += 0.15
        elif action == 2:
            self.theta -= 0.15
        self.current_steps += 1
        self.trail.append((self.x, self.y))
        reward = self._reward(old_dist)
        done   = self._done()
        return self._state(), reward, done

    def _ray(self, angle):
        x, y = self.x, self.y
        dx, dy = math.cos(angle), math.sin(angle)
        dist = 0
        while True:
            x += dx * RAY_STEP
            y += dy * RAY_STEP
            dist += RAY_STEP
            ix, iy = int(x), int(y)
            if ix < 0 or iy < 0 or ix >= 10 or iy >= 10 or self.world[iy][ix] == 1:
                break
        return dist

    def _state(self):
        angles = [self.theta, self.theta + 0.5, self.theta - 0.5]
        rays   = [self._ray(a) for a in angles]
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        gd  = math.hypot(dx, dy)
        ga  = math.atan2(dy, dx)
        rel = math.atan2(math.sin(ga - self.theta), math.cos(ga - self.theta))
        return [rays[0] / 10, rays[1] / 10, rays[2] / 10,
                gd / 14.0, math.sin(self.theta), math.cos(self.theta), rel / math.pi]

    def _dist(self):
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def _collision(self):
        ix, iy = int(self.x), int(self.y)
        if ix < 0 or iy < 0 or ix >= 10 or iy >= 10:
            return True
        return self.world[iy][ix] == 1

    def _reward(self, old):
        if self._dist() < 0.8: return 10
        if self._collision():  return -10
        return (old - self._dist()) - 0.01

    def _done(self):
        return self._dist() < 0.8 or self._collision() or self.current_steps >= self.max_steps

    def reason(self):
        if self._dist() < 0.8:              return "goal"
        if self._collision():               return "collision"
        if self.current_steps >= self.max_steps: return "timeout"
        return None

# ── drawing ───────────────────────────────────────────────────────────────────
def draw_grid(surf, world):
    for gy in range(GRID):
        for gx in range(GRID):
            # env row gy=0 is at screen bottom; flip for draw
            sx, sy = to_screen(gx, GRID - 1 - gy)
            rect = pygame.Rect(sx, sy, CELL, CELL)
            col  = WALL if world[gy][gx] == 1 else FLOOR
            pygame.draw.rect(surf, col, rect)
            pygame.draw.rect(surf, GRID_LINE, rect, 1)

def draw_rays(surf, env):
    ray_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    angles = [env.theta, env.theta + 0.5, env.theta - 0.5]
    sx0, sy0 = to_screen(env.x, env.y)
    for angle in angles:
        length = env._ray(angle) * CELL
        # in screen space y is flipped so sin flips sign
        ex = env.x  + math.cos(angle) * env._ray(angle)
        ey = env.y  + math.sin(angle) * env._ray(angle)
        sx1, sy1 = to_screen(ex, ey)
        pygame.draw.line(ray_surf, RAY_C, (sx0, sy0), (sx1, sy1), 1)
    surf.blit(ray_surf, (0, 0))

def draw_trail(surf, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        x0, y0 = to_screen(*trail[i - 1])
        x1, y1 = to_screen(*trail[i])
        pygame.draw.line(surf, TRAIL_C, (x0, y0), (x1, y1), 1)

def draw_robot(surf, env):
    cx, cy = to_screen(env.x, env.y)
    r = 10
    pygame.draw.circle(surf, ROBOT_C, (cx, cy), r)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), r, 1)
    # heading arrow — flip sin for screen space
    ex = cx + int( math.cos(env.theta) * (r + 6))
    ey = cy + int(-math.sin(env.theta) * (r + 6))   # ← flipped
    pygame.draw.line(surf, (255, 255, 255), (cx, cy), (ex, ey), 2)

def draw_goal(surf, env):
    gx, gy = to_screen(env.goal_x, env.goal_y)
    pygame.draw.circle(surf, GOAL_C, (gx, gy), 8)
    pygame.draw.circle(surf, (255, 255, 255), (gx, gy), 8, 1)
    pygame.draw.circle(surf, (*GOAL_C, 40), (gx, gy), int(0.8 * CELL), 1)

def draw_panel(surf, font_big, font_sm, env, q_vals, action,
               step, episode, ep_reward, outcomes, speed):
    ox = W
    pygame.draw.rect(surf, PANEL_BG, (ox, 0, PANEL, H))
    pygame.draw.line(surf, GRID_LINE, (ox, 0), (ox, H), 1)

    def text(msg, x, y, col=TEXT_C, f=None):
        s = (f or font_sm).render(msg, True, col)
        surf.blit(s, (ox + x, y))

    y = 14
    text("DQN Robot Policy", 12, y, ACCENT, font_big); y += 30
    pygame.draw.line(surf, GRID_LINE, (ox+12, y), (ox+PANEL-12, y), 1); y += 10

    text(f"Episode   {episode}",       12, y); y += 20
    text(f"Step      {step} / 200",    12, y); y += 20
    text(f"Reward    {ep_reward:.2f}", 12, y); y += 20
    text(f"Speed     {speed}x",        12, y); y += 26

    pygame.draw.line(surf, GRID_LINE, (ox+12, y), (ox+PANEL-12, y), 1); y += 10
    text("Q-values", 12, y, DIM_C); y += 18

    labels  = ["forward", "turn L", "turn R"]
    bar_max = 80
    q_min = float(q_vals.min())
    q_range = float(q_vals.max()) - q_min
    for i, (lbl, q) in enumerate(zip(labels, q_vals)):
        col  = ACCENT if i == action else DIM_C
        norm = float(np.clip((q - q_min) / (q_range + 1e-6), 0, 1))
        bw   = int(norm * bar_max)
        text(f"{lbl}", 12, y, col)
        pygame.draw.rect(surf, col,   (ox+90, y+3, bw, 10))
        pygame.draw.rect(surf, DIM_C, (ox+90, y+3, bar_max, 10), 1)
        text(f"{q:.2f}", 175, y, col)
        y += 20
    y += 6

    pygame.draw.line(surf, GRID_LINE, (ox+12, y), (ox+PANEL-12, y), 1); y += 10
    text("outcomes", 12, y, DIM_C); y += 18
    total = max(1, sum(outcomes.values()))
    for label in ["goal", "collision", "timeout"]:
        n   = outcomes[label]
        pct = 100 * n / total
        c   = GOAL_C if label == "goal" else (255,80,80) if label == "collision" else (200,160,60)
        text(f"{label:<10} {n:>4}  {pct:5.1f}%", 12, y, c); y += 18
    y += 6

    pygame.draw.line(surf, GRID_LINE, (ox+12, y), (ox+PANEL-12, y), 1); y += 10
    text("controls", 12, y, DIM_C); y += 18
    for line in ["R  - restart episode", "UP/DN - speed", "Q  - quit"]:
        text(line, 12, y, DIM_C); y += 17

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    weights_path = "/home/prashanna/Documents/project_spatial_awarness/reinforced_model_v3/reinforced_model/qnet_weights.json"

    if not os.path.exists(weights_path):
        print(f"Weights not found at: {weights_path}")
        sys.exit(1)

    print(f"Loading weights from: {weights_path}")
    net = QNet(weights_path)
    env = RobotEnv()

    pygame.init()
    screen = pygame.display.set_mode((W + PANEL, H))
    pygame.display.set_caption("DQN Robot Visualizer")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 12)

    state     = env.reset()
    done      = False
    ep_reward = 0.0
    episode   = 1
    step      = 0
    q_vals    = np.zeros(3)
    action    = 0
    outcomes  = {"goal": 0, "collision": 0, "timeout": 0}
    speeds    = [1, 2, 4, 8]
    sp_idx    = 0
    speed     = speeds[sp_idx]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    state     = env.reset()
                    done      = False
                    ep_reward = 0.0
                    step      = 0
                elif event.key == pygame.K_UP:
                    sp_idx = min(sp_idx + 1, len(speeds) - 1)
                    speed  = speeds[sp_idx]
                elif event.key == pygame.K_DOWN:
                    sp_idx = max(sp_idx - 1, 0)
                    speed  = speeds[sp_idx]

        for _ in range(speed):
            if done:
                reason = env.reason()
                if reason:
                    outcomes[reason] += 1
                episode  += 1
                state     = env.reset()
                done      = False
                ep_reward = 0.0
                step      = 0
                break

            action, q_vals = net.best_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            step      += 1

        screen.fill(BG)
        draw_grid(screen, env.world)
        draw_trail(screen, env.trail)
        draw_rays(screen, env)
        draw_goal(screen, env)
        draw_robot(screen, env)
        draw_panel(screen, font_big, font_sm,
                   env, q_vals, action, step, episode,
                   ep_reward, outcomes, speed)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()