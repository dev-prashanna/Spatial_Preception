import json
import math
import numpy as np
import pygame
import sys
import os
import torch
from deep_q_network import QNet as PyTorchQNet  # Imports your actual network structure

# ─── CONSTANTS & CONFIGURATION ────────────────────────────────────────────────
CELL      = 64
GRID      = 10
W         = CELL * GRID
PANEL     = 260
H         = CELL * GRID
FPS       = 30

# Colors
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

# ─── COORDINATE CONVERSION HELPER ─────────────────────────────────────────────
# Environment uses standard math coords: x→right, y→up.
# Pygame uses y→down, so we invert: screen_y = (GRID - env_y) * CELL
def to_screen(ex, ey):
    return int(ex * CELL), int((GRID - ey) * CELL)

class QNet:
    def __init__(self, path):
        # Automatically select GPU if available, just like training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchQNet().to(self.device)
        
        # Load the exact structural brain saved during training
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Freeze layers into evaluation mode

    def best_action(self, state):
        # Process the state through the exact same PyTorch workflow as training
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
        action = torch.argmax(q_values, dim=1).item()
        
        # Pull the values back to a normal NumPy array so the Pygame panel can read them
        return action, q_values.cpu().squeeze().numpy()
# ─── ENVIRONMENT (Exact Mirror of your Refined Gym) ──────────────────────────
class RobotEnv:
    def __init__(self):
        self.world = np.zeros((10, 10))
        self.world[5, :] = 1
        self.world[5, 4] = 0
        self.world[5, 5] = 0

        self.x = 1.0
        self.y = 1.0
        self.theta = 0.0
        self.ray_step = 0.5
        self.goal_x = 8.5
        self.goal_y = 8.5
        self.max_steps = 200
        self.current_steps = 0
        self.trail = []

    def reset(self):
        self.current_steps = 0
        self.theta = np.random.uniform(0, 2 * np.pi)
        self.trail = []

        while True:
            sx = np.random.randint(1, 9)
            sy = np.random.randint(1, 9)
            if self.world[sy][sx] == 0:
                self.x = float(sx) + 0.5
                self.y = float(sy) + 0.5
                break

        while True:
            gx = np.random.randint(1, 9)
            gy = np.random.randint(1, 9)
            if self.world[gy][gx] == 0:
                if np.sqrt((gx - self.x)**2 + (gy - self.y)**2) > 3:
                    self.goal_x = float(gx) + 0.5
                    self.goal_y = float(gy) + 0.5
                    break
                    
        self.trail.append((self.x, self.y))
        return self.get_state()

    def step(self, action):
        self.current_steps += 1
        old_distance = self.goal_distance()

        if action == 0:
            self.x += np.cos(self.theta) * 0.3
            self.y += np.sin(self.theta) * 0.3
        elif action == 1:
            self.theta += 0.15
            self.x += np.cos(self.theta) * 0.05
            self.y += np.sin(self.theta) * 0.05
        elif action == 2:
            self.theta -= 0.15
            self.x += np.cos(self.theta) * 0.05
            self.y += np.sin(self.theta) * 0.05
        
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.trail.append((self.x, self.y))
        
        reward = self.get_reward(old_distance, action)
        done = self.is_done()

        return self.get_state(), reward, done 

    def get_ray_angles(self):
        return [self.theta, self.theta + 0.5, self.theta - 0.5]

    def ray_distances(self, angle):
        x, y = self.x, self.y
        dx, dy = np.cos(angle), np.sin(angle)
        dist = 0

        while True:
            x += dx * self.ray_step
            y += dy * self.ray_step
            dist += self.ray_step

            ix, iy = math.floor(x), math.floor(y)

            if ix < 0 or iy < 0 or ix >= 10 or iy >= 10:
                break

            if self.world[iy][ix] == 1:
                break

        return dist

    def get_state(self):
        angles = self.get_ray_angles()
        rays = [self.ray_distances(a) for a in angles]

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        goal_distance = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx)

        relative_angle = goal_angle - self.theta
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

        return np.array([
            rays[0]/10,
            rays[1]/10,
            rays[2]/10,
            goal_distance/14.0,
            np.sin(self.theta),
            np.cos(self.theta),
            relative_angle/np.pi
        ], dtype=np.float32)

    def is_collision(self):
        ix, iy = math.floor(self.x), math.floor(self.y)

        if ix < 0 or iy < 0 or ix >= 10 or iy >= 10:
            return True

        return self.world[iy][ix] == 1

    def get_reward(self, old_distance, action):
        new_distance = self.goal_distance()
        shaping = old_distance - new_distance

        if self.is_goal_reached():
            return 20.0 + (self.max_steps - self.current_steps) * 0.05

        if self.is_collision():
            return -10.0
        
        if new_distance < 1.2:
            shaping += 0.02

        if self.current_steps >= self.max_steps:
             return shaping - 0.05

        if action == 0:
            return shaping + 0.02
        else:
            return shaping - 0.05

    def goal_distance(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return np.sqrt(dx ** 2 + dy ** 2)

    def is_goal_reached(self):
        return self.goal_distance() < 0.8

    def is_done(self):
        return self.is_goal_reached() or self.is_collision() or self.current_steps >= self.max_steps
    
    def termination_reason(self):
        if self.is_goal_reached():
            return "goal"
        if self.is_collision():
            return "collision"
        if self.current_steps >= self.max_steps:
            return "timeout"
        return None

# ─── PYGAME DRAWING UTILITIES ─────────────────────────────────────────────────
def draw_grid(surf, world):
    for gy in range(GRID):
        for gx in range(GRID):
            sx, sy = to_screen(gx, 1 + gy)
            rect = pygame.Rect(sx, sy, CELL, CELL)
            col  = WALL if world[gy][gx] == 1 else FLOOR
            pygame.draw.rect(surf, col, rect)
            pygame.draw.rect(surf, GRID_LINE, rect, 1)

def draw_rays(surf, env):
    ray_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    angles = env.get_ray_angles()
    sx0, sy0 = to_screen(env.x, env.y)
    for angle in angles:
        r_dist = env.ray_distances(angle)
        ex = env.x + math.cos(angle) * r_dist
        ey = env.y + math.sin(angle) * r_dist
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
    
    ex = cx + int(math.cos(env.theta) * (r + 6))
    ey = cy + int(-math.sin(env.theta) * (r + 6))
    pygame.draw.line(surf, (255, 255, 255), (cx, cy), (ex, ey), 2)

def draw_goal(surf, env):
    # Removed structural '+ 0.5' duplication because environment coordinates are centered natively
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

# ─── MAIN VISUALIZATION CONTROLLER ────────────────────────────────────────────
def main():
    weights_path = "reinforced_model_v4/reinforced_model/qnet_weights.json"
    if not os.path.exists(weights_path):
        print(f"Weights file not found at path: {os.path.abspath(weights_path)}")
        sys.exit(1)

    print(f"Loading network parameters from: {weights_path}")
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
    speeds    = [1, 2, 4, 8, 16]
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
                reason = env.termination_reason()
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

