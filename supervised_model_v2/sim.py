import pygame
import sys
import math
import csv 
import os
from neural_net import model, predict
import random

# Batch processing configuration
BATCH_SIZE = 5000
TOTAL_DATASET_SIZE = 100000
csv_file = "dataset.csv"

# State tracking for batch processing
batch_state = {
    'dataset': [],
    'total_records': 0,
    'file_initialized': False
}

recording = False
USE_NN = True
pygame.init() 
WIDTH, HEIGHT = 800, 600 
screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
clock = pygame.time.Clock() 
walls = [
    # outer walls
    ((100, 100), (700, 100)),
    ((700, 100), (700, 500)),
    ((700, 500), (100, 500)),
    ((100, 500), (100, 100)),

    # obstacle 1
    ((300, 200), (500, 200)),

    # obstacle 2
    ((250, 350), (550, 350)),

    # obstacle 3
    ((400, 150), (400, 300)),

    # obstacle 4 (box)
    ((550, 250), (650, 250)),
    ((650, 250), (650, 350)),
    ((650, 350), (550, 350)),
    ((550, 350), (550, 250)),
]

class Robot:
    def __init__(self):
        self.respawn()
        self.speed = 2
        self.radius = 8

    def respawn(self):
        """Teleports the robot to a random open position away from immediate walls"""
        self.x = random.randint(150, 650)
        self.y = random.randint(150, 450)
        self.angle = random.uniform(0, 2 * math.pi)

    def cast_ray(self, x, y, angle, walls, max_dist=200):
        dx = math.cos(angle)
        dy = math.sin(angle)

        for i in range(max_dist):
            test_x = x + dx * i
            test_y = y + dy * i

            for wall in walls:
                (x1, y1), (x2, y2) = wall
                if min(x1, x2) - 2 <= test_x <= max(x1, x2) + 2 and \
                   min(y1, y2) - 2 <= test_y <= max(y1, y2) + 2:
                    return i
        return max_dist
    
    def move(self, left_motor, right_motor):
        turn = left_motor - right_motor
        self.angle += turn * 0.05
        forward = (left_motor + right_motor) * 0.5
        self.x += math.cos(self.angle) * forward * self.speed
        self.y += math.sin(self.angle) * forward * self.speed
    
robot = Robot()

while True:
    screen.fill((20, 20, 30))

    # exit handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if batch_state['dataset']:
                if not batch_state['file_initialized']:
                    file_exists = os.path.exists(csv_file)
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists or os.path.getsize(csv_file) == 0:
                            writer.writerow(["L", "F", "R", "vL", "vR"])
                    batch_state['file_initialized'] = True
                
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_state['dataset'])
            pygame.quit()
            sys.exit()

    # draw walls
    for wall in walls:
        pygame.draw.line(screen, (200, 200, 200), wall[0], wall[1], 2)

    # sensors (3 rays)
    sensor_angles = [-0.5, 0, 0.5]
    sensor_values = []
    raw_distances = [] # Keep track of raw distances for crash detection

    for sa in sensor_angles:
        ray_angle = robot.angle + sa
        dist = robot.cast_ray(robot.x, robot.y, ray_angle, walls)
        raw_distances.append(dist)
        
        norm = 1 - (dist / 200)
        noised_val = max(0, min(1, norm + random.uniform(-0.01, 0.01))) # Slightly lowered noise for training stability
        sensor_values.append(noised_val)

        # draw ray
        end_x = robot.x + math.cos(ray_angle) * dist
        end_y = robot.y + math.sin(ray_angle) * dist
        pygame.draw.line(screen, (0, 255, 0), (robot.x, robot.y), (end_x, end_y), 1)

    # CRITICAL: Anti-Stuck / Crash Teleportation
    # If the robot crashes or gets too close to a wall, respawn it to diversify dataset locations
    if min(raw_distances) < 10:
        robot.respawn()
        continue # Skip recording this broken frame

    if USE_NN:
        inputs = sensor_values
        left_motor, right_motor = predict(model, inputs)
    else:
        left = sensor_values[0]
        front = sensor_values[1]
        right = sensor_values[2]

        # Base speed
        left_motor = 0.7
        right_motor = 0.7

        # 1. CORNER TRAP DETECTION
        # If all three sensors see a wall close by, the robot is cornered.
        if front > 0.4 and left > 0.4 and right > 0.4:
            # Force a hard, high-torque spin to the right to break out of the loop
            left_motor = 0.8
            right_motor = -0.8

        # 2. STANDARD OBSTACLE AHEAD
        elif front > 0.3:
            # Turn away aggressively from whichever side is tighter
            if left > right:
                left_motor = 0.8
                right_motor = -0.3  # Pivot right
            else:
                left_motor = -0.3   # Pivot left
                right_motor = 0.8

        # 3. SIDE WALL CUSHIONING
        else:
            if left > 0.4:
                left_motor = 0.7
                right_motor = 0.2   # Nudge right
            elif right > 0.4:
                left_motor = 0.2    # Nudge left
                right_motor = 0.7

        # Safety clip to ensure motor values stay strictly within valid ranges
        left_motor = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))

        # Ensure values stay bounded between -1.0 and 1.0
        left_motor = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))
    
        # Record clean data
        batch_state['dataset'].append([
            sensor_values[0],
            sensor_values[1],
            sensor_values[2],
            left_motor,
            right_motor
        ])

    # Write batch to CSV when batch size is reached
    if len(batch_state['dataset']) >= BATCH_SIZE:
        if not batch_state['file_initialized']:
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(csv_file) == 0:
                    writer.writerow(["L", "F", "R", "vL", "vR"])
            batch_state['file_initialized'] = True
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_state['dataset'])
        
        batch_state['total_records'] += len(batch_state['dataset'])
        print(f"Batch saved! Total records: {batch_state['total_records']}/{TOTAL_DATASET_SIZE}")
        
        batch_state['dataset'] = []
        
        # Periodic map re-shuffling to maximize environment features
        robot.respawn()

        if batch_state['total_records'] >= TOTAL_DATASET_SIZE:
            print("Dataset collection complete!")
            pygame.quit()
            sys.exit()

    robot.move(left_motor, right_motor)
    
    # draw robot
    pygame.draw.circle(screen, (0, 150, 255), (int(robot.x), int(robot.y)), robot.radius)
    pygame.display.flip()
    clock.tick(60)