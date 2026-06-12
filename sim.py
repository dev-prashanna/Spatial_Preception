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
pygame.init() # "Wake up the video game engine!"
WIDTH, HEIGHT = 800, 600 # The size of our classroom (800 pixels wide, 600 tall)
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Building the actual window
clock = pygame.time.Clock() # A stopwatch so the game doesn't run too fast
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
        self.x = random.randint(150, 650)
        self.y = random.randint(150, 450)
        self.angle = random.uniform(0, 2*math.pi)
        self.speed = 2

    def cast_ray(self, x, y, angle, walls, max_dist=200):
        dx = math.cos(angle)
        dy = math.sin(angle)

        for i in range(max_dist):
            test_x = x + dx * i
            test_y = y + dy * i

            # check collision with boundaries
            for wall in walls:
                (x1, y1), (x2, y2) = wall

                # simple bounding box check (approx)
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

    # exit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save any remaining data in batch
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
    

    for sa in sensor_angles:
        ray_angle = robot.angle + sa

        dist = robot.cast_ray(robot.x, robot.y, ray_angle, walls)
        norm=1-(dist/200)
        noised_val=(max(0,min(1,norm+random.uniform(-0.02,0.02))))
        sensor_values.append(noised_val)

        # draw ray
        end_x = robot.x + math.cos(ray_angle) * dist
        end_y = robot.y + math.sin(ray_angle) * dist

        pygame.draw.line(
            screen,
            (0, 255, 0),
            (robot.x, robot.y),
            (end_x, end_y),
            2
        )
    # optional debugging
    if pygame.time.get_ticks() % 500 < 16:
        print(sensor_values)
    
    if USE_NN:
        inputs=sensor_values
        left_motor, right_motor=predict(model,inputs)
    else:
        left = sensor_values[0]
        front = sensor_values[1]
        right = sensor_values[2]

        left_motor = 0.7
        right_motor = 0.7

        # left wall too close
        if left > 0.5:
            left_motor = 0.4
            right_motor = 0.8

        # right wall too close
        elif right > 0.5:
            left_motor = 0.8
            right_motor = 0.4

        # obstacle ahead
        if front > 0.4:

            if left < right:
                left_motor = 0.2
                right_motor = 0.8

            else:
                left_motor = 0.8
                right_motor = 0.2
    
        # Record data
        batch_state['dataset'].append([
            sensor_values[0],
            sensor_values[1],
            sensor_values[2],
            left_motor,
            right_motor
        ])

    # Write batch to CSV when batch size is reached
    if len(batch_state['dataset']) >= BATCH_SIZE:
        
        # Initialize CSV header only once
        if not batch_state['file_initialized']:
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(csv_file) == 0:
                    writer.writerow(["L", "F", "R", "vL", "vR"])
            batch_state['file_initialized'] = True
        
        # Append batch to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_state['dataset'])
        
        batch_state['total_records'] += len(batch_state['dataset'])
        print(f"Batch saved! Total records: {batch_state['total_records']}/{TOTAL_DATASET_SIZE}")
        
        # Clear dataset for next batch
        batch_state['dataset'] = []
        
        # Exit when total dataset size is reached
        if batch_state['total_records'] >= TOTAL_DATASET_SIZE:
            print("Dataset collection complete!")
            pygame.quit()
            sys.exit()

    robot.move(left_motor, right_motor)
    # draw robot
    pygame.draw.circle(screen, (0, 150, 255), (int(robot.x), int(robot.y)), 8)

    pygame.display.flip()
    clock.tick(60)

    