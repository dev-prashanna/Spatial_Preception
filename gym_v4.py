import numpy as np


class RobotEnv:

    def __init__(self):
        self.world = np.zeros((10, 10))
        self.world[5, :] = 1
        self.world[5,4]=0
        self.world[5,5]=0

        self.x = 1.0
        self.y = 1.0
        self.theta = 0.0
        self.ray_step = 0.5
        self.goal_x = 8
        self.goal_y = 8
        self.max_steps = 200
        self.current_steps = 0

    def reset(self):
        self.current_steps = 0
        self.x=1.0
        self.y=1.0
        self.theta=0.0
        
        while True:

            gx=np.random.randint(1,9)
            gy=np.random.randint(1,9)

            if self.world[gy][gx]==0:

                if np.sqrt((gx-1)**2+(gy-1)**2)>3:

                    self.goal_x=gx
                    self.goal_y=gy
                    break

        return self.get_state()

    def step(self, action):
        self.current_steps += 1
        old_distance = self.goal_distance()

        if action==0:

            self.x+=np.cos(self.theta)*0.3
            self.y+=np.sin(self.theta)*0.3

        elif action==1:
            self.theta+=0.15

        elif action==2:
            self.theta-=0.15
        
        reward=self.get_reward(old_distance)
        done=self.is_done()

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

            ix, iy = int(x), int(y)

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
        relative_angle = np.arctan2(
            np.sin(relative_angle),
            np.cos(relative_angle)
        )

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
        ix, iy = int(self.x), int(self.y)

        if ix < 0 or iy < 0 or ix >= 10 or iy >= 10:
            return True

        return self.world[iy][ix] == 1

    def get_reward(self, old_distance):
        new_distance = self.goal_distance()

        shaping = old_distance - new_distance

        if self.is_goal_reached():
            return 10 + (self.max_steps - self.current_steps) * 0.05

        if self.is_collision():
            return -10

        return shaping - 0.01

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
