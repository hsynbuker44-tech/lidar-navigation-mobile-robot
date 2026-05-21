# environment.py

import numpy as np
import random
from config import WORLD_WIDTH, WORLD_HEIGHT, START, GOAL


class Environment:
    def __init__(self, random_obstacles=True):
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT
        self.start = START
        self.goal = GOAL

        if random_obstacles:
            self.obstacles = self.generate_random_obstacles()
        else:
            self.obstacles = [
                (5, 5, 2, 2),
                (7, 3, 2, 2),
                (9, 6, 2, 2),
                (4, 10, 2, 2),
                (8, 12, 2, 2),
                (12, 4, 2, 2),
                (14, 7, 2, 2),
                (11, 11, 2, 2),
                (15, 14, 2, 2),
                (6, 16, 2, 2),
            ]

    def generate_random_obstacles(self):
        obstacles = []
        number_of_obstacles = 10

        attempts = 0
        max_attempts = 1000

        while len(obstacles) < number_of_obstacles and attempts < max_attempts:
            attempts += 1

            w = random.uniform(1.4, 2.4)
            h = random.uniform(1.4, 2.4)

            x = random.uniform(3, self.width - 3)
            y = random.uniform(3, self.height - 3)

            # Başlangıç ve hedef çevresini boş bırak
            start_dist = np.hypot(x - self.start[0], y - self.start[1])
            goal_dist = np.hypot(x - self.goal[0], y - self.goal[1])

            if start_dist < 4.0 or goal_dist < 4.0:
                continue

            # Engeller birbirine çok yakın olmasın
            valid = True
            for ox, oy, ow, oh in obstacles:
                dist = np.hypot(x - ox, y - oy)
                if dist < 3.0:
                    valid = False
                    break

            if valid:
                obstacles.append((x, y, w, h))

        return obstacles

    def is_collision(self, x, y, robot_radius=0.3):
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True

        for ox, oy, w, h in self.obstacles:
            left = ox - w / 2
            right = ox + w / 2
            bottom = oy - h / 2
            top = oy + h / 2

            if (
                x + robot_radius > left and
                x - robot_radius < right and
                y + robot_radius > bottom and
                y - robot_radius < top
            ):
                return True

        return False