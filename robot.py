# robot.py

import numpy as np
from config import MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED, ENCODER_NOISE_STD, IMU_NOISE_STD


class Robot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta], dtype=float)
        self.true_path = []

    def update(self, v, w, dt):
        v = np.clip(v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        w = np.clip(w, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        x, y, theta = self.state

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        theta = self.normalize_angle(theta)

        self.state = np.array([x, y, theta])
        self.true_path.append(self.state.copy())

    def get_noisy_odometry(self):
        """
        Gürültülü odometri ölçümü.
        Gerçek duruma sensör gürültüsü ekleyerek EKF'e ölçüm olarak verir.
        """
        x, y, theta = self.state

        noisy_x = x + np.random.normal(0, ENCODER_NOISE_STD)
        noisy_y = y + np.random.normal(0, ENCODER_NOISE_STD)
        noisy_theta = theta + np.random.normal(0, IMU_NOISE_STD)

        return np.array([noisy_x, noisy_y, noisy_theta])

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi