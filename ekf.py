# ekf.py — Extended Kalman Filter

import numpy as np
from config import EKF_PROCESS_NOISE, EKF_MEASUREMENT_NOISE, DT


class EKF:
    """
    Extended Kalman Filter — diferansiyel sürüş kinematik modeli.

    Durum vektörü: [x, y, θ]
    Kontrol girdisi: [v, ω] (doğrusal ve açısal hız)
    """

    def __init__(self, x0, y0, theta0):
        # Durum vektörü
        self.x = np.array([x0, y0, theta0], dtype=float)

        # Kovaryans matrisi (başlangıçta küçük belirsizlik)
        self.P = np.diag([0.01, 0.01, 0.01])

        # Proses gürültüsü kovaryans matrisi
        self.Q = np.diag([
            EKF_PROCESS_NOISE,
            EKF_PROCESS_NOISE,
            EKF_PROCESS_NOISE * 2
        ])

        # Ölçüm gürültüsü kovaryans matrisi
        self.R = np.diag([
            EKF_MEASUREMENT_NOISE,
            EKF_MEASUREMENT_NOISE,
            EKF_MEASUREMENT_NOISE * 2
        ])

        # Tahmini yol geçmişi
        self.estimated_path = []

    def predict(self, v, w, dt=DT):
        """
        Tahmin adımı: Robot kinematik modeli ile durum tahmini.

        x' = x + v * cos(θ) * dt
        y' = y + v * sin(θ) * dt
        θ' = θ + ω * dt
        """
        theta = self.x[2]

        # Durum geçiş (kinematik model)
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = w * dt

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] += dtheta
        self.x[2] = self.normalize_angle(self.x[2])

        # Jacobian matrisi F (durum geçiş Jacobian'ı)
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])

        # Kontrol Jacobian'ı (gürültü propagasyonu)
        V = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])

        # Kontrol gürültüsü
        M = np.diag([
            (0.1 * abs(v) + 0.01) ** 2,
            (0.1 * abs(w) + 0.01) ** 2
        ])

        # Kovaryans güncelleme
        self.P = F @ self.P @ F.T + V @ M @ V.T + self.Q

        return self.x.copy()

    def update(self, z):
        """
        Güncelleme adımı: Ölçüm ile düzeltme.

        z: [x_measured, y_measured, θ_measured] — gürültülü odometri
        """
        # Ölçüm modeli: H = I (doğrudan gözlem)
        H = np.eye(3)

        # Yenilik (innovation)
        y = z - H @ self.x
        y[2] = self.normalize_angle(y[2])

        # Yenilik kovaryansı
        S = H @ self.P @ H.T + self.R

        # Kalman kazancı
        K = self.P @ H.T @ np.linalg.inv(S)

        # Durum güncelleme
        self.x = self.x + K @ y
        self.x[2] = self.normalize_angle(self.x[2])

        # Kovaryans güncelleme
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

        # Yol geçmişine ekle
        self.estimated_path.append(self.x.copy())

        return self.x.copy()

    def get_state(self):
        """Mevcut tahmini durumu döndür."""
        return self.x.copy()

    def get_covariance(self):
        """Mevcut kovaryans matrisini döndür."""
        return self.P.copy()

    def get_uncertainty(self):
        """Pozisyon belirsizliğini (standart sapma) döndür."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
