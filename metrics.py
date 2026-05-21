# metrics.py — Performans Metrikleri

import numpy as np
import time


class MetricsCollector:
    """
    Simülasyon performans metriklerini toplayan sınıf.

    Toplanan metrikler:
    - Toplam yol uzunluğu
    - Hedefe ulaşma süresi
    - Minimum engel mesafesi (güvenlik skoru)
    - Yol düzgünlüğü (smoothness)
    - Enerji tüketimi
    - Çarpışma sayısı
    - Mod dağılımı
    """

    def __init__(self):
        self.positions = []          # (x, y) pozisyon geçmişi
        self.velocities = []         # (v, w) hız geçmişi
        self.min_obstacles = []      # Her adımda minimum engel mesafesi
        self.modes = []              # Navigasyon mod geçmişi
        self.timestamps = []         # Zaman damgaları

        self.start_time = time.time()
        self.collision_count = 0
        self.step_count = 0

        self.goal_reached = False
        self.goal_time = None

    def record_step(self, robot_state, v, w, min_obstacle_dist, mode):
        """Her simülasyon adımında çağrılır."""
        x, y, theta = robot_state
        self.positions.append((x, y))
        self.velocities.append((v, w))
        self.min_obstacles.append(min_obstacle_dist)
        self.modes.append(mode)
        self.timestamps.append(time.time() - self.start_time)
        self.step_count += 1

    def record_collision(self):
        """Çarpışma olduğunda çağrılır."""
        self.collision_count += 1

    def record_goal_reached(self):
        """Hedefe ulaşıldığında çağrılır."""
        self.goal_reached = True
        self.goal_time = time.time() - self.start_time

    def get_total_path_length(self):
        """Toplam yol uzunluğu (metre)."""
        if len(self.positions) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            total += np.hypot(dx, dy)

        return total

    def get_min_obstacle_distance(self):
        """Tüm simülasyon boyunca minimum engel mesafesi."""
        if not self.min_obstacles:
            return float('inf')
        return min(self.min_obstacles)

    def get_avg_speed(self):
        """Ortalama doğrusal hız."""
        if not self.velocities:
            return 0.0
        return np.mean([abs(v) for v, w in self.velocities])

    def get_path_smoothness(self):
        """
        Yol düzgünlüğü — ardışık yön değişimlerinin karelerinin ortalaması.
        Düşük = daha düzgün yol.
        """
        if len(self.positions) < 3:
            return 0.0

        angle_changes = []
        for i in range(2, len(self.positions)):
            dx1 = self.positions[i - 1][0] - self.positions[i - 2][0]
            dy1 = self.positions[i - 1][1] - self.positions[i - 2][1]
            dx2 = self.positions[i][0] - self.positions[i - 1][0]
            dy2 = self.positions[i][1] - self.positions[i - 1][1]

            a1 = np.arctan2(dy1, dx1)
            a2 = np.arctan2(dy2, dx2)

            diff = abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
            angle_changes.append(diff)

        if not angle_changes:
            return 0.0

        return np.mean(angle_changes)

    def get_energy_consumption(self):
        """
        Basit enerji tüketimi tahmini.
        E = Σ (v² + 0.5 * w²) * dt
        """
        if not self.velocities:
            return 0.0

        energy = 0.0
        dt = 0.1  # config'den de alınabilir
        for v, w in self.velocities:
            energy += (v ** 2 + 0.5 * w ** 2) * dt

        return energy

    def get_mode_distribution(self):
        """Mod dağılımı (yüzde olarak)."""
        if not self.modes:
            return {}

        total = len(self.modes)
        distribution = {}

        for mode in set(self.modes):
            count = self.modes.count(mode)
            distribution[mode] = round(100 * count / total, 1)

        return distribution

    def get_elapsed_time(self):
        """Geçen simülasyon süresi."""
        return self.step_count * 0.1

    def print_summary(self):
        """Simülasyon sonunda özet rapor yazdır."""
        print("\n" + "=" * 55)
        print("       PERFORMANS METRİKLERİ RAPORU")
        print("=" * 55)

        print(f"  Hedefe ulasild mi?    : {'[+] EVET' if self.goal_reached else '[-] HAYIR'}")

        if self.goal_reached:
            print(f"  Hedefe ulasma suresi   : {self.goal_time:.1f} sn")

        print(f"  Simulasyon suresi      : {self.get_elapsed_time():.1f} sn")
        print(f"  Toplam adim           : {self.step_count}")
        print(f"  Toplam yol uzunlugu   : {self.get_total_path_length():.2f} m")
        print(f"  Ortalama hiz          : {self.get_avg_speed():.3f} m/s")
        print(f"  Min. engel mesafesi   : {self.get_min_obstacle_distance():.3f} m")
        print(f"  Yol duzgunlugu        : {self.get_path_smoothness():.4f} rad")
        print(f"  Enerji tuketimi       : {self.get_energy_consumption():.3f} J")
        print(f"  Carpisma sayisi       : {self.collision_count}")

        print(f"\n  Mod Dagilimi:")
        for mode, pct in self.get_mode_distribution().items():
            bar = "#" * int(pct / 3)
            print(f"    {mode:20s} : {pct:5.1f}% {bar}")

        print("=" * 55)

    def get_live_stats(self):
        """Anlık metrikler (görselleştirme için)."""
        return {
            "path_length": self.get_total_path_length(),
            "min_obstacle": self.get_min_obstacle_distance(),
            "avg_speed": self.get_avg_speed(),
            "steps": self.step_count,
            "time": self.get_elapsed_time(),
            "mode": self.modes[-1] if self.modes else "N/A",
            "collisions": self.collision_count,
        }
