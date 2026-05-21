# main.py

import numpy as np

from config import DT, SIM_TIME, START, GOAL, MANUAL_LINEAR_SPEED, MANUAL_ANGULAR_SPEED, LIDAR_RANGE
from environment import Environment
from robot import Robot
from navigation import Navigator
from visualization import PygameVisualizer
from lidar import Lidar
from ekf import EKF
from metrics import MetricsCollector
from analysis import generate_all_figures


def run_simulation(visualizer):
    """Tek bir simülasyon çalıştır. Restart istenirse True döner."""
    env = Environment()
    robot = Robot(*START)
    navigator = Navigator(GOAL, start=START)
    lidar = Lidar()
    ekf = EKF(*START)
    metrics = MetricsCollector()

    visualizer.env = env
    visualizer.restart_requested = False
    visualizer.manual_mode = False

    steps = int(SIM_TIME / DT)

    end_type = None  # 'collision' veya 'goal_reached'

    # Son LiDAR verilerini sakla (analiz için)
    last_raw_points = None
    last_raw_distances = None
    last_filtered_points = None
    last_filtered_distances = None
    last_lidar_angles = None

    for step in range(steps):
        visualizer.handle_events()

        if not visualizer.running:
            return False

        if visualizer.restart_requested:
            print("\n--- Simülasyon yeniden başlatılıyor... ---\n")
            return True

        # LiDAR taraması
        (raw_points, raw_distances, lidar_angles,
         filtered_points, filtered_distances) = lidar.scan(robot.state, env)

        # Son LiDAR verilerini güncelle
        last_raw_points = raw_points
        last_raw_distances = raw_distances
        last_filtered_points = filtered_points
        last_filtered_distances = filtered_distances
        last_lidar_angles = lidar_angles

        # Kontrol
        if visualizer.manual_mode:
            v, w = visualizer.get_manual_control(
                MANUAL_LINEAR_SPEED, MANUAL_ANGULAR_SPEED
            )
            current_mode = "MANUEL"
        else:
            v_cmd, w_cmd = navigator.prev_v, navigator.prev_w
            ekf.predict(v_cmd, w_cmd, DT)
            noisy_measurement = robot.get_noisy_odometry()
            ekf_state = ekf.update(noisy_measurement)

            # Navigatör filtrelenmiş mesafeleri kullanır (daha kararlı)
            v, w = navigator.compute_control(
                ekf_state, filtered_distances, lidar_angles
            )
            current_mode = navigator.mode

        # Metrikler
        min_obstacle_dist = (np.min(filtered_distances)
                             if len(filtered_distances) > 0
                             else float('inf'))
        metrics.record_step(robot.state, v, w, min_obstacle_dist, current_mode)

        # Robot hareketi
        robot.update(v, w, DT)

        x, y, theta = robot.state

        # Çarpışma kontrolü
        if env.is_collision(x, y):
            print("Çarpışma gerçekleşti!")
            metrics.record_collision()
            end_type = 'collision'
            break

        # Hedef kontrolü
        dist_to_goal = np.hypot(x - GOAL[0], y - GOAL[1])
        if dist_to_goal < 0.5:
            print("Hedefe ulaşıldı!")
            metrics.record_goal_reached()
            end_type = 'goal_reached'
            break

        visualizer.draw(robot, GOAL, raw_points,
                        ekf=ekf, navigator=navigator, metrics=metrics)
        visualizer.tick(30)

    print("Simülasyon tamamlandı.")
    print("Son konum:", robot.state)
    metrics.print_summary()

    # ---- Analiz grafiklerini otomatik oluştur ---- #
    try:
        error_stats = generate_all_figures(
            env, robot, ekf, metrics,
            last_raw_points, last_raw_distances,
            last_filtered_points, last_filtered_distances,
            last_lidar_angles, START, GOAL,
            LIDAR_RANGE, dt=DT
        )
        if error_stats:
            print(f"\n[Sonuç] Lokalizasyon RMSE: {error_stats.get('rmse', 0):.4f} m")
            print(f"[Sonuç] Lokalizasyon MAE:  {error_stats.get('mae', 0):.4f} m")
    except Exception as e:
        print(f"[Uyarı] Grafik oluşturma hatası: {e}")

    # ---- Bitiş ekranı ---- #
    if end_type is not None:
        visualizer.draw_end_screen(
            end_type, robot, GOAL, last_raw_points,
            ekf=ekf, navigator=navigator, metrics=metrics
        )

        # Bitiş ekranında kullanıcı kararını bekle
        waiting = True
        while waiting and visualizer.running:
            action = visualizer.handle_end_events()
            if action == 'restart':
                return True
            elif action == 'quit':
                return False
            visualizer.tick(30)

    return False


def main():
    env = Environment()
    visualizer = PygameVisualizer(env)

    while True:
        should_restart = run_simulation(visualizer)
        if not should_restart:
            break

    visualizer.close()


if __name__ == "__main__":
    main()

