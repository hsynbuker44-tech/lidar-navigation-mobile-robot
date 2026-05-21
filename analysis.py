# analysis.py — Simülasyon Sonrası Grafik ve Hata Analizi
#
# Üretilen grafikler:
#   1. environment_map.png      — ortam haritası (engeller, start, goal)
#   2. path_comparison.png      — gerçek yol vs EKF tahmini
#   3. lidar_visualization.png  — ham vs filtrelenmiş LiDAR
#   4. localization_error.png   — zaman serisi konum hatası + RMSE/MAE
#   5. mode_distribution.png    — navigasyon mod dağılımı pasta grafiği
#   6. metrics_summary.png      — hız, enerji, engel mesafesi zaman serisi

import numpy as np
import matplotlib
matplotlib.use("Agg")           # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os

OUTPUT_DIR = "figures"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================ #
#  1. ORTAM HARİTASI
# ============================================================ #

def plot_environment(env, start, goal, save=True):
    """
    2B ortam haritası: engeller, başlangıç, hedef, sınırlar.
    Ödev gereksinimi 6.1
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linewidth=0.5, color="#dddddd", zorder=0)

    # Engeller
    for i, (ox, oy, w, h) in enumerate(env.obstacles):
        rect = patches.Rectangle(
            (ox - w / 2, oy - h / 2), w, h,
            linewidth=1.5, edgecolor="#8B0000", facecolor="#C0392B",
            alpha=0.85, zorder=2, label="Engel" if i == 0 else "_"
        )
        ax.add_patch(rect)
        ax.text(ox, oy, str(i + 1), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=3)

    # Başlangıç
    ax.plot(start[0], start[1], "o", markersize=14, color="#2ECC71",
            markeredgecolor="#1A8A4A", markeredgewidth=1.5,
            zorder=4, label="Başlangıç")
    ax.annotate("  Start", (start[0], start[1]), fontsize=10,
                va="center", color="#1A8A4A", fontweight="bold")

    # Hedef
    ax.plot(goal[0], goal[1], "*", markersize=18, color="#F39C12",
            markeredgecolor="#B7770D", markeredgewidth=1.5,
            zorder=4, label="Hedef")
    ax.annotate("  Goal", (goal[0], goal[1]), fontsize=10,
                va="center", color="#B7770D", fontweight="bold")

    # M-line (start→goal doğrusu)
    ax.plot([start[0], goal[0]], [start[1], goal[1]],
            "--", color="#7F8C8D", linewidth=1.2, alpha=0.6,
            zorder=1, label="M-line (Start→Goal)")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("2B Ortam Haritası — LiDAR Tabanlı Otonom Navigasyon",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    _add_obstacle_count_text(ax, env)

    plt.tight_layout()
    path = _save(fig, "environment_map.png", save)
    return fig, path


def _add_obstacle_count_text(ax, env):
    ax.text(0.02, 0.02,
            f"Engel sayısı: {len(env.obstacles)}",
            transform=ax.transAxes, fontsize=10,
            color="#555", bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", alpha=0.7))


# ============================================================ #
#  2. YOL KARŞILAŞTIRMASI — gerçek vs EKF tahmini
# ============================================================ #

def plot_path_comparison(env, robot_true_path, ekf_estimated_path,
                         start, goal, save=True):
    """
    Gerçek robot yolu ile EKF tahmini yolunun karşılaştırması.
    Ödev gereksinimi 6.2 + 6.4
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linewidth=0.5, color="#dddddd", zorder=0)

    # Engeller (hafif)
    for ox, oy, w, h in env.obstacles:
        rect = patches.Rectangle(
            (ox - w / 2, oy - h / 2), w, h,
            linewidth=1, edgecolor="#8B0000", facecolor="#C0392B",
            alpha=0.4, zorder=2
        )
        ax.add_patch(rect)

    # Gerçek yol
    if len(robot_true_path) > 1:
        true_arr = np.array(robot_true_path)
        ax.plot(true_arr[:, 0], true_arr[:, 1],
                color="#2C3E8A", linewidth=2.2, zorder=3,
                label="Gerçek yol (robot)")

    # EKF tahmini yol
    if len(ekf_estimated_path) > 1:
        ekf_arr = np.array(ekf_estimated_path)
        ax.plot(ekf_arr[:, 0], ekf_arr[:, 1],
                color="#E74C3C", linewidth=1.5, linestyle="--",
                alpha=0.85, zorder=3, label="EKF tahmini yol")

    # Başlangıç & Hedef
    ax.plot(start[0], start[1], "o", markersize=13, color="#2ECC71",
            markeredgecolor="#1A8A4A", markeredgewidth=1.5, zorder=5,
            label="Başlangıç")
    ax.plot(goal[0], goal[1], "*", markersize=17, color="#F39C12",
            markeredgecolor="#B7770D", markeredgewidth=1.5, zorder=5,
            label="Hedef")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Robot Yolu: Gerçek vs EKF Tahmini",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    path = _save(fig, "path_comparison.png", save)
    return fig, path


# ============================================================ #
#  3. LiDAR GÖRSELLEŞTİRME — ham vs filtrelenmiş
# ============================================================ #

def plot_lidar_snapshot(robot_state, raw_points, raw_distances,
                        filtered_points, filtered_distances,
                        angles, lidar_range, save=True):
    """
    Son LiDAR taramasının ham ve filtrelenmiş versiyonunu göster.
    Ödev gereksinimi 6.3
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x, y, theta = robot_state

    for ax, pts, dists, title, color in [
        (axes[0], raw_points, raw_distances,
         "Ham LiDAR Verisi", "#E74C3C"),
        (axes[1], filtered_points, filtered_distances,
         "Filtrelenmiş LiDAR Verisi (Medyan)", "#2ECC71"),
    ]:
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")

        # Robot pozisyonu
        ax.plot(x, y, "o", markersize=10, color="#F39C12",
                zorder=5, label="Robot")

        # LiDAR ışınları
        for i in range(len(pts)):
            if dists[i] < lidar_range * 0.95:
                ax.plot([x, pts[i, 0]], [y, pts[i, 1]],
                        color="#555577", linewidth=0.5, alpha=0.4, zorder=1)

        # Nokta bulutu — mesafeye göre renk
        scatter = ax.scatter(
            pts[:, 0], pts[:, 1],
            c=dists, cmap="plasma_r",
            vmin=0, vmax=lidar_range,
            s=8, zorder=4, label="LiDAR noktaları"
        )
        plt.colorbar(scatter, ax=ax, label="Mesafe (m)", shrink=0.8)

        # Robot yön oku
        hx = x + 0.8 * np.cos(theta)
        hy = y + 0.8 * np.sin(theta)
        ax.annotate("", xy=(hx, hy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#F39C12",
                                   lw=2))

        ax.set_title(title, fontsize=12, fontweight="bold", color="white")
        ax.set_xlabel("X (m)", fontsize=10, color="white")
        ax.set_ylabel("Y (m)", fontsize=10, color="white")
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#2a2a3e", labelcolor="white")

    # Alt grafik: mesafe profili (ham vs filtrelenmiş)
    fig2, ax2 = plt.subplots(figsize=(14, 3))
    deg = np.degrees(angles)
    ax2.plot(deg, raw_distances, color="#E74C3C", linewidth=1,
             alpha=0.7, label="Ham mesafe")
    ax2.plot(deg, filtered_distances, color="#2ECC71", linewidth=1.8,
             label="Filtrelenmiş mesafe")
    ax2.axhline(lidar_range, color="#888", linestyle="--",
                linewidth=0.8, label=f"Max menzil ({lidar_range}m)")
    ax2.set_xlabel("Işın açısı (°)", fontsize=11)
    ax2.set_ylabel("Mesafe (m)", fontsize=11)
    ax2.set_title("LiDAR Mesafe Profili — Ham vs Filtrelenmiş",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    path1 = _save(fig, "lidar_2d_view.png", save)
    path2 = _save(fig2, "lidar_profile.png", save)
    return fig, fig2, path1, path2


# ============================================================ #
#  4. LOKALİZASYON HATASI — zaman serisi + RMSE / MAE
# ============================================================ #

def plot_localization_error(robot_true_path, ekf_estimated_path,
                            dt=0.1, save=True):
    """
    Gerçek yol ile EKF tahmini arasındaki pozisyon hatası.
    RMSE ve MAE hesaplar, zaman serisi olarak çizer.
    Ödev gereksinimi 6.4 + 6.5
    """
    true_arr = np.array(robot_true_path)
    ekf_arr = np.array(ekf_estimated_path)

    # Uzunlukları eşitle
    n = min(len(true_arr), len(ekf_arr))
    if n < 2:
        print("[analysis] Hata analizi için yeterli veri yok.")
        return None, None

    true_xy = true_arr[:n, :2]
    ekf_xy = ekf_arr[:n, :2]

    errors = np.linalg.norm(true_xy - ekf_xy, axis=1)   # her adımda Öklid hatası
    time_axis = np.arange(n) * dt

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(errors)
    max_err = errors.max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Üst: toplam Öklid hatası ---
    axes[0].plot(time_axis, errors, color="#E74C3C", linewidth=1.5,
                 label="Öklid hatası ‖p_true − p_ekf‖")
    axes[0].axhline(rmse, color="#F39C12", linestyle="--", linewidth=1.2,
                    label=f"RMSE = {rmse:.4f} m")
    axes[0].axhline(mae, color="#9B59B6", linestyle=":", linewidth=1.2,
                    label=f"MAE  = {mae:.4f} m")
    axes[0].fill_between(time_axis, 0, errors, alpha=0.15, color="#E74C3C")
    axes[0].set_ylabel("Konum hatası (m)", fontsize=11)
    axes[0].set_title(
        f"Lokalizasyon Hatası — RMSE: {rmse:.4f} m | MAE: {mae:.4f} m | Maks: {max_err:.4f} m",
        fontsize=13, fontweight="bold"
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linewidth=0.5, alpha=0.5)

    # --- Orta: X hatası ---
    x_err = true_xy[:, 0] - ekf_xy[:, 0]
    axes[1].plot(time_axis, x_err, color="#3498DB", linewidth=1.2,
                 label="X hatası")
    axes[1].axhline(0, color="#888", linewidth=0.8, linestyle="--")
    axes[1].fill_between(time_axis, 0, x_err, alpha=0.15, color="#3498DB")
    axes[1].set_ylabel("X hatası (m)", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linewidth=0.5, alpha=0.5)

    # --- Alt: Y hatası ---
    y_err = true_xy[:, 1] - ekf_xy[:, 1]
    axes[2].plot(time_axis, y_err, color="#2ECC71", linewidth=1.2,
                 label="Y hatası")
    axes[2].axhline(0, color="#888", linewidth=0.8, linestyle="--")
    axes[2].fill_between(time_axis, 0, y_err, alpha=0.15, color="#2ECC71")
    axes[2].set_ylabel("Y hatası (m)", fontsize=11)
    axes[2].set_xlabel("Zaman (s)", fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = _save(fig, "localization_error.png", save)

    stats = {"rmse": rmse, "mae": mae, "max_error": max_err, "n_steps": n}
    return fig, stats, path


# ============================================================ #
#  5. MOD DAĞILIMI — pasta grafiği
# ============================================================ #

def plot_mode_distribution(metrics, save=True):
    """Navigasyon mod dağılımı pasta grafiği. Ödev desteği."""
    dist = metrics.get_mode_distribution()
    if not dist:
        return None, None

    mode_color_map = {
        "GO_TO_GOAL":       "#2ECC71",
        "GAP_NAVIGATE":     "#F1C40F",
        "WALL_FOLLOW":      "#9B59B6",
        "ESCAPE":           "#E74C3C",
        "DEADLOCK_RECOVER": "#E67E22",
        "MANUEL":           "#3498DB",
    }

    labels = list(dist.keys())
    sizes = [dist[k] for k in labels]
    colors = [mode_color_map.get(k, "#95A5A6") for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Pasta
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Navigasyon Mod Dağılımı (%)", fontsize=13,
                  fontweight="bold")

    # Yatay bar (daha okunabilir)
    sorted_items = sorted(zip(sizes, labels, colors), reverse=True)
    s_sizes, s_labels, s_colors = zip(*sorted_items)
    bars = ax2.barh(s_labels, s_sizes, color=s_colors,
                    edgecolor="white", linewidth=1)
    ax2.set_xlabel("Kullanım oranı (%)", fontsize=11)
    ax2.set_title("Mod Kullanım Oranları", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linewidth=0.5, alpha=0.5)

    for bar, val in zip(bars, s_sizes):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = _save(fig, "mode_distribution.png", save)
    return fig, path


# ============================================================ #
#  6. METRİK ZAMAN SERİLERİ
# ============================================================ #

def plot_metrics_timeseries(metrics, dt=0.1, save=True):
    """
    Hız, engel mesafesi ve açısal hız zaman serisi grafikleri.
    """
    if not metrics.velocities:
        return None, None

    n = len(metrics.velocities)
    time_axis = np.arange(n) * dt

    v_arr = np.array([v for v, w in metrics.velocities])
    w_arr = np.array([w for v, w in metrics.velocities])
    obs_arr = np.array(metrics.min_obstacles)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Doğrusal hız
    axes[0].plot(time_axis, v_arr, color="#3498DB", linewidth=1.2)
    axes[0].fill_between(time_axis, 0, v_arr, alpha=0.15, color="#3498DB")
    axes[0].set_ylabel("v (m/s)", fontsize=11)
    axes[0].set_title("Doğrusal Hız — Zaman Serisi", fontsize=12,
                       fontweight="bold")
    axes[0].grid(True, linewidth=0.5, alpha=0.5)

    # Açısal hız
    axes[1].plot(time_axis, w_arr, color="#9B59B6", linewidth=1.2)
    axes[1].axhline(0, color="#888", linewidth=0.8, linestyle="--")
    axes[1].fill_between(time_axis, 0, w_arr, alpha=0.15, color="#9B59B6")
    axes[1].set_ylabel("ω (rad/s)", fontsize=11)
    axes[1].set_title("Açısal Hız — Zaman Serisi", fontsize=12,
                       fontweight="bold")
    axes[1].grid(True, linewidth=0.5, alpha=0.5)

    # Minimum engel mesafesi
    axes[2].plot(time_axis, obs_arr, color="#E74C3C", linewidth=1.2,
                 label="Min engel mesafesi")
    axes[2].axhline(0.3, color="#E67E22", linestyle="--", linewidth=1,
                    label="Robot yarıçapı (0.3 m)")
    axes[2].fill_between(time_axis, 0, obs_arr, alpha=0.15, color="#E74C3C")
    axes[2].set_ylabel("Mesafe (m)", fontsize=11)
    axes[2].set_xlabel("Zaman (s)", fontsize=11)
    axes[2].set_title("Minimum Engel Mesafesi — Zaman Serisi",
                       fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = _save(fig, "metrics_timeseries.png", save)
    return fig, path


# ============================================================ #
#  TOPLU ÇALIŞTIRICI — main.py'den çağrılır
# ============================================================ #

def generate_all_figures(env, robot, ekf, metrics,
                         last_raw_points, last_raw_distances,
                         last_filtered_points, last_filtered_distances,
                         last_angles, start, goal,
                         lidar_range, dt=0.1):
    """
    Tüm grafikleri tek seferde üretir ve figures/ klasörüne kaydeder.
    Döndürür: hata istatistikleri sözlüğü (RMSE, MAE, max_error)
    """
    _ensure_output_dir()
    error_stats = {}

    print("\n[analysis] Grafikler üretiliyor...")

    # 1. Ortam haritası
    plot_environment(env, start, goal)
    print("  [OK] environment_map.png")

    # 2. Yol karşılaştırması
    plot_path_comparison(env, robot.true_path, ekf.estimated_path,
                         start, goal)
    print("  [OK] path_comparison.png")

    # 3. LiDAR görselleştirmesi
    if (last_raw_points is not None and
            last_filtered_points is not None):
        plot_lidar_snapshot(
            robot.state,
            last_raw_points, last_raw_distances,
            last_filtered_points, last_filtered_distances,
            last_angles, lidar_range
        )
        print("  [OK] lidar_2d_view.png + lidar_profile.png")

    # 4. Lokalizasyon hatası
    result = plot_localization_error(
        robot.true_path, ekf.estimated_path, dt=dt
    )
    if result[0] is not None:
        _, stats, _ = result
        error_stats = stats
        print(f"  [OK] localization_error.png  "
              f"[RMSE={stats['rmse']:.4f}m  MAE={stats['mae']:.4f}m]")

    # 5. Mod dağılımı
    plot_mode_distribution(metrics)
    print("  [OK] mode_distribution.png")

    # 6. Metrik zaman serileri
    plot_metrics_timeseries(metrics, dt=dt)
    print("  [OK] metrics_timeseries.png")

    print(f"[analysis] Tüm grafikler '{OUTPUT_DIR}/' klasörüne kaydedildi.\n")
    plt.close("all")

    return error_stats


# ============================================================ #
#  YARDIMCI
# ============================================================ #

def _save(fig, filename, save):
    if not save:
        return None
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
