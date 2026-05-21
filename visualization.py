# visualization.py — Gelişmiş Görselleştirme + Manuel Kontrol

import pygame
import numpy as np


class PygameVisualizer:
    def __init__(self, env, scale=35):
        pygame.init()

        self.env = env
        self.scale = scale

        self.screen_width = int(env.width * scale)
        self.screen_height = int(env.height * scale)

        # Bilgi paneli için ekstra genişlik
        self.panel_width = 240
        self.total_width = self.screen_width + self.panel_width

        self.screen = pygame.display.set_mode(
            (self.total_width, self.screen_height)
        )

        pygame.display.set_caption("LiDAR Tabanlı Otonom Navigasyon — Gelişmiş")

        self.clock = pygame.time.Clock()
        self.running = True

        # Font
        self.font_large = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_title = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_overlay_big = pygame.font.SysFont("consolas", 42, bold=True)
        self.font_overlay_med = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_overlay_btn = pygame.font.SysFont("consolas", 18, bold=True)

        # Mod renkleri
        self.mode_colors = {
            "GO_TO_GOAL": (46, 204, 113),
            "GAP_NAVIGATE": (241, 196, 15),
            "WALL_FOLLOW": (155, 89, 182),
            "ESCAPE": (231, 76, 60),
            "DEADLOCK_RECOVER": (230, 126, 34),
            "MANUEL": (52, 152, 219),
            "N/A": (149, 165, 166),
        }

        # ---- Manuel kontrol durumu ---- #
        self.manual_mode = False
        self.restart_requested = False
        self.keys_pressed = {
            "w": False, "s": False,
            "a": False, "d": False,
        }

    def world_to_screen(self, x, y):
        screen_x = int(x * self.scale)
        screen_y = int(self.screen_height - y * self.scale)
        return screen_x, screen_y

    def draw(self, robot, goal, lidar_points=None,
             ekf=None, navigator=None, metrics=None):
        # ---- Ana simülasyon alanı ---- #
        self.screen.fill((30, 30, 35))

        # Simülasyon alanı arka planı
        sim_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height)
        pygame.draw.rect(self.screen, (240, 240, 235), sim_rect)

        # Grid çizgileri
        for i in range(int(self.env.width) + 1):
            sx, sy = self.world_to_screen(i, 0)
            sx2, sy2 = self.world_to_screen(i, self.env.height)
            pygame.draw.line(self.screen, (215, 215, 210), (sx, sy), (sx2, sy2), 1)

        for j in range(int(self.env.height) + 1):
            sx, sy = self.world_to_screen(0, j)
            sx2, sy2 = self.world_to_screen(self.env.width, j)
            pygame.draw.line(self.screen, (215, 215, 210), (sx, sy), (sx2, sy2), 1)

        # Engeller (gölgeli)
        for ox, oy, w, h in self.env.obstacles:
            left = ox - w / 2
            top = oy + h / 2

            screen_x, screen_y = self.world_to_screen(left, top)

            rect = pygame.Rect(
                screen_x, screen_y,
                int(w * self.scale), int(h * self.scale)
            )

            # Gölge
            shadow_rect = rect.copy()
            shadow_rect.x += 3
            shadow_rect.y += 3
            pygame.draw.rect(self.screen, (180, 180, 180), shadow_rect)

            # Engel
            pygame.draw.rect(self.screen, (180, 40, 40), rect)
            pygame.draw.rect(self.screen, (140, 30, 30), rect, 2)

        # EKF tahmini yol (mavi kesikli)
        if ekf is not None and len(ekf.estimated_path) > 1:
            ekf_points = [
                self.world_to_screen(p[0], p[1])
                for p in ekf.estimated_path
            ]
            for i in range(0, len(ekf_points) - 1, 2):
                end_idx = min(i + 1, len(ekf_points) - 1)
                pygame.draw.line(
                    self.screen, (100, 200, 255),
                    ekf_points[i], ekf_points[end_idx], 2
                )

        # Robot gerçek yolu
        if len(robot.true_path) > 1:
            points = [
                self.world_to_screen(p[0], p[1])
                for p in robot.true_path
            ]
            pygame.draw.lines(self.screen, (0, 80, 180), False, points, 3)

        # LiDAR noktaları
        if lidar_points is not None:
            x, y, theta = robot.state
            rx, ry = self.world_to_screen(x, y)

            for px, py in lidar_points:
                sx, sy = self.world_to_screen(px, py)
                pygame.draw.line(self.screen, (200, 200, 195), (rx, ry), (sx, sy), 1)
                pygame.draw.circle(self.screen, (0, 180, 50), (sx, sy), 2)

        # Hedef
        gx, gy = self.world_to_screen(goal[0], goal[1])
        pygame.draw.circle(self.screen, (255, 165, 0), (gx, gy), 20, 2)
        pygame.draw.circle(self.screen, (255, 165, 0), (gx, gy), 12)
        pygame.draw.circle(self.screen, (255, 220, 100), (gx, gy), 6)

        # Robot
        x, y, theta = robot.state
        rx, ry = self.world_to_screen(x, y)

        # Robot gövdesi — mod renginde
        if self.manual_mode:
            mode = "MANUEL"
        elif navigator is not None:
            mode = navigator.mode
        else:
            mode = "GO_TO_GOAL"

        robot_color = self.mode_colors.get(mode, (20, 120, 255))

        pygame.draw.circle(self.screen, robot_color, (rx, ry), int(0.3 * self.scale))
        pygame.draw.circle(self.screen, (0, 0, 0), (rx, ry), int(0.3 * self.scale), 2)

        # Robot yönü
        direction_length = 0.8
        hx = x + direction_length * np.cos(theta)
        hy = y + direction_length * np.sin(theta)
        hx_s, hy_s = self.world_to_screen(hx, hy)
        pygame.draw.line(self.screen, (0, 0, 0), (rx, ry), (hx_s, hy_s), 3)

        # EKF tahmini pozisyon
        if ekf is not None:
            ekf_state = ekf.get_state()
            ex, ey = self.world_to_screen(ekf_state[0], ekf_state[1])
            uncertainty = ekf.get_uncertainty() * self.scale
            if uncertainty > 2:
                pygame.draw.circle(
                    self.screen, (100, 200, 255),
                    (ex, ey), max(3, int(uncertainty)), 1
                )

        # ---- Bilgi Paneli ---- #
        self._draw_info_panel(navigator, metrics, ekf, robot, goal)

        pygame.display.flip()

    def _draw_info_panel(self, navigator, metrics, ekf, robot, goal):
        """Sağ taraftaki bilgi paneli."""
        panel_x = self.screen_width
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.screen_height)
        pygame.draw.rect(self.screen, (35, 35, 45), panel_rect)

        x_offset = panel_x + 15
        y_pos = 15

        # Başlık
        title = self.font_title.render("KONTROL PANELİ", True, (220, 220, 230))
        self.screen.blit(title, (x_offset, y_pos))
        y_pos += 30

        # Ayırıcı çizgi
        pygame.draw.line(
            self.screen, (80, 80, 100),
            (x_offset, y_pos), (panel_x + self.panel_width - 15, y_pos), 1
        )
        y_pos += 15

        # ---- Manuel/Otonom göstergesi ---- #
        if self.manual_mode:
            ctrl_text = "MANUEL (WASD)"
            ctrl_color = (52, 152, 219)
        else:
            ctrl_text = "OTONOM"
            ctrl_color = (46, 204, 113)

        ctrl_label = self.font_small.render("KONTROL:", True, (160, 160, 180))
        self.screen.blit(ctrl_label, (x_offset, y_pos))
        y_pos += 18
        ctrl_surf = self.font_large.render(ctrl_text, True, ctrl_color)
        self.screen.blit(ctrl_surf, (x_offset, y_pos))
        y_pos += 20

        toggle_hint = self.font_small.render("[M] ile değiştir", True, (120, 120, 140))
        self.screen.blit(toggle_hint, (x_offset, y_pos))
        y_pos += 22

        # ---- Mod bilgisi ---- #
        if self.manual_mode:
            mode = "MANUEL"
        elif navigator is not None:
            mode = navigator.mode
        else:
            mode = "N/A"

        mode_label = self.font_small.render("MOD:", True, (160, 160, 180))
        self.screen.blit(mode_label, (x_offset, y_pos))
        y_pos += 18

        mode_color = self.mode_colors.get(mode, (149, 165, 166))
        indicator_rect = pygame.Rect(x_offset, y_pos, 10, 10)
        pygame.draw.rect(self.screen, mode_color, indicator_rect)
        mode_text = self.font_large.render(f" {mode}", True, mode_color)
        self.screen.blit(mode_text, (x_offset + 14, y_pos - 3))
        y_pos += 30

        # ---- Robot durumu ---- #
        x, y, theta = robot.state
        self._draw_section(x_offset, y_pos, "ROBOT DURUMU")
        y_pos += 22
        y_pos = self._draw_stat(x_offset, y_pos, "Konum X", f"{x:.2f} m")
        y_pos = self._draw_stat(x_offset, y_pos, "Konum Y", f"{y:.2f} m")
        y_pos = self._draw_stat(x_offset, y_pos, "Açı θ", f"{np.degrees(theta):.1f}°")

        dist = np.hypot(x - goal[0], y - goal[1])
        y_pos = self._draw_stat(x_offset, y_pos, "Hedefe", f"{dist:.2f} m")
        y_pos += 8

        # ---- EKF bilgisi ---- #
        if ekf is not None:
            self._draw_section(x_offset, y_pos, "EKF TAHMİNİ")
            y_pos += 22
            ekf_state = ekf.get_state()
            y_pos = self._draw_stat(x_offset, y_pos, "Tahmin X", f"{ekf_state[0]:.2f} m")
            y_pos = self._draw_stat(x_offset, y_pos, "Tahmin Y", f"{ekf_state[1]:.2f} m")
            uncertainty = ekf.get_uncertainty()
            y_pos = self._draw_stat(x_offset, y_pos, "Belirsizlik", f"{uncertainty:.4f}")

            error = np.hypot(x - ekf_state[0], y - ekf_state[1])
            y_pos = self._draw_stat(x_offset, y_pos, "EKF Hata", f"{error:.4f} m")
            y_pos += 8

        # ---- Metrikler ---- #
        if metrics is not None:
            stats = metrics.get_live_stats()
            self._draw_section(x_offset, y_pos, "METRİKLER")
            y_pos += 22
            y_pos = self._draw_stat(x_offset, y_pos, "Yol Uz.", f"{stats['path_length']:.2f} m")
            y_pos = self._draw_stat(x_offset, y_pos, "Min Engel", f"{stats['min_obstacle']:.2f} m")
            y_pos = self._draw_stat(x_offset, y_pos, "Ort. Hız", f"{stats['avg_speed']:.3f} m/s")
            y_pos = self._draw_stat(x_offset, y_pos, "Adım", f"{stats['steps']}")
            y_pos = self._draw_stat(x_offset, y_pos, "Süre", f"{stats['time']:.1f} sn")
            y_pos = self._draw_stat(x_offset, y_pos, "Çarpışma", f"{stats['collisions']}")
            y_pos += 8

        # ---- VFH Histogramı ---- #
        if navigator is not None and navigator.vfh_histogram is not None:
            self._draw_section(x_offset, y_pos, "VFH HİSTOGRAM")
            y_pos += 22
            self._draw_vfh_mini(x_offset, y_pos, navigator)
            y_pos += 70

        # ---- WASD görseli (Manuel modda) ---- #
        if self.manual_mode:
            self._draw_wasd_indicator(x_offset, self.screen_height - 170)

        # ---- Renk açıklaması ---- #
        y_pos = self.screen_height - 100
        self._draw_section(x_offset, y_pos, "KISA YOLLAR")
        y_pos += 20
        shortcuts = [
            ("[M] Manuel/Otonom", (180, 180, 190)),
            ("[WASD] Hareket", (52, 152, 219)),
            ("[R] Yeniden Başlat", (241, 196, 15)),
            ("[ESC] Çıkış", (231, 76, 60)),
        ]
        for text, color in shortcuts:
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (x_offset, y_pos))
            y_pos += 16

    def _draw_wasd_indicator(self, x, y):
        """WASD tuş göstergesini çiz."""
        key_size = 28
        gap = 4
        bg_color = (50, 50, 65)
        active_color = (52, 152, 219)
        inactive_color = (80, 80, 100)
        text_color = (220, 220, 230)

        keys_layout = [
            ("W", x + key_size + gap, y, self.keys_pressed["w"]),
            ("A", x, y + key_size + gap, self.keys_pressed["a"]),
            ("S", x + key_size + gap, y + key_size + gap, self.keys_pressed["s"]),
            ("D", x + 2 * (key_size + gap), y + key_size + gap, self.keys_pressed["d"]),
        ]

        for label, kx, ky, pressed in keys_layout:
            color = active_color if pressed else inactive_color
            rect = pygame.Rect(kx, ky, key_size, key_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, bg_color, rect, 2, border_radius=4)

            text = self.font_large.render(label, True, text_color)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def _draw_section(self, x, y, title):
        """Bölüm başlığı çiz."""
        text = self.font_large.render(title, True, (100, 180, 255))
        self.screen.blit(text, (x, y))

    def _draw_stat(self, x, y, label, value):
        """İstatistik satırı çiz."""
        label_surf = self.font_small.render(f"{label}:", True, (140, 140, 160))
        value_surf = self.font_small.render(value, True, (220, 220, 230))
        self.screen.blit(label_surf, (x, y))
        self.screen.blit(value_surf, (x + 110, y))
        return y + 17

    def _draw_vfh_mini(self, x, y, navigator):
        """Küçük VFH histogramı çiz."""
        histogram = navigator.vfh_histogram
        n = len(histogram)
        bar_width = max(1, (self.panel_width - 30) // n)
        max_height = 50

        for i in range(n):
            val = histogram[i]
            h = int(val * max_height)

            bar_x = x + i * bar_width
            bar_y = y + max_height - h

            if val > 0.6:
                color = (231, 76, 60)
            elif val > 0.3:
                color = (241, 196, 15)
            else:
                color = (46, 204, 113)

            if h > 0:
                pygame.draw.rect(
                    self.screen, color,
                    (bar_x, bar_y, max(1, bar_width - 1), h)
                )

        pygame.draw.rect(
            self.screen, (80, 80, 100),
            (x, y, n * bar_width, max_height), 1
        )

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                # M tuşu: Manuel/Otonom geçişi
                elif event.key == pygame.K_m:
                    self.manual_mode = not self.manual_mode

                # R tuşu: Simülasyonu yeniden başlat
                elif event.key == pygame.K_r:
                    self.restart_requested = True

                # WASD tuşları basıldı
                elif event.key == pygame.K_w:
                    self.keys_pressed["w"] = True
                elif event.key == pygame.K_s:
                    self.keys_pressed["s"] = True
                elif event.key == pygame.K_a:
                    self.keys_pressed["a"] = True
                elif event.key == pygame.K_d:
                    self.keys_pressed["d"] = True

            elif event.type == pygame.KEYUP:
                # WASD tuşları bırakıldı
                if event.key == pygame.K_w:
                    self.keys_pressed["w"] = False
                elif event.key == pygame.K_s:
                    self.keys_pressed["s"] = False
                elif event.key == pygame.K_a:
                    self.keys_pressed["a"] = False
                elif event.key == pygame.K_d:
                    self.keys_pressed["d"] = False

    def get_manual_control(self, linear_speed, angular_speed):
        """
        WASD tuşlarına göre manuel kontrol komutları üret.
        W: ileri, S: geri, A: sola dön, D: sağa dön
        """
        v = 0.0
        w = 0.0

        if self.keys_pressed["w"]:
            v += linear_speed
        if self.keys_pressed["s"]:
            v -= linear_speed
        if self.keys_pressed["a"]:
            w += angular_speed
        if self.keys_pressed["d"]:
            w -= angular_speed

        return v, w

    def tick(self, fps=30):
        self.clock.tick(fps)

    def draw_end_screen(self, end_type, robot, goal, lidar_points,
                        ekf=None, navigator=None, metrics=None):
        """
        Simülasyon sonu ekranı.
        end_type: 'collision' veya 'goal_reached'
        """
        # Arka plan olarak son kareyi çiz
        self.draw(robot, goal, lidar_points, ekf=ekf, navigator=navigator, metrics=metrics)

        # Yarı saydam karartma overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        cx = self.screen_width // 2
        cy = self.screen_height // 2

        if end_type == 'collision':
            # Başlık
            title_text = "ÇARPIŞMA OLDU!"
            title_color = (231, 76, 60)
            icon = "💥"
        else:
            title_text = "HEDEFE ULAŞILDI!"
            title_color = (46, 204, 113)
            icon = "🎯"

        # Çerçeve kutusu
        box_w, box_h = 420, 220
        box_rect = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)
        pygame.draw.rect(self.screen, (40, 40, 55), box_rect, border_radius=12)
        pygame.draw.rect(self.screen, title_color, box_rect, 3, border_radius=12)

        # Başlık yazısı
        title_surf = self.font_overlay_big.render(title_text, True, title_color)
        title_rect = title_surf.get_rect(centerx=cx, top=box_rect.top + 25)
        self.screen.blit(title_surf, title_rect)

        # Alt bilgi
        if metrics is not None:
            stats = metrics.get_live_stats()
            info_text = f"Yol: {stats['path_length']:.1f}m  |  Süre: {stats['time']:.1f}sn"
            info_surf = self.font_overlay_med.render(info_text, True, (180, 180, 200))
            info_rect = info_surf.get_rect(centerx=cx, top=title_rect.bottom + 15)
            self.screen.blit(info_surf, info_rect)

        # ---- Butonlar ---- #
        btn_w, btn_h = 160, 45
        btn_y = box_rect.bottom - 70

        # Restart butonu
        self.restart_btn_rect = pygame.Rect(cx - btn_w - 15, btn_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, (241, 196, 15), self.restart_btn_rect, border_radius=8)
        restart_text = self.font_overlay_btn.render("[R] Yeniden", True, (30, 30, 30))
        restart_text_rect = restart_text.get_rect(center=self.restart_btn_rect.center)
        self.screen.blit(restart_text, restart_text_rect)

        # Çıkış butonu
        self.exit_btn_rect = pygame.Rect(cx + 15, btn_y, btn_w, btn_h)
        pygame.draw.rect(self.screen, (231, 76, 60), self.exit_btn_rect, border_radius=8)
        exit_text = self.font_overlay_btn.render("[ESC] Çıkış", True, (255, 255, 255))
        exit_text_rect = exit_text.get_rect(center=self.exit_btn_rect.center)
        self.screen.blit(exit_text, exit_text_rect)

        pygame.display.flip()

    def handle_end_events(self):
        """
        Bitiş ekranındaki olayları işle.
        Returns: 'restart', 'quit', veya None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                elif event.key == pygame.K_r:
                    return 'restart'

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if hasattr(self, 'restart_btn_rect') and self.restart_btn_rect.collidepoint(mx, my):
                    return 'restart'
                if hasattr(self, 'exit_btn_rect') and self.exit_btn_rect.collidepoint(mx, my):
                    return 'quit'

        return None

    def close(self):
        pygame.quit()