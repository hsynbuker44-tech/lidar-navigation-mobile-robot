# navigation.py — Gap-Based Navigation + Güçlendirilmiş Reaktif Güvenlik Katmanı
#
# İyileştirmeler (v2):
#   1. Bölge analizi genişletildi: 5 bölge → 8 bölge (daha hassas ön alan)
#   2. Escape yönü: toplam yerine en uzak sektöre yönelim
#   3. Deadlock recovery: alternating yön (ilk sıkışma ↺, ikinci ↻, ...)
#   4. Hız profili sürekli (smooth) — sabit basamaklar kaldırıldı
#   5. Gap skorlaması: dar geçit genişliği robot çapıyla normalize edildi
#   6. WALL_FOLLOW çıkışı: m-line + hedef açısı birlikte değerlendirildi
#
# İyileştirmeler (v3) — dar geçit + arka çarpışma düzeltmesi:
#   7. ESCAPE: geri gitmeden önce arka mesafe kontrol edilir; arka kapalıysa
#      yalnızca yerinde döner (v=0), kör geri gitmez
#   8. GAP_NAVIGATE: dar geçit için minimum gap açısı 5°'ye düşürüldü ve
#      geçidin derinliği yetersizse (avg_depth < CAUTION_DIST) skordan düşülür
#   9. Tüm arka bölge (135°-180°) ayrı rear_dist olarak izlenir; ESCAPE'de
#      kullanılır ve VERY_CLOSE'a girerse geri hareket engellenir

import numpy as np
from config import (
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    ROBOT_RADIUS, VFH_SAFETY_DISTANCE,
    VERY_CLOSE_DIST, DANGER_DIST, CAUTION_DIST, CLEAR_DIST,
    WALL_FOLLOW_DIST, MLINE_TOLERANCE,
    DEADLOCK_WINDOW, DEADLOCK_MIN_PROGRESS, DEADLOCK_RECOVERY_STEPS,
    SMOOTH_ALPHA_V, SMOOTH_ALPHA_W,
    LIDAR_RANGE
)


class Navigator:
    """
    Gap-based navigasyon + güçlendirilmiş reaktif güvenlik katmanı.

    Modlar:
    - GO_TO_GOAL       : Hedefe doğru git
    - GAP_NAVIGATE     : Gap-based engelden kaçınma
    - WALL_FOLLOW      : Bug2 duvar takibi
    - ESCAPE           : Acil geri çekilme
    - DEADLOCK_RECOVER : Sıkışmadan kurtulma
    """

    def __init__(self, goal, start=None):
        self.goal = np.array(goal)
        self.start = np.array(start[:2]) if start is not None else None

        self.mode = "GO_TO_GOAL"
        self.turn_direction = 1

        self.escape_counter = 0
        self.avoid_counter = 0
        self.wall_follow_counter = 0
        self.deadlock_counter = 0

        # Kaç kez deadlock oldu — alternating yön için
        self.deadlock_occurrences = 0

        self.prev_v = 0.0
        self.prev_w = 0.0

        self.vfh_histogram = None
        self.best_gap_angle = None

        self.position_history = []

        self.mline_start = self.start
        self.wall_follow_start_pos = None
        self.wall_follow_start_dist = None
        self.wall_follow_direction = 1

    # ================================================================== #
    #  ANA KONTROL
    # ================================================================== #

    def compute_control(self, robot_state, lidar_distances=None, lidar_angles=None):
        x, y, theta = robot_state
        pos = np.array([x, y])

        goal_vec = self.goal - pos
        desired_angle = np.arctan2(goal_vec[1], goal_vec[0])
        angle_to_goal = self.normalize_angle(desired_angle - theta)
        distance_to_goal = np.linalg.norm(goal_vec)

        self.position_history.append(pos.copy())

        if lidar_distances is None or lidar_angles is None:
            v = min(MAX_LINEAR_SPEED, 0.5 * distance_to_goal)
            w = 2.5 * angle_to_goal
            return self.smooth_control(v, w)

        # ---- Genişletilmiş bölge analizi (9 bölge) ---- #
        front_narrow  = self.region_min(lidar_distances, lidar_angles,  -12,  12)
        front_left    = self.region_min(lidar_distances, lidar_angles,   12,  50)
        front_right   = self.region_min(lidar_distances, lidar_angles,  -50, -12)
        front         = min(front_narrow, front_left, front_right)
        side_left     = self.region_min(lidar_distances, lidar_angles,   50,  90)
        side_right    = self.region_min(lidar_distances, lidar_angles,  -90, -50)
        rear_left     = self.region_min(lidar_distances, lidar_angles,   90, 135)
        rear_right    = self.region_min(lidar_distances, lidar_angles, -135, -90)
        # YENİ: tam arka bölge — ESCAPE kör geri gidişini önler
        rear_center   = self.region_min(lidar_distances, lidar_angles,  150, 180)
        rear_center   = min(rear_center,
                            self.region_min(lidar_distances, lidar_angles, -180, -150))
        rear_dist     = min(rear_left, rear_right, rear_center)
        left          = min(side_left, rear_left * 0.7)
        right         = min(side_right, rear_right * 0.7)

        # Global minimum — tüm LiDAR'dan en yakın engel
        global_min = np.min(lidar_distances)
        min_front_area = front  # artık tam ön bölge

        # Dar koridor tespiti: her iki yanda yakın engel ama önde boşluk var
        in_narrow_corridor = self._is_in_narrow_corridor(
            front_narrow, left, right, side_left, side_right
        )

        # ---- Gap analizi ---- #
        best_gap_angle, gaps = self.find_best_gap(
            lidar_distances, lidar_angles, angle_to_goal
        )
        self.best_gap_angle = best_gap_angle

        # Görselleştirme histogramı
        self.vfh_histogram = self.build_display_histogram(lidar_distances, lidar_angles)

        # ============================================================== #
        #  REAKTİF GÜVENLİK KATMANI — Her zaman aktif
        # ============================================================== #

        # 1) Temas noktasında → acil ESCAPE
        # Dar koridorda ise: eğer önde gap varsa ve ikinci yandan yaklaşıyorsa
        # ESCAPE tetikleme — robot sığabiliyorsa geçsin
        if global_min < VERY_CLOSE_DIST:
            if in_narrow_corridor and front_narrow > VERY_CLOSE_DIST * 1.5:
                # Dar koridorda önü açık → ESCAPE tetikleme, GAP_NAVIGATE'e geç
                if self.mode not in ("GAP_NAVIGATE", "ESCAPE"):
                    self.mode = "GAP_NAVIGATE"
                    self.avoid_counter = 0
            elif self.mode != "ESCAPE":
                self.mode = "ESCAPE"
                self.escape_counter = 0
                # İYİLEŞTİRME: en uzak sektöre yönelim (toplam yerine)
                self.turn_direction = self._best_escape_direction(
                    lidar_distances, lidar_angles
                )

        # 2) Tehlike bölgesinde → GAP_NAVIGATE'e zorla
        elif global_min < DANGER_DIST and self.mode == "GO_TO_GOAL":
            self.mode = "GAP_NAVIGATE"
            self.avoid_counter = 0

        # ---- Sıkışma algılama ---- #
        if self.detect_deadlock():
            if self.mode != "DEADLOCK_RECOVER":
                self.mode = "DEADLOCK_RECOVER"
                self.deadlock_counter = 0
                self.deadlock_occurrences += 1
                # İYİLEŞTİRME: alternating yön — her sıkışmada farklı taraf
                if self.deadlock_occurrences % 2 == 1:
                    self.turn_direction = 1 if side_left > side_right else -1
                else:
                    self.turn_direction = -1 if side_left > side_right else 1

        # ---- Normal mod geçişleri ---- #
        if self.mode not in ("DEADLOCK_RECOVER", "ESCAPE"):
            if self.mode == "GO_TO_GOAL":
                # Proaktif geçiş: ön alanda CLEAR_DIST'ten önce engel görünürse
                # hemen gap moduna gir — çok yaklaşmayı bekleme
                forward_blocked = (
                    front_narrow < CLEAR_DIST or
                    front_left   < CAUTION_DIST or
                    front_right  < CAUTION_DIST
                )
                if forward_blocked:
                    self.mode = "GAP_NAVIGATE"
                    self.avoid_counter = 0

            elif self.mode == "GAP_NAVIGATE":
                # Ortam tamamen temiz → GO_TO_GOAL
                # Geriye dönmek için CLEAR_DIST'ten daha güvenli bir eşik
                if (self.avoid_counter > 15 and
                        front > CLEAR_DIST * 1.1 and
                        side_left > CAUTION_DIST and
                        side_right > CAUTION_DIST and
                        global_min > DANGER_DIST * 1.5):
                    self.mode = "GO_TO_GOAL"

        # ---- Mod işlemleri ---- #
        if self.mode == "DEADLOCK_RECOVER":
            return self._deadlock_recover(front, angle_to_goal,
                                          side_left, side_right)

        if self.mode == "ESCAPE":
            return self._escape(front_narrow, front_left, front_right,
                                side_left, side_right, global_min, rear_dist)

        if self.mode == "WALL_FOLLOW":
            return self._wall_follow(
                robot_state, front, front_left, front_right,
                side_left, side_right, angle_to_goal, distance_to_goal
            )

        if self.mode == "GAP_NAVIGATE":
            return self._gap_navigate(
                best_gap_angle, angle_to_goal,
                front, front_left, front_right, side_left, side_right,
                min_front_area, distance_to_goal, gaps, global_min,
                lidar_distances, lidar_angles,
                in_narrow_corridor=in_narrow_corridor
            )

        # GO_TO_GOAL — best_gap_angle ile proaktif harmanlama
        return self._go_to_goal(angle_to_goal, distance_to_goal, min_front_area,
                                global_min, front_left, front_right,
                                side_left, side_right,
                                in_narrow_corridor=in_narrow_corridor,
                                best_gap_angle=best_gap_angle)

    # ================================================================== #
    #  İYİLEŞTİRME: En iyi escape yönü — en açık sektöre yönelim
    # ================================================================== #

    def _best_escape_direction(self, lidar_distances, lidar_angles):
        """
        LiDAR'ı 8 sektöre böl, her sektörün ortalama mesafesini hesapla.
        En uzak sektörün sol/sağ tarafını döndür (+1 sol, -1 sağ).
        """
        n_sectors = 8
        sector_size = np.pi / n_sectors   # sadece ±90° (ön yarım daire)
        sector_means = np.zeros(n_sectors)

        for i in range(n_sectors):
            lo = -np.pi / 2 + i * sector_size
            hi = lo + sector_size
            mask = (lidar_angles >= lo) & (lidar_angles < hi)
            if np.any(mask):
                sector_means[i] = np.mean(lidar_distances[mask])
            else:
                sector_means[i] = 0.0

        best_sector = int(np.argmax(sector_means))
        # Sektör 0-3: sağ yarı → -1, sektör 4-7: sol yarı → +1
        return 1 if best_sector >= n_sectors // 2 else -1

    # ================================================================== #
    #  DAR KORİDOR TESPİTİ
    # ================================================================== #

    def _is_in_narrow_corridor(self, front_narrow, left, right,
                                side_left, side_right):
        """
        Her iki yanda da engel var ama ön açık ise dar koridor.
        Robot çapının 1.5 katından geniş bir boşluk varsa geçilebilir sayılır.
        """
        min_passable = 2 * ROBOT_RADIUS  # ~0.6 m
        both_sides_close = (side_left < CAUTION_DIST and side_right < CAUTION_DIST)
        front_open = front_narrow > min_passable * 1.2
        symmetric = abs(side_left - side_right) < CAUTION_DIST * 0.8
        return both_sides_close and front_open and symmetric

    # ================================================================== #
    #  GAP BULMA
    # ================================================================== #

    def find_best_gap(self, lidar_distances, lidar_angles, angle_to_goal):
        """
        LiDAR verisinden boş koridorları bul.
        İYİLEŞTİRME: Minimum gap genişliği robot çapı ile orantılı.
        """
        n = len(lidar_distances)
        if n == 0:
            return angle_to_goal, []

        # Güvenlik mesafesi = robot yarıçapının 1.5 katı + küçük marj
        # Not: DANGER_DIST çok büyük olduğunda dar geçitleri kapatıyor,
        # bu yüzden robot boyutuna dayalı gerçekçi bir eşik kullanıyoruz.
        safety_dist = ROBOT_RADIUS * 2.5 + 0.15   # ~0.9 m

        is_free = lidar_distances > safety_dist

        gaps = []
        gap_start = None

        for i in range(n):
            if is_free[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    self._add_gap(gaps, gap_start, i - 1, lidar_angles, lidar_distances)
                    gap_start = None

        if gap_start is not None:
            self._add_gap(gaps, gap_start, n - 1, lidar_angles, lidar_distances)

        if not gaps:
            best_idx = np.argmax(lidar_distances)
            return lidar_angles[best_idx], []

        best_gap = None
        best_score = -float('inf')

        for center_angle, width_rad, avg_depth in gaps:
            # 1) Hedefe yakınlık — ağırlık artırıldı: hedef yönündeki gap tercihli
            goal_diff = abs(self.normalize_angle(center_angle - angle_to_goal))
            goal_score = -goal_diff * 2.0

            # Bonus: 30° içindeyse ekstra puan
            if goal_diff < np.deg2rad(30):
                goal_score += 1.0

            # 2) Genişlik skoru — robot çapı normalize
            passable_ratio = min(width_rad / np.deg2rad(15), 2.0)
            width_score = 0.7 * passable_ratio

            # 3) Derinlik skoru — EN ÖNEMLİ: önde yeterli yer var mı?
            # Sığ gap'lere büyük ceza, derin gap'lere büyük bonus
            if avg_depth < DANGER_DIST:
                depth_score = -2.0   # çok sığ — bu gap gerçek değil
            elif avg_depth < CAUTION_DIST:
                depth_score = -0.8 * (1.0 - avg_depth / CAUTION_DIST)
            else:
                depth_score = 1.2 * min(avg_depth / LIDAR_RANGE, 1.0)

            score = goal_score + width_score + depth_score

            if score > best_score:
                best_score = score
                best_gap = center_angle

        return best_gap if best_gap is not None else angle_to_goal, gaps

    def _add_gap(self, gaps, start_idx, end_idx, angles, distances):
        width_rad = abs(angles[end_idx] - angles[start_idx])
        # Minimum gap açısı: robot çapını geçmeye yetecek açısal genişlik
        # Yakın engelde (1m) 0.6m = ~34 derece, uzakta daha az
        avg_d = np.mean(distances[start_idx:end_idx + 1])
        min_gap_rad = np.arctan2(ROBOT_RADIUS * 2.2, max(avg_d, 0.5))
        if width_rad < min_gap_rad:
            return

        center_angle = (angles[start_idx] + angles[end_idx]) / 2.0
        gaps.append((center_angle, width_rad, avg_d))

    # ================================================================== #
    #  GO_TO_GOAL — Proaktif gap-blended hedef navigasyonu
    # ================================================================== #

    def _go_to_goal(self, angle_to_goal, distance_to_goal, min_front, global_min,
                    front_left, front_right, left, right,
                    in_narrow_corridor=False, best_gap_angle=None):
        """
        GO_TO_GOAL artık tamamen kör değil: LiDAR'dan gelen best_gap_angle
        ile hedef yönünü harmanlar. Engele ne kadar yakınsa gap'e o kadar
        ağırlık verir. Böylece robot çok önceden sapar — çarpışmaz.
        """
        v = min(MAX_LINEAR_SPEED, 0.6 * distance_to_goal)

        if distance_to_goal < 1.5:
            v *= 0.5

        # Hız profili: global_min'e göre yavaşla
        if global_min < CLEAR_DIST:
            speed_factor = np.clip(
                (global_min - ROBOT_RADIUS) / (CLEAR_DIST - ROBOT_RADIUS),
                0.25, 1.0
            )
            v *= speed_factor

        # ── Proaktif gap yönlendirmesi ─────────────────────────────────
        # gap_blend: 0 = tamamen hedefe bak, 1 = tamamen gap'e bak
        # CLEAR_DIST'ten itibaren gap etkisi devreye girer,
        # CAUTION_DIST'e gelince tam gap modunda çalışır.
        if best_gap_angle is not None and global_min < CLEAR_DIST:
            blend = np.clip(
                (CLEAR_DIST - global_min) / (CLEAR_DIST - CAUTION_DIST),
                0.0, 0.85   # max %85 gap, her zaman biraz hedefe bak
            )
            steering_angle = (1.0 - blend) * angle_to_goal + blend * best_gap_angle
        else:
            steering_angle = angle_to_goal

        w = 2.8 * steering_angle

        # Lateral itme kuvveti — dar koridorda devre dışı
        if not in_narrow_corridor:
            w += self._lateral_repulsion(left, right, front_left, front_right)

        return self.smooth_control(v, w)

    # ================================================================== #
    #  GAP NAVİGASYON
    # ================================================================== #

    def _gap_navigate(self, best_gap_angle, angle_to_goal,
                      front, front_left, front_right, left, right,
                      min_front, distance_to_goal, gaps, global_min,
                      lidar_distances, lidar_angles,
                      in_narrow_corridor=False):
        self.avoid_counter += 1

        if best_gap_angle is not None and len(gaps) > 0:
            gap_error = best_gap_angle

            # Dar koridorda: önü açık ve gap küçük açıda → dümdüz git
            if in_narrow_corridor and abs(gap_error) < np.deg2rad(20):
                v = MAX_LINEAR_SPEED * 0.55
                w = 1.8 * gap_error   # sadece küçük düzeltme
                return self.smooth_control(v, w)

            # Sürekli hız profili
            if global_min < VERY_CLOSE_DIST * 1.3:
                v = 0.06
            else:
                t = np.clip(
                    (global_min - VERY_CLOSE_DIST) /
                    (CAUTION_DIST - VERY_CLOSE_DIST),
                    0.0, 1.0
                )
                v = 0.08 + t * (MAX_LINEAR_SPEED * 0.75 - 0.08)

            # Mesafeye göre ek yavaşlama
            v = min(v, 0.5 * distance_to_goal + 0.05)

            # Dönüş hesapı:
            # Köşe geçişinde gain düşür — büyük gap_error'da daha yavaş dön
            if abs(gap_error) > np.deg2rad(45):
                turn_gain = 1.8   # çok büyük açı — sert dönme, yavaşça çevril
                v *= 0.35
            elif abs(gap_error) > np.deg2rad(25):
                turn_gain = 2.2   # orta açı
                v *= 0.55
            else:
                turn_gain = 3.0   # küçük açı — normal

            w = turn_gain * gap_error + 0.25 * angle_to_goal

            # Lateral itme kuvveti — dar koridorda DEVRE DIŞI
            if not in_narrow_corridor:
                w += self._lateral_repulsion(left, right, front_left, front_right)
            else:
                corridor_k = 0.4
                if left < ROBOT_RADIUS * 3.0 and left < right:
                    w -= corridor_k * (1.0 - left / (ROBOT_RADIUS * 3.0))
                elif right < ROBOT_RADIUS * 3.0 and right < left:
                    w += corridor_k * (1.0 - right / (ROBOT_RADIUS * 3.0))

            # Köşe dampingi: ön alanın her iki tarafı da yakınsa w'yu sınırla
            corner_detected = (front_left < CAUTION_DIST and front_right < CAUTION_DIST)
            if corner_detected:
                w_limit = 1.0   # köşede maksimum açısal hız
                w = np.clip(w, -w_limit, w_limit)

        else:
            # Gap yok → duvar takibine geç
            self.mode = "WALL_FOLLOW"
            self.wall_follow_counter = 0
            self.wall_follow_start_pos = np.array(self.position_history[-1])
            self.wall_follow_start_dist = np.linalg.norm(
                self.goal - self.position_history[-1]
            )
            self.wall_follow_direction = 1 if left > right else -1
            return self.smooth_control(0.08, 1.2 * self.wall_follow_direction)

        return self.smooth_control(v, w)

    # ================================================================== #
    #  BUG2 DUVAR TAKİBİ
    # ================================================================== #

    def _wall_follow(self, robot_state, front, front_left, front_right,
                     left, right, angle_to_goal, distance_to_goal):
        x, y, theta = robot_state
        self.wall_follow_counter += 1

        d = self.wall_follow_direction
        wall_dist = left if d == 1 else right
        wall_front = front_left if d == 1 else front_right

        # M-line kontrolü
        if self.wall_follow_counter > 20 and self.mline_start is not None:
            if self.is_on_mline(x, y) and distance_to_goal < self.wall_follow_start_dist - 0.3:
                self.mode = "GO_TO_GOAL"
                return self.smooth_control(0.25, 2.5 * angle_to_goal)

        # İYİLEŞTİRME: Çıkış koşuluna hedef açısı da eklendi
        # Hem yol açık hem de hedefe bakıyoruz
        if (self.wall_follow_counter > 20 and
                front > CLEAR_DIST and
                front_left > CAUTION_DIST and
                front_right > CAUTION_DIST and
                abs(angle_to_goal) < np.deg2rad(60)):
            self.mode = "GO_TO_GOAL"
            return self.smooth_control(0.25, 2.5 * angle_to_goal)

        v = 0.22

        if front < DANGER_DIST:
            v = 0.05
            w = d * 2.2
        elif front < CAUTION_DIST:
            v = 0.12
            w = d * 1.2
        else:
            wall_error = WALL_FOLLOW_DIST - wall_dist
            wall_front_error = WALL_FOLLOW_DIST - wall_front
            w = -d * (1.8 * wall_error + 0.7 * wall_front_error)
            w += 0.2 * angle_to_goal

        return self.smooth_control(v, w)

    # ================================================================== #
    #  ESCAPE — Acil geri çekilme (arka kontrollü)
    # ================================================================== #

    def _escape(self, front, front_left, front_right, left, right,
                global_min, rear_dist):
        self.escape_counter += 1
        turn = self.turn_direction

        # ── Arka güvenli mi? ──────────────────────────────────────────
        # Eğer arkada da engel varsa (rear_dist < DANGER_DIST) geri
        # gitmek yerine yerinde dönerek gap ara.
        rear_safe = rear_dist > DANGER_DIST

        if self.escape_counter <= 5:
            if rear_safe:
                v = -0.30          # geri git — arka açık
                w = turn * 2.2
            else:
                v = 0.0            # yerinde dön — arka kapalı
                w = turn * 2.8
        elif self.escape_counter <= 12:
            if rear_safe:
                v = 0.08
                w = turn * 2.2
            else:
                v = 0.0
                w = turn * 2.5
        else:
            v = 0.15
            w = turn * 1.5

        # Çıkış: ön açıldı ve yeterince döndük
        if self.escape_counter > 8 and front > DANGER_DIST * 1.2:
            self.mode = "GAP_NAVIGATE"
            self.avoid_counter = 0

        return self.smooth_control(v, w, allow_reverse=rear_safe)

    # ================================================================== #
    #  DEADLOCK KURTARMA — alternating yön
    # ================================================================== #

    def _deadlock_recover(self, front, angle_to_goal, left, right):
        self.deadlock_counter += 1

        if self.deadlock_counter < DEADLOCK_RECOVERY_STEPS // 2:
            v = -0.28
            w = self.turn_direction * 2.0
        else:
            v = 0.30
            w = self.turn_direction * 0.6 + 0.6 * angle_to_goal

        if self.deadlock_counter >= DEADLOCK_RECOVERY_STEPS:
            self.mode = "GO_TO_GOAL"
            self.position_history = self.position_history[-5:]

        return self.smooth_control(v, w, allow_reverse=True)

    # ================================================================== #
    #  LATERAL REPULSION — Yan engellerden itme kuvveti
    # ================================================================== #

    def _lateral_repulsion(self, left, right, front_left, front_right):
        """
        Sol/sağ engellerden kaçmak için açısal itme kuvveti üretir.
        Köşe sendromunun önlenmesi: her iki ön bölge aynı anda tetiklenirse
        (engelin köşesindeyiz) baskalar birbirini iptal eder, sallanma olmaz.
        """
        w_rep = 0.0
        rep_threshold = CAUTION_DIST

        # Yan itme kuvvetleri
        left_rep  = 0.0
        right_rep = 0.0

        if left < rep_threshold:
            strength = ((rep_threshold - left) / rep_threshold) ** 2
            left_rep = 1.4 * strength

        if right < rep_threshold:
            strength = ((rep_threshold - right) / rep_threshold) ** 2
            right_rep = 1.4 * strength

        w_rep = right_rep - left_rep   # pozitif = sağa iten, negatif = sola iten

        # Köşe tespiti: her iki ön taraf aktifse kuvvetleri dengele
        front_left_active  = front_left  < DANGER_DIST
        front_right_active = front_right < DANGER_DIST

        if front_left_active and front_right_active:
            # İki yandan eşit baskı — köşe. Net açısal katkayı sıfırla,
            # sadece mevcut w_rep'i yarıya düşür (hafif yan gerilim bırak)
            w_rep *= 0.3
        else:
            # Sadece bir taraf aktif — normal itme kuvveti
            if front_left_active:
                strength = ((DANGER_DIST - front_left) / DANGER_DIST) ** 2
                w_rep -= 0.9 * strength
            if front_right_active:
                strength = ((DANGER_DIST - front_right) / DANGER_DIST) ** 2
                w_rep += 0.9 * strength

        return w_rep

    # ================================================================== #
    #  GÖRSELLEŞTIRME HİSTOGRAMI
    # ================================================================== #

    def build_display_histogram(self, lidar_distances, lidar_angles):
        n_sectors = 72
        histogram = np.zeros(n_sectors)
        sector_size = np.pi / n_sectors
        half_fov = np.pi / 2

        for dist, angle in zip(lidar_distances, lidar_angles):
            if dist >= LIDAR_RANGE * 0.95:
                continue

            norm = angle + half_fov
            idx = int(norm / sector_size)
            idx = max(0, min(n_sectors - 1, idx))

            val = max(0, 1.0 - dist / LIDAR_RANGE)
            histogram[idx] += val

        max_val = np.max(histogram)
        if max_val > 0:
            histogram /= max_val

        return histogram

    # ================================================================== #
    #  BUG2 M-LINE
    # ================================================================== #

    def is_on_mline(self, x, y):
        if self.mline_start is None:
            return False

        s = self.mline_start
        g = self.goal
        line_vec = g - s
        line_len = np.linalg.norm(line_vec)
        if line_len < 0.01:
            return False

        line_unit = line_vec / line_len
        point_vec = np.array([x, y]) - s
        projection = np.dot(point_vec, line_unit)
        closest = s + projection * line_unit
        dist_to_line = np.linalg.norm(np.array([x, y]) - closest)

        return dist_to_line < MLINE_TOLERANCE and 0 < projection < line_len

    # ================================================================== #
    #  SIKIŞMA ALGILAMA
    # ================================================================== #

    def detect_deadlock(self):
        if len(self.position_history) < DEADLOCK_WINDOW:
            return False
        if self.mode == "DEADLOCK_RECOVER":
            return False

        recent = self.position_history[-DEADLOCK_WINDOW:]
        progress = np.linalg.norm(recent[-1] - recent[0])
        return progress < DEADLOCK_MIN_PROGRESS

    # ================================================================== #
    #  YARDIMCI
    # ================================================================== #

    def region_min(self, distances, angles, min_deg, max_deg):
        """Belirli açı aralığındaki minimum mesafe (10-percentile)."""
        min_rad = np.deg2rad(min_deg)
        max_rad = np.deg2rad(max_deg)
        mask = (angles >= min_rad) & (angles <= max_rad)
        if not np.any(mask):
            return 999.0
        return np.percentile(distances[mask], 10)

    def smooth_control(self, v, w, allow_reverse=False):
        if allow_reverse:
            v = np.clip(v, -0.45, MAX_LINEAR_SPEED)
        else:
            v = np.clip(v, 0.0, MAX_LINEAR_SPEED)

        w = np.clip(w, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        # Köşe sallanması: ani w sıçramalarını kısıtla (rate limiter)
        # Bir adımda açısal hız 0.8 rad/s'den fazla değişemesin
        MAX_W_DELTA = 0.8
        w_delta = w - self.prev_w
        if abs(w_delta) > MAX_W_DELTA:
            w = self.prev_w + np.sign(w_delta) * MAX_W_DELTA

        smooth_v = (1 - SMOOTH_ALPHA_V) * self.prev_v + SMOOTH_ALPHA_V * v
        smooth_w = (1 - SMOOTH_ALPHA_W) * self.prev_w + SMOOTH_ALPHA_W * w

        self.prev_v = smooth_v
        self.prev_w = smooth_w

        return smooth_v, smooth_w

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi