# lidar.py

import numpy as np
from config import LIDAR_RANGE, LIDAR_NUM_BEAMS, LIDAR_FOV, LIDAR_NOISE_STD


class Lidar:
    def __init__(self):
        self.max_range = LIDAR_RANGE
        self.num_beams = LIDAR_NUM_BEAMS
        self.fov = LIDAR_FOV

    def scan(self, robot_state, env):
        x, y, theta = robot_state

        angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_beams)

        points = []
        distances = []

        for angle in angles:
            beam_angle = theta + angle

            distance = self.cast_ray(x, y, beam_angle, env)

            noisy_distance = distance + np.random.normal(0, LIDAR_NOISE_STD)
            noisy_distance = np.clip(noisy_distance, 0.0, self.max_range)

            px = x + noisy_distance * np.cos(beam_angle)
            py = y + noisy_distance * np.sin(beam_angle)

            points.append((px, py))
            distances.append(noisy_distance)

        raw_points = np.array(points)
        raw_distances = np.array(distances)

        # Gürültü filtresi: median filtre ile ham veriyi yumuşat
        filtered_distances = self._median_filter(raw_distances, window=5)
        filtered_points = np.array([
            (x + filtered_distances[i] * np.cos(theta + angles[i]),
             y + filtered_distances[i] * np.sin(theta + angles[i]))
            for i in range(self.num_beams)
        ])

        return raw_points, raw_distances, angles, filtered_points, filtered_distances

    def _median_filter(self, distances, window=5):
        """Basit 1D medyan filtre — ani spike'ları bastırır."""
        half = window // 2
        filtered = distances.copy()
        for i in range(half, len(distances) - half):
            filtered[i] = np.median(distances[i - half: i + half + 1])
        return filtered

    def cast_ray(self, x, y, angle, env):
        step_size = 0.05
        distance = 0.0

        while distance < self.max_range:
            px = x + distance * np.cos(angle)
            py = y + distance * np.sin(angle)

            if self.is_point_in_obstacle(px, py, env):
                return distance

            if px < 0 or px > env.width or py < 0 or py > env.height:
                return distance

            distance += step_size

        return self.max_range

    def is_point_in_obstacle(self, x, y, env):
        for ox, oy, w, h in env.obstacles:
            left = ox - w / 2
            right = ox + w / 2
            bottom = oy - h / 2
            top = oy + h / 2

            if left <= x <= right and bottom <= y <= top:
                return True

        return False

    def cluster_obstacles(self, points, distances, max_range=None, gap_threshold=0.5):
        """
        LiDAR nokta bulutundan engel kümelerini çıkar.

        Yöntem: Ardışık ışın noktaları arasındaki mesafe sıçramalarını
        bölme noktası olarak kullan (sektör-tabanlı kümeleme).

        Parametreler
        ------------
        points          : (N, 2) nokta bulutu (dünya koordinatları)
        distances       : (N,)   her ışına ait mesafe
        max_range       : Bu değerden uzak noktalar engel sayılmaz
        gap_threshold   : Ardışık nokta mesafesi bu değeri aşarsa yeni küme başlar

        Döndürür
        --------
        clusters : list of dict
            Her küme için:
              'points'   — kümedeki (x, y) noktaları
              'centroid' — küme merkezi (x, y)
              'min_dist' — robota en yakın nokta mesafesi
              'size'     — nokta sayısı
        """
        if max_range is None:
            max_range = self.max_range * 0.95   # tam menzil noktaları engel değil

        # Menzil içindeki noktaları filtrele
        mask = distances < max_range
        valid_pts = points[mask]
        valid_dist = distances[mask]
        valid_idx = np.where(mask)[0]

        if len(valid_pts) == 0:
            return []

        clusters = []
        current_cluster_pts = [valid_pts[0]]
        current_cluster_dist = [valid_dist[0]]

        for i in range(1, len(valid_pts)):
            # Ardışık nokta çiftleri arası Öklid mesafesi
            d = np.linalg.norm(valid_pts[i] - valid_pts[i - 1])

            # Işın indeksi süreksizliği de kontrol et (geniş açı atlamaları)
            idx_gap = valid_idx[i] - valid_idx[i - 1]

            if d > gap_threshold or idx_gap > 5:
                # Yeni küme başlat — eskiyi kaydet
                if len(current_cluster_pts) >= 2:  # tek noktalı kümeleri yoksay
                    clusters.append(self._make_cluster(
                        current_cluster_pts, current_cluster_dist
                    ))
                current_cluster_pts = [valid_pts[i]]
                current_cluster_dist = [valid_dist[i]]
            else:
                current_cluster_pts.append(valid_pts[i])
                current_cluster_dist.append(valid_dist[i])

        # Son kümeyi ekle
        if len(current_cluster_pts) >= 2:
            clusters.append(self._make_cluster(
                current_cluster_pts, current_cluster_dist
            ))

        return clusters

    @staticmethod
    def _make_cluster(pts_list, dist_list):
        pts = np.array(pts_list)
        dists = np.array(dist_list)
        return {
            'points': pts,
            'centroid': pts.mean(axis=0),
            'min_dist': dists.min(),
            'size': len(pts),
        }
