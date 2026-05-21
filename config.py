# config.py

DT = 0.1
SIM_TIME = 9999.0  # Pratik olarak sınırsız — hedefe ulaşana kadar çalışır

WORLD_WIDTH = 20.0
WORLD_HEIGHT = 20.0

START = (2.0, 2.0, 0.0)
GOAL = (18.0, 18.0)

ROBOT_RADIUS = 0.3

# Hız limitleri
MAX_LINEAR_SPEED = 0.6
MAX_ANGULAR_SPEED = 2.0

# LiDAR parametreleri
LIDAR_RANGE = 5.0
LIDAR_NUM_BEAMS = 181
LIDAR_FOV = 3.14159

# Sensör gürültü standart sapmaları
LIDAR_NOISE_STD = 0.02
IMU_NOISE_STD = 0.02
ENCODER_NOISE_STD = 0.02

# VFH parametreleri
VFH_SECTOR_COUNT = 72
VFH_THRESHOLD_LOW = 0.3
VFH_THRESHOLD_HIGH = 0.6
VFH_WIDE_OPENING = 10
VFH_SAFETY_DISTANCE = 1.0     # Gap güvenlik mesafesi (artırıldı)

# Engelden kaçınma eşikleri (robot yarıçapı dahil düşünülmeli)
VERY_CLOSE_DIST = 0.75        # Acil kaçış — daha erken tetikle
DANGER_DIST = 1.4              # Tehlike — hemen dönmeye başla
CAUTION_DIST = 2.4             # Dikkat — modu değiştir, yavaşla
CLEAR_DIST = 3.2               # Güvenli — rahatça ilerle

# Bug2 duvar takibi
WALL_FOLLOW_DIST = 1.2        # Duvardan takip mesafesi
MLINE_TOLERANCE = 0.5

# Sıkışma algılama (Deadlock Detection)
DEADLOCK_WINDOW = 80
DEADLOCK_MIN_PROGRESS = 0.3
DEADLOCK_RECOVERY_STEPS = 40

# EKF parametreleri
EKF_PROCESS_NOISE = 0.01
EKF_MEASUREMENT_NOISE = 0.05

# Smooth control — yüksek = hızlı tepki
SMOOTH_ALPHA_V = 0.70   # daha hızlı tepki
SMOOTH_ALPHA_W = 0.65   # daha hızlı dönüş

# Manuel kontrol (WASD)
MANUAL_LINEAR_SPEED = 0.5
MANUAL_ANGULAR_SPEED = 1.5