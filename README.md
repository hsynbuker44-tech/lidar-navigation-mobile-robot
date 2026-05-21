# Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon

**Ders:** Mobil Robotlar  

---

## 1. Giriş ve Senaryo Tanımı

Bu proje, 2B simülasyon ortamında LiDAR tabanlı otonom navigasyon gerçekleştiren bir mobil robot sistemi geliştirmektedir. Robot, sensör füzyonu ve lokalizasyon algoritmaları kullanarak başlangıç noktasından hedef noktasına güvenli şekilde ulaşmayı amaçlar.

### Senaryo

- **Ortam:** 20×20 metre boyutunda 2B düzlem
- **Engel sayısı:** 10 adet rastgele yerleştirilmiş dikdörtgen engel
- **Başlangıç noktası:** (2.0, 2.0)
- **Hedef noktası:** (18.0, 18.0)
- **Robot görevi:** Engelleri algılayarak güvenli bir güzergah izleyip hedefe ulaşmak
- **Sensör gürültüsü:** LiDAR (σ=0.02m), IMU (σ=0.02 rad), Enkoder (σ=0.02m) gürültü eklenmektedir

### Robot Modeli

Non-holonomik diferansiyel sürüş modeli kullanılmaktadır:

```
x' = x + v·cos(θ)·dt
y' = y + v·sin(θ)·dt
θ' = θ + ω·dt
```

- `v`: Doğrusal hız (maks. 0.6 m/s)
- `ω`: Açısal hız (maks. 2.0 rad/s)
- `dt`: Zaman adımı (0.1 s)

---

## 2. Kullanılan Yöntemler

### 2.1 LiDAR İşleme

- **181 ışınlı 2B LiDAR** sensörü, 180° görüş alanında 5 metre menzile sahiptir
- **Medyan filtre** (pencere=5) uygulanarak anlık gürültü spike'ları bastırılır
- **Engel kümeleme** algoritması, ardışık LiDAR noktaları arasındaki mesafe sıçramalarını kullanarak engelleri gruplar

### 2.2 Sensör Füzyonu — Extended Kalman Filter (EKF)

Robot konumu, üç sensörden gelen verilerin füzyonu ile tahmin edilir:

| Sensör | Ölçüm | Gürültü (σ) |
|---|---|---|
| LiDAR | Engel mesafeleri | 0.02 m |
| IMU | Açısal oryantasyon (θ) | 0.02 rad |
| Tekerlek Enkoderi | Konum (x, y) | 0.02 m |

EKF iki adımlı bir döngüde çalışır:

1. **Tahmin (Predict):** Robot kinematik modeli ile durum tahmini yapılır, Jacobian matrisleri ile kovaryans güncellenir
2. **Güncelleme (Update):** Gürültülü odometri ölçümleri ile Kalman kazancı hesaplanarak durum düzeltilir

### 2.3 Navigasyon — Gap-Based Reaktif Navigasyon + Bug2

Navigasyon sistemi, birden fazla modun dinamik olarak geçişiyle çalışır:

| Mod | Açıklama |
|---|---|
| **GO_TO_GOAL** | Hedefe doğru git, gap-blended proaktif yönlendirme |
| **GAP_NAVIGATE** | LiDAR verisinden boş koridorları (gap) tespit ederek engelden kaçın |
| **WALL_FOLLOW** | Bug2 algoritması ile duvar takibi |
| **ESCAPE** | Acil geri çekilme (arka güvenlik kontrolü ile) |
| **DEADLOCK_RECOVER** | Sıkışma tespiti ve alternatif yönlü kurtarma |

#### Gap Skorlaması

Gap (boşluk) analizi üç kritere göre puanlanır:
- **Hedefe yakınlık:** Hedefe yönelen gap'ler tercih edilir
- **Genişlik:** Robot çapına göre normalize edilen açısal genişlik
- **Derinlik:** Gap'in arkasındaki serbest alan miktarı

#### Dar Koridor Tespiti

Robot, her iki yanda engel olduğunu ancak önde boşluk olduğunu algıladığında dar koridor moduna geçerek lateral itme kuvvetlerini devre dışı bırakır.

### 2.4 Lokalizasyon

- **Dead Reckoning:** Robot kinematik modelini kullanarak pozisyon tahmini
- **Sensör füzyonlu konum tahmini:** EKF ile dead reckoning ve sensör ölçümlerinin birleştirilmesi
- **Hata analizi:** Gerçek yol ile tahmin edilen yol arasındaki Öklid mesafesi, RMSE ve MAE metrikleri ile değerlendirilir

---

## 3. Sonuçlar ve Grafikler

Simülasyon sonunda aşağıdaki grafikler `figures/` klasörüne otomatik olarak kaydedilir:

### 3.1 Ortam Haritası (`environment_map.png`)
2B ortam yerleşimi, 10 engel, başlangıç ve hedef noktaları görselleştirilmiştir.

### 3.2 Robot Yol Planı (`path_comparison.png`)
Gerçek robot yolu (mavi düz çizgi) ile EKF tahmini yol (kırmızı kesikli çizgi) aynı grafik üzerinde karşılaştırılmıştır.

### 3.3 Sensör Görselleştirmesi (`lidar_2d_view.png` + `lidar_profile.png`)
- LiDAR tarama noktalarının ham ve filtrelenmiş halleri iki ayrı alt grafikte gösterilmiştir
- Mesafe profili grafiğinde ham (kırmızı) ve filtrelenmiş (yeşil) veriler karşılaştırılmıştır

### 3.4 Lokalizasyon Sonuçları (`localization_error.png`)
- Öklid hata zaman serisi
- X ve Y eksenlerinde ayrı ayrı hata grafikleri
- RMSE ve MAE değerleri grafik üzerinde işaretlenmiştir

### 3.5 Navigasyon Mod Dağılımı (`mode_distribution.png`)
Pasta grafiği ve yatay bar grafiği ile navigasyon modlarının kullanım oranları gösterilmiştir.

### 3.6 Performans Metrikleri (`metrics_timeseries.png`)
Doğrusal hız, açısal hız ve minimum engel mesafesi zaman serileri üç alt grafikte çizilmiştir.

---

## 4. Hata Analizi ve Kısa Tartışma

### Lokalizasyon Hatası

EKF tabanlı sensör füzyonu, dead reckoning'e kıyasla önemli ölçüde daha düşük lokalizasyon hatası sağlamaktadır. Sistemde:

- **RMSE** ve **MAE** değerleri milimetre mertebesinde kalmaktadır
- Hata artışı özellikle keskin dönüşlerde gözlemlenir (açısal hız yüksekken non-linearite artar)
- Kalman kazancı, ölçüm güvenilirliğine göre otomatik olarak ağırlıkları ayarlamaktadır

### Navigasyon Performansı

- Robot, engelli ortamlarda hedefe ulaşabilmektedir
- Gap-based navigasyon, VFH benzeri yaklaşımla boşlukları analiz ederek güvenli geçit seçimi yapar
- Bug2 algoritması, gap bulunamadığında devreye girerek duvar takibi ile çıkış yolu arar
- Deadlock algılama mekanizması, robotun takılı kalmasını önler

### Bilinen Sınırlamalar

- Çok kalabalık ortamlarda nadiren çarpışma yaşanabilir
- Sensör gürültüsü yüksek olduğunda EKF performansı düşebilir
- Dinamik engeller desteklenmemektedir (statik ortam)

## 6. Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

**Bağımlılıklar:**
- `numpy` — Sayısal hesaplamalar
- `matplotlib` — Grafik oluşturma
- `scipy` — Bilimsel hesaplamalar
- `pygame` — Gerçek zamanlı simülasyon görselleştirmesi

### Çalıştırma

```bash
python main.py
```

### Kontroller

| Tuş | İşlev |
|---|---|
| `M` | Manuel/Otonom mod geçişi |
| `W/A/S/D` | Manuel modda robot kontrolü |
| `R` | Simülasyonu yeniden başlat |
| `ESC` | Çıkış |

### Çıktılar

Simülasyon sonunda tüm grafikler `figures/` klasörüne otomatik olarak kaydedilir:
- `environment_map.png` — Ortam haritası
- `path_comparison.png` — Yol karşılaştırması
- `lidar_2d_view.png` — LiDAR 2B görünüm
- `lidar_profile.png` — LiDAR mesafe profili
- `localization_error.png` — Lokalizasyon hatası
- `mode_distribution.png` — Mod dağılımı
- `metrics_timeseries.png` — Performans metrikleri

---

## Proje Yapısı

```
├── main.py              # Ana simülasyon döngüsü
├── config.py            # Tüm konfigürasyon parametreleri
├── environment.py       # 2B ortam ve engel yönetimi
├── robot.py             # Robot kinematik modeli + gürültülü odometri
├── lidar.py             # LiDAR sensörü + medyan filtre + engel kümeleme
├── ekf.py               # Extended Kalman Filter (sensör füzyonu)
├── navigation.py        # Gap-based navigasyon + Bug2 + reaktif davranış
├── visualization.py     # Pygame tabanlı görselleştirme + manuel kontrol
├── metrics.py           # Performans metrikleri toplama
├── analysis.py          # Simülasyon sonrası grafik ve hata analizi
├── requirements.txt     # Python bağımlılıkları
├── figures/             # Oluşturulan grafik dosyaları
│   ├── environment_map.png
│   ├── path_comparison.png
│   ├── lidar_2d_view.png
│   ├── lidar_profile.png
│   ├── localization_error.png
│   ├── mode_distribution.png
│   └── metrics_timeseries.png
└── README.md            # Bu dosya
```
