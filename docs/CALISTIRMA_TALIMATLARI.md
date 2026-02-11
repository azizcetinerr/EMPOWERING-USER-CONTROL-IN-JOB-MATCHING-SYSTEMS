# Çalıştırma Talimatları

## ✅ Yapılan Geliştirmeler

1. **Tüm projeler için S skoru hesaplanıyor**
   - `config.yaml`'de `sincerity_scoring.calculate_all: true` (varsayılan)
   - Hiçbir proje atlanmıyor
   - Gelişmiş NLP analizi: klişe tespiti, spesifik detay bonusu, sentiment analizi
   - Sigmoid normalizasyon ile geniş skor aralığı (0-100)

2. **R skoru her proje için ayrı hesaplanıyor**
   - Her proje için o projenin gereksinimlerinin CV'de doğrulanma oranı
   - Kurum bilgileri (proje sayısı, onay durumu) ile bonus/ceza
   - Eksik veri cezası

3. **Z-Score normalizasyonu**
   - Tüm skorlar (M, S, R) Z-Score ile normalize ediliyor
   - M skoru artık diğer skorları bastırmıyor
   - Küçük farklar daha belirgin hale geliyor
   - Standart sapması 0 olan skorlar otomatik tespit ediliyor

4. **Tüm skorlar veri olarak saklanıyor**
   - CSV çıktısında 14 sütun:
     - Ham skorlar: `M_Skoru_Ham`, `S_Skoru_Ham`, `R_Skoru_Ham`
     - Normalize skorlar: `M_Skoru_Normalize`, `S_Skoru_Normalize`, `R_Skoru_Normalize`
     - Kullanılan skorlar: `M_Skoru`, `S_Skoru`, `R_Skoru` (normalize)
     - Final skorlar: `Final_Skor`, `Adil_Skor`
   - Hiçbir skor eksik kalmıyor

5. **İstatistikler ve kontroller eklendi**
   - Her adımda skor istatistikleri gösteriliyor
   - Eksik skor kontrolü yapılıyor
   - Skor katkıları analiz ediliyor

## 🚀 Çalıştırma Adımları

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Config Dosyasını Kontrol Edin

`config/config.yaml` dosyasını açın ve şunları kontrol edin:
- `sincerity_scoring.calculate_all: true` (tüm projeler için S skoru)
- CV URL'i doğru mu?
- Motivasyon mektubu var mı?

### 3. Pipeline'ı Çalıştırın

#### Seçenek A: Tüm Pipeline (Sıfırdan)

```bash
python src/pipeline.py
```

veya

```bash
python test_pipeline.py
```

#### Seçenek B: Sadece Skorlama (IP1 çıktıları varsa)

Eğer `IP1_Ciktisi_*.pkl` dosyaları mevcutsa:

```bash
python run_scoring_only.py
```

Bu script:
- Mevcut IP1 çıktılarını kullanır
- Sadece IP2-IP5'i çalıştırır (daha hızlı)
- Tüm projeler için skorları hesaplar

## 📊 Beklenen Çıktılar

### CSV Dosyası (`outputs/results/IP5_Nihai_Adil_Siralama.csv`)

Her satırda şu 14 sütun olacak:

**Temel Bilgiler:**
- `link`: Proje URL'i
- `tam_aciklama`: Proje açıklaması
- `cikarilan_beceriler`: Çıkarılan beceriler

**Ham Skorlar (Orijinal Değerler):**
- `M_Skoru_Ham`: Ham eşleştirme skoru (0-100)
- `S_Skoru_Ham`: Ham samimiyet skoru (0-100)
- `R_Skoru_Ham`: Ham güvenilirlik skoru (0-100)

**Normalize Skorlar (Z-Score ile):**
- `M_Skoru_Normalize`: Normalize edilmiş M skoru
- `S_Skoru_Normalize`: Normalize edilmiş S skoru
- `R_Skoru_Normalize`: Normalize edilmiş R skoru

**Kullanılan Skorlar (Normalize Edilmiş):**
- `M_Skoru`: Final hesaplamada kullanılan M skoru (normalize)
- `S_Skoru`: Final hesaplamada kullanılan S skoru (normalize)
- `R_Skoru`: Final hesaplamada kullanılan R skoru (normalize, her proje için farklı)

**Final Skorlar:**
- `Final_Skor`: Ağırlıklı ortalama (M×0.5 + S×0.3 + R×0.2)
- `Adil_Skor`: Final skor + adillik bonusu

### Log Dosyası (`logs/proje.log`)

Tüm işlemler loglanır:
- Her adımın başlangıcı/bitişi
- İlerleme raporları
- Skor istatistikleri
- Hata mesajları

## ⏱️ Süre Tahminleri

- **IP1 (Veri Toplama)**: ~30-40 dakika (952 proje)
- **IP2 (M Skoru)**: ~1 dakika
- **IP3 (S Skoru)**: ~10-15 dakika (tüm projeler için)
- **IP4 (R Skoru)**: ~2-3 dakika
- **IP5 (Final Ranking)**: ~1 dakika

**Toplam**: ~45-60 dakika

## 🔍 Kontrol Listesi

Çalıştırdıktan sonra kontrol edin:

- [ ] CSV dosyasında tüm projeler var mı? (952 proje)
- [ ] Her proje için M_Skoru var mı?
- [ ] Her proje için S_Skoru var mı? (NaN yok mu?)
- [ ] Her proje için R_Skoru var mı?
- [ ] Her proje için Adil_Skor var mı?
- [ ] Skorlar farklılık gösteriyor mu? (hepsi aynı değil mi?)

## 🐛 Sorun Giderme

```bash
pip install pyyaml
```

### "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### "S skoru eksik" uyarısı
- `config.yaml`'de `calculate_all: true` olduğundan emin olun
- IP3'ün tamamlanmasını bekleyin

### Çok yavaş çalışıyor
- S skoru hesaplama uzun sürer (tüm projeler için embedding)
- Sabırlı olun veya `calculate_all: false` yapıp sadece ilk N proje için hesaplayın

## 📝 Örnek Çıktı

```
✅ TÜM SKORLAMA ADIMLARI TAMAMLANDI!

📊 ÖZET:
   Toplam proje sayısı: 952
   M skoru (normalize): 0.00 - 53.15 (Ort: 51.83)
   S skoru (normalize): 20.36 - 78.26 (Ort: 50.00)
   R skoru (normalize): 0.00 - 96.25 (Ort: 50.74) (her proje için farklı)
   Adil skor: 6.13 - 73.01 (Ort: 55.84)

📈 SKOR KATKILARI:
   M katkısı: 25.92 (50.0% ağırlık)
   S katkısı: 15.00 (30.0% ağırlık)
   R katkısı: 10.15 (20.0% ağırlık)

🏆 TOP 5 PROJE:
   1. [Adil: 73.01] [M: 52.9, S: 74.3, R: 96.2] 47139
   2. [Adil: 71.18] [M: 53.0, S: 74.0, R: 87.5] 48151
   3. [Adil: 69.22] [M: 53.1, S: 76.7, R: 73.3] 34309
   ...
```

## ✅ Başarı Kriterleri

Pipeline başarılı sayılır eğer:
1. ✅ Tüm 952 proje işlendi
2. ✅ Hiçbir proje için skor eksik değil
3. ✅ Skorlar farklılık gösteriyor (hepsi aynı değil)
4. ✅ CSV dosyası 14 sütun içeriyor (ham + normalize + final)
5. ✅ Z-Score normalizasyonu uygulandı (M skoru artık diğerlerini bastırmıyor)
6. ✅ Her proje için ayrı R skoru hesaplandı
7. ✅ Log dosyasında hata yok

