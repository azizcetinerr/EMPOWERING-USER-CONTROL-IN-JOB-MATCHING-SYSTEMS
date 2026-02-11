# Changelog

Tüm önemli değişiklikler bu dosyada dokümante edilmiştir.

## [2.0.0] - 2024-12-09

### ✨ Yeni Özellikler

#### Z-Score Normalizasyonu
- Tüm skorlar (M, S, R) Z-Score ile normalize ediliyor
- M skoru artık diğer skorları bastırmıyor
- Küçük farklar daha belirgin hale geliyor
- Standart sapması 0 olan skorlar otomatik tespit ediliyor

#### S Skoru Geliştirmeleri
- Klişe tespiti ve cezalandırma eklendi
- Spesifik detay bonusu (tarih, bütçe, katılımcı sayısı)
- Sentiment analizi (öznellik, robotik dil tespiti)
- Sigmoid normalizasyon ile geniş skor aralığı (0-100)
- S skorları artık 0.48-96.68 aralığında (önceden 58.8-59.9)

#### R Skoru Geliştirmeleri
- Her proje için ayrı R skoru hesaplanıyor
- Kurum bilgileri ile bonus/ceza (proje sayısı, onay durumu)
- Eksik veri cezası (adres, website eksikliği)
- R skorları artık 0-69.67 aralığında (her proje için farklı)

#### CSV Çıktısı Geliştirmeleri
- Ham skorlar kaydediliyor (`*_Ham` sütunları)
- Normalize skorlar kaydediliyor (`*_Normalize` sütunları)
- Toplam 14 sütun: temel bilgiler + ham + normalize + final skorlar

### 🔧 Değişiklikler

- `final_ranking.py`: Z-Score normalizasyonu eklendi
- `sincerity_score.py`: NLP analizi derinleştirildi
- `reliability_score.py`: Proje-spesifik R skoru hesaplama
- `run_scoring_only.py`: Ham ve normalize skorların kaydedilmesi
- `pipeline.py`: Güncellenmiş CSV çıktısı

### 📚 Dokümantasyon

- `README.md`: Yeni özellikler eklendi
- `CALISTIRMA_TALIMATLARI.md`: Z-Score normalizasyonu ve yeni CSV sütunları
- `GELISTIRME_NOTLARI.md`: Tüm geliştirmeler dokümante edildi
- `S_SKORU_GELISTIRME.md`: S skoru geliştirmeleri detayları
- `R_SKORU_GELISTIRME.md`: R skoru geliştirmeleri detayları
- `Z_SCORE_NORMALIZASYON.md`: Z-Score normalizasyonu detayları

### 🐛 Düzeltmeler

- M skoru baskınlığı sorunu çözüldü
- S skoru dar aralık sorunu çözüldü
- R skoru sabit değer sorunu çözüldü
- Standart sapması 0 olan skorlar için hata kontrolü eklendi

## [1.0.0] - 2024-12-01

### ✨ İlk Sürüm

- Modüler Python paketi yapısı
- YAML konfigürasyon dosyası
- Veri temizleme modülü
- Profesyonel logging sistemi
- Kapsamlı dokümantasyon
- Tüm IP adımları (IP1-IP5)

