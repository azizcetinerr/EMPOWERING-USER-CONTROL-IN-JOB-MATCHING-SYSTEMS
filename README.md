# İŞ EŞLEŞTİRME SİSTEMLERİNDE KULLANICI KONTROLÜNÜN GÜÇLENDİRİLMESİ

Bu proje, gönüllüler ve projeler arasında adil, şeffaf ve yetenek bazlı eşleştirme yapan yapay zeka destekli bir sistemdir.

## 🚀 Özellikler

- **Veri Toplama**: Youth Portal ve Europass CV entegrasyonu.
- **Beceri Çıkarma**: NLP (SkillNER, Spacy) kullanarak metinlerden beceri analizi.
- **Skorlama Sistemi**:
  - **M-Skoru (Matching)**: Beceri ve vektör tabanlı uyumluluk.
  - **S-Skoru (Sincerity)**: Samimiyet ve motivasyon analizi.
  - **R-Skoru (Reliability)**: Kurum ve proje güvenilirlik analizi.
- **Adil Sıralama**: Diversity-Aware Ranking (MMR) ve çok amaçlı optimizasyon.

## 📁 Proje Yapısı

```
Tubitak_Proje/
├── config/                 # Sistem ayarları (config.yaml)
├── src/                    # Kaynak kodlar
│   ├── data_collection/    # Veri toplama modülleri
│   ├── skill_extraction/   # Beceri analiz modülleri
│   ├── scoring/            # Skorlama algoritmaları
│   └── pipeline.py         # Ana çalışma dosyası
├── data/                   # Veri dosyaları (Ham ve İşlenmiş)
├── outputs/                # Çıktılar ve Raporlar
├── notebooks/              # Analiz notebookları
└── web/                    # (Opsiyonel) Web arayüzü
```

## 🛠 Kurulum

Kurulum adımları için [KURULUM.md](KURULUM.md) dosyasına bakınız.

Hızlı kurulum:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 💻 Kullanım

Sistemi çalıştırmak için:

```bash
python src/pipeline.py
```

Detaylı kullanım talimatları için [KURULUM.md](KURULUM.md) ve [CALISTIRMA_TALIMATLARI.md](CALISTIRMA_TALIMATLARI.md) dosyalarını inceleyebilirsiniz.

## 📊 Çıktılar

Sistem çalıştıktan sonra `outputs/` klasöründe aşağıdaki verileri üretir:
- Sıralanmış aday listeleri
- Skorlama detayları
- Performans ve adillik grafikleri
