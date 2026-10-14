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

## 📄 Makale verisi ve doğrulama betiği / Paper data and verification script

Bu bölüm FAIR-PASS makalesinin (fairpass_v2) *Data and Code Availability Statement* kısmında atıf yapılan dosyaları içerir.
This section lists the files cited in the Data and Code Availability Statement of the FAIR-PASS paper (fairpass_v2).

| Dosya / File | İçerik / Content |
|---|---|
| `outputs/results/IP5_S_yeniden.csv` | 952 ilanın skorlanmış hali (ham ve standardize M/S/R skorları, `Final_Skor`, `Adil_Skor`, `Insan_Puani`, `Baglam_Puani`) / Scored listing file for the 952 listings |
| `outputs/results/IP5_Nihai_Adil_Siralama.csv` | Önceki sürümün skor dosyası; `recompute_sincerity.py` için girdi / Score file of the earlier version; input to `recompute_sincerity.py` |
| `PAPER/recompute_sincerity.py` | S skorunu repodaki `src/scoring` sınıflarıyla yeniden hesaplayıp `IP5_S_yeniden.csv` dosyasını üretir (model indirme gerekir) / Recomputes the sincerity score with the classes in `src/scoring` and writes `IP5_S_yeniden.csv` (downloads models) |
| `PAPER/verify_numbers.py` | Makalenin Sonuçlar bölümündeki türetilmiş değerleri bu CSV'den yeniden hesaplar / Recomputes the derived values reported in the Results section |
| `config/config.yaml` | Raporlanan çalıştırmada kullanılan niyet mektubunu içerir / Contains the motivation letter used in the reported run |

Çalıştırma (proje kökünden) / Run from the repository root:

```bash
pip install pandas scipy
python PAPER/verify_numbers.py outputs/results/IP5_S_yeniden.csv
```

`IP5_S_yeniden.csv` dosyasını baştan üretmek için / To regenerate `IP5_S_yeniden.csv`:

```bash
pip install -r requirements.txt
python PAPER/recompute_sincerity.py
```

Özgeçmiş (CV) kişisel veri içerdiği için paylaşılmamaktadır. / The curriculum vitae is not shared because it contains personal data.
