# Kurulum ve Kullanım Kılavuzu

## 📁 Klasör Yapısı

```
Tubitak_Proje/
├── config/                    # Konfigürasyon dosyaları
│   └── config.yaml           # Tüm sistem ayarları
│
├── src/                       # Kaynak kodlar
│   ├── data_collection/       # Veri toplama modülleri
│   │   ├── youth_portal_scraper.py      # Proje linklerini çeker
│   │   ├── project_detail_scraper.py    # Proje detaylarını çeker
│   │   └── cv_parser.py                 # CV'yi parse eder
│   │
│   ├── skill_extraction/      # Beceri çıkarma modülleri
│   │   ├── skill_extractor.py           # SkillNER ile beceri çıkarır
│   │   └── vectorizer.py               # Vektörizasyon yapar
│   │
│   ├── scoring/               # Skorlama modülleri
│   │   ├── matching_score.py            # M-Skoru hesaplar
│   │   ├── sincerity_score.py           # S-Skoru hesaplar
│   │   ├── reliability_score.py         # R-Skoru hesaplar
│   │   └── final_ranking.py            # Nihai sıralama
│   │
│   ├── utils/                # Yardımcı fonksiyonlar
│   │   ├── config_loader.py             # Config yükleyici
│   │   ├── logger.py                    # Logging sistemi
│   │   └── data_cleaner.py              # Veri temizleme
│   │
│   └── pipeline.py           # Ana pipeline (tüm işlemleri çalıştırır)
│
├── outputs/                   # Çıktı dosyaları
│   ├── raw/                   # Ham veriler
│   ├── processed/             # İşlenmiş veriler
│   └── results/               # Sonuçlar (CSV, grafikler)
│
├── logs/                      # Log dosyaları
│   └── proje.log             # Sistem logları
│
├── requirements.txt           # Python bağımlılıkları
├── README.md                  # Genel dokümantasyon
├── GELISTIRME_NOTLARI.md      # İyileştirme notları
└── KURULUM.md                 # Bu dosya
```

## 🚀 Hızlı Başlangıç

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Konfigürasyonu Düzenleyin

`config/config.yaml` dosyasını açın ve şunları kontrol edin:
- CV URL'i (`data_sources.europass_cv.url`)
- Motivasyon mektubu (`data_sources.motivation_letter.text`)
- Diğer ayarlar

### 3. Pipeline'ı Çalıştırın

```bash
python src/pipeline.py
```

veya Python'da:

```python
from src.pipeline import ProjectPipeline

pipeline = ProjectPipeline()
df_final = pipeline.run_full_pipeline()
```

## 📝 Adım Adım Kullanım

### Sadece Veri Toplama (IP1)

```python
from src.pipeline import ProjectPipeline

pipeline = ProjectPipeline()
df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection()
```

### Sadece Skorlama (IP2-IP5)

```python
# Önce IP1'i çalıştırın, sonra:
df_projects = pipeline.run_ip2_matching_score(df_projects, candidate_vectors)
df_projects = pipeline.run_ip3_sincerity_score(df_projects)
r_score = pipeline.run_ip4_reliability_score(candidate_skills, cv_text)
df_final = pipeline.run_ip5_final_ranking(df_projects, r_score)
```

## 🔧 Modül Bazlı Kullanım

Her modülü ayrı ayrı da kullanabilirsiniz:

```python
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data_collection.youth_portal_scraper import YouthPortalScraper

# Config ve logger
config = load_config()
logger = setup_logger()

# Sadece link çekme
scraper = YouthPortalScraper(config, logger)
links = scraper.get_project_links()
```

## 📊 Çıktı Dosyaları

Sistem şu dosyaları üretir:

- `outputs/processed/IP1_Ciktisi_*.pkl`: İşlenmiş veriler
- `outputs/results/IP5_Nihai_Adil_Siralama.csv`: Nihai sıralama
- `logs/proje.log`: Sistem logları

## ⚠️ Önemli Notlar

1. **İlk Çalıştırmada**: Tüm modeller indirilecek (birkaç GB)
2. **İnternet Bağlantısı**: Web scraping için gerekli
3. **Chrome/Chromium**: Selenium için gerekli
4. **İşlem Süresi**: 952 proje için ~30-40 dakika sürebilir

## 🐛 Sorun Giderme

### Model Bulunamadı Hatası
```bash
python -m spacy download en_core_web_lg
```

### Import Hatası
```bash
# Proje kök dizininden çalıştırdığınızdan emin olun
python src/pipeline.py
```

### Log Seviyesini Değiştirme
`config/config.yaml` dosyasında:
```yaml
logging:
  level: "DEBUG"  # Daha detaylı loglar için
```

## 📚 Daha Fazla Bilgi

- [README.md](README.md) - Genel proje dokümantasyonu
- [GELISTIRME_NOTLARI.md](GELISTIRME_NOTLARI.md) - İyileştirme detayları
- [config/config.yaml](config/config.yaml) - Tüm ayarlar

