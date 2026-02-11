# Web Arayüzü

Bu klasör web arayüzü ile ilgili tüm dosyaları içerir.

## Dosya Yapısı

```
web/
├── app.py              # Flask web uygulaması
├── templates/          # HTML şablonları
│   ├── index.html     # Ana sayfa
│   └── results.html   # Sonuçlar sayfası
├── static/            # Statik dosyalar (CSS, JS, resimler)
└── README.md          # Bu dosya
```

## Kurulum

1. Gerekli paketleri yükleyin (proje kök dizininde):
```bash
pip install -r requirements.txt
```

veya sadece Flask için:
```bash
pip install flask pyyaml
```

## Kullanım

1. Web uygulamasını başlatın:
```bash
cd web
python app.py
```

veya proje kök dizininden:

```bash
python web/app.py
```

2. Tarayıcınızda şu adresi açın:
```
http://localhost:5000
```

## Kullanım Adımları

1. **CV Linki Girin**: Ana sayfada Europass CV paylaşım linkinizi girin
   - Örnek format: `https://europa.eu/europass/eportfolio/api/eprofile/shared-profile/...`

2. **Analiz Et**: "Analiz Et" butonuna tıklayın

3. **İlerlemeyi Takip Edin**: 
   - İşlem sırasında ilerleme çubuğu ve mesajlar görüntülenecektir
   - İşlem tamamlandığında otomatik olarak sonuçlar sayfasına yönlendirileceksiniz

4. **Sonuçları İnceleyin**:
   - Top 10 proje listelenir
   - Her proje için M, S, R ve Final skorları gösterilir
   - Proje detaylarına link üzerinden ulaşabilirsiniz

## Özellikler

- ✅ Modern ve kullanıcı dostu arayüz
- ✅ Gerçek zamanlı ilerleme takibi
- ✅ Detaylı skor gösterimi (M, S, R, Final)
- ✅ Top 10 proje listesi
- ✅ Responsive tasarım (mobil uyumlu)
- ✅ Hata yönetimi ve kullanıcı geri bildirimi

## Nasıl Çalışır?

1. Kullanıcı Europass CV linkini girer
2. Sistem CV'yi analiz eder ve becerileri çıkarır
3. Tüm projelerle eşleştirme yapılır
4. M (Eşleştirme), S (Samimiyet), R (Güvenilirlik) skorları hesaplanır
5. Final skor ile sıralama yapılır
6. En uygun 10 proje kullanıcıya gösterilir

## Performans Optimizasyonları

Web uygulaması aşağıdaki optimizasyonları içerir (kaliteden ödün vermeden):

1. **Paralel Proje Scraping**: 
   - 5 thread ile paralel scraping
   - ~5x daha hızlı proje toplama
   
2. **Akıllı Proje Önbellekleme**: 
   - Mevcut projeler önbellekten yüklenir
   - Sadece yeni projeler scrape edilir
   - Örnek: 100 proje varsa, 5 yeni eklendiğinde sadece 5'i scrape edilir
   
3. **Optimize S Skoru Hesaplama**: 
   - Sadece top 50 proje için S skoru hesaplanır (en iyi eşleşenler)
   - ~10-20x daha hızlı, kalite korunur
   
4. **Model Ön Yükleme**: Modeller uygulama başlangıcında yüklenir ve önbelleğe alınır
5. **Mevcut IP1 Çıktılarını Kullanma**: Eğer `IP1_Ciktisi_Projeler_Becerili.pkl` dosyası varsa, proje scraping'i atlanır

**Detaylı bilgi için**: `PERFORMANS_OPTIMIZASYONLARI.md` dosyasına bakın.

### İlk Çalıştırma (Yavaş)
- Proje verileri toplanır (~5-10 dakika)
- Modeller yüklenir
- CV analiz edilir
- Skorlar hesaplanır

### Sonraki Çalıştırmalar (Hızlı)
- Sadece CV analiz edilir (~30 saniye - 2 dakika)
- Önbelleğe alınmış proje verileri kullanılır
- Modeller zaten yüklü

### Daha Hızlı Çalıştırmak İçin

1. **IP1 çıktılarını hazırlayın**: Önce bir kez tam pipeline'ı çalıştırın:
   ```bash
   python src/pipeline.py
   ```
   Bu, `IP1_Ciktisi_Projeler_Becerili.pkl` dosyasını oluşturur.

2. **Sadece skorlama yapın**: Web uygulaması artık sadece CV analizi yapacak ve çok daha hızlı çalışacak.

## Notlar

- İlk analiz uzun sürebilir (model yüklemeleri nedeniyle)
- CV linkinin geçerli bir Europass paylaşım linki olması gerekir
- Analiz sırasında tarayıcı penceresini kapatmayın
- Proje verileri önbelleğe alındığı için, proje listesi güncellenmek istendiğinde önbelleği temizleyin

## Sorun Giderme

**Hata: "CV URL gerekli"**
- Lütfen geçerli bir CV linki girdiğinizden emin olun

**Hata: "Session bulunamadı"**
- Sayfayı yenileyin ve tekrar deneyin

**Analiz çok uzun sürüyor**
- Bu normaldir, özellikle ilk çalıştırmada modeller yüklenir
- İlerleme çubuğunu takip edin
