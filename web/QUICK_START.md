# Hızlı Başlangıç Kılavuzu

## Durum Kontrolü

Uygulama başladığında şu mesajları görürsünüz:

### ✅ Hızlı Mod (Önerilen)
```
✅ IP1 çıktıları bulundu - Hızlı mod aktif
   (Proje scraping'i atlanacak, sadece CV analizi yapılacak)
```

### ⚠️ Yavaş Mod (İlk Çalıştırma)
```
⚠️  IP1 çıktıları bulunamadı - Yavaş mod
   (İlk analiz uzun sürebilir, proje verileri toplanacak)
```

## Hızlı Mod İçin IP1 Çıktılarını Oluşturma

### Yöntem 1: Pipeline ile (Önerilen)

1. Terminal'de proje kök dizinine gidin:
```bash
cd /Users/acetiner/Desktop/Tubitak_Proje/Tubitak_Proje_Yeni
```

2. Pipeline'ı çalıştırın (bir kez):
```bash
python src/pipeline.py
```

Bu komut:
- Proje verilerini toplar (~5-10 dakika)
- Becerileri çıkarır
- Vektörleri hesaplar
- `IP1_Ciktisi_Projeler_Becerili.pkl` dosyasını oluşturur

3. Web uygulamasını yeniden başlatın:
```bash
python web/app.py
```

Artık "Hızlı mod aktif" mesajını göreceksiniz!

### Yöntem 2: Web Arayüzünden (İlk Analiz)

1. Web arayüzünü açın: `http://localhost:5001`
2. CV linkinizi girin
3. İlk analiz uzun sürebilir (~10-15 dakika) ama IP1 çıktıları oluşturulacak
4. Sonraki analizler çok daha hızlı olacak (~2-3 dakika)

## Performans Karşılaştırması

| Durum | İlk Analiz | Sonraki Analizler |
|-------|------------|-------------------|
| **Yavaş Mod** | ~10-15 dakika | ~2-3 dakika |
| **Hızlı Mod** | ~2-3 dakika | ~1-2 dakika |

## İpuçları

1. **İlk kez kullanıyorsanız**: Pipeline'ı bir kez çalıştırın, sonra web arayüzünü kullanın
2. **Sadece CV analizi yapmak istiyorsanız**: Web arayüzünden devam edin (ilk analiz uzun sürer)
3. **Proje listesi güncellendi**: Önbellek otomatik olarak güncellenecek, sadece yeni projeler scrape edilecek

## Sorun Giderme

**"IP1 çıktıları bulunamadı" mesajı görüyorum:**
- Normal, ilk çalıştırmada bu mesaj görünür
- İlk analiz uzun sürebilir ama sonraki analizler hızlı olacak

**"Modeller yükleniyor" çok uzun sürüyor:**
- İlk kez model indiriliyor (internet bağlantısına bağlı)
- Sonraki başlatmalarda çok daha hızlı olacak

**Analiz çok uzun sürüyor:**
- İlk analiz: Normal (~10-15 dakika)
- Sonraki analizler: Çok daha hızlı (~2-3 dakika)
