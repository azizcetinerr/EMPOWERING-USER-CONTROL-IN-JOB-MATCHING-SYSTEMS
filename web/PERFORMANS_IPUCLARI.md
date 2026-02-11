# Performans İpuçları

## Sorun: Çok Yavaş veya Çok Hızlı (429 Hatası)

### Çözüm: Adaptive Rate Limiting ✅

Artık sistem otomatik olarak hızını ayarlıyor:
- **429 hatası alırsa**: Otomatik olarak yavaşlar
- **Başarılı istekler**: Otomatik olarak hızlanır
- **Optimal hız**: Sunucunun izin verdiği maksimum hızda çalışır

## Beklenen Süreler

### Yeni Projeler (İlk Kez)
- **100 proje**: ~15-20 dakika (adaptive rate limiting ile)
- **50 proje**: ~8-10 dakika
- **10 proje**: ~2-3 dakika

### Mevcut Projeler (Önbellekten)
- **Sadece CV analizi**: ~2-3 dakika
- **Yeni proje yoksa**: ~1-2 dakika

## Hızlandırma İpuçları

### 1. Mevcut Projeleri Kullanın
Eğer daha önce proje verileri toplandıysa:
- Sadece yeni projeler scrape edilir
- Çok daha hızlı çalışır

### 2. Thread Sayısını Ayarlayın
`config/config.yaml` dosyasında:
```yaml
performance:
  num_workers: 4  # Artırılabilir: 5-6 (ama 429 riski artar)
```

**Öneriler**:
- İyi internet: 5-6 thread
- Orta internet: 4 thread (varsayılan)
- Yavaş internet: 3 thread

### 3. İlk N Projeyi Kullanın
Tüm projeler yerine sadece ilk N projeyi kullanmak için:
- Pipeline'da proje sayısını sınırlayın
- Veya web arayüzünde "sadece top 50 proje" seçeneği ekleyin

## Sorun Giderme

### Çok Fazla 429 Hatası
**Belirtiler**: Log'larda sürekli "429" hatası
**Çözüm**: 
- Thread sayısını azaltın: `num_workers: 3`
- Request delay'i artırın: `request_delay: 2`

### Çok Yavaş İlerleme
**Belirtiler**: 1 saat+ bekliyor
**Çözüm**:
- Thread sayısını artırın: `num_workers: 5-6`
- Request delay'i azaltın: `request_delay: 0.5`
- Mevcut projeleri kullanın (önbellekten)

### İlerleme Görmüyorum
**Belirtiler**: Log'da hiçbir şey yok
**Çözüm**:
- Log seviyesini artırın: `logging.level: DEBUG`
- Terminal çıktısını kontrol edin

## Optimal Ayarlar

### Hızlı İnternet + Güçlü Sunucu
```yaml
performance:
  num_workers: 6

scraping:
  requests:
    request_delay: 0.5
```

### Normal İnternet
```yaml
performance:
  num_workers: 4  # Varsayılan

scraping:
  requests:
    request_delay: 1  # Varsayılan
```

### Yavaş İnternet veya Rate Limiting Sorunları
```yaml
performance:
  num_workers: 3

scraping:
  requests:
    request_delay: 2
    retry_delay: 15
```

## Adaptive Rate Limiting Nasıl Çalışır?

1. **Başlangıç**: 1 saniye delay ile başlar
2. **429 hatası**: Delay'i 1.5x artırır (max 5 saniye)
3. **10 başarılı istek**: Delay'i 0.9x azaltır (min 1 saniye)
4. **Optimal hız**: Sunucunun izin verdiği maksimum hızda çalışır

Bu sayede:
- ✅ Rate limiting'den kaçınır
- ✅ Mümkün olduğunca hızlı çalışır
- ✅ Otomatik olarak ayarlanır
