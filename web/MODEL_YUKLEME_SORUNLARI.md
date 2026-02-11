# Model Yükleme Sorunları ve Çözümleri

## Sorun: Modeller 2 Saat Bekliyor

Bu **kesinlikle normal değil**! Modellerin yüklenmesi birkaç dakika sürmeli.

## Çözüm: Lazy Loading (Uygulandı ✅)

Artık modeller **lazy loading** ile yükleniyor:
- Başlangıçta modeller yüklenmez
- Sadece gerektiğinde yüklenir
- Uygulama başlangıcı çok daha hızlı

## Model Yükleme Süreleri (Normal)

| Model | İlk Yükleme | Sonraki Yüklemeler |
|-------|-------------|-------------------|
| Skill Extractor | ~1-2 dakika | ~10 saniye |
| Vectorizer | ~2-3 dakika | ~30 saniye |
| Sincerity Calculator | ~3-5 dakika | ~1-2 dakika |
| Matching Calculator | ~1 dakika | ~10 saniye |
| Reliability Calculator | ~1-2 dakika | ~30 saniye |
| Ranking Calculator | ~10 saniye | ~5 saniye |

**Toplam (ilk kez)**: ~8-13 dakika
**Toplam (sonraki)**: ~3-4 dakika

## Eğer Hala Yavaşsa

### 1. İnternet Bağlantısını Kontrol Edin
```bash
# Model indirme hızını test edin
curl -o /dev/null -s -w "%{speed_download}\n" https://huggingface.co
```

### 2. HuggingFace Cache'i Kontrol Edin
```bash
du -sh ~/.cache/huggingface
# Normal: 3-5 GB
```

### 3. Model Yükleme Loglarını Kontrol Edin
```bash
tail -f logs/proje.log
```

### 4. Disk Alanını Kontrol Edin
```bash
df -h
# En az 10 GB boş alan olmalı
```

## Hızlı Çözüm

Eğer modeller çok yavaş yükleniyorsa:

1. **Uygulamayı yeniden başlatın** (lazy loading ile artık başlangıç hızlı)
2. **İlk analiz sırasında modeller yüklenecek** (sadece gerektiğinde)
3. **Sonraki analizler çok daha hızlı olacak**

## Model Boyutları

| Model | Boyut | İndirme Süresi (10 Mbps) |
|-------|-------|-------------------------|
| xlm-roberta-base | ~1 GB | ~15 dakika |
| roberta-base-openai-detector | ~500 MB | ~7 dakika |
| cross-encoder/nli-distilroberta-base | ~250 MB | ~4 dakika |
| en_core_web_lg | ~500 MB | ~7 dakika |

**Toplam**: ~2.25 GB, ~33 dakika (ilk kez)

## Öneriler

1. ✅ **Lazy Loading**: Artık aktif (modeller sadece gerektiğinde yüklenir)
2. ✅ **Önbellekleme**: Modeller bir kez yüklendikten sonra önbellekte kalır
3. ⚠️ **İnternet**: İlk yükleme için iyi bir bağlantı gerekir

## Sorun Devam Ederse

1. Log dosyasını kontrol edin: `logs/proje.log`
2. Hangi modelin takıldığını bulun
3. O modeli manuel olarak test edin:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("xlm-roberta-base")
```
