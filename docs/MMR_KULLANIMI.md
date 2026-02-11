# MMR (Maximal Marginal Relevance) Kullanımı

## Durum
MMR (Maximal Marginal Relevance) algoritması eklendi ve aktif edildi. Bu algoritma, Diversity Score'u artırmak için çeşitlilik faktörünü sıralamaya dahil eder.

## Önemli Not ⚠️

**Mevcut metrik sonuçlarınız muhtemelen MMR'sız oluşturulmuştur!** MMR'ın etkisini görmek için:

1. **Pipeline'ı yeniden çalıştırın** (Flask app veya `src/pipeline.py`)
2. **VEYA** `test_mmr.py` script'ini çalıştırarak MMR'ın çalışıp çalışmadığını test edin

## Config Ayarları

`config/config.yaml` dosyasında:

```yaml
diversity_aware_ranking:
  enabled: true   # MMR aktif
  top_k: 10       # MMR ile seçilecek proje sayısı
  lambda: 0.6     # Relevance ağırlığı (0.6 = %60 relevance, %40 diversity)
```

### Lambda Parametresi Ayarı

- **lambda = 1.0**: Sadece relevance (çeşitlilik yok)
- **lambda = 0.5**: Eşit ağırlık (relevance ve diversity)
- **lambda = 0.0**: Sadece diversity (relevance yok)

**Diversity Score'un artması için lambda'yı düşürmeyi deneyin:**
- `lambda: 0.4` → %60 diversity, %40 relevance
- `lambda: 0.3` → %70 diversity, %30 relevance

## Test Script'i

MMR'ın çalışıp çalışmadığını test etmek için:

```bash
python test_mmr.py
```

Bu script:
- MMR ayarlarını kontrol eder
- Mevcut sonuçları yükler
- MMR öncesi ve sonrası metrikleri karşılaştırır
- Diversity Score'un artıp artmadığını gösterir

## MMR Nasıl Çalışır?

1. **İlk proje**: En yüksek Adil_Skor'lu proje seçilir
2. **Sonraki projeler**: Her adımda, kalan projelerden şu formülle en iyi olan seçilir:
   ```
   MMR_score = lambda * relevance + (1 - lambda) * diversity
   ```
   - **relevance**: Adil_Skor (0-1 normalize)
   - **diversity**: Seçili projelere olan maksimum benzerlik (1 - max_similarity)

3. **Top-K**: İlk K proje MMR ile seçilir, kalanlar orijinal sıralamada kalır

## Beklenen Sonuçlar

MMR aktifken:
- **Diversity Score artmalı** (0.0'dan yukarı çıkmalı)
- **Relevance Accuracy biraz düşebilir** (çeşitlilik için trade-off)
- **Confidence değişebilir** (farklı projeler seçildiği için)

## Sorun Giderme

### Diversity Score Hala 0.0

1. **Pipeline'ı yeniden çalıştırın** (eski sonuçlar MMR'sız olabilir)
2. **Lambda'yı düşürün** (örn. 0.4 veya 0.3)
3. **Vektörlerin mevcut olduğunu kontrol edin** (`IP1_Ciktisi_Projeler_Vektorlu.pkl`)
4. **Log mesajlarını kontrol edin** (MMR çalışıyor mu?)

### MMR Çalışmıyor

1. `config.yaml`'da `diversity_aware_ranking.enabled: true` olduğundan emin olun
2. Log mesajlarında "🔄 MMR (Maximal Marginal Relevance) ile diversity-aware ranking uygulanıyor" mesajını arayın
3. `test_mmr.py` script'ini çalıştırın

## Örnek Kullanım

1. **Config'i ayarlayın**:
   ```yaml
   diversity_aware_ranking:
     enabled: true
     top_k: 10
     lambda: 0.5  # Veya daha düşük (0.4, 0.3)
   ```

2. **Pipeline'ı çalıştırın**:
   ```bash
   python src/pipeline.py
   # VEYA
   python web/app.py  # Flask app
   ```

3. **Sonuçları kontrol edin**:
   - Log dosyasında "MMR ranking tamamlandı" mesajını arayın
   - `outputs/metrics/` dizinindeki metrik raporlarını kontrol edin
   - Diversity Score'un artıp artmadığını görün
