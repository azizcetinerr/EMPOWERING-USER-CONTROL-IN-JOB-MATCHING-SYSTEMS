# Metrikleri Kaydetme

Mevcut IP5 sonuçlarından metrikleri hesaplayıp kaydetmek için:

## Kullanım

```bash
python save_metrics_from_existing.py
```

veya

```bash
python3 save_metrics_from_existing.py
```

## Script Ne Yapar?

1. `outputs/results/IP5_Nihai_Adil_Siralama.csv` dosyasını yükler
2. `IP1_Ciktisi_Projeler_Vektorlu.pkl` dosyasından vektörleri yükler (diversity için gerekli)
3. 5 metrik hesaplar:
   - Relevance Accuracy (Ortalama M_Skoru)
   - Diversity Score (Çeşitlilik)
   - Confidence (M skorları std sapması)
   - RC (Recommendation Coverage)
   - UC (User Coverage)
4. Sonuçları `outputs/metrics/` dizinine kaydeder:
   - `metrics_report_YYYYMMDD_HHMMSS.json` (JSON formatında)
   - `metrics_report_YYYYMMDD_HHMMSS.md` (Markdown formatında)
   - `metrics_visualization_YYYYMMDD_HHMMSS.png` (Grafik)

## Not

Script, web uygulamanızın çalıştığı Python interpreter'ı ile aynı Python'u kullanmalıdır.
