# Lambda Test Talimatları

## Sorun
MMR Diversity Score 0.0 kalıyor ve değişmiyor.

## Çözüm Adımları

### 1. Lambda Test Script'ini Çalıştırın

```bash
python test_mmr_lambda.py
```

Bu script:
- 8 farklı lambda değeri test eder (0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
- Her lambda için Diversity Score'u hesaplar
- En iyi lambda değerini bulur ve önerir

### 2. En İyi Lambda Değerini Config'e Ekleyin

Script çıktısında "⭐ EN İYİ" işaretli lambda değerini bulun ve config.yaml'a ekleyin:

```yaml
diversity_aware_ranking:
  enabled: true
  top_k: 10
  lambda: 0.1  # Script'ten bulunan en iyi değer
```

### 3. Pipeline'ı Yeniden Çalıştırın

Yeni lambda değeri ile pipeline'ı çalıştırın.

## Notlar

- Lambda = 0.0 → Sadece diversity (relevance yok)
- Lambda = 1.0 → Sadece relevance (diversity yok)
- Lambda = 0.3 → %30 relevance, %70 diversity
- Lambda = 0.1 → %10 relevance, %90 diversity

Eğer hiçbir lambda Diversity Score'u artırmazsa, bu projelerin gerçekten çok benzer olduğu anlamına gelir. Bu durumda:
- top_k değerini artırmayı deneyin (örn. 20 veya 30)
- Veya proje havuzunun gerçekten çok benzer olduğunu kabul edin
