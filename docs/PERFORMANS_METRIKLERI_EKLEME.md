# Performans Metrikleri Modülü - Dokümantasyon

## Özet

Öneri sisteminin performansını ölçmek için `FinalRankingCalculator` sınıfına `calculate_performance_metrics` fonksiyonu eklendi. Bu fonksiyon, en iyi 5 proje (Top-K) için 3 metrik hesaplıyor:

1. **Relevance Accuracy (Alaka Düzeyi)**: M_Skoru ortalaması
2. **Diversity Score (Çeşitlilik)**: Item-to-Item cosine similarity ile 1 - ortalama benzerlik
3. **Confidence (Güven Aralığı)**: M skorlarının standart sapması

## Yapılan Değişiklikler

### 1. `src/scoring/final_ranking.py`

#### a) Import Eklendi
```python
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
```

#### b) Yeni Fonksiyon: `calculate_performance_metrics`
- **Konum**: `FinalRankingCalculator` sınıfı içinde
- **Parametreler**:
  - `df_ranked`: Sıralanmış proje DataFrame'i
  - `top_k`: Hesaplanacak en iyi proje sayısı (default: 5)
  - `vector_column`: Proje vektörlerinin bulunduğu sütun (default: "beceri_vektorleri")
  - `m_score_column`: Matching Score sütunu (default: "M_Skoru")
- **Return**: `(relevance_accuracy, diversity_score, confidence)` tuple

#### c) Entegrasyon
`calculate_ranking` fonksiyonunun sonunda (sıralama tamamlandıktan sonra) otomatik olarak çağrılıyor.

## Metrik Detayları

### 1. Relevance Accuracy
- **Formül**: `mean(M_Skoru)` (Top-K projeler için)
- **Aralık**: 0-100 (normalize edilmiş skorlar için)
- **Açıklama**: Sistemin kullanıcının CV'si ile ne kadar yüksek eşleşme bulabildiğini gösterir
- **Yüksek = İyi**: Daha yüksek ortalama M skoru, daha alakalı öneriler demektir

### 2. Diversity Score (ILS - Intra-List Similarity)
- **Formül**: `1 - mean(cosine_similarity(pairwise))`
- **Aralık**: 0-1
- **Açıklama**: Önerilen projelerin birbirine ne kadar benzer olduğunu ölçer
- **Yüksek = İyi**: Daha yüksek diversity, daha çeşitli öneriler demektir
- **Hesaplama**:
  1. Top-K projelerin vektörlerini al
  2. Pairwise cosine similarity hesapla
  3. Diagonal'i 0 yap (kendisiyle benzerlik hariç)
  4. Upper triangle'ın ortalamasını al
  5. Diversity = 1 - Ortalama Benzerlik

### 3. Confidence
- **Formül**: `std(M_Skoru)` (Top-K projeler için)
- **Aralık**: 0+ (standart sapma)
- **Açıklama**: Önerilen projelerin M skorlarının tutarlılığını ölçer
- **Yorumlama**:
  - Düşük std: Tutarlı öneriler (benzer kalitede)
  - Yüksek std: Değişken kalitede öneriler

## Çıktı Formatı

Fonksiyon, `calculate_ranking` içinde otomatik olarak çağrılıyor ve sonuçlar logger ile yazdırılıyor:

```
📊 SİSTEM PERFORMANSI: Relevance=75.32 | Diversity=0.68 | Confidence=12.45
```

## Hata Yönetimi

Fonksiyon, hataları gracefully handle eder:
- Vektörler yoksa: Diversity = 0.0
- M_Skoru yoksa: Relevance = 0.0, Confidence = 0.0
- Tek proje varsa: Diversity = 0.0 (çeşitlilik hesaplanamaz)
- Vektör formatı hatalıysa: Uyarı loglanır, Diversity = 0.0

## Kullanım Örneği

Fonksiyon otomatik olarak `calculate_ranking` içinde çağrılıyor:

```python
# Pipeline içinde
ranking_calc = FinalRankingCalculator(config, logger)
df_final = ranking_calc.calculate_ranking(df_projects)
# Otomatik olarak metrikler hesaplanıp loglanıyor
```

Manuel kullanım (isteğe bağlı):

```python
relevance, diversity, confidence = ranking_calc.calculate_performance_metrics(
    df_final,
    top_k=5,
    vector_column="beceri_vektorleri",
    m_score_column="M_Skoru"
)
```

## Notlar

- Vektörler normalize edilerek cosine similarity hesaplanıyor
- Top-K parametresi değiştirilebilir (default: 5)
- Sadece geçerli vektörleri olan projeler diversity hesaplamasına dahil edilir
- Performance impact minimal (sadece Top-K için hesaplama yapılıyor)
