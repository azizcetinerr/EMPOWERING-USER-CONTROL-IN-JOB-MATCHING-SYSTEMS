# Matematiksel Klonlama Sorunu - Çözüm Önerileri

## Sorun

- **Diversity Score: 0.0** → Top 5 proje birbirine çok benzer
- **Confidence: 0.0586** → M_Skoru'lar neredeyse aynı (std sapma çok düşük)

## Ana Neden

**Z-Score normalizasyonu** tüm skorları 50 civarına çekiyor, bu da:
1. Tüm projelerin M_Skoru'larını birbirine yakınlaştırıyor
2. Standart sapmayı düşürüyor (Confidence düşük)
3. Sıralamayı sadece küçük farklara dayandırıyor

## Çözüm 1: Normalizasyon İyileştirmesi (Önerilen)

Z-Score normalizasyonunda daha geniş aralık kullan:

```python
# Mevcut: normalized = 50 + (z_scores * 20)  # ±2.5 std = 0-100
# Önerilen: normalized = 50 + (z_scores * 25)  # ±2 std = 0-100 (daha geniş aralık)
```

## Çözüm 2: Diversity-Aware Ranking (En Etkili)

Top-K seçiminde çeşitlilik faktörü ekle:

```python
def diversity_aware_ranking(df, top_k=5, lambda_param=0.5):
    """
    MMR (Maximal Marginal Relevance) benzeri çeşitlilik faktörlü sıralama
    
    Args:
        df: Proje DataFrame (Adil_Skor ile sıralanmış)
        top_k: Seçilecek proje sayısı
        lambda_param: Relevance (0.0) ile Diversity (1.0) arası denge (0.5 = eşit)
    
    Returns:
        Çeşitlilik faktörlü top-k projeler
    """
    selected = []
    remaining = df.copy()
    
    # İlk proje: En yüksek skorlu
    first = remaining.nlargest(1, 'Adil_Skor')
    selected.append(first.index[0])
    remaining = remaining.drop(first.index[0])
    
    # Kalan projeler için çeşitlilik faktörlü seçim
    while len(selected) < top_k and len(remaining) > 0:
        max_score = -1
        best_idx = None
        
        for idx, row in remaining.iterrows():
            # Relevance score (Adil_Skor)
            relevance = row['Adil_Skor']
            
            # Diversity score (seçili projelere benzerlik)
            diversity = calculate_diversity_score(row, selected, df)
            
            # Kombine score: lambda * relevance + (1-lambda) * diversity
            combined = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if combined > max_score:
                max_score = combined
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining = remaining.drop(best_idx)
    
    return df.loc[selected]
```

## Çözüm 3: Metrik Hesaplamada Ham Skorlar Kullan

Confidence metriği için normalize edilmiş skorlar yerine ham skorları kullan:

```python
# Mevcut: m_score_column="M_Skoru" (normalize edilmiş)
# Önerilen: m_score_column="M_Skoru_Ham" (ham skorlar)
```

## Çözüm 4: Top-K Artırma

Top-K'yi 5'ten 10'a çıkararak daha geniş bir yelpaze göster.

## Uygulama Önceliği

1. **Çözüm 3** (En hızlı): Metriklerde ham skorları kullan
2. **Çözüm 1** (Orta): Normalizasyon aralığını genişlet
3. **Çözüm 2** (En etkili ama kompleks): Diversity-aware ranking ekle
