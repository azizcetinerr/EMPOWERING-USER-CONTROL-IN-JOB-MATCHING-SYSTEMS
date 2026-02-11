# Matematiksel Klonlama Sorunu Analizi

## Mevcut Durum

Metriklerde ciddi bir "Matematiksel Klonlama" sorunu tespit edildi:

- **Diversity Score**: 0.0 (Top 5 proje arasında çeşitlilik yok)
- **Confidence**: 0.0586 (M_Skoru standart sapması çok düşük, skorlar neredeyse aynı)
- **Relevance Accuracy**: 52.98 (Normal)

## Sorunun Nedenleri

1. **Z-Score Normalizasyonu Çok Agresif**
   - Z-Score normalizasyonu tüm skorları 50 civarına çekiyor
   - Formül: `normalized = 50 + (z_score * 20)`
   - Bu, tüm M_Skoru değerlerini birbirine çok yakın hale getiriyor

2. **Vektör Benzerliği**
   - Top 5 projenin vektörleri çok benzer
   - Cosine similarity yüksek → Diversity düşük

3. **Sıralama Algoritması**
   - Sadece skorlara göre sıralama yapılıyor
   - Çeşitlilik faktörü göz ardı ediliyor

## Çözüm Önerileri

### 1. Diversity-Aware Ranking (Çeşitlilik Faktörlü Sıralama)
- Top-K seçiminde çeşitlilik faktörü eklenmeli
- MMR (Maximal Marginal Relevance) algoritması kullanılabilir
- Skor + Diversity bonusu kombinasyonu

### 2. Normalizasyon İyileştirmesi
- Z-Score normalizasyonu yerine Min-Max normalizasyonu
- Veya Z-Score'da daha geniş aralık (mevcut: ±2.5 std, önerilen: ±3 std)

### 3. Vektör Çeşitliliği Kontrolü
- Diversity Score'u sıralamaya entegre et
- Benzer projeleri filtrele veya çeşitlilik bonusu ekle

