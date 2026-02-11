# MMR (Maximal Marginal Relevance) Algoritması Entegrasyonu ve Sorun Çözüm Raporu

## 📋 İçindekiler
1. [Proje Özeti](#proje-özeti)
2. [Başlangıç Durumu ve Tespit Edilen Sorunlar](#başlangıç-durumu-ve-tespit-edilen-sorunlar)
3. [MMR Algoritması Entegrasyonu](#mmr-algoritması-entegrasyonu)
4. [Karşılaşılan Teknik Sorunlar ve Çözümleri](#karşılaşılan-teknik-sorunlar-ve-çözümleri)
5. [Yapılan Testler ve Sonuçlar](#yapılan-testler-ve-sonuçlar)
6. [Mevcut Durum](#mevcut-durum)
7. [Gelecek Öneriler](#gelecek-öneriler)
8. [Teknik Detaylar](#teknik-detaylar)

---

## 1. Proje Özeti

Bu proje, TÜBİTAK destekli bir öneri sistemi geliştirme çalışmasıdır. Sistem, adayların CV'lerine göre uygun projeleri önermektedir. Sistem şu bileşenlerden oluşmaktadır:

- **Matching Score (M-Skoru)**: Aday becerileri ile proje gereksinimleri arasındaki uyum skoru
- **Sincerity Score (S-Skoru)**: Adayın samimiyet skoru
- **Reliability Score (R-Skoru)**: Proje bilgilerinin güvenilirlik skoru
- **Final Ranking**: Tüm skorların ağırlıklı ortalaması ile nihai sıralama
- **Performance Metrics**: Relevance Accuracy, Diversity Score, Confidence, Recommendation Coverage, User Coverage

### Sistemin Amacı
Sistem, adaylara en uygun projeleri önermekle birlikte, önerilen projelerin birbirinden farklı (diverse) olmasını da sağlamalıdır. Çünkü aynı tür projeleri önermek, kullanıcı deneyimini düşürür ve sistemin etkinliğini azaltır.

---

## 2. Başlangıç Durumu ve Tespit Edilen Sorunlar

### 2.1. İlk Tespit Edilen Sorun: "Matematiksel Klonlama"

**Sorun Tanımı:**
Sistem metrikleri hesaplandığında şu sonuçlar elde ediliyordu:

```json
{
  "metrics": {
    "relevance_accuracy": 52.9798,
    "diversity_score": 0.0,
    "confidence": 0.0586,
    "recommendation_coverage": 0.0053,
    "user_coverage": 0.0054
  }
}
```

**Sorunun Belirtileri:**
1. **Diversity Score = 0.0**: Bu, önerilen projelerin birbirine neredeyse identik olduğunu gösteriyordu
2. **Confidence = 0.0586**: Çok düşük bir güven skoru, skorların birbirine çok yakın olduğunu (düşük standart sapma) gösteriyordu
3. Bu iki durum birlikte "Matematiksel Klonlama" problemi olarak adlandırıldı

**Sorunun Nedenleri:**
- Sistem, sadece relevance (uyum) skoruna göre sıralama yapıyordu
- Diversity (çeşitlilik) faktörü hiç dikkate alınmıyordu
- En yüksek skorlu projeler birbirine benzer olduğu için, top-5 öneriler birbirinin kopyası gibiydi

### 2.2. İlk Çözüm Denemeleri

#### Deneme 1: Confidence Hesaplamasını Düzeltme
- **Yapılan**: Confidence metrik hesaplamasında normalize edilmiş skorlar yerine ham skorlar kullanılmaya başlandı
- **Sonuç**: Confidence değeri biraz arttı (0.0586 → 0.0429) ama Diversity Score hala 0.0 kaldı
- **Neden Başarısız**: Sorun confidence'da değil, diversity'deydi

#### Deneme 2: Logging ve Debugging
- **Yapılan**: Diversity hesaplama koduna detaylı logging eklendi
- **Sonuç**: Sorunun kaynağı daha iyi anlaşıldı ama çözüm sağlanamadı
- **Neden Başarısız**: Logging sorunu gösterdi ama çözmedi

---

## 3. MMR Algoritması Entegrasyonu

### 3.1. MMR Algoritması Nedir?

**MMR (Maximal Marginal Relevance)** algoritması, information retrieval ve recommendation sistemlerinde kullanılan bir yeniden sıralama (re-ranking) algoritmasıdır.

**Amaç**: Relevance (uygunluk) ve Diversity (çeşitlilik) arasında optimal dengeyi sağlamak.

**Formül**:
```
MMR_Score = λ * Relevance + (1 - λ) * Diversity
```

Burada:
- **λ (lambda)**: Relevance ağırlığı (0.0 - 1.0 arası)
  - λ = 1.0 → Sadece relevance önemli (diversity yok)
  - λ = 0.5 → Relevance ve diversity eşit ağırlıkta
  - λ = 0.0 → Sadece diversity önemli (relevance yok)
- **Relevance**: Projenin aday profiliyle uyumu (Adil_Skor)
- **Diversity**: Projenin daha önce seçilen projelere olan farklılığı (1 - maksimum cosine similarity)

### 3.2. MMR Algoritmasının Çalışma Mantığı

1. **İlk Proje Seçimi**: En yüksek Adil_Skor'a sahip proje seçilir (zaten sıralanmış listede ilk sırada)

2. **Kalan Projeler için İterasyon**:
   - Her kalan proje için:
     - **Relevance**: Adil_Skor (normalize edilmiş: 0-1 arası)
     - **Diversity**: Seçili projelere olan maksimum cosine similarity'nin tersi (1 - max_similarity)
     - **MMR_Score**: λ * Relevance + (1 - λ) * Diversity
   - En yüksek MMR_Score'a sahip proje seçilir
   - Seçilen proje "seçili projeler" listesine eklenir

3. **Sonuç**: Top-K proje, relevance ve diversity dengesi gözetilerek seçilmiş olur

### 3.3. MMR Entegrasyonu - Kod Değişiklikleri

#### 3.3.1. Config Dosyasına MMR Ayarları Eklendi

**Dosya**: `config/config.yaml`

```yaml
# Diversity-Aware Ranking (MMR) Ayarları
diversity_aware_ranking:
  enabled: true  # MMR algoritmasını etkinleştir
  top_k: 10  # MMR ile seçilecek proje sayısı (top-K)
  lambda: 0.3  # Relevance ağırlığı (0.0-1.0), 0.3 = %30 relevance, %70 diversity
```

**Neden Bu Değerler?**
- `enabled: true`: MMR'ı aktif hale getirmek için
- `top_k: 10`: İlk 10 projeyi MMR ile seçmek (daha fazla çeşitlilik için yeterli sayı)
- `lambda: 0.3`: Diversity'yi artırmak için relevance'den daha fazla ağırlık verdik (%70 diversity, %30 relevance)

#### 3.3.2. FinalRankingCalculator Sınıfına MMR Metodu Eklendi

**Dosya**: `src/scoring/final_ranking.py`

**Yapılan Değişiklikler**:

1. **`__init__` metoduna MMR config yükleme eklendi**:
```python
# MMR (Diversity-Aware Ranking) ayarları
mmr_config = config.get('diversity_aware_ranking', {})
self.mmr_enabled = mmr_config.get('enabled', False)
self.mmr_top_k = mmr_config.get('top_k', 10)
self.mmr_lambda = mmr_config.get('lambda', 0.5)
```

2. **`calculate_ranking` metoduna MMR çağrısı eklendi**:
```python
# Sırala
df_result = df_result.sort_values(by='Adil_Skor', ascending=False).reset_index(drop=True)

# MMR (Maximal Marginal Relevance) ile diversity-aware top-K seçimi
if self.mmr_enabled and self.mmr_top_k > 0:
    self.logger.info(f"🔄 MMR (Maximal Marginal Relevance) ile diversity-aware ranking uygulanıyor (Top-{self.mmr_top_k})...")
    df_result = self._apply_mmr_ranking(
        df_result,
        top_k=self.mmr_top_k,
        lambda_param=self.mmr_lambda
    )
```

3. **`_apply_mmr_ranking` metodu eklendi**:
   - İlk projeyi seçer (en yüksek Adil_Skor)
   - Kalan projeler için MMR skoru hesaplar
   - En yüksek MMR skorlu projeleri seçer
   - Seçilen projeleri başa alır, kalanları orijinal sırada bırakır

**Neden Bu Yaklaşım?**
- Top-K projeyi MMR ile seçmek, çeşitliliği artırır
- Kalan projeleri orijinal sırada bırakmak, sistemin genel performansını korur
- Sadece önerilen projelerin çeşitliliği artırılır, tüm sistem etkilenmez

---

## 4. Karşılaşılan Teknik Sorunlar ve Çözümleri

### 4.1. Sorun: Vektör Formatları Tutarsız

**Sorun Tanımı**:
`beceri_vektorleri` sütunundaki vektörler farklı formatlardaydı:
- Bazıları 1D array: `(768,)`
- Bazıları 2D array: `(1, 768)` veya `(7, 768)` veya `(9, 768)`
- Bazıları scalar float (NaN)

**Hata Mesajı**:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 1, the array at index 0 has size 768 and the array at index 1 has size 5376
```

**Neden Bu Sorun Oluştu?**
- Farklı projeler farklı sayıda beceriye sahip
- Her beceri için ayrı embedding vektörü oluşturulmuş
- Bazı projelerde 1 beceri var → `(1, 768)` formatı
- Bazı projelerde 7 beceri var → `(7, 768)` formatı
- Cosine similarity hesaplamak için tüm vektörler aynı boyutta olmalı

**Çözüm**:
Tüm vektörleri standart bir formata çevirdik (1D array, 768 boyutunda):

```python
# 2D vektör (n_skills, 768) → 1D vektör (768,)
if vec_array.ndim == 2:
    if vec_array.shape[1] == 768:
        normalized_vec = vec_array.mean(axis=0)  # Ortalamasını al → (768,)
    else:
        normalized_vec = vec_array.flatten()
elif vec_array.ndim == 1:
    normalized_vec = vec_array  # Zaten 1D
elif vec_array.ndim > 2:
    normalized_vec = vec_array.flatten()  # 3D+ → flatten
```

**Neden Ortalama Alma?**
- Bir proje birden fazla beceriye sahipse, her beceri için ayrı embedding var
- Bu embedding'lerin ortalamasını alarak, projenin "temsili vektörü"nü oluşturuyoruz
- Bu, projenin genel beceri profilini temsil eder

### 4.2. Sorun: pd.isna() NumPy Array'lerde Çalışmıyor

**Sorun Tanımı**:
```python
if pd.isna(vector):  # Hata veriyor!
    continue
```

**Hata Mesajı**:
```
ValueError: The truth value of an array with more than one element is ambiguous. 
Use a.any() or a.all()
```

**Neden Bu Sorun Oluştu?**
- `pd.isna()` NumPy array'lerde boolean array döndürür, tek bir boolean değil
- `if` statement'ta boolean array kullanılamaz (ambiguous)

**Çözüm**:
```python
# Önce None kontrolü
if vector is None:
    continue

# NaN float kontrolü (pandas merge sonrası NaN float olabilir)
if isinstance(vector, float) and (np.isnan(vector) or pd.isna(vector)):
    continue

# NumPy array kontrolü için pd.isna() kullanmıyoruz, direkt np.array() ile dönüştürüyoruz
```

### 4.3. Sorun: Çift Normalizasyon

**Sorun Tanımı**:
Vektörler iki kez normalize ediliyordu:
1. İlk döngüde: `vectors_list`'e eklenirken (normalize edilmiş olarak)
2. İkinci döngüde: `normalized_vectors`'a eklenirken tekrar normalize ediliyordu

**Sonuç**: Vektörler yanlış normalize ediliyordu, bu da similarity hesaplamalarını bozuyordu.

**Çözüm**:
İlk döngüde sadece vektörleri 1D'ye çevirdik (normalize etmeden):
```python
# İlk döngü: Sadece 1D'ye çevir (normalize etme)
normalized_vec = vec_array.mean(axis=0)  # (768,)
vectors_list.append(normalized_vec)  # Henüz normalize edilmemiş

# İkinci döngü: Normalize et (unit vector yap)
norm = np.linalg.norm(vec_array)
if norm > 1e-8:
    vec_normalized = vec_array / norm  # Unit vector
    normalized_vectors.append(vec_normalized)
```

### 4.4. Sorun: Indentation Hatası (if-else Blok Yapısı)

**Sorun Tanımı**:
Diversity hesaplama kodunda, `vectors_matrix` oluşturulduktan sonraki kodlar `else` bloğunun dışındaydı:

```python
if len(normalized_vectors) < 2:
    diversity_score = 0.0
else:
    vectors_matrix = np.vstack(normalized_vectors)

# Bu kod else dışında! Hata!
similarity_matrix = cosine_similarity(vectors_matrix)
```

**Sonuç**: `normalized_vectors < 2` durumunda `vectors_matrix` tanımsız kalıyordu ve hata oluşuyordu.

**Çözüm**:
Tüm similarity hesaplama kodunu `else` bloğu içine aldık:
```python
if len(normalized_vectors) < 2:
    diversity_score = 0.0
else:
    vectors_matrix = np.vstack(normalized_vectors)
    similarity_matrix = cosine_similarity(vectors_matrix)
    # ... diğer hesaplamalar ...
    diversity_score = 1.0 - avg_similarity
```

### 4.5. Sorun: Vektörlerin DataFrame Merge Sonrası Kaybolması

**Sorun Tanımı**:
Test script'lerinde vektörler DataFrame'den okunamıyordu çünkü:
- Vektörler pickle dosyasında DataFrame formatında saklanıyordu
- CSV dosyasında vektörler yoktu
- Merge işlemi sırasında link formatları uyuşmuyordu

**Çözüm**:
```python
# Link'leri string'e çevir (uyumsuzluk olmasın)
vectors_df['link'] = vectors_df['link'].astype(str)
df['link'] = df['link'].astype(str)
df = df.merge(vectors_df, on='link', how='left')
```

---

## 5. Yapılan Testler ve Sonuçlar

### 5.1. Test Script'leri Oluşturuldu

#### 5.1.1. `test_mmr.py`
**Amaç**: MMR'ın mevcut lambda değeriyle çalışıp çalışmadığını test etmek

**Yaptığı İşlemler**:
1. CSV ve vektör dosyalarını yükler
2. Baseline metrikleri hesaplar (MMR öncesi)
3. MMR uygular
4. MMR sonrası metrikleri hesaplar
5. Karşılaştırma yapar

**Sonuç**: MMR çalışıyor ama Diversity Score hala düşük

#### 5.1.2. `test_mmr_lambda.py`
**Amaç**: Farklı lambda değerlerini test edip en iyisini bulmak

**Yaptığı İşlemler**:
1. Lambda değerlerini test eder: [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
2. Her lambda için MMR uygular
3. Diversity ve Relevance metriklerini hesaplar
4. En iyi lambda değerini önerir

**Sonuçlar**:
```
Lambda     Diversity    Relevance    Değişim
Baseline   0.0028       52.98        -
0.9        0.0000       52.98        -0.0028
0.7        0.0000       52.98        -0.0028
...
0.0        0.0000       52.98        -0.0028
```

**Gözlem**: Hiçbir lambda değeri Diversity Score'u artıramadı. Bu, MMR sonrası vektörlerin diversity hesaplama kodunda işlenemediğini gösteriyor.

#### 5.1.3. `debug_diversity.py`
**Amaç**: Diversity Score'un neden 0.0 olduğunu detaylı analiz etmek

**Yaptığı İşlemler**:
1. Top-5 projelerin vektörlerini inceler
2. Her vektörün tipini, shape'ini, dtype'ını kontrol eder
3. Pairwise similarity'leri hesaplar
4. Detaylı debug bilgisi sağlar

**Sonuç**: Vektör formatları tutarsızlığı tespit edildi

### 5.2. Test Sonuçları

#### Baseline (MMR Öncesi)
```
Diversity Score: 0.0028
Relevance Accuracy: 52.98
Confidence: 0.0429
```

**Yorum**: 
- Diversity çok düşük (neredeyse 0)
- Bu, top-5 projelerin birbirine çok benzer olduğunu gösteriyor
- 3/5 projede vektör var (2 projede NaN)

#### MMR Sonrası (Lambda=0.0 - Sadece Diversity)
```
Diversity Score: 0.0000
Relevance Accuracy: 52.98
```

**Yorum**:
- MMR çalışıyor (4/5 proje pozisyonu değişti)
- Ancak Diversity Score hala 0.0
- Bu, diversity hesaplama kodunda bir sorun olduğunu gösteriyor
- MMR sonrası top-5'te tüm projelerde vektör var

---

## 6. Mevcut Durum

### 6.1. Başarılan İyileştirmeler

✅ **MMR Algoritması Entegre Edildi**:
- `_apply_mmr_ranking` metodu eklendi
- Config dosyasına ayarlar eklendi
- `calculate_ranking` metoduna entegre edildi

✅ **Vektör İşleme Düzeltildi**:
- 2D vektörler → 1D'ye dönüştürülüyor (mean ile)
- NaN float kontrolü eklendi
- `pd.isna()` sorunu çözüldü
- Çift normalizasyon sorunu çözüldü

✅ **Kod Hataları Düzeltildi**:
- Indentation hatası düzeltildi
- Exception handling iyileştirildi
- Vector processing mantığı sadeleştirildi

✅ **Test Altyapısı Oluşturuldu**:
- `test_mmr.py`: Temel MMR testi
- `test_mmr_lambda.py`: Lambda parametresi optimizasyonu
- `debug_diversity.py`: Detaylı diversity analizi

### 6.2. Kalan Sorunlar

⚠️ **MMR Sonrası Diversity Score Hala 0.0**:
- **Sorun**: MMR sonrası top-5'te tüm projelerde vektör var, ancak diversity hesaplama kodunda sadece 1 vektör işleniyor
- **Hata Mesajı**: "⚠️ Diversity hesaplanamadı: Yeterli vektör yok (1/5)"
- **Olası Nedenler**:
  1. `pd.concat` sonrası vektörlerin referansları kaybolmuş olabilir
  2. Exception handling çok geniş, vektörler sessizce atlanıyor olabilir
  3. Vektör işleme kodunda bir edge case atlanmış olabilir

⚠️ **Baseline Diversity Çok Düşük**:
- Baseline'da Diversity = 0.0028 (neredeyse 0)
- Bu, top-5 projelerin gerçekten çok benzer olduğunu gösteriyor
- MMR olmadan bile diversity düşük

### 6.3. Kod Durumu

**Çalışan Kodlar**:
- ✅ MMR algoritması implementasyonu
- ✅ Vektör normalizasyon mantığı
- ✅ Baseline diversity hesaplama
- ✅ Test script'leri

**Sorunlu Kodlar**:
- ⚠️ MMR sonrası diversity hesaplama (vektörler işlenemiyor)

---

## 7. Gelecek Öneriler

### 7.1. Kısa Vadeli Çözümler

#### Öneri 1: Vektör İşleme Kodunu Detaylı Debug Etme
```python
# Diversity hesaplama koduna detaylı logging ekle
for idx, row in top_projects.iterrows():
    vector = row[vector_column]
    self.logger.debug(f"Proje {idx}: vector type={type(vector)}, shape={getattr(vector, 'shape', 'N/A')}")
    # ... vektör işleme ...
    self.logger.debug(f"Proje {idx}: vectors_list length={len(vectors_list)}")
```

#### Öneri 2: Exception Handling'i Daraltma
```python
# Geniş exception yerine spesifik exception'lar yakala
except ValueError as e:
    self.logger.warning(f"Vektör işleme hatası (ValueError): {e}")
except TypeError as e:
    self.logger.warning(f"Vektör işleme hatası (TypeError): {e}")
except AttributeError as e:
    self.logger.warning(f"Vektör işleme hatası (AttributeError): {e}")
```

#### Öneri 3: Vektörleri DataFrame'de Doğrudan Saklama
MMR sonrası vektörlerin kaybolmaması için, vektörleri DataFrame'de doğrudan saklamak yerine, her zaman pickle dosyasından yeniden yüklemek:

```python
# MMR sonrası vektörleri yeniden yükle
df_mmr = ranking_calc._apply_mmr_ranking(...)
# Vektörleri yeniden merge et
df_mmr = df_mmr.merge(vectors_df[['link', 'beceri_vektorleri']], on='link', how='left')
```

### 7.2. Orta Vadeli İyileştirmeler

#### Öneri 1: Vektör Formatını Standardize Etme
Tüm vektörleri preprocessing aşamasında standart formata çevirmek:
- Tüm vektörleri 1D array (768,) formatına çevir
- NaN vektörleri temizle veya default vektör kullan

#### Öneri 2: Alternatif Diversity Metrikleri Deneme
- **Intra-List Diversity (ILD)**: Önerilen liste içindeki çeşitlilik
- **Coverage-based Diversity**: Farklı kategorilerden proje önerme
- **Content-based Diversity**: Proje içeriklerine göre çeşitlilik

#### Öneri 3: Lambda Parametresini Dinamik Yapma
Kullanıcı tercihlerine göre lambda'yı ayarlama:
- Kullanıcı "daha fazla çeşitlilik" isterse → lambda düşür
- Kullanıcı "daha uyumlu projeler" isterse → lambda artır

### 7.3. Uzun Vadeli İyileştirmeler

#### Öneri 1: Embedding Kalitesini Artırma
- Daha iyi bir embedding modeli kullanmak
- Domain-specific fine-tuning yapmak
- Multi-modal embedding (text + metadata) kullanmak

#### Öneri 2: Hybrid Recommendation Sistemi
- Content-based filtering + Collaborative filtering
- MMR + Clustering (kategori bazlı çeşitlilik)
- User feedback'e göre dinamik ağırlıklandırma

#### Öneri 3: A/B Testing Altyapısı
Farklı lambda değerlerini gerçek kullanıcılarla test etmek:
- Kullanıcı memnuniyeti metrikleri
- Click-through rate (CTR)
- Engagement metrikleri

---

## 8. Teknik Detaylar

### 8.1. Cosine Similarity Hesaplama

**Formül**:
```
similarity(A, B) = (A · B) / (||A|| * ||B||)
```

**Kodda Kullanım**:
```python
# Vektörleri normalize et (unit vector)
vec_normalized = vec_array / np.linalg.norm(vec_array)

# Cosine similarity = dot product (normalize edilmiş vektörler için)
similarity = np.dot(vec1_normalized, vec2_normalized)
```

**Neden Cosine Similarity?**
- Vektörlerin büyüklüğünden bağımsız (sadece yön önemli)
- Embedding vektörlerinde çok etkili
- 0-1 aralığında değer verir (kolay yorumlama)

### 8.2. Diversity Score Hesaplama

**Formül**:
```
Diversity = 1 - Ortalama_Cosine_Similarity
```

**Kodda Kullanım**:
```python
# Pairwise similarity matrix
similarity_matrix = cosine_similarity(vectors_matrix)

# Diagonal'i 0 yap (kendisiyle benzerlik = 1, bunu istemiyoruz)
np.fill_diagonal(similarity_matrix, 0.0)

# Upper triangle'dan ortalama al (sadece bir kez saymak için)
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
avg_similarity = similarity_matrix[mask].mean()

# Diversity = 1 - Ortalama Benzerlik
diversity_score = 1.0 - avg_similarity
```

**Neden Upper Triangle?**
- Similarity matrix simetrik (A-B = B-A)
- Her çifti sadece bir kez saymak için
- Diagonal'i 0 yapmak: Kendisiyle benzerlik = 1, bu çeşitlilik hesaplamasında anlamsız

### 8.3. MMR Algoritması Detaylı Açıklama

**Pseudo-code**:
```
1. selected = [first_project]  # En yüksek skorlu
2. remaining = all_projects - selected
3. WHILE len(selected) < top_k:
    4. best_score = -inf
    5. best_project = None
    6. FOR each project in remaining:
        7. relevance = project.score / max_score
        8. diversity = 1 - max(similarity(project, s) for s in selected)
        9. mmr_score = lambda * relevance + (1-lambda) * diversity
        10. IF mmr_score > best_score:
            11. best_score = mmr_score
            12. best_project = project
    13. selected.append(best_project)
    14. remaining.remove(best_project)
15. RETURN selected + remaining (in original order)
```

**Zaman Karmaşıklığı**:
- O(k * n * d) where:
  - k = top_k (seçilecek proje sayısı)
  - n = toplam proje sayısı
  - d = embedding boyutu (768)
- Her iterasyonda tüm kalan projeler için similarity hesaplanıyor

**Optimizasyon Önerileri**:
- Vectorized operations kullanmak (NumPy)
- Batch processing yapmak
- Early stopping (yeterli diversity'ye ulaşınca dur)

### 8.4. Vektör İşleme Pipeline'ı

```
1. Vektör Yükleme
   ├─ Pickle dosyasından oku
   ├─ DataFrame formatında mı? → Merge et
   └─ Dict formatında mı? → Apply ile ekle

2. Vektör Formatını Standartlaştırma
   ├─ 2D array (n, 768)? → mean(axis=0) → (768,)
   ├─ 1D array (768,)? → Direkt kullan
   ├─ 3D+ array? → flatten()
   └─ Scalar/NaN? → Skip

3. Boyut Kontrolü
   ├─ > 768? → [:768] (truncate)
   ├─ < 768? → Padding (zeros)
   └─ = 768? → OK

4. Normalizasyon
   ├─ Norm hesapla
   ├─ Norm > 1e-8? → Normalize et (unit vector)
   └─ Norm <= 1e-8? → Skip (sıfır vektör)

5. Similarity Hesaplama
   ├─ Normalize edilmiş vektörler → Cosine similarity
   └─ Diversity = 1 - Ortalama Similarity
```

---

## 9. Sonuç ve Özet

### 9.1. Yapılanlar

1. ✅ **MMR Algoritması Entegre Edildi**: Diversity-aware ranking için MMR algoritması eklendi
2. ✅ **Vektör İşleme Düzeltildi**: Tutarsız vektör formatları standardize edildi
3. ✅ **Kod Hataları Giderildi**: Indentation, exception handling, normalizasyon sorunları çözüldü
4. ✅ **Test Altyapısı Oluşturuldu**: 3 farklı test script'i yazıldı

### 9.2. Kalan İşler

1. ⚠️ **MMR Sonrası Diversity Hesaplama**: Vektörlerin neden işlenemediğini bulmak ve düzeltmek
2. ⚠️ **Baseline Diversity İyileştirme**: MMR olmadan bile diversity'yi artırmak
3. 📝 **Lambda Optimizasyonu**: En iyi lambda değerini bulmak (şu anda test ediliyor ama sonuç vermiyor)

### 9.3. Öğrenilenler

1. **Vektör Formatları Tutarlı Olmalı**: Farklı formatlardaki vektörler sorun çıkarıyor
2. **Exception Handling Dikkatli Olmalı**: Çok geniş exception handling, gerçek sorunları gizleyebilir
3. **Test Altyapısı Önemli**: Detaylı test script'leri olmadan sorunları bulmak çok zor
4. **Diversity Hesaplama Hassas**: Vektör normalizasyonu, similarity hesaplama, her adım kritik

### 9.4. Tavsiyeler

1. **Kısa Vadede**: MMR sonrası diversity hesaplama sorununu çözmek için detaylı debugging yapılmalı
2. **Orta Vadede**: Vektör preprocessing pipeline'ını güçlendirmek, tüm vektörleri baştan standardize etmek
3. **Uzun Vadede**: Daha iyi embedding modeli, hybrid recommendation sistemi, A/B testing altyapısı

---

## 10. Ek Bilgiler

### 10.1. Kullanılan Teknolojiler

- **Python 3.x**
- **NumPy**: Vektör işlemleri, matematiksel hesaplamalar
- **Pandas**: DataFrame işlemleri
- **scikit-learn**: Cosine similarity hesaplama
- **XLM-RoBERTa**: Embedding modeli (768 boyutlu)

### 10.2. Dosya Yapısı

```
Tubitak_Proje_Yeni/
├── src/
│   └── scoring/
│       └── final_ranking.py          # MMR implementasyonu
├── config/
│   └── config.yaml                   # MMR ayarları
├── test_mmr.py                       # Temel MMR testi
├── test_mmr_lambda.py                # Lambda optimizasyonu
├── debug_diversity.py                # Diversity analizi
└── IP1_Ciktisi_Projeler_Vektorlu.pkl # Vektör dosyası
```

### 10.3. Referanslar

- **MMR Algoritması**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries.
- **Diversity Metrics**: Vargas, S., & Castells, P. (2011). Rank and relevance in novelty and diversity metrics for recommendation.
- **Cosine Similarity**: Scikit-learn documentation

---

**Rapor Tarihi**: 31 Aralık 2025  
**Hazırlayan**: AI Assistant (Cursor)  
**Proje**: TÜBİTAK Öneri Sistemi - MMR Entegrasyonu
