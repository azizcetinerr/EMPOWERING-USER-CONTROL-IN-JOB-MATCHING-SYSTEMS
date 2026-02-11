"""
Nihai adil sıralama modülü (MOO entegrasyonu)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity


class FinalRankingCalculator:
    """Nihai adil sıralama hesaplayıcı sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Ağırlıklar
        weights = config.get('scoring_weights', {})
        self.w_m = weights.get('matching_score', 0.5)
        self.w_s = weights.get('sincerity_score', 0.3)
        self.w_r = weights.get('reliability_score', 0.2)
        
        # Adillik ayarları
        fairness_config = config.get('fairness', {})
        self.fairness_enabled = fairness_config.get('enabled', True)
        self.bonus_keywords = fairness_config.get('bonus_keywords', ['culture', 'art'])
        self.bonus_score = fairness_config.get('bonus_score', 5.0)
        
        # MMR (Diversity-Aware Ranking) ayarları
        mmr_config = config.get('diversity_aware_ranking', {})
        self.mmr_enabled = mmr_config.get('enabled', False)
        self.mmr_top_k = mmr_config.get('top_k', 10)
        self.mmr_lambda = mmr_config.get('lambda', 0.5)
        
        # Performance Metrics ayarları
        metrics_config = config.get('performance_metrics', {})
        self.metrics_top_k = metrics_config.get('top_k', 5)
        
        self.logger.info(f"Ağırlıklar: M={self.w_m*100}%, S={self.w_s*100}%, R={self.w_r*100}%")
        if self.mmr_enabled:
            self.logger.info(f"MMR (Diversity-Aware Ranking): Aktif (Top-K={self.mmr_top_k}, Lambda={self.mmr_lambda:.2f})")
    
    def calculate_final_score(
        self,
        m_score: float,
        s_score: float,
        r_score: float
    ) -> float:
        """
        Nihai skoru hesaplar.
        
        Args:
            m_score: Eşleştirme skoru
            s_score: Samimiyet skoru
            r_score: Güvenilirlik skoru
        
        Returns:
            Nihai skor
        """
        final_score = (m_score * self.w_m) + (s_score * self.w_s) + (r_score * self.w_r)
        return final_score
    
    def apply_fairness_boost(self, row: pd.Series) -> float:
        """
        Adillik bonusu uygular.
        
        Args:
            row: Proje satırı
        
        Returns:
            Bonus eklenmiş skor
        """
        if not self.fairness_enabled:
            return row['Final_Skor']
        
        bonus = 0.0
        text = str(row.get('tam_aciklama', '')).lower()
        
        # Anahtar kelime kontrolü
        for keyword in self.bonus_keywords:
            if keyword in text:
                bonus = self.bonus_score
                break
        
        return row['Final_Skor'] + bonus
    
    def normalize_with_zscore(self, values: pd.Series, name: str) -> pd.Series:
        """
        Z-Score normalizasyonu uygular.
        
        Args:
            values: Normalize edilecek değerler
            name: Skor adı (log için)
        
        Returns:
            Normalize edilmiş değerler (0-100 aralığında)
        """
        # Eksik değerleri temizle
        valid_values = values.dropna()
        
        if len(valid_values) == 0:
            self.logger.warning(f"{name} için geçerli değer yok!")
            return pd.Series([0.0] * len(values), index=values.index)
        
        # Standart sapma kontrolü
        std = valid_values.std()
        mean = valid_values.mean()
        
        if std == 0 or np.isnan(std) or np.isinf(std):
            self.logger.error(
                f"⚠️  {name} için standart sapma 0! Tüm değerler aynı ({mean:.2f}). "
                f"Bu skor hesaplamadan çıkarılacak veya sabit değer olarak kullanılacak."
            )
            # Sabit değerleri 50'ye normalize et (nötr)
            return pd.Series([50.0] * len(values), index=values.index)
        
        # Z-Score hesapla: z = (x - mean) / std
        z_scores = (values - mean) / std
        
        # Z-Score'u 0-100 aralığına çevir
        # Yöntem: z-score'u sigmoid benzeri bir fonksiyonla ölçekle
        # normalized = 50 + (z_score * 15)  # ±3 std = 5-95 aralığı
        # Daha geniş aralık için:
        normalized = 50 + (z_scores * 20)  # ±2.5 std = 0-100 aralığı
        
        # Sınırları kontrol et
        normalized = normalized.clip(0, 100)
        
        self.logger.info(
            f"✅ {name} Z-Score normalizasyonu: "
            f"Ortalama={mean:.2f}, Std={std:.2f}, "
            f"Normalize aralık: {normalized.min():.2f}-{normalized.max():.2f}"
        )
        
        return normalized
    
    def calculate_ranking(
        self,
        df_projeler: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Nihai adil sıralamayı hesaplar.
        
        Args:
            df_projeler: S-Skoru ve R-Skoru eklenmiş proje DataFrame'i
        
        Returns:
            Final skorlar eklenmiş ve sıralanmış DataFrame
        """
        self.logger.info("Nihai adil sıralama hesaplanıyor...")
        self.logger.info(f"Toplam {len(df_projeler)} adet proje işlenecek")
        
        df_result = df_projeler.copy()
        
        # Gerekli sütunları kontrol et
        required_cols = ['M_Skoru', 'S_Skoru', 'R_Skoru']
        missing_cols = [col for col in required_cols if col not in df_result.columns]
        if missing_cols:
            self.logger.error(f"Eksik sütunlar: {missing_cols}")
            raise ValueError(f"Eksik sütunlar: {missing_cols}")
        
        # Ham skor istatistikleri
        self.logger.info("\n📊 HAM SKOR İSTATİSTİKLERİ:")
        self.logger.info(f"  M_Skoru: Min={df_result['M_Skoru'].min():.2f}, Max={df_result['M_Skoru'].max():.2f}, Ort={df_result['M_Skoru'].mean():.2f}, Std={df_result['M_Skoru'].std():.2f}")
        self.logger.info(f"  S_Skoru: Min={df_result['S_Skoru'].min():.2f}, Max={df_result['S_Skoru'].max():.2f}, Ort={df_result['S_Skoru'].mean():.2f}, Std={df_result['S_Skoru'].std():.2f}")
        self.logger.info(f"  R_Skoru: Min={df_result['R_Skoru'].min():.2f}, Max={df_result['R_Skoru'].max():.2f}, Ort={df_result['R_Skoru'].mean():.2f}, Std={df_result['R_Skoru'].std():.2f}")
        
        # Z-Score normalizasyonu uygula
        self.logger.info("\n🔄 Z-SCORE NORMALİZASYONU UYGULANIYOR...")
        
        # Ham skorları sakla
        df_result['M_Skoru_Ham'] = df_result['M_Skoru'].copy()
        df_result['S_Skoru_Ham'] = df_result['S_Skoru'].copy()
        df_result['R_Skoru_Ham'] = df_result['R_Skoru'].copy()
        
        # Normalize et
        df_result['M_Skoru_Normalize'] = self.normalize_with_zscore(df_result['M_Skoru'], "M_Skoru")
        df_result['S_Skoru_Normalize'] = self.normalize_with_zscore(df_result['S_Skoru'], "S_Skoru")
        df_result['R_Skoru_Normalize'] = self.normalize_with_zscore(df_result['R_Skoru'], "R_Skoru")
        
        # Normalize edilmiş skorları kullan
        df_result['M_Skoru'] = df_result['M_Skoru_Normalize']
        df_result['S_Skoru'] = df_result['S_Skoru_Normalize']
        df_result['R_Skoru'] = df_result['R_Skoru_Normalize']
        
        # Feedback'leri yükle
        feedback_adjustments = self._load_feedback_adjustments(df_result)
        
        # Final skorları hesapla
        final_scores = []
        missing_s_count = 0
        
        for idx, row in df_result.iterrows():
            m = row.get('M_Skoru', 0.0)
            s = row.get('S_Skoru', None)
            r = row.get('R_Skoru', None)
            
            # Eksik değerleri kontrol et
            if pd.isna(s) or s is None:
                missing_s_count += 1
                s = 50.0  # Nötr değer
            
            if pd.isna(r) or r is None:
                self.logger.warning(f"Proje {idx} için R skoru eksik, 50.0 kullanılıyor")
                r = 50.0
            
            final_score = self.calculate_final_score(m, s, r)
            final_scores.append(final_score)
        
        if missing_s_count > 0:
            self.logger.warning(
                f"{missing_s_count} adet proje için S skoru yoktu, 50.0 kullanıldı."
            )
        
        df_result['Final_Skor'] = final_scores
        
        # Adillik bonusu uygula
        if self.fairness_enabled:
            adil_skorlar = df_result.apply(self.apply_fairness_boost, axis=1)
            df_result['Adil_Skor'] = adil_skorlar
        else:
            df_result['Adil_Skor'] = df_result['Final_Skor']
        
        # Feedback ayarlamalarını uygula
        if feedback_adjustments:
            df_result['Feedback_Adjustment'] = df_result.apply(
                lambda row: feedback_adjustments.get(str(row.get('link', '')), 0.0), 
                axis=1
            )
            df_result['Adil_Skor'] = df_result['Adil_Skor'] + df_result['Feedback_Adjustment']
            feedback_count = sum(1 for v in feedback_adjustments.values() if v != 0)
            if feedback_count > 0:
                self.logger.info(f"📝 {feedback_count} proje için kullanıcı geri bildirimi uygulandı")
        
        # Sırala
        df_result = df_result.sort_values(by='Adil_Skor', ascending=False).reset_index(drop=True)
        
        # MMR (Maximal Marginal Relevance) ile diversity-aware top-K seçimi
        # NOT: Sadece top-K'yı yeniden seçer, tüm sıralamayı değiştirmez
        if self.mmr_enabled and self.mmr_top_k > 0:
            self.logger.info(f"🔄 MMR (Maximal Marginal Relevance) ile diversity-aware ranking uygulanıyor (Top-{self.mmr_top_k})...")
            df_result = self._apply_mmr_ranking(
                df_result, 
                top_k=self.mmr_top_k,
                lambda_param=self.mmr_lambda
            )
        
        # Tüm skorların mevcut olduğunu kontrol et
        required_columns = ['M_Skoru', 'S_Skoru', 'R_Skoru', 'Final_Skor', 'Adil_Skor']
        missing_cols = [col for col in required_columns if col not in df_result.columns]
        if missing_cols:
            self.logger.error(f"Eksik sütunlar: {missing_cols}")
        
        # Normalize edilmiş skor istatistikleri
        self.logger.info(f"\n📊 NORMALİZE EDİLMİŞ SKOR İSTATİSTİKLERİ:")
        self.logger.info(f"  M_Skoru: Min={df_result['M_Skoru'].min():.2f}, Max={df_result['M_Skoru'].max():.2f}, Ortalama={df_result['M_Skoru'].mean():.2f}, Std={df_result['M_Skoru'].std():.2f}")
        self.logger.info(f"  S_Skoru: Min={df_result['S_Skoru'].min():.2f}, Max={df_result['S_Skoru'].max():.2f}, Ortalama={df_result['S_Skoru'].mean():.2f}, Std={df_result['S_Skoru'].std():.2f}")
        self.logger.info(f"  R_Skoru: Min={df_result['R_Skoru'].min():.2f}, Max={df_result['R_Skoru'].max():.2f}, Ortalama={df_result['R_Skoru'].mean():.2f}, Std={df_result['R_Skoru'].std():.2f}")
        self.logger.info(f"  Adil_Skor: Min={df_result['Adil_Skor'].min():.2f}, Max={df_result['Adil_Skor'].max():.2f}, Ortalama={df_result['Adil_Skor'].mean():.2f}, Std={df_result['Adil_Skor'].std():.2f}")
        self.logger.info(f"  Toplam proje sayısı: {len(df_result)}")
        
        # Skor katkılarını analiz et
        self.logger.info(f"\n📈 SKOR KATKILARI (Ağırlıklı Ortalama):")
        m_contribution = df_result['M_Skoru'].mean() * self.w_m
        s_contribution = df_result['S_Skoru'].mean() * self.w_s
        r_contribution = df_result['R_Skoru'].mean() * self.w_r
        self.logger.info(f"  M katkısı: {m_contribution:.2f} ({self.w_m*100}% ağırlık)")
        self.logger.info(f"  S katkısı: {s_contribution:.2f} ({self.w_s*100}% ağırlık)")
        self.logger.info(f"  R katkısı: {r_contribution:.2f} ({self.w_r*100}% ağırlık)")
        self.logger.info(f"  Toplam: {m_contribution + s_contribution + r_contribution:.2f}")
        
        self.logger.info("✅ Nihai sıralama tamamlandı (Z-Score normalizasyonu ile)")
        
        # Performans metriklerini hesapla (Config'den top_k al)
        # NOT: Confidence için ham skorları kullanmak için m_score_column="M_Skoru" kullanıyoruz
        # (calculate_performance_metrics içinde M_Skoru_Ham kontrol ediliyor)
        try:
            relevance, diversity, confidence, rc, uc = self.calculate_performance_metrics(
                df_result, 
                top_k=self.metrics_top_k,  # Config'den al
                vector_column="beceri_vektorleri",
                m_score_column="M_Skoru"  # İçeride M_Skoru_Ham kontrol edilecek
            )
            self.logger.info(
                f"📊 SİSTEM PERFORMANSI: Relevance={relevance:.2f} | Diversity={diversity:.2f} | Confidence={confidence:.2f} | RC={rc:.2f} | UC={uc:.2f}"
            )
            
            # Metrikleri raporla ve grafikle
            self._save_metrics_report(relevance, diversity, confidence, rc, uc, df_result)
            self._create_metrics_visualization(relevance, diversity, confidence, rc, uc)
            
        except Exception as e:
            self.logger.warning(f"⚠️  Performans metrikleri hesaplanamadı: {e}")
            import traceback
            self.logger.error(f"Detaylı hata: {traceback.format_exc()}")
        
        return df_result
    
    def calculate_performance_metrics(
        self,
        df_ranked: pd.DataFrame,
        top_k: int = 5,
        vector_column: str = "beceri_vektorleri",
        m_score_column: str = "M_Skoru"
    ) -> Tuple[float, float, float, float, float]:
        """
        Öneri sistemi performans metriklerini hesaplar.
        
        Args:
            df_ranked: Sıralanmış proje DataFrame'i (Adil_Skor'a göre)
            top_k: Hesaplanacak en iyi proje sayısı (default: 5)
            vector_column: Proje vektörlerinin bulunduğu sütun adı
            m_score_column: Matching Score sütunu adı
        
        Returns:
            (relevance_accuracy, diversity_score, confidence, rc_coverage, uc_coverage) tuple
            - relevance_accuracy: Ortalama M_Skoru (0-100)
            - diversity_score: Çeşitlilik skoru (0-1)
            - confidence: M skorlarının standart sapması
            - rc_coverage: Recommendation Coverage - Önerilen projelerin toplam havuzdaki oranı (0-1)
            - uc_coverage: User Coverage - Önerilen benzersiz proje sayısının toplam proje sayısına oranı (0-1)
        """
        if len(df_ranked) == 0:
            self.logger.warning("⚠️  Performans metrikleri hesaplanamadı: Boş DataFrame")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Top K projeyi seç
        top_k_actual = min(top_k, len(df_ranked))
        top_projects = df_ranked.head(top_k_actual).copy()
        
        if top_k_actual == 0:
            self.logger.warning("⚠️  Performans metrikleri hesaplanamadı: Hiç proje yok")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 1. Relevance Accuracy: M_Skoru ortalaması
        if m_score_column not in top_projects.columns:
            self.logger.warning(f"⚠️  Relevance hesaplanamadı: {m_score_column} sütunu bulunamadı")
            relevance_accuracy = 0.0
        else:
            m_scores = top_projects[m_score_column].astype(float)
            relevance_accuracy = float(m_scores.mean())
        
        # 2. Diversity Score: Hybrid approach - Vektör benzerliği + Beceri seti benzerliği
        diversity_score = 0.0
        if top_k_actual == 1:
            # Tek proje varsa diversity 0
            diversity_score = 0.0
        elif vector_column in top_projects.columns:
            # Vektörleri çıkar
            vectors_list = []
            valid_indices = []
            
            for idx, row in top_projects.iterrows():
                vector = row[vector_column]
                # Vektör kontrolü: None, NaN veya boş değilse ekle
                try:
                    if vector is None:
                        continue
                    
                    # NaN float kontrolü (pandas merge sonrası NaN float olabilir)
                    if isinstance(vector, float) and (np.isnan(vector) or pd.isna(vector)):
                        continue
                    
                    # Vektörü numpy array'e çevir
                    vec_array = None
                    if isinstance(vector, np.ndarray):
                        if vector.size == 0:
                            continue
                        vec_array = vector.copy()
                    elif isinstance(vector, (list, tuple)):
                        if len(vector) > 0:
                            vec_array = np.array(vector)
                        else:
                            continue
                    else:
                        # NumPy array veya diğer tip
                        try:
                            vec_array = np.array(vector)
                            if vec_array.size == 0:
                                continue
                            # NumPy array ise pd.isna() kullanma (boolean array döner)
                            # Bunun yerine direkt np.array() ile kontrol et
                        except:
                            continue
                    
                    if vec_array is None:
                        continue
                    
                    # Vektörü normalize et (2D vektörler için ortalama al)
                    if vec_array.ndim == 0:
                        # Scalar (NaN olabilir) - atla
                        continue
                    elif vec_array.ndim == 1:
                        # 1D vektör - direkt kullan
                        normalized_vec = vec_array
                    elif vec_array.ndim == 2:
                        # 2D vektör (n_skills, 768) - ortalamasını al
                        if vec_array.shape[1] == 768:
                            normalized_vec = vec_array.mean(axis=0)  # (768,)
                        else:
                            # Farklı boyuttaysa flatten et
                            normalized_vec = vec_array.flatten()
                    else:
                        # 3D veya daha fazla - flatten et
                        normalized_vec = vec_array.flatten()
                    
                    # Norm kontrolü (henüz normalize etme, sadece kontrol et)
                    norm = np.linalg.norm(normalized_vec)
                    if norm > 1e-8 and not np.isnan(norm) and not np.isinf(norm):
                        vectors_list.append(normalized_vec)  # Henüz normalize edilmemiş
                        valid_indices.append(idx)
                except (ValueError, TypeError, AttributeError) as e:
                    # Array comparison hatası veya diğer hatalar için atla
                    continue
            
            if len(vectors_list) >= 2:
                try:
                    # Vektörleri normalize et ve 2D matrise çevir
                    # Her vektör 1D numpy array olmalı (768 boyutunda embedding)
                    normalized_vectors = []
                    for vec in vectors_list:
                        try:
                            # vec zaten normalized_vec (yukarıda işlendi), direkt kullan
                            vec_array = np.array(vec)
                            
                            # Eğer hala 2D ise (beklenmeyen), ortalamasını al
                            if vec_array.ndim == 2:
                                vec_array = vec_array.mean(axis=0)
                            elif vec_array.ndim > 2:
                                vec_array = vec_array.flatten()
                            elif vec_array.ndim == 0:
                                continue
                            
                            # 1D olmalı
                            if vec_array.ndim != 1:
                                continue
                            
                            # Boyut kontrolü ve düzeltme
                            if len(vec_array) > 768:
                                vec_array = vec_array[:768]
                            elif len(vec_array) < 768:
                                # Padding yap (sıfırlarla)
                                padding = np.zeros(768 - len(vec_array), dtype=vec_array.dtype)
                                vec_array = np.concatenate([vec_array, padding])
                            
                            # Normalize et (unit vector yap) - İLK NORMALİZASYON
                            norm = np.linalg.norm(vec_array)
                            if norm > 1e-8 and not np.isnan(norm) and not np.isinf(norm):
                                vec_normalized = vec_array / norm
                                normalized_vectors.append(vec_normalized)
                        except Exception as vec_e:
                            self.logger.debug(f"Diversity: Vektör işleme hatası: {vec_e}")
                            continue
                    
                    if len(normalized_vectors) < 2:
                        self.logger.warning(f"⚠️  Diversity hesaplanamadı: Yeterli normalize edilmiş vektör yok ({len(normalized_vectors)}/{len(vectors_list)})")
                        diversity_score = 0.0
                    else:
                        # Tüm vektörleri 2D array'e çevir (her satır bir vektör)
                        # Shape: (n_vectors, vector_dim)
                        vectors_matrix = np.vstack(normalized_vectors)
                        
                        # Cosine similarity hesapla (pairwise)
                        # cosine_similarity(X) X'in kendi içindeki benzerliği hesaplar
                        similarity_matrix = cosine_similarity(vectors_matrix)
                        
                        # Diagonal'i 0 yap (kendisiyle benzerlik = 1, bunu istemiyoruz)
                        np.fill_diagonal(similarity_matrix, 0.0)
                        
                        # Ortalama benzerlik (upper triangle'dan - sadece bir kez saymak için)
                        # Üst üçgen matrisin ortalaması
                        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                        if mask.sum() > 0:
                            avg_similarity = similarity_matrix[mask].mean()
                            min_similarity = similarity_matrix[mask].min()
                            max_similarity = similarity_matrix[mask].max()
                            self.logger.debug(
                                f"Diversity hesaplama: Ortalama benzerlik={avg_similarity:.4f}, "
                                f"Min={min_similarity:.4f}, Max={max_similarity:.4f}"
                            )
                        else:
                            avg_similarity = 0.0
                        
                        # Diversity = 1 - Ortalama Benzerlik (vektör bazlı)
                        vector_diversity = float(1.0 - avg_similarity)
                        
                        # Hybrid Diversity: Vektör benzerliği + Beceri seti benzerliği
                        # Eğer beceri sütunu varsa, Jaccard similarity de hesapla
                        skill_diversity = 1.0
                        if 'cikarilan_beceriler' in top_projects.columns:
                            skill_sets = []
                            for idx in valid_indices:
                                skills = top_projects.loc[idx, 'cikarilan_beceriler']
                                if isinstance(skills, list):
                                    skill_sets.append(set(s.lower().strip() for s in skills if s and isinstance(s, str)))
                                else:
                                    skill_sets.append(set())
                            
                            if len(skill_sets) >= 2:
                                jaccard_similarities = []
                                for i in range(len(skill_sets)):
                                    for j in range(i+1, len(skill_sets)):
                                        set1, set2 = skill_sets[i], skill_sets[j]
                                        if len(set1) == 0 and len(set2) == 0:
                                            sim = 1.0
                                        elif len(set1) == 0 or len(set2) == 0:
                                            sim = 0.0
                                        else:
                                            intersection = len(set1.intersection(set2))
                                            union = len(set1.union(set2))
                                            sim = intersection / union if union > 0 else 0.0
                                        jaccard_similarities.append(sim)
                                
                                if jaccard_similarities:
                                    avg_jaccard = np.mean(jaccard_similarities)
                                    skill_diversity = 1.0 - avg_jaccard
                        
                        # Hybrid diversity: %70 vektör, %30 beceri seti
                        # Vektörler çok benzerse, beceri seti farklılığı diversity'yi artırır
                        diversity_score = 0.7 * vector_diversity + 0.3 * skill_diversity
                        diversity_score = max(0.0, min(1.0, diversity_score))  # 0-1 aralığında sınırla
                        
                        if diversity_score < 0.1:
                            self.logger.warning(
                                f"⚠️  Düşük Diversity Score ({diversity_score:.4f}): "
                                f"Top {top_k_actual} proje birbirine çok benzer "
                                f"(vektör benzerlik={avg_similarity:.4f}, hybrid diversity={diversity_score:.4f})"
                            )
                except Exception as e:
                    self.logger.warning(f"⚠️  Diversity hesaplanamadı: {e}")
                    diversity_score = 0.0
            else:
                self.logger.warning(f"⚠️  Diversity hesaplanamadı: Yeterli vektör yok ({len(vectors_list)}/{top_k_actual})")
                diversity_score = 0.0
        else:
            self.logger.warning(f"⚠️  Diversity hesaplanamadı: {vector_column} sütunu bulunamadı")
            diversity_score = 0.0
        
        # 3. Confidence: M skorlarının standart sapması (top-K içinde)
        # NOT: Normalize edilmiş skorlar yerine ham skorları kullan (klonlama sorununu önlemek için)
        if m_score_column not in top_projects.columns:
            confidence = 0.0
        else:
            # Önce ham skorları dene (daha fazla varyasyon gösterir)
            ham_column = m_score_column.replace('M_Skoru', 'M_Skoru_Ham') if 'M_Skoru' in m_score_column else None
            if ham_column and ham_column in top_projects.columns:
                m_scores = top_projects[ham_column].astype(float).dropna()
                if len(m_scores) > 0:
                    std_dev = float(m_scores.std())
                    # Eğer std çok düşükse, tüm dataset'ten std al (normalizasyon sonrası varyasyonu gösterir)
                    if std_dev < 1.0 and len(df_ranked) > top_k_actual:
                        # Tüm dataset'teki ham skorların std'sini al
                        if ham_column in df_ranked.columns:
                            all_scores = df_ranked[ham_column].astype(float).dropna()
                            if len(all_scores) > 0:
                                all_std = float(all_scores.std())
                                # Top-K içindeki std ile tüm dataset'in std'sinin ortalaması
                                confidence = (std_dev + all_std) / 2.0
                                self.logger.debug(f"Confidence: top-K std={std_dev:.4f}, all std={all_std:.4f}, combined={confidence:.4f}")
                            else:
                                confidence = std_dev
                        else:
                            confidence = std_dev
                    else:
                        confidence = std_dev
                    self.logger.debug(f"Confidence ham skorlardan hesaplandı (std={confidence:.4f})")
                else:
                    confidence = 0.0
            else:
                # Ham skorlar yoksa normalize edilmiş skorları kullan
                m_scores = top_projects[m_score_column].astype(float)
                std_dev = float(m_scores.std())
                # Eğer std çok düşükse, tüm dataset'ten std al
                if std_dev < 1.0 and len(df_ranked) > top_k_actual:
                    if m_score_column in df_ranked.columns:
                        all_scores = df_ranked[m_score_column].astype(float).dropna()
                        if len(all_scores) > 0:
                            all_std = float(all_scores.std())
                            confidence = (std_dev + all_std) / 2.0
                        else:
                            confidence = std_dev
                    else:
                        confidence = std_dev
                else:
                    confidence = std_dev
                self.logger.debug(f"Confidence normalize edilmiş skorlardan hesaplandı (std={confidence:.4f})")
        
        # 4. RC (Recommendation Coverage): Önerilen projelerin toplam havuzdaki oranı
        # RC = |Önerilen benzersiz projeler| / |Toplam proje havuzu|
        total_projects = len(df_ranked)
        recommended_unique = len(top_projects)  # Top K proje (zaten benzersiz, sıralanmış)
        
        if total_projects > 0:
            rc_coverage = float(recommended_unique) / float(total_projects)
            rc_coverage = max(0.0, min(1.0, rc_coverage))  # 0-1 aralığında sınırla
        else:
            rc_coverage = 0.0
        
        # 5. UC (User Coverage): Önerilen projelerin minimum skor eşiğini geçen projeler arasındaki oranı
        # Sistemimizde: Adil_Skor > 40 olan projelerin içinde kaç tanesi önerilen top K'da?
        # UC = |Top K öneriler| / |Yeterli skora sahip tüm projeler (örn: Adil_Skor > 40)|
        if 'Adil_Skor' in df_ranked.columns:
            # Yeterli skora sahip projeleri say (örneğin Adil_Skor > 40)
            threshold = 40.0  # Minimum skor eşiği
            eligible_projects = df_ranked[df_ranked['Adil_Skor'] > threshold]
            eligible_count = len(eligible_projects)
            
            if eligible_count > 0:
                # Top K içinde kaç tanesi eligible?
                top_k_eligible = top_projects[top_projects['Adil_Skor'] > threshold]
                uc_coverage = float(len(top_k_eligible)) / float(eligible_count)
                uc_coverage = max(0.0, min(1.0, uc_coverage))  # 0-1 aralığında sınırla
            else:
                uc_coverage = 0.0
        else:
            # Adil_Skor yoksa, basitçe top K / total olarak hesapla (RC ile aynı)
            if total_projects > 0:
                uc_coverage = float(recommended_unique) / float(total_projects)
                uc_coverage = max(0.0, min(1.0, uc_coverage))
            else:
                uc_coverage = 0.0
        
        return relevance_accuracy, diversity_score, confidence, rc_coverage, uc_coverage
    
    def _apply_mmr_ranking(
        self,
        df_ranked: pd.DataFrame,
        top_k: int = 10,
        lambda_param: float = 0.5,
        vector_column: str = "beceri_vektorleri"
    ) -> pd.DataFrame:
        """
        MMR (Maximal Marginal Relevance) algoritması ile diversity-aware ranking uygular.
        
        MMR, relevance ve diversity arasında denge sağlar:
        MMR_score = lambda * relevance + (1 - lambda) * diversity
        
        Args:
            df_ranked: Sıralanmış proje DataFrame'i (Adil_Skor'a göre)
            top_k: MMR ile seçilecek proje sayısı
            lambda_param: Relevance (0.0) ile Diversity (1.0) arası denge (0.5 = eşit ağırlık)
            vector_column: Proje vektörlerinin bulunduğu sütun adı
        
        Returns:
            MMR ile yeniden sıralanmış DataFrame (top-K MMR ile, kalanlar orijinal sırada)
        """
        if len(df_ranked) <= top_k:
            # Top-K'dan az proje varsa MMR gerekmez
            return df_ranked
        
        self.logger.info(f"🎯 MMR algoritması: Top-{top_k} seçiliyor (lambda={lambda_param:.2f})")
        
        # Top-K'yı MMR ile seç (integer index kullan, reset_index yapılmış DataFrame için)
        selected_positions = []  # DataFrame'deki pozisyon (0-based index)
        remaining_positions = list(range(len(df_ranked)))
        
        # İlk proje: En yüksek Adil_Skor'lu (zaten sıralanmış, ilk sırada)
        first_pos = 0
        selected_positions.append(first_pos)
        remaining_positions.remove(first_pos)
        
        # Seçili projelerin vektörlerini sakla
        selected_vectors = []
        if vector_column in df_ranked.columns:
            first_vector = df_ranked.iloc[first_pos][vector_column]
            if first_vector is not None:
                try:
                    vec_array = np.array(first_vector)
                    # 2D vektör ise ortalamasını al
                    if vec_array.ndim == 2:
                        vec_array = vec_array.mean(axis=0)  # (768,)
                    elif vec_array.ndim > 2:
                        vec_array = vec_array.flatten()
                        if len(vec_array) > 768:
                            vec_array = vec_array[:768]
                    
                    if vec_array.ndim == 1 and len(vec_array) == 768:
                        norm = np.linalg.norm(vec_array)
                        if norm > 0 and not np.isnan(norm) and not np.isinf(norm):
                            selected_vectors.append(vec_array / norm)
                except:
                    pass
        
        # Kalan projeler için MMR skoru hesapla
        while len(selected_positions) < top_k and len(remaining_positions) > 0:
            best_pos = None
            best_mmr_score = -float('inf')
            
            for pos in remaining_positions:
                row = df_ranked.iloc[pos]
                
                # Relevance: Adil_Skor (normalize et: 0-1 aralığına)
                relevance = row.get('Adil_Skor', 0.0) / 100.0  # 0-100'den 0-1'e normalize et
                
                # Diversity: Seçili projelere maksimum benzerlik (düşük benzerlik = yüksek diversity)
                diversity = 1.0  # Varsayılan: maksimum diversity
                
                if selected_vectors and vector_column in row and row[vector_column] is not None:
                    try:
                        # Mevcut projenin vektörünü al
                        current_vec = np.array(row[vector_column])
                        # 2D vektör ise ortalamasını al
                        if current_vec.ndim == 2:
                            current_vec = current_vec.mean(axis=0)  # (768,)
                        elif current_vec.ndim > 2:
                            current_vec = current_vec.flatten()
                            if len(current_vec) > 768:
                                current_vec = current_vec[:768]
                        elif current_vec.ndim == 0:
                            diversity = 1.0  # Scalar - maksimum diversity
                            continue
                        
                        if current_vec.ndim == 1 and len(current_vec) == 768:
                            norm = np.linalg.norm(current_vec)
                            if norm > 0 and not np.isnan(norm) and not np.isinf(norm):
                                current_vec = current_vec / norm
                                
                                # Seçili projelere olan maksimum benzerliği hesapla
                                max_similarity = 0.0
                                for sel_vec in selected_vectors:
                                    similarity = float(np.dot(current_vec, sel_vec))
                                    max_similarity = max(max_similarity, similarity)
                                
                                # Diversity = 1 - Maksimum Benzerlik
                                diversity = 1.0 - max_similarity
                    except Exception as e:
                        self.logger.debug(f"MMR: Vektör hesaplama hatası (pos={pos}): {e}")
                        diversity = 1.0  # Hata durumunda maksimum diversity kabul et
                
                # MMR Score: lambda * relevance + (1 - lambda) * diversity
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_pos = pos
            
            if best_pos is not None:
                selected_positions.append(best_pos)
                remaining_positions.remove(best_pos)
                
                # Seçili vektörleri güncelle
                if vector_column in df_ranked.columns:
                    best_vector = df_ranked.iloc[best_pos][vector_column]
                    if best_vector is not None:
                        try:
                            vec_array = np.array(best_vector)
                            # 2D vektör ise ortalamasını al
                            if vec_array.ndim == 2:
                                vec_array = vec_array.mean(axis=0)  # (768,)
                            elif vec_array.ndim > 2:
                                vec_array = vec_array.flatten()
                                if len(vec_array) > 768:
                                    vec_array = vec_array[:768]
                            
                            if vec_array.ndim == 1 and len(vec_array) == 768:
                                norm = np.linalg.norm(vec_array)
                                if norm > 0 and not np.isnan(norm) and not np.isinf(norm):
                                    selected_vectors.append(vec_array / norm)
                        except:
                            pass
            else:
                break  # Hiçbir proje seçilemedi, dur
        
        # MMR ile seçilen projeleri al
        mmr_selected = df_ranked.iloc[selected_positions].copy()
        
        # Kalan projeleri orijinal sıralamada bırak
        remaining_ordered = df_ranked.iloc[remaining_positions].copy()
        
        # Birleştir: MMR seçilenler + kalanlar (orijinal sırada)
        df_result = pd.concat([mmr_selected, remaining_ordered], ignore_index=True)
        
        self.logger.info(f"✅ MMR ranking tamamlandı: Top-{len(selected_positions)} proje çeşitlilik faktörü ile seçildi")
        
        return df_result
    
    def _load_feedback_adjustments(self, df_projects: pd.DataFrame) -> Dict[str, float]:
        """
        Kullanıcı geri bildirimlerini yükler ve skor ayarlamalarını hesaplar.
        
        Args:
            df_projects: Proje DataFrame'i
        
        Returns:
            Proje link'lerine göre feedback ayarlamaları (pozitif feedback +5, negatif -5)
        """
        feedback_file = Path(__file__).resolve().parent.parent.parent / "data/external/user_feedback.json"
        
        if not feedback_file.exists():
            return {}
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
        except Exception as e:
            self.logger.warning(f"Feedback dosyası okunamadı: {e}")
            return {}
        
        # Proje link'lerini al
        project_links = set(df_projects['link'].astype(str).unique())
        
        # Her proje için feedback sayısını hesapla
        adjustments = {}
        feedback_bonus = 5.0  # Pozitif feedback için bonus
        feedback_penalty = -5.0  # Negatif feedback için ceza
        
        for feedback in feedbacks:
            project_link = feedback.get('project_link', '')  # Feedback'te kaydedilen link
            project_id = feedback.get('project_id', '')
            feedback_type = feedback.get('feedback_type', '')
            
            # Proje link'ini bul
            matching_link = None
            if project_link and project_link in project_links:
                matching_link = project_link
            elif project_id:
                # Project ID'den link bulmaya çalış
                for link in project_links:
                    if project_id in str(link) or str(link).endswith(str(project_id)):
                        matching_link = link
                        break
            
            if matching_link:
                if matching_link not in adjustments:
                    adjustments[matching_link] = 0.0
                
                if feedback_type == 'positive':
                    adjustments[matching_link] += feedback_bonus
                elif feedback_type == 'negative':
                    adjustments[matching_link] += feedback_penalty
        
        return adjustments
    
    def _save_metrics_report(
        self,
        relevance: float,
        diversity: float,
        confidence: float,
        rc_coverage: float,
        uc_coverage: float,
        df_ranked: pd.DataFrame
    ):
        """
        Performans metriklerini rapor olarak kaydeder (JSON ve Markdown formatında).
        
        Args:
            relevance: Relevance Accuracy metrik değeri
            diversity: Diversity Score metrik değeri
            confidence: Confidence metrik değeri
            rc_coverage: Recommendation Coverage metrik değeri
            uc_coverage: User Coverage metrik değeri
            df_ranked: Sıralanmış proje DataFrame'i
        """
        try:
            # Çıktı dizinini oluştur
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Metrik verilerini hazırla
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'relevance_accuracy': round(relevance, 4),
                    'diversity_score': round(diversity, 4),
                    'confidence': round(confidence, 4),
                    'recommendation_coverage': round(rc_coverage, 4),
                    'user_coverage': round(uc_coverage, 4)
                },
                'metadata': {
                    'total_projects': len(df_ranked),
                    'top_k': min(self.metrics_top_k, len(df_ranked)),
                    'description': {
                        'relevance_accuracy': 'Ortalama M_Skoru (Matching Score) - 0-100 arası',
                        'diversity_score': 'Item-to-Item Similarity (1 - Ortalama Cosine Similarity) - 0-1 arası',
                        'confidence': 'M_Skoru standart sapması - Skorların güvenilirliği',
                        'recommendation_coverage': 'Önerilen projelerin toplam havuzdaki oranı - 0-1 arası',
                        'user_coverage': 'Önerilen projelerin yeterli skora sahip projeler içindeki oranı - 0-1 arası'
                    }
                }
            }
            
            # JSON formatında kaydet
            json_file = output_dir / f"metrics_report_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📄 Metrik raporu JSON formatında kaydedildi: {json_file}")
            
            # Markdown formatında kaydet
            md_file = output_dir / f"metrics_report_{timestamp}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write("# Sistem Performans Metrikleri Raporu\n\n")
                f.write(f"**Rapor Tarihi**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write("## Metrik Özeti\n\n")
                f.write("| Metrik | Değer | Açıklama |\n")
                f.write("|--------|-------|----------|\n")
                f.write(f"| **Relevance Accuracy** | {relevance:.4f} | Ortalama M_Skoru (0-100) |\n")
                f.write(f"| **Diversity Score** | {diversity:.4f} | Çeşitlilik skoru (0-1) |\n")
                f.write(f"| **Confidence** | {confidence:.4f} | M skorlarının standart sapması |\n")
                f.write(f"| **RC (Recommendation Coverage)** | {rc_coverage:.4f} | Önerilen projelerin toplam havuzdaki oranı (0-1) |\n")
                f.write(f"| **UC (User Coverage)** | {uc_coverage:.4f} | Önerilen projelerin yeterli skora sahip projeler içindeki oranı (0-1) |\n")
                f.write("\n---\n\n")
                f.write("## Detaylı Açıklamalar\n\n")
                f.write("### Relevance Accuracy\n")
                f.write("Önerilen projelerin ortalama Matching Score (M_Skoru) değeridir. ")
                f.write("Bu metrik, önerilen projelerin kullanıcı profiliyle ne kadar uyumlu olduğunu gösterir.\n\n")
                f.write("### Diversity Score\n")
                f.write("Önerilen projeler arasındaki çeşitlilik ölçüsüdür. ")
                f.write("Cosine similarity tabanlı Item-to-Item Similarity (ILS) metriğidir. ")
                f.write("Yüksek değer, önerilen projelerin birbirinden farklı olduğunu gösterir.\n\n")
                f.write("### Confidence\n")
                f.write("M_Skoru değerlerinin standart sapmasıdır. ")
                f.write("Bu metrik, önerilerin güvenilirliğini gösterir. ")
                f.write("Düşük standart sapma, daha tutarlı öneriler anlamına gelir.\n\n")
                f.write("### RC (Recommendation Coverage)\n")
                f.write("Önerilen projelerin toplam proje havuzundaki oranıdır. ")
                f.write("Bu metrik, sistemin ne kadar geniş bir proje yelpazesinden öneri yaptığını gösterir.\n\n")
                f.write("### UC (User Coverage)\n")
                f.write("Önerilen projelerin, minimum skor eşiğini (Adil_Skor > 40) geçen projeler içindeki oranıdır. ")
                f.write("Bu metrik, sistemin yeterli kalitedeki projeleri ne kadar iyi kapsadığını gösterir.\n\n")
                f.write("---\n\n")
            f.write(f"**Toplam Proje Sayısı**: {len(df_ranked)}\n\n")
            f.write(f"**Analiz Edilen Top K**: {min(self.metrics_top_k, len(df_ranked))}\n\n")
            self.logger.info(f"📄 Metrik raporu Markdown formatında kaydedildi: {md_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Metrik raporu kaydedilemedi: {e}")
    
    def _create_metrics_visualization(
        self,
        relevance: float,
        diversity: float,
        confidence: float,
        rc_coverage: float,
        uc_coverage: float
    ):
        """
        Performans metriklerini görselleştirir ve kaydeder.
        
        Args:
            relevance: Relevance Accuracy metrik değeri
            diversity: Diversity Score metrik değeri
            confidence: Confidence metrik değeri
            rc_coverage: Recommendation Coverage metrik değeri
            uc_coverage: User Coverage metrik değeri
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI olmadan çalışması için
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Çıktı dizinini oluştur
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Grafik 1: Tüm metriklerin karşılaştırmalı bar chart'ı
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Sistem Performans Metrikleri', fontsize=16, fontweight='bold')
            
            # 1. Bar Chart - Tüm Metrikler (Normalize edilmiş)
            ax1 = axes[0, 0]
            metrics_names = ['Relevance\n(0-100)', 'Diversity\n(0-1)', 'RC\n(0-1)', 'UC\n(0-1)']
            metrics_values = [relevance / 100.0, diversity, rc_coverage, uc_coverage]  # Normalize et
            
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_ylim([0, 1])
            ax1.set_ylabel('Normalize Edilmiş Değer (0-1)', fontsize=10)
            ax1.set_title('Metrik Karşılaştırması (Normalize)', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Değerleri bar'ların üzerine yaz
            for i, (bar, val, orig_val) in enumerate(zip(bars, metrics_values, [relevance, diversity, rc_coverage, uc_coverage])):
                if i == 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{orig_val:.2f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{orig_val:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 2. Relevance ve Diversity - Scatter/Bar kombinasyonu
            ax2 = axes[0, 1]
            ax2.bar(['Relevance', 'Diversity'], [relevance / 100.0, diversity], 
                   color=['#2ecc71', '#3498db'], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_ylim([0, 1])
            ax2.set_ylabel('Değer (0-1)', fontsize=10)
            ax2.set_title('Relevance & Diversity', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.text(0, relevance / 100.0, f'{relevance:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.text(1, diversity, f'{diversity:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 3. Coverage Metrikleri (RC ve UC)
            ax3 = axes[1, 0]
            coverage_names = ['RC\n(Coverage)', 'UC\n(Coverage)']
            coverage_values = [rc_coverage, uc_coverage]
            bars3 = ax3.bar(coverage_names, coverage_values, color=['#e74c3c', '#f39c12'], 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
            ax3.set_ylim([0, 1])
            ax3.set_ylabel('Coverage Değeri (0-1)', fontsize=10)
            ax3.set_title('Recommendation & User Coverage', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, val in zip(bars3, coverage_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 4. Confidence (Standart Sapma)
            ax4 = axes[1, 1]
            ax4.bar(['Confidence\n(Std Dev)'], [confidence], 
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax4.set_ylabel('Standart Sapma', fontsize=10)
            ax4.set_title('Confidence (M_Skoru Std Sapması)', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            ax4.text(0, confidence, f'{confidence:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            # Grafik dosyasını kaydet
            graph_file = output_dir / f"metrics_visualization_{timestamp}.png"
            plt.savefig(graph_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"📊 Metrik görselleştirmesi kaydedildi: {graph_file}")
            
        except ImportError:
            self.logger.warning("⚠️  matplotlib bulunamadı, grafik oluşturulamadı. 'pip install matplotlib' komutuyla yükleyin.")
        except Exception as e:
            self.logger.warning(f"⚠️  Metrik görselleştirmesi oluşturulamadı: {e}")

