"""
Eşleştirme skoru (M-Skoru) hesaplayan modül
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Union


class MatchingScoreCalculator:
    """Eşleştirme skoru hesaplayıcı sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_matching_score(
        self,
        proje_vektorleri: np.ndarray,
        aday_vektorleri: np.ndarray
    ) -> float:
        """
        Proje gereksinimleri ile aday becerileri arasındaki uyumu hesaplar.
        
        Args:
            proje_vektorleri: Proje beceri vektörleri (n_proje_skills x embedding_dim)
            aday_vektorleri: Aday beceri vektörleri (n_aday_skills x embedding_dim)
        
        Returns:
            Eşleştirme skoru (0-100)
        """
        # Boş vektör kontrolü
        if proje_vektorleri is None or len(proje_vektorleri) == 0:
            return 0.0
        
        # Boyut kontrolü
        if len(proje_vektorleri.shape) == 1:
            proje_vektorleri = proje_vektorleri.reshape(1, -1)
        
        # Benzerlik matrisi hesapla
        benzerlik_matrisi = cosine_similarity(proje_vektorleri, aday_vektorleri)
        
        # Her proje gereksinimi için adayın en iyi karşılayan becerisini bul
        en_iyi_eslesmeler = benzerlik_matrisi.max(axis=1)
        
        # Ortalama al ve 0-100 aralığına çevir
        skor = en_iyi_eslesmeler.mean() * 100
        
        return float(skor)
    
    def calculate_all_scores(
        self,
        df_projeler: pd.DataFrame,
        aday_vektorleri: np.ndarray,
        vector_column: str = "beceri_vektorleri"
    ) -> pd.DataFrame:
        """
        Tüm projeler için eşleştirme skorlarını hesaplar (batch processing ile optimize edildi).
        
        Args:
            df_projeler: Proje DataFrame'i
            aday_vektorleri: Aday beceri vektörleri
            vector_column: Vektör sütunu adı
        
        Returns:
            M_Skoru eklenmiş DataFrame
        """
        self.logger.info(f"{len(df_projeler)} adet proje puanlanıyor...")
        
        # Batch size'ı config'den al
        batch_size = self.config.get('performance', {}).get('matching_score_batch_size', 100)
        
        scores = []
        total = len(df_projeler)
        
        # Batch processing ile optimize et
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df_projeler.iloc[batch_start:batch_end]
            
            # Batch içindeki tüm proje vektörlerini topla
            batch_scores = []
            for idx, row in batch_df.iterrows():
                proje_vec = row[vector_column]
                score = self.calculate_matching_score(proje_vec, aday_vektorleri)
                batch_scores.append(score)
            
            scores.extend(batch_scores)
            
            if batch_end % (batch_size * 2) == 0 or batch_end == total:
                self.logger.info(f"   İlerleme: {batch_end}/{total} proje işlendi ({batch_end*100//total}%)")
        
        df_result = df_projeler.copy()
        df_result["M_Skoru"] = scores
        
        # Skora göre sırala
        df_result = df_result.sort_values(by="M_Skoru", ascending=False).reset_index(drop=True)
        
        self.logger.info("Tüm projeler puanlandı")
        return df_result

