"""
Nihai adil sıralama modülü (MOO entegrasyonu)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict


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
        
        self.logger.info(f"Ağırlıklar: M={self.w_m*100}%, S={self.w_s*100}%, R={self.w_r*100}%")
    
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
    
    def calculate_ranking(
        self,
        df_projeler: pd.DataFrame,
        r_score: float
    ) -> pd.DataFrame:
        """
        Nihai adil sıralamayı hesaplar.
        
        Args:
            df_projeler: S-Skoru eklenmiş proje DataFrame'i
            r_score: Güvenilirlik skoru (tek değer)
        
        Returns:
            Final skorlar eklenmiş ve sıralanmış DataFrame
        """
        self.logger.info("Nihai adil sıralama hesaplanıyor...")
        
        df_result = df_projeler.copy()
        
        # Final skorları hesapla
        final_scores = []
        for idx, row in df_result.iterrows():
            m = row.get('M_Skoru', 0.0)
            s = row.get('S_Skoru', 0.0)
            final_score = self.calculate_final_score(m, s, r_score)
            final_scores.append(final_score)
        
        df_result['Final_Skor'] = final_scores
        
        # Adillik bonusu uygula
        if self.fairness_enabled:
            adil_skorlar = df_result.apply(self.apply_fairness_boost, axis=1)
            df_result['Adil_Skor'] = adil_skorlar
        else:
            df_result['Adil_Skor'] = df_result['Final_Skor']
        
        # Sırala
        df_result = df_result.sort_values(by='Adil_Skor', ascending=False).reset_index(drop=True)
        
        self.logger.info("Nihai sıralama tamamlandı")
        return df_result

