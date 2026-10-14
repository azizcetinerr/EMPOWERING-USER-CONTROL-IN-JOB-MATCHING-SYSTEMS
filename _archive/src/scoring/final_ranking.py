"""
Final fair ranking module (MOO integration)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict


class FinalRankingCalculator:
    """Final fair ranking calculator class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Weights
        weights = config.get('scoring_weights', {})
        self.w_m = weights.get('matching_score', 0.5)
        self.w_s = weights.get('sincerity_score', 0.3)
        self.w_r = weights.get('reliability_score', 0.2)
        
        # Fairness settings
        fairness_config = config.get('fairness', {})
        self.fairness_enabled = fairness_config.get('enabled', True)
        self.bonus_keywords = fairness_config.get('bonus_keywords', ['culture', 'art'])
        self.bonus_score = fairness_config.get('bonus_score', 5.0)
        
        self.logger.info(f"Weights: M={self.w_m*100}%, S={self.w_s*100}%, R={self.w_r*100}%")
    
    def calculate_final_score(
        self,
        m_score: float,
        s_score: float,
        r_score: float
    ) -> float:
        """
        Computes the final score.
        
        Args:
            m_score: Matching score
            s_score: Sincerity score
            r_score: Reliability score
        
        Returns:
            Final score
        """
        final_score = (m_score * self.w_m) + (s_score * self.w_s) + (r_score * self.w_r)
        return final_score
    
    def apply_fairness_boost(self, row: pd.Series) -> float:
        """
        Applies the fairness bonus.
        
        Args:
            row: Project row
        
        Returns:
            Score with the bonus added
        """
        if not self.fairness_enabled:
            return row['Final_Score']
        
        bonus = 0.0
        text = str(row.get('full_description', '')).lower()
        
        # Keyword check
        for keyword in self.bonus_keywords:
            if keyword in text:
                bonus = self.bonus_score
                break
        
        return row['Final_Score'] + bonus
    
    def calculate_ranking(
        self,
        df_projects: pd.DataFrame,
        r_score: float
    ) -> pd.DataFrame:
        """
        Computes the final fair ranking.
        
        Args:
            df_projects: Project DataFrame with S-Score added
            r_score: Reliability score (single value)
        
        Returns:
            DataFrame with final scores added and sorted
        """
        self.logger.info("Computing the final fair ranking...")
        
        df_result = df_projects.copy()
        
        # Compute the final scores
        final_scores = []
        for idx, row in df_result.iterrows():
            m = row.get('M_Score', 0.0)
            s = row.get('S_Score', 0.0)
            final_score = self.calculate_final_score(m, s, r_score)
            final_scores.append(final_score)
        
        df_result['Final_Score'] = final_scores
        
        # Apply the fairness bonus
        if self.fairness_enabled:
            fair_scores = df_result.apply(self.apply_fairness_boost, axis=1)
            df_result['Fair_Score'] = fair_scores
        else:
            df_result['Fair_Score'] = df_result['Final_Score']
        
        # Sort
        df_result = df_result.sort_values(by='Fair_Score', ascending=False).reset_index(drop=True)
        
        self.logger.info("Final ranking completed")
        return df_result

