"""
Module that computes the matching score (M-Score)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Union


class MatchingScoreCalculator:
    """Matching score calculator class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_matching_score(
        self,
        project_vectors: np.ndarray,
        candidate_vectors: np.ndarray
    ) -> float:
        """
        Computes how well the candidate's skills cover the project's requirements.
        
        Args:
            project_vectors: Project skill vectors (n_project_skills x embedding_dim)
            candidate_vectors: Candidate skill vectors (n_candidate_skills x embedding_dim)
        
        Returns:
            Matching score (0-100)
        """
        # Empty vector check
        if project_vectors is None or len(project_vectors) == 0:
            return 0.0
        
        # Shape check
        if len(project_vectors.shape) == 1:
            project_vectors = project_vectors.reshape(1, -1)
        
        # Compute the similarity matrix
        similarity_matrix = cosine_similarity(project_vectors, candidate_vectors)
        
        # For each project requirement, find the candidate skill that covers it best
        best_matches = similarity_matrix.max(axis=1)
        
        # Take the mean and map it to the 0-100 range
        score = best_matches.mean() * 100
        
        return float(score)
    
    def calculate_all_scores(
        self,
        df_projects: pd.DataFrame,
        candidate_vectors: np.ndarray,
        vector_column: str = "skill_vectors"
    ) -> pd.DataFrame:
        """
        Computes matching scores for all projects.
        
        Args:
            df_projects: Project DataFrame
            candidate_vectors: Candidate skill vectors
            vector_column: Name of the vector column
        
        Returns:
            DataFrame with M_Score added
        """
        self.logger.info(f"Scoring {len(df_projects)} projects...")
        
        scores = []
        for idx, row in df_projects.iterrows():
            project_vec = row[vector_column]
            score = self.calculate_matching_score(project_vec, candidate_vectors)
            scores.append(score)
            
            if (idx + 1) % 100 == 0:
                self.logger.debug(f"Progress: {idx+1}/{len(df_projects)} projects scored")
        
        df_result = df_projects.copy()
        df_result["M_Score"] = scores
        
        # Sort by score
        df_result = df_result.sort_values(by="M_Score", ascending=False).reset_index(drop=True)
        
        self.logger.info("All projects scored")
        return df_result

