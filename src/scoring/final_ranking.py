"""
Final fair ranking module (MOO integration)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path
import json
import re
from sklearn.metrics.pairwise import cosine_similarity


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
        
        # MMR (Diversity-Aware Ranking) settings
        mmr_config = config.get('diversity_aware_ranking', {})
        self.mmr_enabled = mmr_config.get('enabled', False)
        self.mmr_top_k = mmr_config.get('top_k', 10)
        self.mmr_lambda = mmr_config.get('lambda', 0.5)
        
        # Performance Metrics settings
        metrics_config = config.get('performance_metrics', {})
        self.metrics_top_k = metrics_config.get('top_k', 5)
        
        self.logger.info(f"Weights: M={self.w_m*100}%, S={self.w_s*100}%, R={self.w_r*100}%")
        if self.mmr_enabled:
            self.logger.info(f"MMR (Diversity-Aware Ranking): active (Top-K={self.mmr_top_k}, Lambda={self.mmr_lambda:.2f})")
    
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
        
        # Keyword check on whole words (a substring test matches "art" inside "start", "part", ...)
        for keyword in self.bonus_keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text):
                bonus = self.bonus_score
                break
        
        return row['Final_Score'] + bonus
    
    def normalize_with_zscore(self, values: pd.Series, name: str) -> pd.Series:
        """
        Applies Z-Score normalization.
        
        Args:
            values: Values to normalize
            name: Score name (for logging)
        
        Returns:
            Normalized values (in the 0-100 range)
        """
        # Remove missing values
        valid_values = values.dropna()
        
        if len(valid_values) == 0:
            self.logger.warning(f"No valid values for {name}!")
            return pd.Series([0.0] * len(values), index=values.index)
        
        # Standard deviation check
        std = valid_values.std()
        mean = valid_values.mean()
        
        if std == 0 or np.isnan(std) or np.isinf(std):
            self.logger.error(
                f"⚠️  Standard deviation of {name} is 0! All values are equal ({mean:.2f}). "
                f"This score will be excluded from the computation or used as a constant."
            )
            # Normalize constant values to 50 (neutral)
            return pd.Series([50.0] * len(values), index=values.index)
        
        # Compute the Z-Score: z = (x - mean) / std
        z_scores = (values - mean) / std
        
        # Map the Z-Score to the 0-100 range
        # Method: scale the z-score with a sigmoid-like function
        # normalized = 50 + (z_score * 15)  # ±3 std = range 5-95
        # For a wider range:
        normalized = 50 + (z_scores * 20)  # ±2.5 std = range 0-100
        
        # Check the bounds
        normalized = normalized.clip(0, 100)
        
        self.logger.info(
            f"✅ {name} Z-Score normalizasyonu: "
            f"Mean={mean:.2f}, Std={std:.2f}, "
            f"Normalized range: {normalized.min():.2f}-{normalized.max():.2f}"
        )
        
        return normalized
    
    def calculate_ranking(
        self,
        df_projects: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the final fair ranking.
        
        Args:
            df_projects: Project DataFrame with S-Score and R-Score added
        
        Returns:
            DataFrame with final scores added and sorted
        """
        self.logger.info("Computing the final fair ranking...")
        self.logger.info(f"{len(df_projects)} projects will be processed in total")
        
        df_result = df_projects.copy()
        
        # Check the required columns
        required_cols = ['M_Score', 'S_Score', 'R_Score']
        missing_cols = [col for col in required_cols if col not in df_result.columns]
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Raw score statistics
        self.logger.info("\n📊 RAW SCORE STATISTICS:")
        self.logger.info(f"  M_Score: Min={df_result['M_Score'].min():.2f}, Max={df_result['M_Score'].max():.2f}, Mean={df_result['M_Score'].mean():.2f}, Std={df_result['M_Score'].std():.2f}")
        self.logger.info(f"  S_Score: Min={df_result['S_Score'].min():.2f}, Max={df_result['S_Score'].max():.2f}, Mean={df_result['S_Score'].mean():.2f}, Std={df_result['S_Score'].std():.2f}")
        self.logger.info(f"  R_Score: Min={df_result['R_Score'].min():.2f}, Max={df_result['R_Score'].max():.2f}, Mean={df_result['R_Score'].mean():.2f}, Std={df_result['R_Score'].std():.2f}")
        
        # Apply Z-Score normalization
        self.logger.info("\n🔄 APPLYING Z-SCORE NORMALIZATION...")
        
        # Keep the raw scores
        df_result['M_Score_Raw'] = df_result['M_Score'].copy()
        df_result['S_Score_Raw'] = df_result['S_Score'].copy()
        df_result['R_Score_Raw'] = df_result['R_Score'].copy()
        
        # Normalize et
        df_result['M_Score_Normalized'] = self.normalize_with_zscore(df_result['M_Score'], "M_Score")
        df_result['S_Score_Normalized'] = self.normalize_with_zscore(df_result['S_Score'], "S_Score")
        df_result['R_Score_Normalized'] = self.normalize_with_zscore(df_result['R_Score'], "R_Score")
        
        # Use the normalized scores
        df_result['M_Score'] = df_result['M_Score_Normalized']
        df_result['S_Score'] = df_result['S_Score_Normalized']
        df_result['R_Score'] = df_result['R_Score_Normalized']
        
        # Load the feedback
        feedback_adjustments = self._load_feedback_adjustments(df_result)
        
        # Compute the final scores
        final_scores = []
        missing_s_count = 0
        
        for idx, row in df_result.iterrows():
            m = row.get('M_Score', 0.0)
            s = row.get('S_Score', None)
            r = row.get('R_Score', None)
            
            # Check for missing values
            if pd.isna(s) or s is None:
                missing_s_count += 1
                s = 50.0  # Neutral value
            
            if pd.isna(r) or r is None:
                self.logger.warning(f"R score missing for project {idx}, using 50.0")
                r = 50.0
            
            final_score = self.calculate_final_score(m, s, r)
            final_scores.append(final_score)
        
        if missing_s_count > 0:
            self.logger.warning(
                f"No S score for {missing_s_count} projects, 50.0 was used."
            )
        
        df_result['Final_Score'] = final_scores
        
        # Apply the fairness bonus
        if self.fairness_enabled:
            fair_scores = df_result.apply(self.apply_fairness_boost, axis=1)
            df_result['Fair_Score'] = fair_scores
        else:
            df_result['Fair_Score'] = df_result['Final_Score']
        
        # Apply the feedback adjustments
        if feedback_adjustments:
            df_result['Feedback_Adjustment'] = df_result.apply(
                lambda row: feedback_adjustments.get(str(row.get('link', '')), 0.0), 
                axis=1
            )
            df_result['Fair_Score'] = df_result['Fair_Score'] + df_result['Feedback_Adjustment']
            feedback_count = sum(1 for v in feedback_adjustments.values() if v != 0)
            if feedback_count > 0:
                self.logger.info(f"📝 User feedback applied to {feedback_count} projects")
        
        # Sort
        df_result = df_result.sort_values(by='Fair_Score', ascending=False).reset_index(drop=True)
        
        # Diversity-aware top-K selection with MMR (Maximal Marginal Relevance)
        # NOTE: Only re-selects the top-K; it does not change the whole ranking
        if self.mmr_enabled and self.mmr_top_k > 0:
            self.logger.info(f"🔄 Applying diversity-aware ranking with MMR (Maximal Marginal Relevance) (Top-{self.mmr_top_k})...")
            df_result = self._apply_mmr_ranking(
                df_result, 
                top_k=self.mmr_top_k,
                lambda_param=self.mmr_lambda
            )
        
        # Check that all scores are present
        required_columns = ['M_Score', 'S_Score', 'R_Score', 'Final_Score', 'Fair_Score']
        missing_cols = [col for col in required_columns if col not in df_result.columns]
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
        
        # Normalized score statistics
        self.logger.info(f"\n📊 NORMALIZED SCORE STATISTICS:")
        self.logger.info(f"  M_Score: Min={df_result['M_Score'].min():.2f}, Max={df_result['M_Score'].max():.2f}, Mean={df_result['M_Score'].mean():.2f}, Std={df_result['M_Score'].std():.2f}")
        self.logger.info(f"  S_Score: Min={df_result['S_Score'].min():.2f}, Max={df_result['S_Score'].max():.2f}, Mean={df_result['S_Score'].mean():.2f}, Std={df_result['S_Score'].std():.2f}")
        self.logger.info(f"  R_Score: Min={df_result['R_Score'].min():.2f}, Max={df_result['R_Score'].max():.2f}, Mean={df_result['R_Score'].mean():.2f}, Std={df_result['R_Score'].std():.2f}")
        self.logger.info(f"  Fair_Score: Min={df_result['Fair_Score'].min():.2f}, Max={df_result['Fair_Score'].max():.2f}, Mean={df_result['Fair_Score'].mean():.2f}, Std={df_result['Fair_Score'].std():.2f}")
        self.logger.info(f"  Total number of projects: {len(df_result)}")
        
        # Analyze the score contributions
        self.logger.info(f"\n📈 SCORE CONTRIBUTIONS (weighted mean):")
        m_contribution = df_result['M_Score'].mean() * self.w_m
        s_contribution = df_result['S_Score'].mean() * self.w_s
        r_contribution = df_result['R_Score'].mean() * self.w_r
        self.logger.info(f"  M contribution: {m_contribution:.2f} ({self.w_m*100}% weight)")
        self.logger.info(f"  S contribution: {s_contribution:.2f} ({self.w_s*100}% weight)")
        self.logger.info(f"  R contribution: {r_contribution:.2f} ({self.w_r*100}% weight)")
        self.logger.info(f"  Total: {m_contribution + s_contribution + r_contribution:.2f}")
        
        self.logger.info("✅ Final ranking completed (with Z-Score normalization)")
        
        # Compute the performance metrics (top_k from the config)
        # NOTE: m_score_column="M_Score" is used so that raw scores are used for confidence
        # (M_Score_Raw is checked inside calculate_performance_metrics)
        try:
            relevance, diversity, confidence, rc, uc = self.calculate_performance_metrics(
                df_result, 
                top_k=self.metrics_top_k,  # From the config
                vector_column="skill_vectors",
                m_score_column="M_Score"  # M_Score_Raw is checked inside
            )
            self.logger.info(
                f"📊 SYSTEM PERFORMANCE: Relevance={relevance:.2f} | Diversity={diversity:.2f} | Confidence={confidence:.2f} | RC={rc:.2f} | UC={uc:.2f}"
            )
            
            # Report and plot the metrics
            self._save_metrics_report(relevance, diversity, confidence, rc, uc, df_result)
            self._create_metrics_visualization(relevance, diversity, confidence, rc, uc)
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not compute performance metrics: {e}")
            import traceback
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
        
        return df_result
    
    def calculate_performance_metrics(
        self,
        df_ranked: pd.DataFrame,
        top_k: int = 5,
        vector_column: str = "skill_vectors",
        m_score_column: str = "M_Score"
    ) -> Tuple[float, float, float, float, float]:
        """
        Computes the recommender system's performance metrics.
        
        Args:
            df_ranked: Ranked project DataFrame (by Fair_Score)
            top_k: Number of best projects to evaluate (default: 5)
            vector_column: Name of the column holding the project vectors
            m_score_column: Name of the Matching Score column
        
        Returns:
            (relevance_accuracy, diversity_score, confidence, rc_coverage, uc_coverage) tuple
            - relevance_accuracy: Mean M_Score (0-100)
            - diversity_score: Diversity score (0-1)
            - confidence: Standard deviation of the M scores
            - rc_coverage: Recommendation Coverage - share of the recommended projects in the whole pool (0-1)
            - uc_coverage: User Coverage - ratio of unique recommended projects to the total number of projects (0-1)
        """
        if len(df_ranked) == 0:
            self.logger.warning("⚠️  Could not compute performance metrics: empty DataFrame")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Select the top K projects
        top_k_actual = min(top_k, len(df_ranked))
        top_projects = df_ranked.head(top_k_actual).copy()
        
        if top_k_actual == 0:
            self.logger.warning("⚠️  Could not compute performance metrics: no projects")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 1. Relevance Accuracy: mean of M_Score
        if m_score_column not in top_projects.columns:
            self.logger.warning(f"⚠️  Could not compute relevance: column {m_score_column} not found")
            relevance_accuracy = 0.0
        else:
            m_scores = top_projects[m_score_column].astype(float)
            relevance_accuracy = float(m_scores.mean())
        
        # 2. Diversity Score: hybrid approach - vector similarity + skill set similarity
        diversity_score = 0.0
        if top_k_actual == 1:
            # Diversity is 0 if there is a single project
            diversity_score = 0.0
        elif vector_column in top_projects.columns:
            # Extract the vectors
            vectors_list = []
            valid_indices = []
            
            for idx, row in top_projects.iterrows():
                vector = row[vector_column]
                # Vector check: add if not None, NaN or empty
                try:
                    if vector is None:
                        continue
                    
                    # NaN float check (can be a NaN float after a pandas merge)
                    if isinstance(vector, float) and (np.isnan(vector) or pd.isna(vector)):
                        continue
                    
                    # Convert the vector to a numpy array
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
                        # NumPy array or another type
                        try:
                            vec_array = np.array(vector)
                            if vec_array.size == 0:
                                continue
                            # Do not use pd.isna() on a NumPy array (it returns a boolean array)
                            # Check directly with np.array() instead
                        except:
                            continue
                    
                    if vec_array is None:
                        continue
                    
                    # Normalize the vector (take the mean for 2D vectors)
                    if vec_array.ndim == 0:
                        # Scalar (could be NaN) - skip
                        continue
                    elif vec_array.ndim == 1:
                        # 1D vector - use directly
                        normalized_vec = vec_array
                    elif vec_array.ndim == 2:
                        # 2D vector (n_skills, 768) - take its mean
                        if vec_array.shape[1] == 768:
                            normalized_vec = vec_array.mean(axis=0)  # (768,)
                        else:
                            # Flatten if the shape differs
                            normalized_vec = vec_array.flatten()
                    else:
                        # 3D or more - flatten
                        normalized_vec = vec_array.flatten()
                    
                    # Norm check (do not normalize yet, only check)
                    norm = np.linalg.norm(normalized_vec)
                    if norm > 1e-8 and not np.isnan(norm) and not np.isinf(norm):
                        vectors_list.append(normalized_vec)  # Not normalized yet
                        valid_indices.append(idx)
                except (ValueError, TypeError, AttributeError) as e:
                    # Skip on array comparison errors or other errors
                    continue
            
            if len(vectors_list) >= 2:
                try:
                    # Normalize the vectors and stack them into a 2D matrix
                    # Each vector must be a 1D numpy array (768-dimensional embedding)
                    normalized_vectors = []
                    for vec in vectors_list:
                        try:
                            # vec is already normalized_vec (processed above), use directly
                            vec_array = np.array(vec)
                            
                            # If it is still 2D (unexpected), take its mean
                            if vec_array.ndim == 2:
                                vec_array = vec_array.mean(axis=0)
                            elif vec_array.ndim > 2:
                                vec_array = vec_array.flatten()
                            elif vec_array.ndim == 0:
                                continue
                            
                            # Must be 1D
                            if vec_array.ndim != 1:
                                continue
                            
                            # Shape check and correction
                            if len(vec_array) > 768:
                                vec_array = vec_array[:768]
                            elif len(vec_array) < 768:
                                # Pad (with zeros)
                                padding = np.zeros(768 - len(vec_array), dtype=vec_array.dtype)
                                vec_array = np.concatenate([vec_array, padding])
                            
                            # Normalize (make it a unit vector) - FIRST NORMALIZATION
                            norm = np.linalg.norm(vec_array)
                            if norm > 1e-8 and not np.isnan(norm) and not np.isinf(norm):
                                vec_normalized = vec_array / norm
                                normalized_vectors.append(vec_normalized)
                        except Exception as vec_e:
                            self.logger.debug(f"Diversity: vector processing error: {vec_e}")
                            continue
                    
                    if len(normalized_vectors) < 2:
                        self.logger.warning(f"⚠️  Could not compute diversity: not enough normalized vectors ({len(normalized_vectors)}/{len(vectors_list)})")
                        diversity_score = 0.0
                    else:
                        # Stack all vectors into a 2D array (one vector per row)
                        # Shape: (n_vectors, vector_dim)
                        vectors_matrix = np.vstack(normalized_vectors)
                        
                        # Compute cosine similarity (pairwise)
                        # cosine_similarity(X) computes the similarity within X
                        similarity_matrix = cosine_similarity(vectors_matrix)
                        
                        # Set the diagonal to 0 (self-similarity = 1, which we do not want)
                        np.fill_diagonal(similarity_matrix, 0.0)
                        
                        # Mean similarity (from the upper triangle - to count each pair once)
                        # Mean of the upper triangular matrix
                        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                        if mask.sum() > 0:
                            avg_similarity = similarity_matrix[mask].mean()
                            min_similarity = similarity_matrix[mask].min()
                            max_similarity = similarity_matrix[mask].max()
                            self.logger.debug(
                                f"Diversity computation: mean similarity={avg_similarity:.4f}, "
                                f"Min={min_similarity:.4f}, Max={max_similarity:.4f}"
                            )
                        else:
                            avg_similarity = 0.0
                        
                        # Diversity = 1 - mean similarity (vector-based)
                        vector_diversity = float(1.0 - avg_similarity)
                        
                        # Hybrid diversity: vector similarity + skill set similarity
                        # If a skill column exists, also compute the Jaccard similarity
                        skill_diversity = 1.0
                        if 'extracted_skills' in top_projects.columns:
                            skill_sets = []
                            for idx in valid_indices:
                                skills = top_projects.loc[idx, 'extracted_skills']
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
                        
                        # Hybrid diversity: 70% vector, 30% skill set
                        # If the vectors are very similar, skill set differences increase the diversity
                        diversity_score = 0.7 * vector_diversity + 0.3 * skill_diversity
                        diversity_score = max(0.0, min(1.0, diversity_score))  # Clip to the 0-1 range
                        
                        if diversity_score < 0.1:
                            self.logger.warning(
                                f"⚠️  Low Diversity Score ({diversity_score:.4f}): "
                                f"the top {top_k_actual} projects are very similar to each other "
                                f"(vector similarity={avg_similarity:.4f}, hybrid diversity={diversity_score:.4f})"
                            )
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not compute diversity: {e}")
                    diversity_score = 0.0
            else:
                self.logger.warning(f"⚠️  Could not compute diversity: not enough vectors ({len(vectors_list)}/{top_k_actual})")
                diversity_score = 0.0
        else:
            self.logger.warning(f"⚠️  Could not compute diversity: column {vector_column} not found")
            diversity_score = 0.0
        
        # 3. Confidence: standard deviation of the M scores (within the top-K)
        # NOTE: Use raw scores instead of normalized scores (to avoid the cloning problem)
        if m_score_column not in top_projects.columns:
            confidence = 0.0
        else:
            # Try the raw scores first (they show more variation)
            ham_column = m_score_column.replace('M_Score', 'M_Score_Raw') if 'M_Score' in m_score_column else None
            if ham_column and ham_column in top_projects.columns:
                m_scores = top_projects[ham_column].astype(float).dropna()
                if len(m_scores) > 0:
                    std_dev = float(m_scores.std())
                    # If the std is very low, take the std of the whole dataset (shows the variation after normalization)
                    if std_dev < 1.0 and len(df_ranked) > top_k_actual:
                        # Take the std of the raw scores over the whole dataset
                        if ham_column in df_ranked.columns:
                            all_scores = df_ranked[ham_column].astype(float).dropna()
                            if len(all_scores) > 0:
                                all_std = float(all_scores.std())
                                # Mean of the top-K std and the whole-dataset std
                                confidence = (std_dev + all_std) / 2.0
                                self.logger.debug(f"Confidence: top-K std={std_dev:.4f}, all std={all_std:.4f}, combined={confidence:.4f}")
                            else:
                                confidence = std_dev
                        else:
                            confidence = std_dev
                    else:
                        confidence = std_dev
                    self.logger.debug(f"Confidence computed from raw scores (std={confidence:.4f})")
                else:
                    confidence = 0.0
            else:
                # If there are no raw scores, use the normalized scores
                m_scores = top_projects[m_score_column].astype(float)
                std_dev = float(m_scores.std())
                # If the std is very low, take the std of the whole dataset
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
                self.logger.debug(f"Confidence computed from normalized scores (std={confidence:.4f})")
        
        # 4. RC (Recommendation Coverage): share of the recommended projects in the whole pool
        # RC = |unique recommended projects| / |total project pool|
        total_projects = len(df_ranked)
        recommended_unique = len(top_projects)  # Top K projects (already unique and sorted)
        
        if total_projects > 0:
            rc_coverage = float(recommended_unique) / float(total_projects)
            rc_coverage = max(0.0, min(1.0, rc_coverage))  # Clip to the 0-1 range
        else:
            rc_coverage = 0.0
        
        # 5. UC (User Coverage): share of recommended projects among the projects above the minimum score threshold
        # In our system: how many of the projects with Fair_Score > 40 are in the recommended top K?
        # UC = |top K recommendations| / |all projects with a sufficient score (e.g. Fair_Score > 40)|
        if 'Fair_Score' in df_ranked.columns:
            # Count the projects with a sufficient score (e.g. Fair_Score > 40)
            threshold = 40.0  # Minimum score threshold
            eligible_projects = df_ranked[df_ranked['Fair_Score'] > threshold]
            eligible_count = len(eligible_projects)
            
            if eligible_count > 0:
                # How many of the top K are eligible?
                top_k_eligible = top_projects[top_projects['Fair_Score'] > threshold]
                uc_coverage = float(len(top_k_eligible)) / float(eligible_count)
                uc_coverage = max(0.0, min(1.0, uc_coverage))  # Clip to the 0-1 range
            else:
                uc_coverage = 0.0
        else:
            # Without Fair_Score, compute simply as top K / total (same as RC)
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
        vector_column: str = "skill_vectors"
    ) -> pd.DataFrame:
        """
        Applies diversity-aware ranking with the MMR (Maximal Marginal Relevance) algorithm.
        
        MMR balances relevance and diversity:
        MMR_score = lambda * relevance + (1 - lambda) * diversity
        
        Args:
            df_ranked: Ranked project DataFrame (by Fair_Score)
            top_k: Number of projects to select with MMR
            lambda_param: Balance between relevance (0.0) and diversity (1.0) (0.5 = equal weight)
            vector_column: Name of the column holding the project vectors
        
        Returns:
            DataFrame re-ranked with MMR (top-K by MMR, the rest in the original order)
        """
        if len(df_ranked) <= top_k:
            # MMR is not needed if there are fewer projects than top-K
            return df_ranked
        
        self.logger.info(f"🎯 MMR algorithm: selecting Top-{top_k} (lambda={lambda_param:.2f})")
        
        # Select the top-K with MMR (use integer positions, for a DataFrame after reset_index)
        selected_positions = []  # Position in the DataFrame (0-based index)
        remaining_positions = list(range(len(df_ranked)))
        
        # First project: the one with the highest Fair_Score (already sorted, first row)
        first_pos = 0
        selected_positions.append(first_pos)
        remaining_positions.remove(first_pos)
        
        # Store the vectors of the selected projects
        selected_vectors = []
        if vector_column in df_ranked.columns:
            first_vector = df_ranked.iloc[first_pos][vector_column]
            if first_vector is not None:
                try:
                    vec_array = np.array(first_vector)
                    # If it is a 2D vector, take its mean
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
        
        # Compute the MMR score for the remaining projects
        while len(selected_positions) < top_k and len(remaining_positions) > 0:
            best_pos = None
            best_mmr_score = -float('inf')
            
            for pos in remaining_positions:
                row = df_ranked.iloc[pos]
                
                # Relevance: Fair_Score (normalized to the 0-1 range)
                relevance = row.get('Fair_Score', 0.0) / 100.0  # Normalize from 0-100 to 0-1
                
                # Diversity: maximum similarity to the selected projects (low similarity = high diversity)
                diversity = 1.0  # Default: maximum diversity
                
                if selected_vectors and vector_column in row and row[vector_column] is not None:
                    try:
                        # Get the current project's vector
                        current_vec = np.array(row[vector_column])
                        # If it is a 2D vector, take its mean
                        if current_vec.ndim == 2:
                            current_vec = current_vec.mean(axis=0)  # (768,)
                        elif current_vec.ndim > 2:
                            current_vec = current_vec.flatten()
                            if len(current_vec) > 768:
                                current_vec = current_vec[:768]
                        elif current_vec.ndim == 0:
                            diversity = 1.0  # Scalar - maximum diversity
                            continue
                        
                        if current_vec.ndim == 1 and len(current_vec) == 768:
                            norm = np.linalg.norm(current_vec)
                            if norm > 0 and not np.isnan(norm) and not np.isinf(norm):
                                current_vec = current_vec / norm
                                
                                # Compute the maximum similarity to the selected projects
                                max_similarity = 0.0
                                for sel_vec in selected_vectors:
                                    similarity = float(np.dot(current_vec, sel_vec))
                                    max_similarity = max(max_similarity, similarity)
                                
                                # Diversity = 1 - maximum similarity
                                diversity = 1.0 - max_similarity
                    except Exception as e:
                        self.logger.debug(f"MMR: vector computation error (pos={pos}): {e}")
                        diversity = 1.0  # On error, assume maximum diversity
                
                # MMR Score: lambda * relevance + (1 - lambda) * diversity
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_pos = pos
            
            if best_pos is not None:
                selected_positions.append(best_pos)
                remaining_positions.remove(best_pos)
                
                # Update the selected vectors
                if vector_column in df_ranked.columns:
                    best_vector = df_ranked.iloc[best_pos][vector_column]
                    if best_vector is not None:
                        try:
                            vec_array = np.array(best_vector)
                            # If it is a 2D vector, take its mean
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
                break  # No project could be selected, stop
        
        # Take the projects selected by MMR
        mmr_selected = df_ranked.iloc[selected_positions].copy()
        
        # Keep the remaining projects in the original order
        remaining_ordered = df_ranked.iloc[remaining_positions].copy()
        
        # Merge: MMR-selected + remaining (in the original order)
        df_result = pd.concat([mmr_selected, remaining_ordered], ignore_index=True)
        
        self.logger.info(f"✅ MMR ranking completed: Top-{len(selected_positions)} projects selected with the diversity factor")
        
        return df_result
    
    def _load_feedback_adjustments(self, df_projects: pd.DataFrame) -> Dict[str, float]:
        """
        Loads user feedback and computes score adjustments.
        
        Args:
            df_projects: Project DataFrame
        
        Returns:
            Feedback adjustments by project link (positive feedback +5, negative -5)
        """
        feedback_file = Path(__file__).resolve().parent.parent.parent / "data/external/user_feedback.json"
        
        if not feedback_file.exists():
            return {}
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not read the feedback file: {e}")
            return {}
        
        # Get the project links
        project_links = set(df_projects['link'].astype(str).unique())
        
        # Compute the number of feedback entries for each project
        adjustments = {}
        feedback_bonus = 5.0  # Bonus for positive feedback
        feedback_penalty = -5.0  # Penalty for negative feedback
        
        for feedback in feedbacks:
            project_link = feedback.get('project_link', '')  # Link stored in the feedback
            project_id = feedback.get('project_id', '')
            feedback_type = feedback.get('feedback_type', '')
            
            # Find the project link
            matching_link = None
            if project_link and project_link in project_links:
                matching_link = project_link
            elif project_id:
                # Try to find the link from the project ID
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
        Saves the performance metrics as a report (in JSON and Markdown format).
        
        Args:
            relevance: Relevance Accuracy metric value
            diversity: Diversity Score metric value
            confidence: Confidence metric value
            rc_coverage: Recommendation Coverage metric value
            uc_coverage: User Coverage metric value
            df_ranked: Ranked project DataFrame
        """
        try:
            # Create the output directory
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare the metric data
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
                        'relevance_accuracy': 'Mean M_Score (Matching Score) - range 0-100',
                        'diversity_score': 'Item-to-Item Similarity (1 - mean cosine similarity) - range 0-1',
                        'confidence': 'Standard deviation of M_Score - reliability of the scores',
                        'recommendation_coverage': 'Share of the recommended projects in the whole pool - range 0-1',
                        'user_coverage': 'Share of the recommended projects among projects with a sufficient score - range 0-1'
                    }
                }
            }
            
            # Save in JSON format
            json_file = output_dir / f"metrics_report_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📄 Metric report saved in JSON format: {json_file}")
            
            # Save in Markdown format
            md_file = output_dir / f"metrics_report_{timestamp}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write("# System Performance Metrics Report\n\n")
                f.write(f"**Report date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write("## Metric Summary\n\n")
                f.write("| Metric | Value | Description |\n")
                f.write("|--------|-------|----------|\n")
                f.write(f"| **Relevance Accuracy** | {relevance:.4f} | Mean M_Score (0-100) |\n")
                f.write(f"| **Diversity Score** | {diversity:.4f} | Diversity score (0-1) |\n")
                f.write(f"| **Confidence** | {confidence:.4f} | Standard deviation of the M scores |\n")
                f.write(f"| **RC (Recommendation Coverage)** | {rc_coverage:.4f} | Share of the recommended projects in the whole pool (0-1) |\n")
                f.write(f"| **UC (User Coverage)** | {uc_coverage:.4f} | Share of the recommended projects among projects with a sufficient score (0-1) |\n")
                f.write("\n---\n\n")
                f.write("## Detailed Descriptions\n\n")
                f.write("### Relevance Accuracy\n")
                f.write("The mean Matching Score (M_Score) of the recommended projects. ")
                f.write("This metric shows how well the recommended projects fit the user profile.\n\n")
                f.write("### Diversity Score\n")
                f.write("A measure of diversity among the recommended projects. ")
                f.write("It is the cosine-similarity-based Item-to-Item Similarity (ILS) metric. ")
                f.write("A high value shows that the recommended projects differ from each other.\n\n")
                f.write("### Confidence\n")
                f.write("The standard deviation of the M_Score values. ")
                f.write("This metric shows the reliability of the recommendations. ")
                f.write("A low standard deviation means more consistent recommendations.\n\n")
                f.write("### RC (Recommendation Coverage)\n")
                f.write("The share of the recommended projects in the whole project pool. ")
                f.write("This metric shows how wide a range of projects the system recommends from.\n\n")
                f.write("### UC (User Coverage)\n")
                f.write("The share of the recommended projects among the projects above the minimum score threshold (Fair_Score > 40). ")
                f.write("This metric shows how well the system covers projects of sufficient quality.\n\n")
                f.write("---\n\n")
            f.write(f"**Total number of projects**: {len(df_ranked)}\n\n")
            f.write(f"**Top K analyzed**: {min(self.metrics_top_k, len(df_ranked))}\n\n")
            self.logger.info(f"📄 Metric report saved in Markdown format: {md_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save the metric report: {e}")
    
    def _create_metrics_visualization(
        self,
        relevance: float,
        diversity: float,
        confidence: float,
        rc_coverage: float,
        uc_coverage: float
    ):
        """
        Visualizes and saves the performance metrics.
        
        Args:
            relevance: Relevance Accuracy metric value
            diversity: Diversity Score metric value
            confidence: Confidence metric value
            rc_coverage: Recommendation Coverage metric value
            uc_coverage: User Coverage metric value
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # So that it works without a GUI
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Create the output directory
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Chart 1: comparative bar chart of all metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('System Performance Metrics', fontsize=16, fontweight='bold')
            
            # 1. Bar chart - all metrics (normalized)
            ax1 = axes[0, 0]
            metrics_names = ['Relevance\n(0-100)', 'Diversity\n(0-1)', 'RC\n(0-1)', 'UC\n(0-1)']
            metrics_values = [relevance / 100.0, diversity, rc_coverage, uc_coverage]  # Normalize et
            
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_ylim([0, 1])
            ax1.set_ylabel('Normalized value (0-1)', fontsize=10)
            ax1.set_title('Metric Comparison (normalized)', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Write the values above the bars
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
            
            # 2. Relevance and Diversity - scatter/bar combination
            ax2 = axes[0, 1]
            ax2.bar(['Relevance', 'Diversity'], [relevance / 100.0, diversity], 
                   color=['#2ecc71', '#3498db'], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_ylim([0, 1])
            ax2.set_ylabel('Value (0-1)', fontsize=10)
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
            ax3.set_ylabel('Coverage value (0-1)', fontsize=10)
            ax3.set_title('Recommendation & User Coverage', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, val in zip(bars3, coverage_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 4. Confidence (standard deviation)
            ax4 = axes[1, 1]
            ax4.bar(['Confidence\n(Std Dev)'], [confidence], 
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax4.set_ylabel('Standard deviation', fontsize=10)
            ax4.set_title('Confidence (M_Score std. deviation)', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            ax4.text(0, confidence, f'{confidence:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the chart file
            graph_file = output_dir / f"metrics_visualization_{timestamp}.png"
            plt.savefig(graph_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"📊 Metric visualization saved: {graph_file}")
            
        except ImportError:
            self.logger.warning("⚠️  matplotlib not found, the chart could not be created. Install it with 'pip install matplotlib'.")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not create the metric visualization: {e}")

