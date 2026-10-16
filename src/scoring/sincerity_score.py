"""
Module that computes the sincerity score (S-Score)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import logging


class SincerityScoreCalculator:
    """Sincerity score calculator class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # AI detection model (disabled by default: the GPT-2 output detector returns a single
        # constant for the letter, so it cannot change the order of the listings)
        ai_config = config.get('models', {}).get('ai_detection', {})
        self.ai_enabled = ai_config.get('enabled', False)
        if self.ai_enabled:
            ai_model_name = ai_config.get('model_name', 'roberta-base-openai-detector')
            self.logger.info(f"Loading AI detection model: {ai_model_name}...")
            self.ai_tokenizer = AutoTokenizer.from_pretrained(ai_model_name)
            self.ai_model = AutoModelForSequenceClassification.from_pretrained(ai_model_name)
            self.ai_model.eval()
        
        # Embedding model
        embed_config = config.get('models', {}).get('embedding', {})
        embed_model_name = embed_config.get('model_name', 'xlm-roberta-base')
        device_str = embed_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        
        self.logger.info(f"Loading embedding model: {embed_model_name}...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(self.device)
        self.embed_model.eval()
        
        # Scoring settings
        sincerity_config = config.get('sincerity_scoring', {})
        self.human_weight = sincerity_config.get('human_score_weight', 0.4)
        self.context_weight = sincerity_config.get('context_score_weight', 0.6)
        self.top_n = sincerity_config.get('top_n_projects', 20)
        if not self.ai_enabled:
            self.human_weight, self.context_weight = 0.0, 1.0
        self.query_prefix = embed_config.get('query_prefix', '')
        self.passage_prefix = embed_config.get('passage_prefix', '')
        
        # Motivation letter
        self.motivation_letter = config.get('data_sources', {}).get('motivation_letter', {}).get('text', '')
        if not self.motivation_letter:
            self.logger.warning("Motivation letter not found!")
        
        # Precompute the letter's vector
        if self.motivation_letter:
            self.letter_vector = self._get_embedding(self.query_prefix + self.motivation_letter)
            self.human_score = self._detect_ai_generated(self.motivation_letter) if self.ai_enabled else 0.0
            if self.ai_enabled:
                self.logger.info(f"Motivation letter AI score: {self.human_score:.4f}")
    
    def _detect_ai_generated(self, text: str) -> float:
        """Returns the probability that the text was written by AI"""
        inputs = self.ai_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.ai_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        ai_probability = probs[0][1].item()
        return ai_probability
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Returns the embedding of the text"""
        inputs = self.embed_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)

        # Mean pooling over real tokens only
        mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        return (summed / mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
    
    def calculate_sincerity_score(self, project_text: str) -> tuple:
        """
        Computes the sincerity score for a project.
        
        Args:
            project_text: Project description text
        
        Returns:
            (S_Score, Human_Score, Context_Score) tuple
        """
        # Get the project vector
        project_vector = self._get_embedding(self.passage_prefix + str(project_text))
        
        # Cosine similarity
        similarity = cosine_similarity(self.letter_vector, project_vector)[0][0]
        context_score = similarity * 100
        
        # Human-written score (complement); 0 when the detector is disabled
        human_score = (1 - self.human_score) * 100 if self.ai_enabled else 0.0
        
        # Final S score
        final_s = (human_score * self.human_weight) + (context_score * self.context_weight)
        
        return final_s, human_score, context_score
    
    def calculate_all_scores(
        self,
        df_projects: pd.DataFrame,
        text_column: str = "full_description",
        calculate_all: bool = False
    ) -> pd.DataFrame:
        """
        Computes sincerity scores for all projects (only the top N, or all).
        
        Args:
            df_projects: Project DataFrame (must be sorted by M_Score)
            text_column: Name of the text column
            calculate_all: If True, compute for all projects; if False, only for the top N
        
        Returns:
            DataFrame with S_Score added
        """
        # Fast mode: process only the best N projects (default)
        if calculate_all:
            df_top = df_projects.copy()
            self.logger.info(f"Computing sincerity scores for {len(df_top)} projects (all projects)...")
        else:
            df_top = df_projects.head(self.top_n).copy()
            self.logger.info(f"Computing sincerity scores for {len(df_top)} projects (top {self.top_n} projects)...")
        
        scores = []
        human_scores = []
        context_scores = []
        
        for idx, row in df_top.iterrows():
            s_score, h_score, c_score = self.calculate_sincerity_score(row[text_column])
            scores.append(s_score)
            human_scores.append(h_score)
            context_scores.append(c_score)
        
        df_result = df_top.copy()
        df_result["S_Score"] = scores
        df_result["Human_Score"] = human_scores
        df_result["Context_Score"] = context_scores
        
        # If only the top N were computed, add NaN values for the other projects
        if not calculate_all and len(df_projects) > len(df_result):
            # Get the missing projects from the original DataFrame
            remaining_df = df_projects.iloc[len(df_result):].copy()
            remaining_df["S_Score"] = None
            remaining_df["Human_Score"] = None
            remaining_df["Context_Score"] = None
            
            # Merge the results
            df_result = pd.concat([df_result, remaining_df], ignore_index=True)
        
        self.logger.info("Sincerity scores computed")
        return df_result

