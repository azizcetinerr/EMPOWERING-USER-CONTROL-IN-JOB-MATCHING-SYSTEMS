"""
Module that converts skills into vectors (XLM-RoBERTa)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import logging


class SkillVectorizer:
    """Skill vectorization class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        model_config = config.get('models', {}).get('embedding', {})
        self.model_name = model_config.get('model_name', 'xlm-roberta-base')
        self.batch_size = model_config.get('batch_size', 32)
        
        # Device selection
        device_str = model_config.get('device', 'auto')
        if device_str == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.logger.info(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error while loading model: {e}")
            raise
    
    def vectorize_skills(self, skills: Union[List[str], List[List[str]]]) -> np.ndarray:
        """
        Converts a skill list into vectors.
        
        Args:
            skills: List of skills, or a list of skill lists
        
        Returns:
            Vector matrix (numpy array)
        """
        if not skills or len(skills) == 0:
            return np.zeros((1, 768))  # XLM-R-base dimension
        
        # If it is a nested list (one skill list per project)
        if isinstance(skills[0], list):
            # Join the skills of each project
            skill_texts = [' '.join(skill_list) if skill_list else '' for skill_list in skills]
        else:
            skill_texts = skills
        
        # Batch processing
        all_embeddings = []
        
        for i in range(0, len(skill_texts), self.batch_size):
            batch = skill_texts[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        
        result = np.vstack(all_embeddings)
        return result
    
    def vectorize_dataframe(self, df: pd.DataFrame, skill_column: str = "extracted_skills") -> pd.DataFrame:
        """
        Converts the skill lists in a DataFrame into vectors.
        
        Args:
            df: DataFrame to process
            skill_column: Name of the skill list column
        
        Returns:
            DataFrame with vectors added
        """
        self.logger.info(f"Vectorizing {len(df)} projects...")
        
        skill_lists = df[skill_column].tolist()
        vectors = []
        
        for skill_list in skill_lists:
            if not skill_list or len(skill_list) == 0:
                vectors.append(np.zeros((1, 768)))
            else:
                vec = self.vectorize_skills(skill_list)
                vectors.append(vec)
        
        df_result = df.copy()
        df_result["skill_vectors"] = vectors
        
        self.logger.info("All projects vectorized")
        return df_result

