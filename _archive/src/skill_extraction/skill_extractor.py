"""
Module that extracts skills with SkillNER
"""

import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
import pandas as pd
from typing import List
import logging
from pathlib import Path

from ..utils.data_cleaner import filter_skills


class SkillExtractorModule:
    """SkillNER skill extraction class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        model_config = config.get('models', {}).get('skill_extraction', {})
        self.spacy_model = model_config.get('spacy_model', 'en_core_web_lg')
        
        # Load the spaCy model
        try:
            self.nlp = spacy.load(self.spacy_model)
            self.logger.info(f"spaCy model loaded: {self.spacy_model}")
        except IOError:
            self.logger.error(f"spaCy model not found: {self.spacy_model}")
            raise
        
        # Initialize the SkillExtractor
        try:
            self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
            self.logger.info("SkillExtractor initialized successfully")
        except Exception as e:
            self.logger.error(f"Error while initializing SkillExtractor: {e}")
            raise
        
        # Filtering settings
        data_config = config.get('data_processing', {})
        self.min_skill_length = data_config.get('min_skill_length', 2)
        self.skill_stopwords = set(data_config.get('skill_stopwords', []))
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extracts skills from a text.
        
        Args:
            text: Text to process
        
        Returns:
            List of extracted skills
        """
        try:
            annotations = self.skill_extractor.annotate(text)
            
            skills = []
            
            # Full matches
            for match in annotations['results']['full_matches']:
                skills.append(match['doc_node_value'])
            
            # N-gram matches
            for match in annotations['results']['ngram_scored']:
                skills.append(match['doc_node_value'])
            
            # Deduplicate and normalize
            skills = list(set([s.lower() for s in skills]))
            
            # Filtrele
            skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
            
            return skills
            
        except Exception as e:
            self.logger.error(f"Skill extraction error: {e}", exc_info=True)
            return []
    
    def extract_skills_from_dataframe(self, df: pd.DataFrame, text_column: str = "full_description") -> pd.DataFrame:
        """
        Extracts skills from all texts in a DataFrame.
        
        Args:
            df: DataFrame to process
            text_column: Name of the text column
        
        Returns:
            DataFrame with the skill list added
        """
        self.logger.info(f"Processing {len(df)} texts...")
        
        skill_lists = []
        total = len(df)
        
        for idx, text in enumerate(df[text_column].astype(str)):
            try:
                skills = self.extract_skills_from_text(text)
                skill_lists.append(skills)
            except Exception as e:
                self.logger.warning(f"Error while processing row {idx}: {e}")
                skill_lists.append([])
            
            # Progress report
            if (idx + 1) % 50 == 0:
                self.logger.info(f"Progress: {idx+1}/{total} texts processed")
        
        df_result = df.copy()
        df_result["extracted_skills"] = skill_lists
        
        self.logger.info(f"All texts processed")
        return df_result

