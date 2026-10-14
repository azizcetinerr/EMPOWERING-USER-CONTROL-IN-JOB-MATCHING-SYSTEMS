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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

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
        
        # Filtering settings
        data_config = config.get('data_processing', {})
        self.min_skill_length = data_config.get('min_skill_length', 2)
        self.skill_stopwords = set(data_config.get('skill_stopwords', []))
        
        model_config = config.get('models', {}).get('skill_extraction', {})
        self.spacy_model = model_config.get('spacy_model', 'en_core_web_sm')
        
        # Use SkillNER
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
        
        # Performance optimization parameters (read from the config)
        skill_config = self.config.get('models', {}).get('skill_extraction', {})
        self.max_text_length = skill_config.get('max_text_length', 8000)  # 0 = unlimited
        self.max_ngram_results = skill_config.get('max_ngram_results', 40)  # 0 = unlimited
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extracts skills from a text.
        
        Args:
            text: Text to process
        
        Returns:
            List of extracted skills
        """
        # Use SkillNER
        try:
            # Performance optimization: truncate very long texts
            original_text = text
            if self.max_text_length > 0 and len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                self.logger.debug(f"Text truncated from {len(original_text)} to {len(text)} characters (for performance)")
            
            annotations = self.skill_extractor.annotate(text)
            
            skills = []
            
            # Full matches (faster, take priority)
            for match in annotations['results']['full_matches']:
                skills.append(match['doc_node_value'])
            
            # N-gram matches (limited for performance)
            ngram_results = annotations.get('results', {}).get('ngram_scored', [])
            # Take the best N n-grams (for performance)
            if self.max_ngram_results > 0 and len(ngram_results) > self.max_ngram_results:
                ngram_results = ngram_results[:self.max_ngram_results]
            for match in ngram_results:
                skills.append(match['doc_node_value'])
            
            # Deduplicate and normalize
            skills = list(set([s.lower() for s in skills]))
            
            # Filtrele
            skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
            
            return skills
            
        except (IndexError, ValueError, KeyError) as e:
            # Expected errors from the skillNer library (for some texts)
            # These errors are normal; an empty list is returned and processing continues
            self.logger.debug(f"Skill extraction error (text skipped): {type(e).__name__}")
            return []
        except Exception as e:
            # Warn on unexpected errors
            self.logger.warning(f"Skill extraction error: {e}")
            return []
    
    def extract_skills_from_dataframe(self, df: pd.DataFrame, text_column: str = "full_description", parallel: bool = True) -> pd.DataFrame:
        """
        Extracts skills from all texts in a DataFrame (parallel or sequential).
        
        Args:
            df: DataFrame to process
            text_column: Name of the text column
            parallel: Use parallel processing (True) or sequential (False) - IGNORED, parallel is always used
        
        Returns:
            DataFrame with the skill list added
        """
        total = len(df)
        self.logger.info(f"Processing {total} texts...")
        
        # Always use parallel mode (if there are more than 10 projects) - the parallel parameter is no longer used
        if total > 10:
            self.logger.info(f"🚀 Parallel skill extraction mode active (more than 10 projects)")
            return self._extract_skills_parallel(df, text_column)
        else:
            self.logger.info(f"📝 Sequential skill extraction mode (10 or fewer projects)")
            return self._extract_skills_sequential(df, text_column)
    
    def _extract_skills_sequential(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Sequential skill extraction (old method)"""
        skill_lists = []
        total = len(df)
        error_count = 0
        
        for idx, text in enumerate(df[text_column].astype(str)):
            try:
                skills = self.extract_skills_from_text(text)
                skill_lists.append(skills)
            except Exception as e:
                error_count += 1
                skill_lists.append([])
            
            # Progress report (shown more often in sequential mode)
            if (idx + 1) % 25 == 0:
                self.logger.info(f"Progress: {idx+1}/{total} texts processed ({(idx+1)*100//total}%)")
        
        # Error summary
        if error_count > 0:
            self.logger.info(f"⚠️  Skill extraction error for {error_count}/{total} texts (normal; some texts cannot be processed)")
        
        df_result = df.copy()
        df_result["extracted_skills"] = skill_lists
        
        self.logger.info(f"✅ All texts processed")
        return df_result
    
    def _extract_skills_parallel(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Parallel skill extraction (fast method)"""
        total = len(df)
        
        # Read the number of workers from the config (skill_extraction_workers if set, otherwise num_workers)
        perf_config = self.config.get('performance', {})
        num_workers = perf_config.get('skill_extraction_workers') or perf_config.get('num_workers', 8)
        # Limit by the number of CPUs (too many threads can cause memory problems)
        import os
        cpu_count = os.cpu_count() or 4
        # Optimal thread count: number of CPUs or the config value, max 16 (more threads can cause memory problems)
        max_workers = min(num_workers, cpu_count * 2, 16)  # Max 16 thread
        
        self.logger.info(f"🚀 Starting parallel skill extraction: {max_workers} threads")
        
        # Create a separate model instance for each worker
        def create_worker_extractor():
            """Creates a separate model instance for each thread"""
            nlp = spacy.load(self.spacy_model)
            skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
            return skill_extractor
        
        # Create the worker pool - sequential loading (safer for memory)
        self.logger.info(f"   ⏳ Loading {max_workers} worker models...")
        worker_extractors = []
        
        for i in range(max_workers):
            try:
                extractor = create_worker_extractor()
                worker_extractors.append(extractor)
                if (i + 1) % 4 == 0 or (i + 1) == max_workers:
                    self.logger.info(f"   ✅ {i + 1}/{max_workers} worker models loaded")
            except Exception as e:
                self.logger.warning(f"   ⚠️  Worker {i+1} could not be loaded: {e}, continuing with fewer workers")
                if len(worker_extractors) == 0:
                    self.logger.error("❌ No worker model could be loaded, switching to sequential mode")
                    return self._extract_skills_sequential(df, text_column)
                break
        
        max_workers = len(worker_extractors)  # Actual number of workers
        self.logger.info(f"   ✅ {max_workers} worker models ready, starting parallel processing")
        
        worker_lock = threading.Lock()
        worker_idx = 0
        
        def get_worker_extractor():
            """Returns a worker extractor in round-robin order"""
            nonlocal worker_idx
            with worker_lock:
                extractor = worker_extractors[worker_idx]
                worker_idx = (worker_idx + 1) % max_workers
                return extractor
        
        # Prepare the data to process
        texts = df[text_column].astype(str).tolist()
        
        def process_single_text(idx_text_pair):
            """Processes a single text"""
            idx, text = idx_text_pair
            try:
                # Use SkillNER
                worker_extractor = get_worker_extractor()
                
                # Performance optimization: truncate very long texts
                max_text_len = self.config.get('models', {}).get('skill_extraction', {}).get('max_text_length', 8000)
                if max_text_len > 0 and len(text) > max_text_len:
                    text = text[:max_text_len]
                
                # Extract the skills
                annotations = worker_extractor.annotate(text)
                
                skills = []
                # Full matches (faster, take priority)
                for match in annotations['results']['full_matches']:
                    skills.append(match['doc_node_value'])
                
                # N-gram matches (limited for performance)
                ngram_results = annotations.get('results', {}).get('ngram_scored', [])
                max_ngram = self.config.get('models', {}).get('skill_extraction', {}).get('max_ngram_results', 40)
                # Take the best N n-grams (for performance)
                if max_ngram > 0 and len(ngram_results) > max_ngram:
                    ngram_results = ngram_results[:max_ngram]
                for match in ngram_results:
                    skills.append(match['doc_node_value'])
                
                # Deduplicate and normalize (optimized)
                skills = list(set([s.lower() for s in skills]))
                
                # Filtrele
                skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
                
                return (idx, skills)
            except (IndexError, ValueError, KeyError):
                # Expected errors (normal)
                return (idx, [])
            except Exception as e:
                # Unexpected errors
                self.logger.debug(f"Skill extraction error (idx {idx}): {type(e).__name__}")
                return (idx, [])
        
        # Parallel processing - true parallelism with as_completed
        skill_lists = [None] * total
        error_count = 0
        completed = 0
        start_time = time.time()
        
        self.logger.info(f"🚀 Parallel skill extraction started: {total} texts, {max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks (they all start at once)
            future_to_idx = {
                executor.submit(process_single_text, (idx, text)): idx 
                for idx, text in enumerate(texts)
            }
            
            # Collect the results (as_completed - the first finished task is taken first)
            for future in as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                try:
                    idx, skills = future.result()
                    skill_lists[idx] = skills
                    completed += 1
                    
                    # Progress report (every 20 items or on completion - logging too often slows things down)
                    if completed % 20 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        remaining = total - completed
                        avg_time = elapsed / completed if completed > 0 else 0
                        estimated_remaining = avg_time * remaining / max_workers
                        self.logger.info(
                            f"📊 Progress: {completed}/{total} texts processed "
                            f"({completed*100//total}%) | "
                            f"Estimated remaining: {int(estimated_remaining//60)}m {int(estimated_remaining%60)}s | "
                            f"Parallel: {max_workers} threads"
                        )
                except Exception as e:
                    error_count += 1
                    skill_lists[original_idx] = []
                    self.logger.debug(f"Skill extraction error (idx {original_idx}): {type(e).__name__}")
        
        # Error summary
        if error_count > 0:
            self.logger.info(f"⚠️  Skill extraction error for {error_count}/{total} texts (normal; some texts cannot be processed)")
        
        df_result = df.copy()
        df_result["extracted_skills"] = skill_lists
        
        self.logger.info(f"✅ All texts processed in parallel ({max_workers} threads)")
        return df_result

