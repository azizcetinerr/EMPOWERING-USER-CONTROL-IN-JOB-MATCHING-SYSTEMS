#!/usr/bin/env python3
"""
EMPOWERING USER CONTROL IN JOB MATCHING SYSTEMS - Web Interface
Project matching system based on a CV link
"""

import os
# Fix the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import threading
import pandas as pd
import numpy as np

# Add the project root directory to the path (the parent directory of the web folder)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.pipeline import ProjectPipeline
from src.data_collection.cv_parser import EuropassCVParser
from src.skill_extraction.skill_extractor import SkillExtractorModule
from src.skill_extraction.vectorizer import SkillVectorizer
from src.scoring.matching_score import MatchingScoreCalculator
from src.scoring.sincerity_score import SincerityScoreCalculator
from src.scoring.reliability_score import ReliabilityScoreCalculator
from src.scoring.final_ranking import FinalRankingCalculator

# Start the Flask app together with the template folder
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'tubitak-project-secret-key-2024'  # Change this in production!

# Global variables
processing_status = {}
processing_results = {}

# Cache: project data and models
_cached_projects = None
_cached_models = {}
_cached_api_data = None  # Raw data from the API
_api_cache_file = None   # Cache file path
_config = None
_logger = None

def get_api_cache_file():
    """Return the API cache file path"""
    global _api_cache_file
    if _api_cache_file is None:
        _api_cache_file = project_root / "outputs" / "api_cache.json"
        _api_cache_file.parent.mkdir(parents=True, exist_ok=True)
    return _api_cache_file

def load_api_cache():
    """Load the API cache from disk"""
    import json
    cache_file = get_api_cache_file()
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        if _logger:
            _logger.warning(f"Could not read the API cache file: {e}")
        return None

def save_api_cache(api_data: dict, project_count: int):
    """Save the API data to the cache"""
    import json
    from datetime import datetime
    import hashlib
    
    cache_file = get_api_cache_file()
    
    # Take the hash of the API data (for change detection)
    api_data_str = json.dumps(api_data, sort_keys=True)
    data_hash = hashlib.md5(api_data_str.encode()).hexdigest()
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'project_count': project_count,
        'data_hash': data_hash,
        'api_data': api_data
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        if _logger:
            _logger.info(f"✅ API cache saved: {project_count} projects ({cache_file})")
        return True
    except Exception as e:
        if _logger:
            _logger.error(f"API cache kaydedilemedi: {e}")
        return False

def check_api_for_updates():
    """Compare the API data with the cached data and update if they differ"""
    global _cached_api_data, _config, _logger
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    # Load the cache
    cached_data = load_api_cache()
    
    # Fetch data from the API
    try:
        from src.data_collection.youth_portal_scraper import YouthPortalScraper
        scraper = YouthPortalScraper(_config, _logger)
        api_data = scraper.get_project_data_from_api()
        
        if api_data is None:
            _logger.warning("⚠️  Could not fetch data from the API, the cache will be used")
            if cached_data:
                _cached_api_data = cached_data.get('api_data')
            return False
        
        project_count = api_data.get('hits', {}).get('total', {}).get('value', 0)
        
        # If there is no cache, save directly
        if cached_data is None:
            _logger.info("📥 Creating the API cache for the first time...")
            save_api_cache(api_data, project_count)
            _cached_api_data = api_data
            return True
        
        # Hash comparison
        import json
        import hashlib
        api_data_str = json.dumps(api_data, sort_keys=True)
        current_hash = hashlib.md5(api_data_str.encode()).hexdigest()
        cached_hash = cached_data.get('data_hash')
        
        if current_hash != cached_hash:
            _logger.info(f"🔄 Change detected in the API data (cache: {cached_data.get('project_count')}, API: {project_count}), updating...")
            save_api_cache(api_data, project_count)
            _cached_api_data = api_data
            # Also invalidate the cached projects (they will be processed again)
            global _cached_projects
            _cached_projects = None
            return True
        else:
            _logger.info(f"✅ No change in the API data, using the cache ({project_count} projects)")
            _cached_api_data = cached_data.get('api_data')
            return False
            
    except Exception as e:
        _logger.error(f"API update check error: {e}", exc_info=True)
        # Use the cache on error
        if cached_data:
            _cached_api_data = cached_data.get('api_data')
        return False

def get_cached_projects():
    """Return or load the cached project data (checks CSV and pickle)"""
    global _cached_projects, _cached_api_data, _config, _logger
    
    # Check the API cache on the first load
    if _cached_api_data is None:
        check_api_for_updates()
    
    if _cached_projects is not None:
        return _cached_projects
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    from src.utils.config_loader import get_output_path
    
    # Check the CSV file first
    csv_projects_file = get_output_path(_config, 'project_details')
    ip1_projects_file = project_root / _config['output']['files']['projects_with_skills']
    
    # Use the pickle file if it exists (more complete)
    if ip1_projects_file.exists():
        try:
            _cached_projects = pd.read_pickle(ip1_projects_file)
            
            # Check and log the presence of skills
            if 'extracted_skills' in _cached_projects.columns:
                skills_count = _cached_projects['extracted_skills'].notna().sum()
                valid_skills = _cached_projects['extracted_skills'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                ).sum()
                _logger.info(f"✅ {len(_cached_projects)} projects loaded from pickle")
                _logger.info(f"   📊 {valid_skills}/{len(_cached_projects)} projects have valid skills (from cache)")
            else:
                _logger.warning(f"⚠️  No skill column in the pickle!")
            
            # Add vectors if they are missing
            if "skill_vectors" not in _cached_projects.columns:
                _logger.info("   No vectors, computing...")
                vectorizer = _get_model('vectorizer')
                _cached_projects = vectorizer.vectorize_dataframe(_cached_projects)
            else:
                _logger.info("   ✅ Vectors available (from cache)")
            
            return _cached_projects
        except Exception as e:
            _logger.warning(f"Could not read the pickle file: {e}")
    
    # Use the CSV file if it exists
    if csv_projects_file.exists():
        try:
            _cached_projects = pd.read_csv(csv_projects_file, encoding='utf-8-sig')
            _logger.info(f"✅ {len(_cached_projects)} projects loaded from CSV")
            # Extract the skills and compute the vectors
            if 'extracted_skills' not in _cached_projects.columns:
                skill_extractor = _get_model('skill_extractor')
                _cached_projects = skill_extractor.extract_skills_from_dataframe(_cached_projects)
            if "skill_vectors" not in _cached_projects.columns:
                vectorizer = _get_model('vectorizer')
                _cached_projects = vectorizer.vectorize_dataframe(_cached_projects)
            return _cached_projects
        except Exception as e:
            _logger.warning(f"Could not read the CSV file: {e}")
    
    return None

def get_cached_models():
    """Load the models lazily (only when needed)"""
    global _cached_models, _config, _logger
    
    if _cached_models:
        return _cached_models
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    # Lazy loading: load the models only when needed
    # Set to None at the start; they are loaded on first use
    _cached_models = {
        'vectorizer': None,
        'matching_calc': None,
        'sincerity_calc': None,
        'reliability_calc': None,
        'ranking_calc': None,
        'skill_extractor': None,
        '_config': _config,
        '_logger': _logger
    }
    
    return _cached_models

def _get_model(model_key: str):
    """Load a model lazily"""
    global _cached_models
    
    if _cached_models.get(model_key) is not None:
        return _cached_models[model_key]
    
    _config = _cached_models.get('_config')
    _logger = _cached_models.get('_logger')
    
    if _config is None or _logger is None:
        _config = load_config()
        _logger = setup_logger()
        _cached_models['_config'] = _config
        _cached_models['_logger'] = _logger
    
    # Load the model
    if model_key == 'vectorizer':
        _logger.info("Loading the vectorizer model...")
        _cached_models[model_key] = SkillVectorizer(_config, _logger)
    elif model_key == 'matching_calc':
        _logger.info("Loading the matching calculator...")
        _cached_models[model_key] = MatchingScoreCalculator(_config, _logger)
    elif model_key == 'sincerity_calc':
        _logger.info("Loading the sincerity calculator (this may take a while)...")
        _cached_models[model_key] = SincerityScoreCalculator(_config, _logger)
    elif model_key == 'reliability_calc':
        _logger.info("Loading the reliability calculator...")
        _cached_models[model_key] = ReliabilityScoreCalculator(_config, _logger)
    elif model_key == 'ranking_calc':
        _logger.info("Loading the ranking calculator...")
        _cached_models[model_key] = FinalRankingCalculator(_config, _logger)
    elif model_key == 'skill_extractor':
        _logger.info("Loading the skill extractor...")
        _cached_models[model_key] = SkillExtractorModule(_config, _logger)
    
    return _cached_models[model_key]

def _clean_description(text, max_length=200):
    """Cleans and shortens a description text"""
    if not text or not isinstance(text, str):
        return ''
    
    import re
    
    # Do the basic cleaning first
    cleaned = _clean_language_codes(text)
    
    # Extra cleaning: if the title and description are mixed up, find the first sentence
    # If the description is very short or looks like a title, find the first real sentence
    
    # Find the first sentence (ending with a period, question mark or exclamation mark)
    sentences = re.split(r'[.!?]+\s+', cleaned)
    if len(sentences) > 1:
        # Find the first meaningful sentence (at least 20 characters)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.lower().startswith(('apply', 'share', 'sign', 'join', 'activity', 'deadline')):
                cleaned = sentence
                break
    
    # Truncate to the maximum length, cutting at a word boundary
    if len(cleaned) > max_length:
        truncated = cleaned[:max_length - 3]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # Keep at least 70%
            return truncated[:last_space] + '...'
        return truncated + '...'
    
    return cleaned

def _clean_language_codes(text):
    """Removes language codes and unnecessary characters from a text, splits merged words, removes page texts and produces clean sentences"""
    if not text or not isinstance(text, str):
        return text if text else ''
    
    import re
    
    # 0. Remove page texts (such as Apply, Share this page, Sign in/Join)
    page_ui_texts = [
        r'\bApply\b', r'\bShare this page\b', r'\bSign in\s*/?\s*Join\b', 
        r'\bSign in\b', r'\bJoin the Corps\b', r'\bJoin\b.*Corps',
        r'\bLooking for participants\b', r'\bActivity dates\b', r'\bActivity location\b',
        r'\bActivity type\b', r'\bActivity topics\b', r'\bDeadline for applications\b',
        r'\bNo application deadline\b', r'\bPlease explain\b.*\bbox below\b',
        r'\bThe feedback collection\b.*\banonymous\b', r'\bPlease do not submit\b',
        r'\bDiscover more\b.*\boneuropa\.eu\b', r'\bIn order to check\b.*\bSign In\b'
    ]
    for pattern in page_ui_texts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 0.5. Remove broken characters (isolated characters such as Ų, ų)
    text = re.sub(r'[ŲųŪūŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ]+', '', text)
    # Isolated single-character Unicode characters (not regular letters)
    text = re.sub(r'\b[^\w\s]\s+', '', text)  # Isolated special characters at the start of a word
    text = re.sub(r'\s+[^\w\s]\b', '', text)  # Isolated special characters at the end of a word
    
    # 1. Remove language codes and the characters attached to them
    # Patterns such as BGбългарскиCSčeština (2-3 letter code + non-ASCII characters)
    text = re.sub(r'[A-Z]{2,3}[а-яА-ЯёЁa-záčďéěíňóřšťúůýžàáâãäåæ\u00e7èéêëìíîïðñòóôõ\u00f6øùúû\u00fcýþÿ\u0131\u011f\u015f\u00fc\u00f6\u00e7ąćęłńóśźżăî\u015fț]+', '', text)
    
    # Abbreviations such as DEL, EL + Greek characters (ελληνικά)
    text = re.sub(r'[A-Z]{2,3}[ελληνικάα-ωΑ-Ω]+', '', text)
    
    # Repeated upper-case codes (BGCSDADE...) - these are usually language selectors
    text = re.sub(r'\b[A-Z]{2,10}\b(?=[а-яА-ЯёЁa-záčďéěíňóřšťúůýžελληνικάα-ωΑ-Ωăî\u015fț])', '', text)
    
    # 1.5. Remove isolated upper-case codes (such as VZW, NGO, ESC - but these can be meaningful in context, be careful)
    # Remove only those at word boundaries surrounded by spaces
    text = re.sub(r'\s+[A-Z]{2,5}\s+', ' ', text)  # Isolated codes in the middle
    
    # 2. Unnecessary characters and codes left at the start/end
    text = re.sub(r'^[A-Z]{2,10}\s+', '', text)  # Upper-case codes at the start
    text = re.sub(r'\s+[A-Z]{2,10}$', '', text)  # Upper-case codes at the end
    text = re.sub(r'^[^\w]+\s*', '', text)  # Special characters at the start
    text = re.sub(r'\s*[^\w]+$', '', text)  # Special characters at the end
    
    # 3. Split merged words (camelCase, PascalCase and special cases)
    # First split merged capitalized words (e.g. "BordersGiosef" -> "Borders Giosef")
    text = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', text)
    # Add a space when an upper-case letter follows a lower-case letter (Descriptionvzw -> Description vzw)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add a space after single-letter words (such as A, I) (but keep abbreviations such as "AI", "IT")
    text = re.sub(r'\b([A-Z])([A-Z][a-z]{2,})', r'\1 \2', text)
    
    # 3.5. Special merged cases (such as Descriptionvzw, ApplyDescription)
    # Split merged texts that follow words such as Description, Apply
    text = re.sub(r'\b(Description|Apply|Share|Sign|Join|Activity|Deadline|Looking)([A-Z][a-z]+)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Description|Apply|Share|Sign|Join|Activity|Deadline|Looking)([a-z]{3,})', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 4. Remove extra whitespace and unnecessary characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text)  # Whitespace at the start/end
    
    # 5. Remove isolated single letters and special characters left at the start (such as E, D, L, Ų)
    text = re.sub(r'^[A-ZŲŪŮŰŴŶŸŹŻŽſ]\s+', '', text)
    
    # 6. Capitalize the start of the sentence (if it starts with a lower-case letter)
    if text and len(text) > 0:
        if text[0].islower():
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        # If it starts with a broken character, remove it
        elif ord(text[0]) > 127 and text[0] not in 'ÀÁÂÃÄÅÆ\u00c7ÈÉÊËÌÍÎÏÐÑÒÓÔÕ\u00d6ØÙÚÛ\u00dcÝÞßàáâãäåæ\u00e7èéêëìíîïðñòóôõ\u00f6øùúû\u00fcýþ':
            # If it is not a regular Unicode character, start from the second character
            if len(text) > 1:
                text = text[1:].strip()
                if text and text[0].islower():
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text.strip()

def _get_project_title(row):
    """Gets the project title (derives it from the description if project_name is missing)"""
    import re
    
    # Check project_name first (clean it if present)
    title = row.get('project_name')
    if title and str(title).strip() and str(title) != 'Project title not found':
        cleaned_title = _clean_language_codes(str(title).strip())
        # Use the cleaned title if it is meaningful (at least 5 characters)
        if len(cleaned_title) > 5:
            # Maximum 150 characters, do not cut the end of a sentence
            if len(cleaned_title) > 150:
                # Do not cut the last word, take up to the nearest space
                truncated = cleaned_title[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:  # Take at least 100 characters
                    return truncated[:last_space] + '...'
                return cleaned_title[:150]
            return cleaned_title
    
    # Take the first meaningful line from the description
    description = str(row.get('full_description', ''))
    if description:
        # Clean the full description first
        clean_desc = _clean_language_codes(description)
        
        # Take the first line
        first_line = clean_desc.split('\n')[0].strip()
        
        # Find the first meaningful sentence (ending with punctuation or 50+ characters)
        # Use the cleaned line if it is meaningful (more than 20 characters)
        if len(first_line) > 20:
            # Maximum 150 characters, do not cut the end of a sentence
            if len(first_line) > 150:
                truncated = first_line[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:
                    return truncated[:last_space] + '...'
                return first_line[:150]
            return first_line
        
        # Otherwise take the first meaningful part of the cleaned description
        if len(clean_desc) > 20:
            if len(clean_desc) > 150:
                truncated = clean_desc[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:
                    return truncated[:last_space] + '...'
                return clean_desc[:150]
            return clean_desc
    
    return 'No project title'

def run_pipeline_with_cv_url(cv_url: str, session_id: str):
    """Runs the pipeline with a CV URL (optimized version)"""
    global _config, _logger
    
    # Make sure the session ID is in processing_status
    if session_id not in processing_status:
        processing_status[session_id] = {
            'status': 'processing',
            'message': 'Starting...',
            'progress': 5
        }
    
    try:
        # Update the status
        processing_status[session_id].update({
            'status': 'processing',
            'message': 'Starting...',
            'progress': 5
        })
        
        # Load the config and logger
        if _config is None:
            _config = load_config()
        if _logger is None:
            _logger = setup_logger()
        
        # ✅ RESUME: checkpoint check - resume from where it stopped
        checkpoint_dir = project_root / "outputs" / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{session_id}.pkl"
        
        skip_to_ip3 = False
        if checkpoint_file.exists():
            try:
                import pickle
                _logger.info(f"🔄 Checkpoint found, resuming from where it stopped: {checkpoint_file}")
                
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                df_projects = checkpoint_data['df_projects']
                candidate_vectors = checkpoint_data['candidate_vectors']
                cv_text = checkpoint_data['cv_text']
                candidate_skills = checkpoint_data['candidate_skills']
                checkpoint_stage = checkpoint_data.get('checkpoint_stage', 'IP2_COMPLETED')
                
                _logger.info(f"✅ Checkpoint loaded: Stage={checkpoint_stage}, project count={len(df_projects)}")
                _logger.info(f"   Skipping IP1 and IP2, resuming from stage {checkpoint_stage}...")
                
                # IP1 and IP2 skipped, continue from IP3
                processing_status[session_id]['message'] = 'Checkpoint loaded, resuming from IP3...'
                processing_status[session_id]['progress'] = 70
                
                # Delete the checkpoint (used once)
                checkpoint_file.unlink()
                _logger.info(f"🗑️  Checkpoint used and deleted: {checkpoint_file}")
                
                # Go to IP3 (jump to the IP3 code below)
                skip_to_ip3 = True
                
            except Exception as e:
                _logger.warning(f"⚠️  Could not load the checkpoint: {e}, continuing with the normal pipeline")
                skip_to_ip3 = False
        
        if not skip_to_ip3:
            # Normal pipeline - start from IP1
            # Cache the models
            processing_status[session_id]['message'] = 'Loading models...'
            processing_status[session_id]['progress'] = 10
            models = get_cached_models()
            
            # Check the cached project data and the API
            processing_status[session_id]['message'] = 'Checking project data and the API...'
            processing_status[session_id]['progress'] = 15
            # API check (in the background, the cache is used)
            check_api_for_updates()
            df_projects = get_cached_projects()
            
            if df_projects is None:
                # No project data, run the full pipeline (cache-aware)
                processing_status[session_id]['message'] = 'Checking project data (existing data will be loaded from the cache)...'
                processing_status[session_id]['progress'] = 20
                pipeline = ProjectPipeline()
                pipeline.config['data_sources']['europass_cv']['url'] = cv_url
                # Pass the cached API data to the pipeline
                if _cached_api_data:
                    pipeline.config['_cached_api_data'] = _cached_api_data
                df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection()
                
                # Update the cache
                global _cached_projects
                _cached_projects = df_projects
            else:
                # Project data exists, give the preloaded projects to the pipeline
                # If skills and vectors exist, the pipeline only parses the CV
                _logger.info(f"🔍 get_cached_projects() returned: {len(df_projects)} projects")
                if 'extracted_skills' in df_projects.columns:
                    skills_count = df_projects['extracted_skills'].notna().sum()
                    _logger.info(f"   📊 {skills_count}/{len(df_projects)} projects have a skill column")
                if 'skill_vectors' in df_projects.columns:
                    vectors_count = df_projects['skill_vectors'].notna().sum()
                    _logger.info(f"   📊 {vectors_count}/{len(df_projects)} projects have a vector column")
                
                processing_status[session_id]['message'] = 'Project data loaded from the cache, analyzing the CV...'
                processing_status[session_id]['progress'] = 20
                pipeline = ProjectPipeline()
                pipeline.config['data_sources']['europass_cv']['url'] = cv_url
                # Pass the cached API data to the pipeline
                if _cached_api_data:
                    pipeline.config['_cached_api_data'] = _cached_api_data
                _logger.info(f"🚀 Passing preloaded_projects to the pipeline: {len(df_projects)} projects")
                df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection(preloaded_projects=df_projects)
            # Note: the preloaded_projects check is now done inside the pipeline
            # If the projects were loaded from the cache and skills/vectors exist,
            # the pipeline only parses the CV and skips skill extraction/vectorization
            
            # IP1 completed, the projects were saved to file
            # Now moving on to IP2
            _logger.info("✅ IP1 completed! Projects saved to file. Moving on to IP2...")
            
            # IP2 and IP4: compute the M and R scores in parallel
            processing_status[session_id]['message'] = 'IP2 & IP4: computing M and R scores in parallel (2 concurrent jobs)...'
            processing_status[session_id]['progress'] = 50
            
            from concurrent.futures import ThreadPoolExecutor
            
            matching_calc = _get_model('matching_calc')
            reliability_calc = _get_model('reliability_calc')
            
            def calculate_m_score():
                """Compute the M score"""
                return matching_calc.calculate_all_scores(df_projects.copy(), candidate_vectors)
            
            def calculate_r_score():
                """Compute the R score (independent of the M score)"""
                # Cache the CV paragraphs (do not parse them every time)
                cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
                # Truncate very long paragraphs (for performance)
                cv_paragraphs = [p[:500] for p in cv_paragraphs]  # Max 500 characters
                return reliability_calc.calculate_project_reliability_scores(
                    df_projects.copy(), 
                    cv_paragraphs,
                    skill_column="extracted_skills"
                )
            
            # Parallel computation (2 threads)
            with ThreadPoolExecutor(max_workers=2) as executor:
                m_future = executor.submit(calculate_m_score)
                r_future = executor.submit(calculate_r_score)
                
                # Wait for the results and merge them
                df_projects = m_future.result()
                df_projects_r = r_future.result()
                
                # Add the R score
                if 'R_Score' in df_projects_r.columns:
                    df_projects['R_Score'] = df_projects_r['R_Score']
            
            # ✅ CHECKPOINT: save the state after IP2 (to resume from where it stopped)
            try:
                checkpoint_dir = project_root / "outputs" / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a checkpoint file with the session ID
                checkpoint_file = checkpoint_dir / f"checkpoint_{session_id}.pkl"
                
                import pickle
                checkpoint_data = {
                    'df_projects': df_projects,
                    'candidate_vectors': candidate_vectors,
                    'cv_text': cv_text,
                    'candidate_skills': candidate_skills,
                    'checkpoint_stage': 'IP2_COMPLETED',  # Record which stage we are at
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                _logger.info(f"💾 Checkpoint saved: {checkpoint_file} (IP2 completed)")
            except Exception as e:
                _logger.warning(f"⚠️  Checkpoint kaydedilemedi: {e}")
                # Continue even if the checkpoint could not be saved
        
        # IP2 completed (from the normal pipeline or the checkpoint), go to IP3
        processing_status[session_id]['progress'] = 70
        processing_status[session_id]['message'] = 'IP2 & IP4 completed! Starting IP3...'
        
        # IP3: S score (after the M score, needed for ranking)
        processing_status[session_id]['message'] = 'IP3: computing the sincerity score (for all projects)...'
        processing_status[session_id]['progress'] = 75
        # Compute the S score for all projects (user request: no project without an S score)
        calculate_all = True  # Compute for all projects
        sincerity_calc = _get_model('sincerity_calc')  # Loading this model may take a while
        df_projects = sincerity_calc.calculate_all_scores(df_projects, calculate_all=calculate_all)
        
        # IP5: Final Ranking
        processing_status[session_id]['message'] = 'IP5: computing the final ranking...'
        processing_status[session_id]['progress'] = 85
        ranking_calc = _get_model('ranking_calc')
        df_final = ranking_calc.calculate_ranking(df_projects)
        
        processing_status[session_id]['message'] = 'Preparing the results...'
        processing_status[session_id]['progress'] = 95
        
        # Prepare the results (set the status even on error)
        try:
            results = {
                'total_projects': len(df_final),
                'top_projects': []
            }
            
            # Prepare the top 10 projects
            top_10 = df_final.head(10)
            for idx, row in top_10.iterrows():
                try:
                    # Extract the project ID from the link (last part)
                    link = row.get('link', '')
                    project_id = link.split('/')[-1].split('_')[0] if link else str(idx)
                    
                    project_data = {
                        'rank': idx + 1,
                        'project_id': project_id,
                        'link': link,
                        'title': _get_project_title(row),
                        'description': _clean_description(str(row.get('full_description', ''))),
                        'm_score': round(float(row.get('M_Score', 0)), 2),
                        's_score': round(float(row.get('S_Score')), 2) if pd.notna(row.get('S_Score')) else None,
                        'r_score': round(float(row.get('R_Score', 0)), 2),
                        'final_score': round(float(row.get('Fair_Score', 0)), 2),
                        'skills': row.get('extracted_skills', [])[:5] if isinstance(row.get('extracted_skills', []), list) else []
                    }
                    results['top_projects'].append(project_data)
                except Exception as e:
                    _logger.warning(f"Error preparing the result for project {idx}: {e}")
                    continue
            
            # Prepare all relevant projects (those with Fair_Score > 40, or the top 50)
            all_relevant = df_final[df_final['Fair_Score'] > 40].head(100)  # First 100 relevant projects
            results['all_relevant_projects'] = []
            for idx, row in all_relevant.iterrows():
                try:
                    # Extract the project ID from the link (last part)
                    link = row.get('link', '')
                    project_id = link.split('/')[-1].split('_')[0] if link else str(idx)
                    
                    project_data = {
                        'rank': idx + 1,
                        'project_id': project_id,
                        'link': link,
                        'title': _get_project_title(row),
                        'description': _clean_description(str(row.get('full_description', ''))),
                        'm_score': round(float(row.get('M_Score', 0)), 2),
                        's_score': round(float(row.get('S_Score')), 2) if pd.notna(row.get('S_Score')) else None,
                        'r_score': round(float(row.get('R_Score', 0)), 2),
                        'final_score': round(float(row.get('Fair_Score', 0)), 2),
                        'skills': row.get('extracted_skills', [])[:5] if isinstance(row.get('extracted_skills', []), list) else []
                    }
                    results['all_relevant_projects'].append(project_data)
                except Exception as e:
                    _logger.warning(f"Error preparing the result for project {idx}: {e}")
                    continue
            
            # Statistics
            results['stats'] = {
                'm_score': {
                    'min': round(float(df_final['M_Score'].min()), 2),
                    'max': round(float(df_final['M_Score'].max()), 2),
                    'avg': round(float(df_final['M_Score'].mean()), 2)
                },
                's_score': {
                    'min': round(float(df_final['S_Score'].min()), 2) if pd.notna(df_final['S_Score'].min()) else None,
                    'max': round(float(df_final['S_Score'].max()), 2) if pd.notna(df_final['S_Score'].max()) else None,
                    'avg': round(float(df_final['S_Score'].mean()), 2) if pd.notna(df_final['S_Score'].mean()) else None
                },
                'r_score': {
                    'min': round(float(df_final['R_Score'].min()), 2),
                    'max': round(float(df_final['R_Score'].max()), 2),
                    'avg': round(float(df_final['R_Score'].mean()), 2)
                },
                'final_score': {
                    'min': round(float(df_final['Fair_Score'].min()), 2),
                    'max': round(float(df_final['Fair_Score'].max()), 2),
                    'avg': round(float(df_final['Fair_Score'].mean()), 2)
                }
            }
            
            processing_results[session_id] = results
        except Exception as e:
            _logger.error(f"Result preparation error: {e}", exc_info=True)
            # The results could not be prepared but the pipeline completed; return a simple result
            processing_results[session_id] = {
                'total_projects': len(df_final),
                'top_projects': [],
                'stats': {},
                'warning': 'The results could not be fully prepared, but the analysis completed.'
            }
        
        # ALWAYS set the status (whether or not the results were prepared)
        processing_status[session_id]['status'] = 'completed'
        processing_status[session_id]['message'] = 'Completed!'
        processing_status[session_id]['progress'] = 100
        _logger.info(f"✅ Pipeline completed successfully (session: {session_id}, project count: {len(df_final)})")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        _logger.error(f"❌ Pipeline error (session: {session_id}): {error_trace}")
        
        # Make sure the session ID is in processing_status
        if session_id not in processing_status:
            processing_status[session_id] = {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'progress': 0
            }
        else:
            processing_status[session_id].update({
                'status': 'error',
                'message': f'Error: {str(e)}',
                'progress': 0
            })
        processing_results[session_id] = {'error': str(e)}
    finally:
        # Make sure the status is set in every case
        if session_id in processing_status:
            final_status = processing_status[session_id].get('status', 'unknown')
            _logger.info(f"🔍 Final status check (session: {session_id}): {final_status}, progress: {processing_status[session_id].get('progress', 0)}")
            
            # If the status is still 'processing' or 'queued' and progress is 95+, it probably completed but the status was not set
            if final_status in ['processing', 'queued'] and processing_status[session_id].get('progress', 0) >= 95:
                _logger.warning(f"⚠️  Status not marked as 'completed' but progress is 95%+. Fixing...")
                processing_status[session_id]['status'] = 'completed'
                processing_status[session_id]['message'] = 'Completed! (automatic correction)'
                processing_status[session_id]['progress'] = 100


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_cv():
    """Take the CV URL and start processing"""
    data = request.get_json()
    cv_url = data.get('cv_url', '').strip()
    
    if not cv_url:
        return jsonify({'error': 'CV URL is required'}), 400
    
    # Create the session ID
    import uuid
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # Initialize the processing status (before the thread starts)
    processing_status[session_id] = {
        'status': 'queued',
        'message': 'Queued...',
        'progress': 0
    }
    
    # Run the pipeline in the background
    try:
        thread = threading.Thread(
            target=run_pipeline_with_cv_url,
            args=(cv_url, session_id),
            name=f"Pipeline-{session_id[:8]}"
        )
        thread.daemon = True
        thread.start()
        
        # Verify that the thread started
        if thread.is_alive():
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Processing started'
            })
        else:
            # Error if the thread did not start
            processing_status[session_id] = {
                'status': 'error',
                'message': 'Thread could not be started',
                'progress': 0
            }
            return jsonify({
                'error': 'Processing could not be started'
            }), 500
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'message': f'Error: {str(e)}',
            'progress': 0
        }
        return jsonify({
            'error': f'Processing could not be started: {str(e)}'
        }), 500


@app.route('/status/<session_id>')
def get_status(session_id):
    """Return the processing status"""
    if session_id not in processing_status:
        return jsonify({'error': 'Session not found'}), 404
    
    status = processing_status[session_id].copy()
    
    # Automatic correction: if progress is 95+ and the status is still 'processing'
    if status.get('status') in ['processing', 'queued'] and status.get('progress', 0) >= 95:
        # If there are results, it probably completed
        if session_id in processing_results:
            status['status'] = 'completed'
            status['message'] = 'Completed! (automatic correction)'
            status['progress'] = 100
            # Also update the global status
            processing_status[session_id].update(status)
            _logger.info(f"🔧 Status corrected automatically (session: {session_id})")
    
    # If it completed, also add the results
    if status['status'] == 'completed' and session_id in processing_results:
        status['results'] = processing_results[session_id]
    
    return jsonify(status)


@app.route('/results/<session_id>')
def show_results(session_id):
    """Show the results"""
    if session_id not in processing_results:
        return redirect(url_for('index'))
    
    results = processing_results[session_id]
    return render_template('results.html', results=results, session_id=session_id)


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Save user feedback"""
    try:
        data = request.get_json()
        project_id = data.get('project_id')
        feedback_type = data.get('feedback_type')  # 'positive' or 'negative'
        session_id = data.get('session_id')
        
        if not all([project_id, feedback_type, session_id]):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400
        
        if feedback_type not in ['positive', 'negative']:
            return jsonify({'success': False, 'error': 'Invalid feedback type'}), 400
        
        # Feedback file path
        feedback_file = project_root / "user_feedback.json"
        
        # Load the existing feedback
        feedbacks = []
        if feedback_file.exists():
            try:
                import json
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
            except Exception as e:
                _logger.warning(f"Could not read the feedback file: {e}")
                feedbacks = []
        
        # Find the project link (from the session)
        project_link = ''
        if session_id in processing_results:
            # Search in the top 10 and all_relevant_projects
            for project in processing_results[session_id].get('top_projects', []):
                if str(project.get('project_id', '')) == str(project_id):
                    project_link = project.get('link', '')
                    break
            if not project_link:
                for project in processing_results[session_id].get('all_relevant_projects', []):
                    if str(project.get('project_id', '')) == str(project_id):
                        project_link = project.get('link', '')
                        break
        
        # Add the new feedback
        from datetime import datetime
        new_feedback = {
            'project_id': project_id,
            'project_link': project_link,
            'feedback_type': feedback_type,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        feedbacks.append(new_feedback)
        
        # Save the feedback
        import json
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
        
        _logger.info(f"✅ Feedback saved: {project_id} - {feedback_type} (session: {session_id})")
        
        return jsonify({
            'success': True,
            'message': 'Your feedback has been saved'
        })
        
    except Exception as e:
        _logger.error(f"Feedback saving error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Feedback could not be saved: {str(e)}'
        }), 500


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the cache (to reload the project data)"""
    global _cached_projects, _cached_models
    _cached_projects = None
    _cached_models = {}
    return jsonify({'success': True, 'message': 'Cache cleared'})


@app.route('/cache-status')
def cache_status():
    """Check the cache status"""
    global _cached_projects, _cached_models
    
    ip1_file = project_root / "IP1_output_listings_with_skills.pkl"
    has_ip1 = ip1_file.exists()
    has_cached_projects = _cached_projects is not None
    has_cached_models = len(_cached_models) > 0
    
    return jsonify({
        'ip1_file_exists': has_ip1,
        'projects_cached': has_cached_projects,
        'models_cached': has_cached_models,
        'message': 'Fast mode active' if (has_ip1 and has_cached_models) else 'Slow mode (first run)'
    })


if __name__ == '__main__':
    import os
    
    # Read the port from an environment variable or use 5001 by default
    port = int(os.environ.get('PORT', 5001))
    
    print("=" * 80)
    print("EMPOWERING USER CONTROL IN JOB MATCHING SYSTEMS - Starting the web interface...")
    print("=" * 80)
    
    # Load the models lazily (only when needed)
    print("\nModels will be loaded lazily (only when needed)")
    print("   This makes the startup much faster")
    try:
        # Only prepare the model structure, do not load the models
        get_cached_models()
        print("✅ Model structure prepared (models will be loaded when needed)")
    except Exception as e:
        print(f"⚠️  Error while preparing the model structure: {e}")
    
    # Check and load the API cache
    print("\n📥 Checking the API cache...")
    try:
        check_api_for_updates()
        cache_data = load_api_cache()
        if cache_data:
            print(f"✅ API cache loaded: {cache_data.get('project_count', 0)} projects")
            print(f"   Last update: {cache_data.get('timestamp', 'Unknown')}")
        else:
            print("⚠️  API cache not found, it will be created on the first request")
    except Exception as e:
        print(f"⚠️  API cache check error: {e}")
    
    # Check the IP1 outputs
    ip1_file = project_root / "IP1_output_listings_with_skills.pkl"
    if ip1_file.exists():
        print("✅ IP1 outputs found - fast mode active")
        print("   (Project scraping will be skipped, only the CV analysis will run)")
    else:
        print("⚠️  IP1 outputs not found - slow mode")
        print("   (The first analysis may take long; project data will be collected)")
        print("   Tip: first create the IP1 outputs by running 'python src/pipeline.py'")
    
    print(f"\nOpen http://localhost:{port} in your browser")
    print("Press Ctrl+C to quit\n")
    
    # Run in production mode (debug=False) - no automatic restart on file changes
    # Debug mode can interrupt the pipeline threads and lose sessions
    # Note: gunicorn should be used in production; app.run() is only for development
    app.run(debug=False, host='0.0.0.0', port=port)
