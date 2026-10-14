"""
Main pipeline - runs all steps in order
"""

import sys
from pathlib import Path

# Add the project root directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config, get_output_path
from src.utils.logger import setup_logger
from src.data_collection.youth_portal_scraper import YouthPortalScraper
from src.data_collection.project_detail_scraper import ProjectDetailScraper
from src.data_collection.cv_parser import EuropassCVParser
from src.skill_extraction.skill_extractor import SkillExtractorModule
from src.skill_extraction.vectorizer import SkillVectorizer
from src.scoring.matching_score import MatchingScoreCalculator
from src.scoring.sincerity_score import SincerityScoreCalculator
from src.scoring.reliability_score import ReliabilityScoreCalculator
from src.scoring.final_ranking import FinalRankingCalculator

import pandas as pd
import numpy as np
from pathlib import Path


class ProjectPipeline:
    """Main pipeline class"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to the configuration file (if None, the default)
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            name="pipeline",
            log_level=self.config['logging']['level'],
            log_file=self.config['logging']['file'],
            console=self.config['logging']['console']
        )
        
        # Create the output directories
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Creates the output directories"""
        project_root = Path(__file__).resolve().parent.parent
        output_config = self.config.get('output', {})
        
        dirs = [
            output_config.get('base_dir', 'outputs'),
            output_config.get('raw_data_dir', 'data/raw'),
            output_config.get('processed_data_dir', 'data/processed'),
            output_config.get('results_dir', 'outputs/results'),
            'logs'
        ]
        
        for dir_path in dirs:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_ip1_data_collection(self, preloaded_projects=None):
        """
        IP1: Data collection stage (cache-aware)
        
        Args:
            preloaded_projects: Preloaded project DataFrame (optional, from the cache)
        """
        self.logger.info("=" * 80)
        self.logger.info("IP1: DATA COLLECTION AND STANDARDIZATION STARTED")
        self.logger.info("=" * 80)
        
        project_root = Path(__file__).parent.parent
        
        # If preloaded projects exist and have skills, use them
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            self.logger.info(f"🔍 Starting preloaded projects check: {len(preloaded_projects)} projects")
            
            # Check the presence of skills in detail
            has_skills = False
            has_vectors = False
            skills_info = ""
            vectors_info = ""
            
            if 'extracted_skills' in preloaded_projects.columns:
                non_null_skills = preloaded_projects['extracted_skills'].notna()
                non_null_count = non_null_skills.sum()
                if non_null_count > 0:
                    valid_skills = preloaded_projects[non_null_skills]['extracted_skills'].apply(
                        lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                    )
                    valid_count = valid_skills.sum()
                    has_skills = valid_count > 0
                    skills_info = f"{valid_count}/{len(preloaded_projects)} projects have valid skills"
                else:
                    skills_info = "Skill column exists but is empty for all projects"
            else:
                skills_info = "No skill column"
            
            if 'skill_vectors' in preloaded_projects.columns:
                non_null_vectors = preloaded_projects['skill_vectors'].notna()
                vector_count = non_null_vectors.sum()
                has_vectors = vector_count > 0
                vectors_info = f"{vector_count}/{len(preloaded_projects)} projects have vectors"
            else:
                vectors_info = "No vector column"
            
            self.logger.info(f"🔍 Preloaded projects check: {len(preloaded_projects)} projects")
            self.logger.info(f"   Skills: {has_skills} ({skills_info})")
            self.logger.info(f"   Vectors: {has_vectors} ({vectors_info})")
            
            if has_skills and has_vectors:
                self.logger.info(f"✅ Using {len(preloaded_projects)} preloaded projects (skills and vectors available)")
                self.logger.info("   📊 Checking for new projects...")
                
                # CV parse et
                cv_parser = EuropassCVParser(self.config, self.logger)
                cv_text, cv_skills = cv_parser.parse_cv()
                
                # Extract skills from the CV
                skill_extractor = SkillExtractorModule(self.config, self.logger)
                candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
                
                # Compute the candidate vectors
                vectorizer = SkillVectorizer(self.config, self.logger)
                candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
                
                # Read the latest project publication date from the metadata
                metadata_file = project_root / "data/external/last_scrape_metadata.json"
                last_published_date = None
                if metadata_file.exists():
                    try:
                        import json
                        from datetime import datetime
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            last_published_date_str = metadata.get('last_published_date')
                            if last_published_date_str:
                                last_published_date = datetime.fromisoformat(last_published_date_str)
                                self.logger.info(f"📅 Latest project publication date: {last_published_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        self.logger.warning(f"Could not read metadata: {e}")
                
                # Check for new projects and add them
                df_final_projects = preloaded_projects.copy()
                existing_links = set(preloaded_projects['link'].tolist()) if 'link' in preloaded_projects.columns else set()
                
                # Fetch the project links
                scraper = YouthPortalScraper(self.config, self.logger)
                cached_api_data = self.config.get('_cached_api_data')
                all_links = scraper.get_project_links(cached_api_data=cached_api_data)
                new_links = [link for link in all_links if link not in existing_links]
                
                if new_links:
                    self.logger.info(f"🆕 Found {len(new_links)} new projects, fetching details...")
                    
                    # Fetch details of the new projects (use the parallel version)
                    try:
                        from src.data_collection.project_detail_scraper_parallel import ProjectDetailScraperParallel
                        max_workers = self.config.get('performance', {}).get('num_workers', 5)
                        detail_scraper = ProjectDetailScraperParallel(self.config, self.logger, max_workers=max_workers)
                        self.logger.info(f"   Parallel scraping active: {max_workers} threads")
                    except ImportError:
                        # Fallback: Normal scraper
                        self.logger.info("   Parallel scraper not found, using the regular scraper")
                        detail_scraper = ProjectDetailScraper(self.config, self.logger)
                    df_new_projects = detail_scraper.scrape_all_projects(new_links)
                    
                    if len(df_new_projects) > 0:
                        # Extract skills for the new projects
                        self.logger.info(f"   🔧 Extracting skills for {len(df_new_projects)} new projects...")
                        df_new_projects = skill_extractor.extract_skills_from_dataframe(df_new_projects)
                        
                        # Vectorize the new projects
                        self.logger.info(f"   🔧 Vectorizing {len(df_new_projects)} new projects...")
                        df_new_projects = vectorizer.vectorize_dataframe(df_new_projects)
                        
                        # Merge the old and new projects
                        df_final_projects = pd.concat([df_final_projects, df_new_projects], ignore_index=True)
                        self.logger.info(f"   ✅ {len(df_new_projects)} new projects added, total: {len(df_final_projects)} projects")
                else:
                    self.logger.info("   ✅ No new projects, using the existing projects")
                
                # Filter out projects whose application deadline has passed
                from datetime import datetime
                if 'application_deadline' in df_final_projects.columns:
                    df_final_projects['application_deadline'] = pd.to_datetime(df_final_projects['application_deadline'], errors='coerce')
                    now = datetime.now()
                    expired_projects = df_final_projects[df_final_projects['application_deadline'].notna() & (df_final_projects['application_deadline'] < now)]
                    if len(expired_projects) > 0:
                        self.logger.info(f"🗑️  {len(expired_projects)} projects are past their application deadline, removing...")
                        df_final_projects = df_final_projects[df_final_projects['application_deadline'].isna() | (df_final_projects['application_deadline'] >= now)]
                        self.logger.info(f"   ✅ {len(df_final_projects)} active projects remain")
                
                # Save the projects to file (current state)
                projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
                projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
                df_final_projects.to_pickle(projects_vectors_path)
                df_final_projects.to_pickle(projects_skills_path)
                
                csv_projects_file = get_output_path(self.config, 'project_details')
                df_final_projects.to_csv(csv_projects_file, index=False, encoding='utf-8-sig')
                
                # Update the metadata (latest addition date and publication date)
                try:
                    import json
                    from datetime import datetime
                    last_added_date = datetime.now()  # Date the latest projects were added
                    last_published_date = None
                    
                    if 'published_date' in df_final_projects.columns:
                        df_final_projects['published_date'] = pd.to_datetime(df_final_projects['published_date'], errors='coerce')
                        published_projects = df_final_projects[df_final_projects['published_date'].notna()]
                        if len(published_projects) > 0:
                            last_published_date = published_projects['published_date'].max()
                    
                    metadata = {
                        'last_added_date': last_added_date.isoformat(),  # Date the latest project was added
                        'last_published_date': last_published_date.isoformat() if last_published_date is not None and pd.notna(last_published_date) else None,
                        'last_scrape_date': datetime.now().isoformat(),
                        'last_scrape_count': len(new_links) if new_links else 0,
                        'total_projects': len(df_final_projects),
                        'last_update': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"📅 Metadata updated: last addition {last_added_date.strftime('%Y-%m-%d %H:%M:%S')}, latest publication date {last_published_date.strftime('%Y-%m-%d') if last_published_date is not None and pd.notna(last_published_date) else 'N/A'} ({len(new_links) if new_links else 0} new projects)")
                except Exception as e:
                    self.logger.warning(f"Could not update metadata: {e}")
                
                self.logger.info("=" * 80)
                self.logger.info("✅ IP1 COMPLETED! Projects saved to file.")
                self.logger.info(f"   📁 CSV: {csv_projects_file}")
                self.logger.info(f"   📁 Pickle (skills): {projects_skills_path}")
                self.logger.info(f"   📁 Pickle (vectors): {projects_vectors_path}")
                self.logger.info(f"   📊 {len(df_final_projects)} projects saved in total ({len(new_links) if new_links else 0} newly added)")
                self.logger.info("=" * 80)
                
                return df_final_projects, candidate_vectors, cv_text, candidate_skills
            else:
                self.logger.info(f"ℹ️  Preloaded projects exist but skills/vectors are missing")
                self.logger.info(f"   has_skills={has_skills}, has_vectors={has_vectors}")
                self.logger.info("   The normal pipeline will run (scraping/skill extraction only for new projects)")
                # Use preloaded_projects as df_existing_projects and continue with the normal pipeline
                df_existing_projects = preloaded_projects.copy()
                # Collect the existing links
                if 'link' in df_existing_projects.columns:
                    existing_links = set(df_existing_projects['link'].tolist())
                else:
                    existing_links = set()
                new_links = []  # Empty at the start
                # Read the metadata file (latest project publication date - takes priority)
                metadata_file = project_root / "data/external/last_scrape_metadata.json"
                last_scrape_date = None
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            last_published_date_str = metadata.get('last_published_date')
                            if last_published_date_str:
                                from datetime import datetime
                                last_published_date = datetime.fromisoformat(last_published_date_str)
                                self.logger.info(f"📅 Latest project publication date: {last_published_date.strftime('%Y-%m-%d')} (cache active)")
                                last_scrape_date = last_published_date
                            else:
                                last_scrape_date_str = metadata.get('last_scrape_date')
                                if last_scrape_date_str:
                                    from datetime import datetime
                                    last_scrape_date = datetime.fromisoformat(last_scrape_date_str)
                                    self.logger.info(f"📅 Latest scraping date: {last_scrape_date.strftime('%Y-%m-%d %H:%M:%S')} (fallback)")
                    except Exception as e:
                        self.logger.warning(f"Could not read the metadata file: {e}")
                # Continue with the normal pipeline but use df_existing_projects
                # (the code below continues)
        
        # If the preloaded_projects check succeeded, execution does not reach here (returned above)
        # If preloaded_projects is missing or incomplete, the normal pipeline runs
        
        # Metadata file (for the latest scraping date)
        if preloaded_projects is None:
            metadata_file = project_root / "data/external/last_scrape_metadata.json"
            last_scrape_date = None
            
        # Check the existing project data (both CSV and pickle)
        # If preloaded_projects exists, use it
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            # Use the preloaded projects as df_existing_projects
            df_existing_projects = preloaded_projects.copy()
            if 'link' in df_existing_projects.columns:
                existing_links = set(df_existing_projects['link'].tolist())
            else:
                existing_links = set()
            new_links = []  # Empty at the start
            self.logger.info(f"✅ Using preloaded projects: {len(df_existing_projects)} projects, {len(existing_links)} links")
        else:
            # Normal flow: load from CSV and pickle
            ip1_projects_file = project_root / self.config['output']['files']['projects_with_skills']
            csv_projects_file = get_output_path(self.config, 'project_details')
            df_existing_projects = None
            existing_links = set()
            new_links = []  # Empty at the start
        
        # Read the metadata file (latest project publication date - takes priority)
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Check published_date first (more accurate - actual publication date)
                    last_published_date_str = metadata.get('last_published_date')
                    if last_published_date_str:
                        from datetime import datetime
                        last_published_date = datetime.fromisoformat(last_published_date_str)
                        self.logger.info(f"📅 Latest project publication date: {last_published_date.strftime('%Y-%m-%d')} (cache active)")
                        last_scrape_date = last_published_date  # Use published_date (more accurate)
                    else:
                        # Fallback: scraping date
                        last_scrape_date_str = metadata.get('last_scrape_date')
                        if last_scrape_date_str:
                            from datetime import datetime
                            last_scrape_date = datetime.fromisoformat(last_scrape_date_str)
                            self.logger.info(f"📅 Latest scraping date: {last_scrape_date.strftime('%Y-%m-%d %H:%M:%S')} (fallback)")
            except Exception as e:
                self.logger.warning(f"Could not read the metadata file: {e}")
        
        # Check the CSV file first (faster) - only if preloaded_projects is missing
        if preloaded_projects is None and csv_projects_file.exists():
            try:
                df_csv = pd.read_csv(csv_projects_file, encoding='utf-8-sig')
                if 'link' in df_csv.columns:
                    # If date-based filtering is available, take the projects before that date
                    if last_scrape_date:
                        # Check published_date first (more accurate)
                        if 'published_date' in df_csv.columns:
                            df_csv['published_date'] = pd.to_datetime(df_csv['published_date'], errors='coerce')
                            # Filter the projects before that date
                            df_old = df_csv[df_csv['published_date'] < last_scrape_date]
                            df_new = df_csv[df_csv['published_date'] >= last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Loaded {len(df_old)} old projects from CSV (publication-date cache: < {last_scrape_date.strftime('%Y-%m-%d')})")
                                # Use the old projects
                                df_existing_projects = df_old
                                existing_links.update(set(df_old['link'].tolist()))
                            
                            if len(df_new) > 0:
                                self.logger.info(f"ℹ️  CSV has {len(df_new)} new projects (publication date >= {last_scrape_date.strftime('%Y-%m-%d')}, will be scraped again)")
                        elif 'scraped_at' in df_csv.columns:
                            # Fallback: use scraped_at
                            df_csv['scraped_at'] = pd.to_datetime(df_csv['scraped_at'], errors='coerce')
                            df_old = df_csv[df_csv['scraped_at'] < last_scrape_date]
                            df_new = df_csv[df_csv['scraped_at'] >= last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Loaded {len(df_old)} old projects from CSV (scraping-date cache)")
                                df_existing_projects = df_old
                                existing_links.update(set(df_old['link'].tolist()))
                            
                            if len(df_new) > 0:
                                self.logger.info(f"ℹ️  CSV has {len(df_new)} new projects (will be scraped again)")
                        else:
                            # No date information, take all projects
                            csv_links = set(df_csv['link'].tolist())
                            existing_links.update(csv_links)
                            self.logger.info(f"✅ Loaded {len(csv_links)} projects from CSV (no date information)")
                            df_existing_projects = df_csv
                    else:
                        # No date information, take all projects
                        csv_links = set(df_csv['link'].tolist())
                        existing_links.update(csv_links)
                        self.logger.info(f"✅ Loaded {len(csv_links)} projects from CSV")
                        df_existing_projects = df_csv
            except Exception as e:
                self.logger.warning(f"Could not read the CSV file: {e}")
        
        # Then check the pickle file (for skills and vectors) - only if preloaded_projects is missing
        if preloaded_projects is None and ip1_projects_file.exists():
            try:
                df_pickle = pd.read_pickle(ip1_projects_file)
                if 'link' in df_pickle.columns:
                    # Date-based filtering (published_date first, then scraped_at)
                    if last_scrape_date:
                        # Check published_date first (more accurate)
                        if 'published_date' in df_pickle.columns:
                            df_pickle['published_date'] = pd.to_datetime(df_pickle['published_date'], errors='coerce')
                            df_old = df_pickle[df_pickle['published_date'] < last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Loaded {len(df_old)} old projects from pickle (publication-date cache: < {last_scrape_date.strftime('%Y-%m-%d')})")
                                # The pickle is more complete (has skills and vectors), use it
                                df_existing_projects = df_old.copy()  # Make a copy
                                existing_links.update(set(df_old['link'].tolist()))
                                # Check and log the presence of skills
                                if 'extracted_skills' in df_existing_projects.columns:
                                    skills_count = df_existing_projects['extracted_skills'].notna().sum()
                                    self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projects have skills (from cache)")
                                else:
                                    self.logger.warning(f"   ⚠️  No skill column in the pickle!")
                        elif 'scraped_at' in df_pickle.columns:
                            # Fallback: use scraped_at
                            df_pickle['scraped_at'] = pd.to_datetime(df_pickle['scraped_at'], errors='coerce')
                            df_old = df_pickle[df_pickle['scraped_at'] < last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Loaded {len(df_old)} old projects from pickle (scraping-date cache)")
                                df_existing_projects = df_old.copy()  # Make a copy
                                existing_links.update(set(df_old['link'].tolist()))
                                # Check and log the presence of skills
                                if 'extracted_skills' in df_existing_projects.columns:
                                    skills_count = df_existing_projects['extracted_skills'].notna().sum()
                                    self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projects have skills (from cache)")
                                else:
                                    self.logger.warning(f"   ⚠️  No skill column in the pickle!")
                        else:
                            # No date information, take all projects
                            pickle_links = set(df_pickle['link'].tolist())
                            existing_links.update(pickle_links)
                            self.logger.info(f"✅ Loaded {len(pickle_links)} projects from pickle (no date information)")
                            df_existing_projects = df_pickle.copy()  # Make a copy
                            # Check and log the presence of skills
                            if 'extracted_skills' in df_existing_projects.columns:
                                skills_count = df_existing_projects['extracted_skills'].notna().sum()
                                self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projects have skills (from cache)")
                            else:
                                self.logger.warning(f"   ⚠️  No skill column in the pickle!")
                    else:
                        # No date information, take all projects
                        pickle_links = set(df_pickle['link'].tolist())
                        existing_links.update(pickle_links)
                        self.logger.info(f"✅ Loaded {len(pickle_links)} projects from pickle")
                        df_existing_projects = df_pickle.copy()  # Make a copy
                        # Check and log the presence of skills
                        if 'extracted_skills' in df_existing_projects.columns:
                            skills_count = df_existing_projects['extracted_skills'].notna().sum()
                            self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projects have skills (from cache)")
                        else:
                            self.logger.warning(f"   ⚠️  No skill column in the pickle!")
            except Exception as e:
                self.logger.warning(f"Could not read the pickle file: {e}")
        
        if len(existing_links) > 0:
            self.logger.info(f"✅ Found {len(existing_links)} existing projects in total (date-based cache)")
        
        # 1. Fetch the project links (only if needed for new projects)
        if preloaded_projects is None or len(new_links) == 0:
            # Normal flow: fetch the links
            self.logger.info("Step 1: Fetching project links...")
            scraper = YouthPortalScraper(self.config, self.logger)
            # Use cached API data from the config if available
            cached_api_data = self.config.get('_cached_api_data')  # Temporary config variable
            all_links = scraper.get_project_links(cached_api_data=cached_api_data)
            
            # Find the new links
            if 'existing_links' not in locals():
                existing_links = set()
            new_links = [link for link in all_links if link not in existing_links]
        else:
            # Preloaded projects exist; only check for new links
            self.logger.info("Step 1: Checking project links (from cache)...")
            scraper = YouthPortalScraper(self.config, self.logger)
            cached_api_data = self.config.get('_cached_api_data')
            all_links = scraper.get_project_links(cached_api_data=cached_api_data)
            new_links = [link for link in all_links if link not in existing_links]
        
        self.logger.info(f"   Found {len(all_links)} project links in total")
        self.logger.info(f"   Existing: {len(existing_links)}, New: {len(new_links)}")
        
        # Save the links
        output_file = self.config['data_sources']['youth_portal']['output_file']
        links_file = project_root / output_file
        with open(links_file, 'w', encoding='utf-8') as f:
            for link in all_links:
                f.write(link + "\n")
        self.logger.info(f"Links saved: {links_file}")
        
        # 2. Fetch the project details (only the new ones) - parallel version
        if new_links:
            self.logger.info(f"Step 2: Fetching details of {len(new_links)} new projects in parallel...")
            # Use parallel scraping (faster)
            try:
                from src.data_collection.project_detail_scraper_parallel import ProjectDetailScraperParallel
                max_workers = self.config.get('performance', {}).get('num_workers', 5)
                detail_scraper = ProjectDetailScraperParallel(self.config, self.logger, max_workers=max_workers)
                self.logger.info(f"   Parallel scraping active: {max_workers} threads")
            except ImportError:
                # Fallback: Normal scraper
                self.logger.info("   Parallel scraper not found, using the regular scraper")
                detail_scraper = ProjectDetailScraper(self.config, self.logger)
            df_new_projects = detail_scraper.scrape_all_projects(new_links)
            
            # Merge the old and new projects
            if df_existing_projects is not None and len(df_existing_projects) > 0:
                # Align the columns (for published_date and scraped_at)
                all_columns = set(df_existing_projects.columns) | set(df_new_projects.columns)
                
                # Add the missing columns
                for col in all_columns:
                    if col not in df_existing_projects.columns:
                        df_existing_projects[col] = None
                    if col not in df_new_projects.columns:
                        if col == 'scraped_at':
                            from datetime import datetime
                            df_new_projects[col] = datetime.now().isoformat()
                        else:
                            df_new_projects[col] = None
                
                # Put the columns in the same order
                df_existing_projects = df_existing_projects.reindex(columns=sorted(all_columns))
                df_new_projects = df_new_projects.reindex(columns=sorted(all_columns))
                
                df_projects = pd.concat([df_existing_projects, df_new_projects], ignore_index=True)
                self.logger.info(f"   ✅ {len(df_existing_projects)} old + {len(df_new_projects)} new = {len(df_projects)} projects in total")
            else:
                df_projects = df_new_projects
                self.logger.info(f"   ✅ {len(df_projects)} new projects added")
        else:
            self.logger.info("Step 2: No new projects, using the existing data")
            if df_existing_projects is not None:
                df_projects = df_existing_projects.copy()  # Copy so that the original is not modified
                # If there is no published_date column, it will be added on the next scraping
                if 'published_date' not in df_projects.columns:
                    self.logger.info(f"   ℹ️  Using {len(df_projects)} existing projects (no published_date; it will be added on the next scraping)")
                else:
                    self.logger.info(f"   ✅ Using {len(df_projects)} existing projects (date-based cache active)")
                
                # Check and log the presence of skills
                if 'extracted_skills' in df_projects.columns:
                    skills_count = df_projects['extracted_skills'].notna().sum()
                    self.logger.info(f"   ✅ {skills_count}/{len(df_projects)} projects have skills (from cache)")
                else:
                    self.logger.info(f"   ⚠️  No skill column, skill extraction will run")
            else:
                self.logger.warning("⚠️  No existing project data found and no new projects either!")
                # Create an empty DataFrame
                df_projects = pd.DataFrame(columns=['link', 'full_description'])
        
        # Save (update only if new projects were added)
        output_path = get_output_path(self.config, 'project_details')
        new_projects_count = len(new_links) if new_links else 0
        
        # Update the CSV in every case (so that the published_date column is added)
        if new_projects_count > 0 or not output_path.exists():
            # New projects were added or this is the first run, update the CSV
            df_projects.to_csv(output_path, index=False, encoding='utf-8-sig')
            if new_projects_count > 0:
                self.logger.info(f"Project details saved: {output_path} ({len(df_projects)} projects in total, {new_projects_count} new)")
            else:
                self.logger.info(f"Project details saved: {output_path} ({len(df_projects)} projects in total)")
        
        # Update the metadata file (latest project publication date - more accurate)
        # Update if new projects were added or this is the first run
        # Also, if projects exist but there is no metadata, derive the date from the existing projects
        should_update_metadata = False
        if new_projects_count > 0:
            should_update_metadata = True
        elif not metadata_file.exists() and len(df_projects) > 0:
            # First run and projects exist, create the metadata
            should_update_metadata = True
            self.logger.info("📅 First run: creating the metadata file...")
        
        if should_update_metadata:
            try:
                import json
                from datetime import datetime
                # Check published_date first (more accurate - actual publication date)
                if 'published_date' in df_projects.columns:
                    df_projects['published_date'] = pd.to_datetime(df_projects['published_date'], errors='coerce')
                    # Take only the dates of newly scraped projects (non-None)
                    new_published = df_projects[df_projects['published_date'].notna()]
                    if len(new_published) > 0:
                        latest_published_date = new_published['published_date'].max()
                        if pd.notna(latest_published_date):
                            metadata = {
                                'last_published_date': latest_published_date.isoformat(),  # Actual publication date (takes priority)
                                'last_scrape_date': datetime.now().isoformat(),  # Scraping time (for fallback)
                                'last_scrape_count': new_projects_count,
                                'total_projects': len(df_projects),
                                'last_update': datetime.now().isoformat()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"📅 Metadata updated: latest project publication date {latest_published_date.strftime('%Y-%m-%d')} ({new_projects_count} new projects)")
                    else:
                        # No published_date, use scraped_at
                        if 'scraped_at' in df_projects.columns:
                            df_projects['scraped_at'] = pd.to_datetime(df_projects['scraped_at'], errors='coerce')
                            new_scraped = df_projects[df_projects['scraped_at'].notna()]
                            if len(new_scraped) > 0:
                                latest_date = new_scraped['scraped_at'].max()
                                if pd.notna(latest_date):
                                    metadata = {
                                        'last_scrape_date': latest_date.isoformat(),
                                        'last_scrape_count': new_projects_count,
                                        'total_projects': len(df_projects),
                                        'last_update': datetime.now().isoformat()
                                    }
                                    with open(metadata_file, 'w', encoding='utf-8') as f:
                                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                                    self.logger.info(f"📅 Metadata updated: latest scraping {latest_date.strftime('%Y-%m-%d %H:%M:%S')} ({new_projects_count} new projects)")
                            else:
                                # No date information, use the current time
                                now = datetime.now()
                                metadata = {
                                    'last_scrape_date': now.isoformat(),
                                    'last_scrape_count': new_projects_count,
                                    'total_projects': len(df_projects),
                                    'last_update': now.isoformat()
                                }
                                with open(metadata_file, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                self.logger.info(f"📅 Metadata created: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            # No date information at all
                            now = datetime.now()
                            metadata = {
                                'last_scrape_date': now.isoformat(),
                                'last_scrape_count': new_projects_count,
                                'total_projects': len(df_projects),
                                'last_update': now.isoformat()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"📅 Metadata created: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                elif 'scraped_at' in df_projects.columns:
                    # Fallback: use scraped_at
                    df_projects['scraped_at'] = pd.to_datetime(df_projects['scraped_at'], errors='coerce')
                    new_scraped = df_projects[df_projects['scraped_at'].notna()]
                    if len(new_scraped) > 0:
                        latest_date = new_scraped['scraped_at'].max()
                        if pd.notna(latest_date):
                            metadata = {
                                'last_scrape_date': latest_date.isoformat(),
                                'last_scrape_count': new_projects_count,
                                'total_projects': len(df_projects),
                                'last_update': datetime.now().isoformat()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"📅 Metadata updated: latest scraping {latest_date.strftime('%Y-%m-%d %H:%M:%S')} ({new_projects_count} new projects)")
                else:
                    # No date information at all, use the current time
                    now = datetime.now()
                    metadata = {
                        'last_scrape_date': now.isoformat(),
                        'last_scrape_count': new_projects_count,
                        'total_projects': len(df_projects),
                        'last_update': now.isoformat()
                    }
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"📅 Metadata created: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                self.logger.warning(f"Could not update metadata: {e}")
        
        # If projects exist but there is no metadata, derive the date from the existing projects
        elif not metadata_file.exists() and len(df_projects) > 0 and 'published_date' in df_projects.columns:
            try:
                import json
                from datetime import datetime
                df_projects['published_date'] = pd.to_datetime(df_projects['published_date'], errors='coerce')
                published_projects = df_projects[df_projects['published_date'].notna()]
                if len(published_projects) > 0:
                    latest_date = published_projects['published_date'].max()
                    if pd.notna(latest_date):
                        metadata = {
                            'last_published_date': latest_date.isoformat(),
                            'last_scrape_date': datetime.now().isoformat(),
                            'last_scrape_count': 0,
                            'total_projects': len(df_projects),
                            'last_update': datetime.now().isoformat(),
                            'note': 'Date derived from the existing projects'
                        }
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"📅 Metadata created (from existing projects): latest publication date {latest_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                self.logger.warning(f"Could not create metadata from the existing projects: {e}")
        else:
            # No new projects, the CSV already exists
            self.logger.info(f"Using the existing CSV: {output_path} ({len(df_projects)} projects)")
        
        # 3. Parse the CV (could run in parallel with project scraping, sequential for now)
        self.logger.info("Step 3: Parsing the CV...")
        
        cv_parser = EuropassCVParser(self.config, self.logger)
        cv_text, cv_skills = cv_parser.parse_cv()
        
        # Save the CV data
        project_root = Path(__file__).resolve().parent.parent
        cv_text_file = project_root / "data/raw/IP1_output_candidate_full_text.txt"
        with open(cv_text_file, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        
        cv_skills_df = pd.DataFrame(cv_skills, columns=["raw_skill_name"])
        cv_skills_path = project_root / self.config['output']['files']['candidate_skills_raw']
        cv_skills_df.to_pickle(cv_skills_path)
        self.logger.info(f"CV data saved")
        
        # 4. Extract skills
        # If preloaded_projects exists and has skills, skip this step
        # But first check whether df_projects has skills
        skip_skill_extraction = False
        
        if 'extracted_skills' in df_projects.columns:
            # Skill column exists, check whether there are valid skills
            non_null_skills = df_projects['extracted_skills'].notna()
            if non_null_skills.any():
                valid_skills = df_projects[non_null_skills]['extracted_skills'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                )
                if valid_skills.any():
                    valid_count = valid_skills.sum()
                    self.logger.info(f"✅ Skills already available: {valid_count}/{len(df_projects)} projects have valid skills")
                    self.logger.info("   📊 Skipping skill extraction (from cache)")
                    skip_skill_extraction = True
        
        if not skip_skill_extraction:
            self.logger.info("Step 4: Extracting skills...")
            skill_extractor = SkillExtractorModule(self.config, self.logger)
        else:
            skill_extractor = None  # Not needed
        
        # Extract skills from the CV
        if skill_extractor is not None:
            candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
            candidate_skills_df = pd.DataFrame(candidate_skills, columns=["extracted_skill_name"])
            candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
            candidate_skills_df.to_pickle(candidate_skills_path)
            self.logger.info(f"Candidate skills saved: {candidate_skills_path}")
        else:
            # No skill extractor; extract the CV skills in a simple way (or take them from the cache)
            # In fact the CV skills should always be extracted; only the project skills can be skipped
            skill_extractor = SkillExtractorModule(self.config, self.logger)
            candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
            candidate_skills_df = pd.DataFrame(candidate_skills, columns=["extracted_skill_name"])
            candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
            candidate_skills_df.to_pickle(candidate_skills_path)
            self.logger.info(f"Candidate skills saved: {candidate_skills_path}")
        
        # Extract skills from the projects (only for new projects)
        # Check whether df_projects has skills (in more detail)
        has_skills_in_df = False
        skills_info = "No skill column"
        
        if 'extracted_skills' in df_projects.columns:
            # The column exists; check whether it contains data
            non_null_skills = df_projects['extracted_skills'].notna()
            non_null_count = non_null_skills.sum()
            
            if non_null_count > 0:
                # At least one project has skills; check whether they are valid
                valid_skills = df_projects[non_null_skills]['extracted_skills'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                )
                valid_count = valid_skills.sum()
                has_skills_in_df = valid_count > 0
                skills_info = f"{valid_count}/{len(df_projects)} projects have valid skills"
            else:
                skills_info = f"Skill column exists but is empty for all projects"
        else:
            skills_info = "No skill column"
        
        self.logger.info(f"🔍 Skill check: {skills_info}, skill extraction needed? {not has_skills_in_df}")
        
        if skip_skill_extraction:
            # Skill extraction skipped, just continue
            self.logger.info("   ✅ Using project skills from the cache, no extraction")
        elif new_links and len(new_links) > 0:
            # There are new projects
            df_new_projects_only = df_projects[df_projects['link'].isin(new_links)].copy()
            df_existing_projects_only = df_projects[~df_projects['link'].isin(new_links)].copy()
            
            if len(df_new_projects_only) > 0:
                # Extract skills only for the new projects
                self.logger.info(f"   Extracting skills for {len(df_new_projects_only)} new projects...")
                df_new_projects_only = skill_extractor.extract_skills_from_dataframe(df_new_projects_only)
                
                # Keep the skills of the existing projects and add the new projects
                if has_skills_in_df and len(df_existing_projects_only) > 0:
                    # The existing projects already have skills; only add the new projects
                    df_projects = pd.concat([df_existing_projects_only, df_new_projects_only], ignore_index=True)
                    self.logger.info(f"   ✅ Skills merged: {len(df_existing_projects_only)} old (skills from cache) + {len(df_new_projects_only)} new")
                else:
                    # The existing projects have no skills; extract skills for all projects
                    self.logger.info(f"   Extracting skills for all {len(df_projects)} projects (first run)...")
                    df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
        else:
            # No new projects
            if skip_skill_extraction:
                # Skill extraction was already skipped (checked above)
                self.logger.info("   ✅ Skill extraction skipped (from cache)")
            elif has_skills_in_df:
                # Skills already exist (loaded from pickle), skip skill extraction
                skills_count = df_projects['extracted_skills'].notna().sum()
                valid_skills_count = df_projects['extracted_skills'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                ).sum()
                self.logger.info(f"   ✅ Using the existing projects' skills from the cache (skill extraction skipped)")
                self.logger.info(f"      📊 {valid_skills_count}/{len(df_projects)} projects have valid skills, {skills_count - valid_skills_count} are empty")
                # Do not extract skills, just continue
            elif 'extracted_skills' not in df_projects.columns:
                # No skill column, extract
                self.logger.warning(f"   ⚠️  No skill column, extracting skills for all {len(df_projects)} projects...")
                df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
            else:
                # The skill column exists but is empty or invalid, extract
                empty_count = df_projects['extracted_skills'].isna().sum()
                self.logger.warning(f"   ⚠️  Skill column exists but is empty for {empty_count}/{len(df_projects)} projects, extracting skills for all projects...")
                df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
        
        projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
        df_projects.to_pickle(projects_skills_path)
        self.logger.info(f"Project skills saved: {projects_skills_path} ({len(df_projects)} projects in total)")
        
        # 5. Vectorization
        self.logger.info("Step 5: Vectorizing...")
        vectorizer = SkillVectorizer(self.config, self.logger)
        
        # Vectorize the candidate skills
        candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
        candidate_vectors_path = project_root / self.config['output']['files']['candidate_vectors']
        np.save(candidate_vectors_path, candidate_vectors)
        self.logger.info(f"Candidate vectors saved: {candidate_vectors_path}")
        
        # Vectorize the projects (only the new projects)
        # Check whether the vectors exist and are filled
        has_vectors = ('skill_vectors' in df_projects.columns and 
                      df_projects['skill_vectors'].notna().any())
        
        if has_vectors:
            # Existing vectors are available; compute vectors only for the new projects
            if new_links and len(new_links) > 0:
                df_new_projects_only = df_projects[df_projects['link'].isin(new_links)].copy()
                # Check whether the new projects have vectors
                new_has_vectors = ('skill_vectors' in df_new_projects_only.columns and 
                                  df_new_projects_only['skill_vectors'].notna().any())
                
                if len(df_new_projects_only) > 0 and not new_has_vectors:
                    self.logger.info(f"   Computing vectors for {len(df_new_projects_only)} new projects...")
                    df_new_projects_only = vectorizer.vectorize_dataframe(df_new_projects_only)
                    
                    # Keep the vectors of the existing projects and add the new projects
                    df_existing_with_vectors = df_projects[~df_projects['link'].isin(new_links)].copy()
                    df_projects = pd.concat([df_existing_with_vectors, df_new_projects_only], ignore_index=True)
                    self.logger.info(f"   ✅ Vectors merged: {len(df_existing_with_vectors)} old (from cache) + {len(df_new_projects_only)} new")
                else:
                    self.logger.info(f"   ✅ Vectors available for all projects (from cache)")
            else:
                # No new projects, use the existing vectors
                self.logger.info(f"   ✅ Using the existing projects' vectors from the cache (vectorization skipped - {len(df_projects)} projects)")
        else:
            # No vectors, compute for all projects
            self.logger.info(f"   Computing vectors for all {len(df_projects)} projects...")
            df_projects = vectorizer.vectorize_dataframe(df_projects)
        
        projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
        df_projects.to_pickle(projects_vectors_path)
        self.logger.info(f"Project vectors saved: {projects_vectors_path} ({len(df_projects)} projects in total)")
        
        # Filter out projects whose application deadline has passed
        from datetime import datetime
        if 'application_deadline' in df_projects.columns:
            df_projects['application_deadline'] = pd.to_datetime(df_projects['application_deadline'], errors='coerce')
            now = datetime.now()
            expired_projects = df_projects[df_projects['application_deadline'].notna() & (df_projects['application_deadline'] < now)]
            if len(expired_projects) > 0:
                self.logger.info(f"🗑️  {len(expired_projects)} projects are past their application deadline, removing...")
                df_projects = df_projects[df_projects['application_deadline'].isna() | (df_projects['application_deadline'] >= now)]
                self.logger.info(f"   ✅ {len(df_projects)} active projects remain")
        
        # Also save the projects to CSV (current state)
        csv_projects_file = get_output_path(self.config, 'project_details')
        df_projects.to_csv(csv_projects_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"Projects saved to CSV: {csv_projects_file} ({len(df_projects)} projects in total)")
        
        # Update the metadata (latest addition date and publication date)
        metadata_file = project_root / "last_scrape_metadata.json"
        try:
            import json
            from datetime import datetime
            last_added_date = datetime.now()  # Date the latest projects were added
            last_published_date = None
            
            if 'published_date' in df_projects.columns:
                df_projects['published_date'] = pd.to_datetime(df_projects['published_date'], errors='coerce')
                published_projects = df_projects[df_projects['published_date'].notna()]
                if len(published_projects) > 0:
                    latest_date = published_projects['published_date'].max()
                    if pd.notna(latest_date):
                        last_published_date = latest_date
            
            metadata = {
                'last_added_date': last_added_date.isoformat(),  # Date the latest project was added
                'last_published_date': last_published_date.isoformat() if last_published_date is not None else None,
                'last_scrape_date': datetime.now().isoformat(),
                'last_scrape_count': len(new_links) if 'new_links' in locals() and new_links else 0,
                'total_projects': len(df_projects),
                'last_update': datetime.now().isoformat()
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📅 Metadata updated: last addition {last_added_date.strftime('%Y-%m-%d %H:%M:%S')}, latest publication date {last_published_date.strftime('%Y-%m-%d') if last_published_date is not None else 'N/A'} ({len(new_links) if 'new_links' in locals() and new_links else 0} new projects)")
        except Exception as e:
            self.logger.warning(f"Could not update metadata: {e}")
        
        self.logger.info("=" * 80)
        self.logger.info("✅ IP1 COMPLETED! Projects saved to file.")
        self.logger.info(f"   📁 CSV: {csv_projects_file}")
        self.logger.info(f"   📁 Pickle (skills): {projects_skills_path}")
        self.logger.info(f"   📁 Pickle (vectors): {projects_vectors_path}")
        self.logger.info(f"   📊 {len(df_projects)} projects saved in total")
        self.logger.info("=" * 80)
        
        return df_projects, candidate_vectors, cv_text, candidate_skills
    
    def run_ip2_matching_score(self, df_projects: pd.DataFrame, candidate_vectors: np.ndarray):
        """IP2: Matching score computation"""
        self.logger.info("=" * 80)
        self.logger.info("IP2: COMPUTING THE MATCHING SCORE (M-SCORE)")
        self.logger.info("=" * 80)
        
        matching_calc = MatchingScoreCalculator(self.config, self.logger)
        df_projects = matching_calc.calculate_all_scores(df_projects, candidate_vectors)
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['matching_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"M scores saved: {output_path}")
        
        self.logger.info("IP2 COMPLETED!")
        return df_projects
    
    def run_ip3_sincerity_score(self, df_projects: pd.DataFrame):
        """IP3: Sincerity score computation"""
        self.logger.info("=" * 80)
        self.logger.info("IP3: COMPUTING THE SINCERITY SCORE (S-SCORE)")
        self.logger.info("=" * 80)
        
        sincerity_calc = SincerityScoreCalculator(self.config, self.logger)
        # Read the calculate_all setting from the config
        calculate_all = self.config.get('sincerity_scoring', {}).get('calculate_all', False)
        df_projects = sincerity_calc.calculate_all_scores(df_projects, calculate_all=calculate_all)
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['sincerity_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"S scores saved: {output_path}")
        
        self.logger.info("IP3 COMPLETED!")
        return df_projects
    
    def run_ip2_and_ip4_parallel(self, df_projects: pd.DataFrame, candidate_vectors: np.ndarray, cv_text: str):
        """Runs IP2 and IP4 in parallel"""
        self.logger.info("=" * 80)
        self.logger.info("IP2 & IP4: COMPUTING M AND R SCORES IN PARALLEL")
        self.logger.info("=" * 80)
        
        from concurrent.futures import ThreadPoolExecutor
        
        def calculate_m_score():
            """Compute the M score"""
            self.logger.info("IP2: Computing the M score (parallel)...")
            matching_calc = MatchingScoreCalculator(self.config, self.logger)
            return matching_calc.calculate_all_scores(df_projects.copy(), candidate_vectors)
        
        def calculate_r_score():
            """Compute the R score"""
            self.logger.info("IP4: Computing the R score (parallel)...")
            reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
            cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
            return reliability_calc.calculate_project_reliability_scores(
                df_projects.copy(),
                cv_paragraphs,
                skill_column="extracted_skills"
            )
        
        # Parallel computation
        with ThreadPoolExecutor(max_workers=2) as executor:
            m_future = executor.submit(calculate_m_score)
            r_future = executor.submit(calculate_r_score)
            
            # Wait for the results
            df_projects_m = m_future.result()
            df_projects_r = r_future.result()
            
            # Merge the scores
            df_projects = df_projects_m.copy()
            if 'R_Score' in df_projects_r.columns:
                df_projects['R_Score'] = df_projects_r['R_Score']
        
        # Save
        project_root = Path(__file__).parent.parent
        m_output_path = project_root / self.config['output']['files']['matching_scores']
        df_projects.to_pickle(m_output_path)
        self.logger.info(f"M and R scores saved: {m_output_path}")
        
        self.logger.info("IP2 & IP4 PARALLEL COMPLETED!")
        return df_projects
    
    def run_ip4_reliability_score(self, df_projects: pd.DataFrame, cv_text: str):
        """IP4: Reliability score computation (for each project)"""
        self.logger.info("=" * 80)
        self.logger.info("IP4: COMPUTING THE RELIABILITY SCORE (R-SCORE)")
        self.logger.info("=" * 80)
        self.logger.info("NOTE: For each project, the share of that project's requirements verified in the CV will be computed")
        
        reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
        cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
        
        # Compute the R score for each project
        df_projects = reliability_calc.calculate_project_reliability_scores(
            df_projects, 
            cv_paragraphs,
            skill_column="extracted_skills"
        )
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['reliability_details']
        df_projects[['link', 'R_Score']].to_pickle(output_path)
        self.logger.info(f"R scores saved: {output_path}")
        
        self.logger.info("IP4 COMPLETED!")
        return df_projects
    
    def run_ip5_final_ranking(self, df_projects: pd.DataFrame):
        """IP5: Final fair ranking"""
        self.logger.info("=" * 80)
        self.logger.info("IP5: COMPUTING THE FAIR RANKING (MOO)")
        self.logger.info("=" * 80)
        
        ranking_calc = FinalRankingCalculator(self.config, self.logger)
        df_final = ranking_calc.calculate_ranking(df_projects)
        
        # Select and save all important columns
        columns_to_save = [
            'link', 'full_description', 'extracted_skills',
            # Raw scores (original values)
            'M_Score_Raw', 'S_Score_Raw', 'R_Score_Raw',
            # Normalized scores (with Z-Score)
            'M_Score_Normalized', 'S_Score_Normalized', 'R_Score_Normalized',
            # Scores in use (normalized, as M_Score, S_Score, R_Score)
            'M_Score', 'S_Score', 'R_Score',
            # Final skorlar
            'Final_Score', 'Fair_Score'
        ]
        
        # Check the existing columns
        available_columns = [col for col in columns_to_save if col in df_final.columns]
        missing_columns = [col for col in columns_to_save if col not in df_final.columns]
        
        if missing_columns:
            self.logger.warning(f"Some columns are missing: {missing_columns}")
        
        # Warn if there are no raw scores (they should be created in final_ranking.py)
        if 'M_Score_Raw' not in df_final.columns:
            self.logger.warning("Raw scores not found! Z-Score normalization should have been applied in the final ranking.")
        
        # Save only the existing columns
        df_to_save = df_final[available_columns].copy()
        
        # If there are no raw scores but normalized scores exist, also add the normalized scores as raw
        if 'M_Score_Raw' not in df_final.columns and 'M_Score_Normalized' in df_final.columns:
            self.logger.info("Raw scores missing, also saving the normalized scores as raw...")
            # In this case the normalized scores already exist as M_Score, S_Score, R_Score
            # The raw scores could be taken from earlier stages, but for now use the normalized scores
            pass
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['final_ranking']
        df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Final ranking saved: {output_path}")
        self.logger.info(f"Saved columns: {list(df_to_save.columns)}")
        self.logger.info(f"{len(df_to_save)} projects saved in total")
        
        # Show the top 10
        self.logger.info("\n🏆 TOP 10 PROJECTS:")
        for i, row in df_final.head(10).iterrows():
            project_id = row['link'].split('/')[-1].replace('_en', '')
            self.logger.info(
                f"{i+1}. [Fair score: {row['Fair_Score']:.2f}] "
                f"[M: {row['M_Score']:.1f}, S: {row['S_Score']:.1f}, R: {row['R_Score']:.1f}] "
                f"{project_id}"
            )
        
        self.logger.info("IP5 COMPLETED!")
        return df_final
    
    def run_full_pipeline(self):
        """Runs the whole pipeline"""
        try:
            # IP1
            df_projects, candidate_vectors, cv_text, candidate_skills = self.run_ip1_data_collection()
            
            # IP2 and IP4 - parallel computation (M and R scores are independent)
            self.logger.info("Starting IP2 and IP4 in parallel...")
            df_projects = self.run_ip2_and_ip4_parallel(df_projects, candidate_vectors, cv_text)
            
            # Check: does every project have an R score?
            r_missing = df_projects['R_Score'].isna().sum() if 'R_Score' in df_projects.columns else len(df_projects)
            if r_missing > 0:
                self.logger.warning(f"R score missing for {r_missing} projects!")
            else:
                self.logger.info(f"✅ R score computed for all {len(df_projects)} projects")
            
            # IP3 - compute the S score for ALL projects (after the M score, for ranking)
            self.logger.info("Before IP3: the S score will be computed for all projects")
            df_projects = self.run_ip3_sincerity_score(df_projects)
            
            # Check: does every project have an S score?
            s_missing = df_projects['S_Score'].isna().sum() if 'S_Score' in df_projects.columns else len(df_projects)
            if s_missing > 0:
                self.logger.warning(f"S score missing for {s_missing} projects!")
            else:
                self.logger.info(f"✅ S score computed for all {len(df_projects)} projects")
            
            # IP5
            df_final = self.run_ip5_final_ranking(df_projects)
            
            self.logger.info("=" * 80)
            self.logger.info("✅ THE WHOLE PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    pipeline = ProjectPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

