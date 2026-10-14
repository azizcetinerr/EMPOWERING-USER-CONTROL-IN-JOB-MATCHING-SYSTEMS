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
        project_root = Path(__file__).parent.parent
        output_config = self.config.get('output', {})
        
        dirs = [
            output_config.get('base_dir', 'outputs'),
            output_config.get('raw_data_dir', 'outputs/raw'),
            output_config.get('processed_data_dir', 'outputs/processed'),
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
        
        # If preloaded projects exist and have skills/vectors, use them
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            self.logger.info(f"🔍 Using preloaded projects: {len(preloaded_projects)} projects")
            
            # Check the presence of skills and vectors
            has_skills = 'extracted_skills' in preloaded_projects.columns
            has_vectors = 'skill_vectors' in preloaded_projects.columns
            
            if has_skills and has_vectors:
                self.logger.info("✅ Projects have skills and vectors, skipping scraping...")
                df_projects = preloaded_projects.copy()
            else:
                self.logger.info("⚠️  Projects are missing skills or vectors, only the missing parts will be processed...")
                df_projects = preloaded_projects.copy()
        else:
            # 1. Fetch the project links
            self.logger.info("Step 1: Fetching project links...")
            scraper = YouthPortalScraper(self.config, self.logger)
            links = scraper.get_project_links()
            
            # Save the links
            output_file = self.config['data_sources']['youth_portal']['output_file']
            links_file = project_root / output_file
            with open(links_file, 'w', encoding='utf-8') as f:
                for link in links:
                    f.write(link + "\n")
            self.logger.info(f"Links saved: {links_file}")
            
            # 2. Fetch the project details
            self.logger.info("Step 2: Fetching project details...")
            detail_scraper = ProjectDetailScraper(self.config, self.logger)
            df_projects = detail_scraper.scrape_all_projects(links)
            
            # Save
            output_path = get_output_path(self.config, 'project_details')
            df_projects.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Project details saved: {output_path}")
        
        # 3. Parse the CV (must always run - for a new CV)
        self.logger.info("Step 3: Parsing the CV...")
        try:
            cv_parser = EuropassCVParser(self.config, self.logger)
            cv_text, cv_skills = cv_parser.parse_cv()
        except Exception as cv_error:
            self.logger.error(f"❌ CV parsing error: {cv_error}", exc_info=True)
            raise
        
        # Save the CV data
        project_root = Path(__file__).parent.parent
        cv_text_file = project_root / self.config['output']['files']['candidate_skills_raw'].replace('.pkl', '.txt')
        with open(cv_text_file, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        
        cv_skills_df = pd.DataFrame(cv_skills, columns=["raw_skill_name"])
        cv_skills_path = project_root / self.config['output']['files']['candidate_skills_raw']
        cv_skills_df.to_pickle(cv_skills_path)
        self.logger.info(f"CV data saved")
        
        # 4. Extract skills
        self.logger.info("Step 4: Extracting skills...")
        skill_extractor = SkillExtractorModule(self.config, self.logger)
        
        # Extract skills from the CV (must always run - for a new CV)
        candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
        candidate_skills_df = pd.DataFrame(candidate_skills, columns=["extracted_skill_name"])
        candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
        candidate_skills_df.to_pickle(candidate_skills_path)
        self.logger.info(f"Candidate skills saved: {candidate_skills_path}")
        
        # Extract skills from the projects (only if missing)
        if 'extracted_skills' not in df_projects.columns:
            self.logger.info("Extracting skills from the projects...")
            df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
            projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
            df_projects.to_pickle(projects_skills_path)
            self.logger.info(f"Project skills saved: {projects_skills_path}")
        else:
            self.logger.info("✅ Projects already have skills, skipping...")
        
        # 5. Vectorization
        self.logger.info("Step 5: Vectorizing...")
        vectorizer = SkillVectorizer(self.config, self.logger)
        
        # Vectorize the candidate skills (must always run - for a new CV)
        candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
        candidate_vectors_path = project_root / self.config['output']['files']['candidate_vectors']
        np.save(candidate_vectors_path, candidate_vectors)
        self.logger.info(f"Candidate vectors saved: {candidate_vectors_path}")
        
        # Vectorize the projects (only if missing)
        if 'skill_vectors' not in df_projects.columns:
            self.logger.info("Computing project vectors...")
            df_projects = vectorizer.vectorize_dataframe(df_projects)
            projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
            df_projects.to_pickle(projects_vectors_path)
            self.logger.info(f"Project vectors saved: {projects_vectors_path}")
        else:
            self.logger.info("✅ Project vectors already exist, skipping...")
        
        self.logger.info("IP1 COMPLETED!")
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
        df_projects = sincerity_calc.calculate_all_scores(df_projects)
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['sincerity_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"S scores saved: {output_path}")
        
        self.logger.info("IP3 COMPLETED!")
        return df_projects
    
    def run_ip4_reliability_score(self, candidate_skills: list, cv_text: str):
        """IP4: Reliability score computation"""
        self.logger.info("=" * 80)
        self.logger.info("IP4: COMPUTING THE RELIABILITY SCORE (R-SCORE)")
        self.logger.info("=" * 80)
        
        reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
        cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
        r_score, r_details = reliability_calc.calculate_reliability_score(candidate_skills, cv_paragraphs)
        
        # Save
        project_root = Path(__file__).parent.parent
        details_path = project_root / self.config['output']['files']['reliability_details']
        r_details.to_pickle(details_path)
        self.logger.info(f"R score details saved: {details_path}")
        
        score_path = project_root / self.config['output']['files']['reliability_score']
        with open(score_path, 'w') as f:
            f.write(str(r_score))
        self.logger.info(f"R score saved: {score_path}")
        
        self.logger.info("IP4 COMPLETED!")
        return r_score
    
    def run_ip5_final_ranking(self, df_projects: pd.DataFrame, r_score: float):
        """IP5: Final fair ranking"""
        self.logger.info("=" * 80)
        self.logger.info("IP5: COMPUTING THE FAIR RANKING (MOO)")
        self.logger.info("=" * 80)
        
        ranking_calc = FinalRankingCalculator(self.config, self.logger)
        df_final = ranking_calc.calculate_ranking(df_projects, r_score)
        
        # Save
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['final_ranking']
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Final ranking saved: {output_path}")
        
        # Show the top 10
        self.logger.info("\n🏆 TOP 10 PROJECTS:")
        for i, row in df_final.head(10).iterrows():
            project_id = row['link'].split('/')[-1].replace('_en', '')
            self.logger.info(
                f"{i+1}. [Fair score: {row['Fair_Score']:.2f}] "
                f"[M: {row['M_Score']:.1f}, S: {row['S_Score']:.1f}, R: {r_score:.1f}] "
                f"{project_id}"
            )
        
        self.logger.info("IP5 COMPLETED!")
        return df_final
    
    def run_full_pipeline(self):
        """Runs the whole pipeline"""
        try:
            # IP1
            df_projects, candidate_vectors, cv_text, candidate_skills = self.run_ip1_data_collection()
            
            # IP2
            df_projects = self.run_ip2_matching_score(df_projects, candidate_vectors)
            
            # IP3
            df_projects = self.run_ip3_sincerity_score(df_projects)
            
            # IP4
            r_score = self.run_ip4_reliability_score(candidate_skills, cv_text)
            
            # IP5
            df_final = self.run_ip5_final_ranking(df_projects, r_score)
            
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

