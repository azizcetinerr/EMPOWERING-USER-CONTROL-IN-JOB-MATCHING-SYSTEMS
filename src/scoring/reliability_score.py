"""
Module that computes the reliability score (R-Score)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import logging


class ReliabilityScoreCalculator:
    """Reliability score calculator class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # NLI model
        nli_config = config.get('models', {}).get('nli', {})
        nli_model_name = nli_config.get('model_name', 'cross-encoder/nli-distilroberta-base')
        device_str = nli_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        
        self.logger.info(f"Loading NLI model: {nli_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model loaded on device {self.device}")
    
    def verify_claim(self, claim: str, evidence_list: List[str]) -> float:
        """
        Verifies whether a claim (skill) is supported by a list of evidence.
        Optimized: batch processing and a limit on the number of evidence items.
        
        Args:
            claim: Claim (skill name)
            evidence_list: List of evidence (CV paragraphs)
        
        Returns:
            Verification score (0-100)
        """
        if not evidence_list:
            return 0.0
        
        # Turn the claim into a hypothesis
        hypothesis = f"I have the skill of {claim}."
        
        # If there is too much evidence, select the most important items (for performance)
        # Use the first 10 paragraphs (usually sufficient)
        max_evidence = 10
        limited_evidence = evidence_list[:max_evidence] if len(evidence_list) > max_evidence else evidence_list
        
        # Build pairs with the evidence
        pairs = [(evidence, hypothesis) for evidence in limited_evidence]
        
        # Feed the model (batch processing is already applied)
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Take the entailment score (index 1)
        entailment_scores = logits[:, 1].cpu().numpy()
        
        # Find the highest verification score
        max_score = entailment_scores.max()
        
        # Map to a 0-100 score with a sigmoid-like transform
        probability = 1 / (1 + np.exp(-max_score))
        
        return probability * 100
    
    def calculate_reliability_score(
        self,
        skills: List[str],
        cv_paragraphs: List[str]
    ) -> tuple:
        """
        Computes the reliability score over all skills.
        
        Args:
            skills: List of skills
            cv_paragraphs: CV paragraphs
        
        Returns:
            (final_r_score, details_df) tuple
        """
        self.logger.info(f"Verifying {len(skills)} skills...")
        
        verification_results = []
        total_score = 0
        
        for skill in skills:
            score = self.verify_claim(skill, cv_paragraphs)
            total_score += score
            
            status = "✅ Verified" if score > 50 else "❓ Weak evidence"
            verification_results.append({
                "Skill": skill,
                "R_Score": score,
                "Status": status
            })
            
            self.logger.debug(f"Skill: {skill.ljust(30)} -> Score: {score:.1f} ({status})")
        
        # Overall reliability score
        final_r_score = total_score / len(skills) if skills else 0.0
        
        self.logger.info(f"Final reliability score (R): {final_r_score:.2f} / 100")
        
        details_df = pd.DataFrame(verification_results)
        
        return final_r_score, details_df
    
    def calculate_enhanced_reliability_score(
        self,
        base_r_score: float,
        organization_name: str = None,
        verified: bool = False,
        address: str = None,
        website: str = None,
        project_count: int = 0
    ) -> float:
        """
        Updates the base R score using the organization information.
        
        Args:
            base_r_score: Base reliability score (0-100) - skill verification score
            organization_name: Organization name
            verified: Is the organization verified?
            address: Organization address
            website: Organization website
            project_count: Number of past projects of the organization
        
        Returns:
            Updated R score (0-100)
        """
        reliability_config = self.config.get('reliability_scoring', {})
        enhanced_score = base_r_score
        
        # 1. Bonus for the organization's number of past projects
        if reliability_config.get('project_count_bonus', {}).get('enabled', True):
            thresholds = reliability_config.get('project_count_bonus', {}).get('thresholds', [1, 5, 10, 20, 50])
            bonuses = reliability_config.get('project_count_bonus', {}).get('bonuses', [2.0, 5.0, 10.0, 15.0, 20.0])
            
            project_bonus = 0.0
            for i, threshold in enumerate(reversed(thresholds)):
                if project_count >= threshold:
                    project_bonus = bonuses[len(thresholds) - 1 - i]
                    break
            
            enhanced_score += project_bonus
            if project_bonus > 0:
                self.logger.debug(f"Project count bonus: +{project_bonus:.1f} (project count: {project_count})")
        
        # 2. Verification status bonus
        if reliability_config.get('verified_bonus', {}).get('enabled', True):
            verified_bonus_value = reliability_config.get('verified_bonus', {}).get('bonus_score', 15.0) if verified else 0.0
            enhanced_score += verified_bonus_value
            if verified_bonus_value > 0:
                self.logger.debug(f"Verification status bonus: +{verified_bonus_value:.1f}")
        
        # 3. Missing data penalty
        if reliability_config.get('missing_data_penalties', {}).get('enabled', True):
            penalties = reliability_config.get('missing_data_penalties', {})
            missing_data_penalty = 0.0
            
            # If the address is missing
            if not address or (isinstance(address, str) and address.strip() == ""):
                penalty = penalties.get('address', 10.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Missing address penalty: -{penalty:.1f}")
            
            # If the website is missing
            if not website or (isinstance(website, str) and website.strip() == ""):
                penalty = penalties.get('website', 10.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Missing website penalty: -{penalty:.1f}")
            
            # If the organization name is missing
            if not organization_name or (isinstance(organization_name, str) and organization_name.strip() == ""):
                penalty = penalties.get('organization_name', 5.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Missing organization name penalty: -{penalty:.1f}")
            
            enhanced_score -= missing_data_penalty
        
        # 4. Normalize to the 0-100 range
        if reliability_config.get('normalize_to_0_100', True):
            enhanced_score = max(0.0, min(100.0, enhanced_score))
        
        return enhanced_score
    
    def calculate_project_reliability_scores(
        self,
        df_projects: pd.DataFrame,
        cv_paragraphs: List[str],
        skill_column: str = "extracted_skills"
    ) -> pd.DataFrame:
        """
        Computes the reliability score for each project.
        Takes the requirements (skills) of each project and computes the share of them verified in the CV.
        Updates the score with the organization information (project count, verification status, missing data penalties).
        
        Args:
            df_projects: Project DataFrame (must contain skill lists and organization information)
            cv_paragraphs: CV paragraphs
            skill_column: Name of the skill list column
        
        Returns:
            DataFrame with R_Score added
        """
        self.logger.info(f"Computing R scores for {len(df_projects)} projects...")
        self.logger.info("For each project: share of the project's requirements verified in the CV + organization information")
        
        df_projects = df_projects.copy()
        
        # Check the organization information and set defaults
        if 'organization_name' not in df_projects.columns:
            df_projects['organization_name'] = None
            self.logger.warning("organization_name column not found, set to None")
        
        if 'verified' not in df_projects.columns:
            df_projects['verified'] = False
            self.logger.warning("verified column not found, set to False")
        
        if 'address' not in df_projects.columns:
            df_projects['address'] = None
            self.logger.warning("address column not found, set to None")
        
        if 'website' not in df_projects.columns:
            df_projects['website'] = None
            self.logger.warning("website column not found, set to None")
        
        # Compute the number of projects per organization (if organization_name exists)
        if 'organization_name' in df_projects.columns:
            # Count only organization names that are not None
            valid_orgs = df_projects[df_projects['organization_name'].notna()]['organization_name']
            if len(valid_orgs) > 0:
                org_counts = valid_orgs.value_counts().to_dict()
                df_projects['project_count'] = df_projects['organization_name'].map(org_counts).fillna(0).astype(int)
                self.logger.info(f"Project counts per organization computed ({len(org_counts)} distinct organizations in total)")
            else:
                df_projects['project_count'] = 0
                self.logger.warning("No valid organization name found, project count set to 0")
        else:
            df_projects['project_count'] = 0
        
        r_scores = []
        base_r_scores = []
        total_projects = len(df_projects)
        
        # For progress reporting
        processed = 0
        
        for idx, row in df_projects.iterrows():
            project_skills = row.get(skill_column, [])
            
            # 1. Base R score: skill verification
            if not project_skills or not isinstance(project_skills, list) or len(project_skills) == 0:
                base_r_score = 0.0
            else:
                # Compute the share of this project's skills verified in the CV
                # Combine all skills and use batch processing
                valid_skills = [s.strip() for s in project_skills if s and isinstance(s, str) and len(s.strip()) > 0]
                
                if valid_skills:
                    # Batch processing: verify all skills at once
                    total_score = 0.0
                    for skill in valid_skills:
                        score = self.verify_claim(skill, cv_paragraphs)
                        total_score += score
                
                # Mean score
                    base_r_score = total_score / len(valid_skills)
                else:
                    base_r_score = 0.0
            
            base_r_scores.append(base_r_score)
            processed += 1
            
            # Progress reporting (every 100 projects)
            if processed % 100 == 0 or processed == total_projects:
                self.logger.info(f"   Progress: {processed}/{total_projects} projects processed ({processed*100//total_projects}%)")
            
            # 2. Extended R score: update with the organization information
            enhanced_r_score = self.calculate_enhanced_reliability_score(
                base_r_score=base_r_score,
                organization_name=row.get('organization_name'),
                verified=row.get('verified', False),
                address=row.get('address'),
                website=row.get('website'),
                project_count=row.get('project_count', 0)
            )
            
            r_scores.append(enhanced_r_score)
            
            # Progress report
            if (idx + 1) % 100 == 0:
                self.logger.info(f"   Progress: {idx + 1}/{len(df_projects)} projects processed")
        
        df_result = df_projects.copy()
        df_result["R_Score"] = r_scores
        df_result["R_Score_Base"] = base_r_scores  # Also store the base score (for analysis)
        
        # Statistics
        self.logger.info(f"\n📊 R score statistics:")
        self.logger.info(f"  Base R score (skills only): Min={min(base_r_scores):.2f}, Max={max(base_r_scores):.2f}, Mean={sum(base_r_scores)/len(base_r_scores):.2f}")
        self.logger.info(f"  Extended R score (with organization information): Min={min(r_scores):.2f}, Max={max(r_scores):.2f}, Mean={sum(r_scores)/len(r_scores):.2f}")
        self.logger.info(f"  Mean difference: {(sum(r_scores)/len(r_scores)) - (sum(base_r_scores)/len(base_r_scores)):.2f} points")
        self.logger.info(f"✅ R score computed for all {len(df_projects)} projects")
        
        return df_result

