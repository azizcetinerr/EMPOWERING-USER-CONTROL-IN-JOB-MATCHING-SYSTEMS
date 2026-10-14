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
        
        # Build pairs with the evidence
        pairs = [(evidence, hypothesis) for evidence in evidence_list]
        
        # Feed the model
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

