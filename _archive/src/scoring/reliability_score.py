"""
Güvenilirlik skoru (R-Skoru) hesaplayan modül
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import logging


class ReliabilityScoreCalculator:
    """Güvenilirlik skoru hesaplayıcı sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # NLI modeli
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
        
        self.logger.info(f"NLI modeli yükleniyor: {nli_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model {self.device} cihazına yüklendi")
    
    def verify_claim(self, claim: str, evidence_list: List[str]) -> float:
        """
        Bir iddianın (becerinin) kanıt listesi içinde geçip geçmediğini doğrular.
        
        Args:
            claim: İddia (beceri adı)
            evidence_list: Kanıt listesi (CV paragrafları)
        
        Returns:
            Doğrulama skoru (0-100)
        """
        if not evidence_list:
            return 0.0
        
        # İddiayı hipoteze çevir
        hypothesis = f"I have the skill of {claim}."
        
        # Kanıtlarla çiftler oluştur
        pairs = [(evidence, hypothesis) for evidence in evidence_list]
        
        # Modele ver
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Entailment skorunu al (1. indeks)
        entailment_scores = logits[:, 1].cpu().numpy()
        
        # En yüksek doğrulama skorunu bul
        max_score = entailment_scores.max()
        
        # Sigmoid benzeri dönüşümle 0-100 arası puana çevir
        probability = 1 / (1 + np.exp(-max_score))
        
        return probability * 100
    
    def calculate_reliability_score(
        self,
        skills: List[str],
        cv_paragraphs: List[str]
    ) -> tuple:
        """
        Tüm beceriler için güvenilirlik skorunu hesaplar.
        
        Args:
            skills: Beceri listesi
            cv_paragraphs: CV paragrafları
        
        Returns:
            (final_r_score, details_df) tuple'ı
        """
        self.logger.info(f"{len(skills)} adet beceri doğrulanıyor...")
        
        dogrulama_sonuclari = []
        toplam_skor = 0
        
        for skill in skills:
            score = self.verify_claim(skill, cv_paragraphs)
            toplam_skor += score
            
            durum = "✅ Doğrulandı" if score > 50 else "❓ Kanıt Zayıf"
            dogrulama_sonuclari.append({
                "Beceri": skill,
                "R_Skoru": score,
                "Durum": durum
            })
            
            self.logger.debug(f"Skill: {skill.ljust(30)} -> Skor: {score:.1f} ({durum})")
        
        # Genel güvenilirlik skoru
        final_r_score = toplam_skor / len(skills) if skills else 0.0
        
        self.logger.info(f"Nihai Güvenilirlik Skoru (R): {final_r_score:.2f} / 100")
        
        details_df = pd.DataFrame(dogrulama_sonuclari)
        
        return final_r_score, details_df

