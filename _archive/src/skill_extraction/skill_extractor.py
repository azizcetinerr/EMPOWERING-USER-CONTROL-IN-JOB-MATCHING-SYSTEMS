"""
SkillNER kullanarak becerileri çıkaran modül
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
    """SkillNER beceri çıkarma sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        model_config = config.get('models', {}).get('skill_extraction', {})
        self.spacy_model = model_config.get('spacy_model', 'en_core_web_lg')
        
        # Spacy modelini yükle
        try:
            self.nlp = spacy.load(self.spacy_model)
            self.logger.info(f"Spacy modeli yüklendi: {self.spacy_model}")
        except IOError:
            self.logger.error(f"Spacy modeli bulunamadı: {self.spacy_model}")
            raise
        
        # SkillExtractor'ı başlat
        try:
            self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
            self.logger.info("SkillExtractor başarıyla başlatıldı")
        except Exception as e:
            self.logger.error(f"SkillExtractor başlatılırken hata: {e}")
            raise
        
        # Filtreleme ayarları
        data_config = config.get('data_processing', {})
        self.min_skill_length = data_config.get('min_skill_length', 2)
        self.skill_stopwords = set(data_config.get('skill_stopwords', []))
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Metinden becerileri çıkarır.
        
        Args:
            text: İşlenecek metin
        
        Returns:
            Çıkarılan beceri listesi
        """
        try:
            annotations = self.skill_extractor.annotate(text)
            
            skills = []
            
            # Tam eşleşmeler
            for match in annotations['results']['full_matches']:
                skills.append(match['doc_node_value'])
            
            # Ngram eşleşmeler
            for match in annotations['results']['ngram_scored']:
                skills.append(match['doc_node_value'])
            
            # Benzersiz yap ve normalize et
            skills = list(set([s.lower() for s in skills]))
            
            # Filtrele
            skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
            
            return skills
            
        except Exception as e:
            self.logger.error(f"Beceri çıkarma hatası: {e}", exc_info=True)
            return []
    
    def extract_skills_from_dataframe(self, df: pd.DataFrame, text_column: str = "tam_aciklama") -> pd.DataFrame:
        """
        DataFrame'deki tüm metinlerden becerileri çıkarır.
        
        Args:
            df: İşlenecek DataFrame
            text_column: Metin sütunu adı
        
        Returns:
            Beceri listesi eklenmiş DataFrame
        """
        self.logger.info(f"{len(df)} adet metin işleniyor...")
        
        skill_lists = []
        total = len(df)
        
        for idx, text in enumerate(df[text_column].astype(str)):
            try:
                skills = self.extract_skills_from_text(text)
                skill_lists.append(skills)
            except Exception as e:
                self.logger.warning(f"Satır {idx} işlenirken hata: {e}")
                skill_lists.append([])
            
            # İlerleme raporu
            if (idx + 1) % 50 == 0:
                self.logger.info(f"İlerleme: {idx+1}/{total} metin işlendi")
        
        df_result = df.copy()
        df_result["cikarilan_beceriler"] = skill_lists
        
        self.logger.info(f"Tüm metinler işlendi")
        return df_result

