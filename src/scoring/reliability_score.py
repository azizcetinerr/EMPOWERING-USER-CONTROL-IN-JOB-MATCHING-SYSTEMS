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
        Optimize edilmiş: Batch processing ve evidence sayısı sınırlaması.
        
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
        
        # Çok fazla evidence varsa en önemli olanları seç (performans için)
        # İlk 10 paragrafı kullan (genellikle yeterli)
        max_evidence = 10
        limited_evidence = evidence_list[:max_evidence] if len(evidence_list) > max_evidence else evidence_list
        
        # Kanıtlarla çiftler oluştur
        pairs = [(evidence, hypothesis) for evidence in limited_evidence]
        
        # Modele ver (batch processing zaten yapılıyor)
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
        Temel R skorunu kurum bilgilerine göre günceller.
        
        Args:
            base_r_score: Temel güvenilirlik skoru (0-100) - Beceri doğrulama skoru
            organization_name: Kurum adı
            verified: Kurum onaylı mı?
            address: Kurum adresi
            website: Kurum web sitesi
            project_count: Kurumun geçmiş proje sayısı
        
        Returns:
            Güncellenmiş R skoru (0-100)
        """
        reliability_config = self.config.get('reliability_scoring', {})
        enhanced_score = base_r_score
        
        # 1. Kurumun geçmiş proje sayısı bonusu
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
                self.logger.debug(f"Proje sayısı bonusu: +{project_bonus:.1f} (Proje sayısı: {project_count})")
        
        # 2. Onay statüsü bonusu
        if reliability_config.get('verified_bonus', {}).get('enabled', True):
            verified_bonus_value = reliability_config.get('verified_bonus', {}).get('bonus_score', 15.0) if verified else 0.0
            enhanced_score += verified_bonus_value
            if verified_bonus_value > 0:
                self.logger.debug(f"Onay statüsü bonusu: +{verified_bonus_value:.1f}")
        
        # 3. Eksik veri cezası
        if reliability_config.get('missing_data_penalties', {}).get('enabled', True):
            penalties = reliability_config.get('missing_data_penalties', {})
            missing_data_penalty = 0.0
            
            # Adres eksikse
            if not address or (isinstance(address, str) and address.strip() == ""):
                penalty = penalties.get('address', 10.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Adres eksik cezası: -{penalty:.1f}")
            
            # Web sitesi eksikse
            if not website or (isinstance(website, str) and website.strip() == ""):
                penalty = penalties.get('website', 10.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Web sitesi eksik cezası: -{penalty:.1f}")
            
            # Kurum adı eksikse
            if not organization_name or (isinstance(organization_name, str) and organization_name.strip() == ""):
                penalty = penalties.get('organization_name', 5.0)
                missing_data_penalty += penalty
                self.logger.debug(f"Kurum adı eksik cezası: -{penalty:.1f}")
            
            enhanced_score -= missing_data_penalty
        
        # 4. 0-100 aralığına normalize et
        if reliability_config.get('normalize_to_0_100', True):
            enhanced_score = max(0.0, min(100.0, enhanced_score))
        
        return enhanced_score
    
    def calculate_project_reliability_scores(
        self,
        df_projects: pd.DataFrame,
        cv_paragraphs: List[str],
        skill_column: str = "cikarilan_beceriler"
    ) -> pd.DataFrame:
        """
        Her proje için güvenilirlik skorunu hesaplar.
        Her projenin gereksinimlerini (becerilerini) alır ve bunların CV'de doğrulanma oranını hesaplar.
        Kurum bilgilerine göre skoru günceller (proje sayısı, onay statüsü, eksik veri cezaları).
        
        Args:
            df_projects: Proje DataFrame'i (beceri listeleri ve kurum bilgileri içermeli)
            cv_paragraphs: CV paragrafları
            skill_column: Beceri listesi sütunu adı
        
        Returns:
            R_Skoru eklenmiş DataFrame
        """
        self.logger.info(f"{len(df_projects)} adet proje için R skorları hesaplanıyor...")
        self.logger.info("Her proje için: Projenin gereksinimlerinin CV'de doğrulanma oranı + kurum bilgileri")
        
        df_projects = df_projects.copy()
        
        # Kurum bilgileri kontrolü ve varsayılan değerler
        if 'organization_name' not in df_projects.columns:
            df_projects['organization_name'] = None
            self.logger.warning("organization_name sütunu bulunamadı, None olarak ayarlandı")
        
        if 'verified' not in df_projects.columns:
            df_projects['verified'] = False
            self.logger.warning("verified sütunu bulunamadı, False olarak ayarlandı")
        
        if 'address' not in df_projects.columns:
            df_projects['address'] = None
            self.logger.warning("address sütunu bulunamadı, None olarak ayarlandı")
        
        if 'website' not in df_projects.columns:
            df_projects['website'] = None
            self.logger.warning("website sütunu bulunamadı, None olarak ayarlandı")
        
        # Kurum bazlı proje sayısını hesapla (eğer organization_name varsa)
        if 'organization_name' in df_projects.columns:
            # Sadece None olmayan kurum adlarını say
            valid_orgs = df_projects[df_projects['organization_name'].notna()]['organization_name']
            if len(valid_orgs) > 0:
                org_counts = valid_orgs.value_counts().to_dict()
                df_projects['project_count'] = df_projects['organization_name'].map(org_counts).fillna(0).astype(int)
                self.logger.info(f"Kurum bazlı proje sayıları hesaplandı (Toplam {len(org_counts)} farklı kurum)")
            else:
                df_projects['project_count'] = 0
                self.logger.warning("Geçerli kurum adı bulunamadı, proje sayısı 0 olarak ayarlandı")
        else:
            df_projects['project_count'] = 0
        
        r_scores = []
        base_r_scores = []
        total_projects = len(df_projects)
        
        # Progress reporting için
        processed = 0
        
        for idx, row in df_projects.iterrows():
            project_skills = row.get(skill_column, [])
            
            # 1. Temel R skoru: Beceri doğrulama
            if not project_skills or not isinstance(project_skills, list) or len(project_skills) == 0:
                base_r_score = 0.0
            else:
                # Bu projenin becerilerinin CV'de doğrulanma oranını hesapla
                # Tüm becerileri birleştirip batch processing yap
                valid_skills = [s.strip() for s in project_skills if s and isinstance(s, str) and len(s.strip()) > 0]
                
                if valid_skills:
                    # Batch processing: Tüm becerileri tek seferde doğrula
                    total_score = 0.0
                    for skill in valid_skills:
                        score = self.verify_claim(skill, cv_paragraphs)
                        total_score += score
                
                # Ortalama skor
                    base_r_score = total_score / len(valid_skills)
                else:
                    base_r_score = 0.0
            
            base_r_scores.append(base_r_score)
            processed += 1
            
            # Progress reporting (her 100 projede bir)
            if processed % 100 == 0 or processed == total_projects:
                self.logger.info(f"   İlerleme: {processed}/{total_projects} proje işlendi ({processed*100//total_projects}%)")
            
            # 2. Gelişmiş R skoru: Kurum bilgilerine göre güncelle
            enhanced_r_score = self.calculate_enhanced_reliability_score(
                base_r_score=base_r_score,
                organization_name=row.get('organization_name'),
                verified=row.get('verified', False),
                address=row.get('address'),
                website=row.get('website'),
                project_count=row.get('project_count', 0)
            )
            
            r_scores.append(enhanced_r_score)
            
            # İlerleme raporu
            if (idx + 1) % 100 == 0:
                self.logger.info(f"   İlerleme: {idx + 1}/{len(df_projects)} proje işlendi")
        
        df_result = df_projects.copy()
        df_result["R_Skoru"] = r_scores
        df_result["R_Skoru_Base"] = base_r_scores  # Temel skoru da kaydet (analiz için)
        
        # İstatistikler
        self.logger.info(f"\n📊 R Skoru İstatistikleri:")
        self.logger.info(f"  Temel R skoru (sadece beceri): Min={min(base_r_scores):.2f}, Max={max(base_r_scores):.2f}, Ortalama={sum(base_r_scores)/len(base_r_scores):.2f}")
        self.logger.info(f"  Gelişmiş R skoru (kurum bilgileri ile): Min={min(r_scores):.2f}, Max={max(r_scores):.2f}, Ortalama={sum(r_scores)/len(r_scores):.2f}")
        self.logger.info(f"  Ortalama fark: {(sum(r_scores)/len(r_scores)) - (sum(base_r_scores)/len(base_r_scores)):.2f} puan")
        self.logger.info(f"✅ Tüm {len(df_projects)} proje için R skoru hesaplandı")
        
        return df_result

