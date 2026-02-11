"""
Ana pipeline - Tüm işlemleri sırayla çalıştırır
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
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
    """Ana pipeline sınıfı"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Konfigürasyon dosyası yolu (None ise varsayılan)
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            name="pipeline",
            log_level=self.config['logging']['level'],
            log_file=self.config['logging']['file'],
            console=self.config['logging']['console']
        )
        
        # Çıktı dizinlerini oluştur
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Çıktı dizinlerini oluşturur"""
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
        IP1: Veri toplama aşaması (önbellek destekli)
        
        Args:
            preloaded_projects: Önceden yüklenmiş proje DataFrame'i (opsiyonel, önbellekten)
        """
        self.logger.info("=" * 80)
        self.logger.info("IP1: VERİ TOPLAMA VE STANDARDİZASYON BAŞLADI")
        self.logger.info("=" * 80)
        
        project_root = Path(__file__).parent.parent
        
        # Eğer önceden yüklenmiş projeler varsa ve becerileri/vektörleri varsa, onları kullan
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            self.logger.info(f"🔍 Önceden yüklenmiş projeler kullanılıyor: {len(preloaded_projects)} proje")
            
            # Becerilerin ve vektörlerin varlığını kontrol et
            has_skills = 'cikarilan_beceriler' in preloaded_projects.columns
            has_vectors = 'beceri_vektorleri' in preloaded_projects.columns
            
            if has_skills and has_vectors:
                self.logger.info("✅ Projelerde beceriler ve vektörler mevcut, scraping atlanıyor...")
                df_projects = preloaded_projects.copy()
            else:
                self.logger.info("⚠️  Projelerde beceriler veya vektörler eksik, sadece eksik kısımlar işlenecek...")
                df_projects = preloaded_projects.copy()
        else:
            # 1. Proje linklerini çek
            self.logger.info("Adım 1: Proje linkleri çekiliyor...")
            scraper = YouthPortalScraper(self.config, self.logger)
            links = scraper.get_project_links()
            
            # Linkleri kaydet
            output_file = self.config['data_sources']['youth_portal']['output_file']
            links_file = project_root / output_file
            with open(links_file, 'w', encoding='utf-8') as f:
                for link in links:
                    f.write(link + "\n")
            self.logger.info(f"Linkler kaydedildi: {links_file}")
            
            # 2. Proje detaylarını çek
            self.logger.info("Adım 2: Proje detayları çekiliyor...")
            detail_scraper = ProjectDetailScraper(self.config, self.logger)
            df_projects = detail_scraper.scrape_all_projects(links)
            
            # Kaydet
            output_path = get_output_path(self.config, 'project_details')
            df_projects.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Proje detayları kaydedildi: {output_path}")
        
        # 3. CV'yi parse et (her zaman yapılmalı - yeni CV için)
        self.logger.info("Adım 3: CV parse ediliyor...")
        try:
            cv_parser = EuropassCVParser(self.config, self.logger)
            cv_text, cv_skills = cv_parser.parse_cv()
        except Exception as cv_error:
            self.logger.error(f"❌ CV parsing hatası: {cv_error}", exc_info=True)
            raise
        
        # CV verilerini kaydet
        project_root = Path(__file__).parent.parent
        cv_text_file = project_root / self.config['output']['files']['candidate_skills_raw'].replace('.pkl', '.txt')
        with open(cv_text_file, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        
        cv_skills_df = pd.DataFrame(cv_skills, columns=["ham_beceri_adi"])
        cv_skills_path = project_root / self.config['output']['files']['candidate_skills_raw']
        cv_skills_df.to_pickle(cv_skills_path)
        self.logger.info(f"CV verileri kaydedildi")
        
        # 4. Becerileri çıkar
        self.logger.info("Adım 4: Beceriler çıkarılıyor...")
        skill_extractor = SkillExtractorModule(self.config, self.logger)
        
        # CV'den becerileri çıkar (her zaman yapılmalı - yeni CV için)
        candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
        candidate_skills_df = pd.DataFrame(candidate_skills, columns=["cikarilan_beceri_adi"])
        candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
        candidate_skills_df.to_pickle(candidate_skills_path)
        self.logger.info(f"Aday becerileri kaydedildi: {candidate_skills_path}")
        
        # Projelerden becerileri çıkar (sadece yoksa)
        if 'cikarilan_beceriler' not in df_projects.columns:
            self.logger.info("Projelerden beceriler çıkarılıyor...")
            df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
            projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
            df_projects.to_pickle(projects_skills_path)
            self.logger.info(f"Proje becerileri kaydedildi: {projects_skills_path}")
        else:
            self.logger.info("✅ Projelerde beceriler zaten mevcut, atlanıyor...")
        
        # 5. Vektörizasyon
        self.logger.info("Adım 5: Vektörizasyon yapılıyor...")
        vectorizer = SkillVectorizer(self.config, self.logger)
        
        # Aday becerilerini vektörize et (her zaman yapılmalı - yeni CV için)
        candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
        candidate_vectors_path = project_root / self.config['output']['files']['candidate_vectors']
        np.save(candidate_vectors_path, candidate_vectors)
        self.logger.info(f"Aday vektörleri kaydedildi: {candidate_vectors_path}")
        
        # Projeleri vektörize et (sadece yoksa)
        if 'beceri_vektorleri' not in df_projects.columns:
            self.logger.info("Proje vektörleri hesaplanıyor...")
            df_projects = vectorizer.vectorize_dataframe(df_projects)
            projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
            df_projects.to_pickle(projects_vectors_path)
            self.logger.info(f"Proje vektörleri kaydedildi: {projects_vectors_path}")
        else:
            self.logger.info("✅ Proje vektörleri zaten mevcut, atlanıyor...")
        
        self.logger.info("IP1 TAMAMLANDI!")
        return df_projects, candidate_vectors, cv_text, candidate_skills
    
    def run_ip2_matching_score(self, df_projects: pd.DataFrame, candidate_vectors: np.ndarray):
        """IP2: Eşleştirme skoru hesaplama"""
        self.logger.info("=" * 80)
        self.logger.info("IP2: EŞLEŞTİRME SKORU (M-SKORU) HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        matching_calc = MatchingScoreCalculator(self.config, self.logger)
        df_projects = matching_calc.calculate_all_scores(df_projects, candidate_vectors)
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['matching_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"M skorları kaydedildi: {output_path}")
        
        self.logger.info("IP2 TAMAMLANDI!")
        return df_projects
    
    def run_ip3_sincerity_score(self, df_projects: pd.DataFrame):
        """IP3: Samimiyet skoru hesaplama"""
        self.logger.info("=" * 80)
        self.logger.info("IP3: SAMİMİYET SKORU (S-SKORU) HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        sincerity_calc = SincerityScoreCalculator(self.config, self.logger)
        df_projects = sincerity_calc.calculate_all_scores(df_projects)
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['sincerity_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"S skorları kaydedildi: {output_path}")
        
        self.logger.info("IP3 TAMAMLANDI!")
        return df_projects
    
    def run_ip4_reliability_score(self, candidate_skills: list, cv_text: str):
        """IP4: Güvenilirlik skoru hesaplama"""
        self.logger.info("=" * 80)
        self.logger.info("IP4: GÜVENİLİRLİK SKORU (R-SKORU) HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
        cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
        r_score, r_details = reliability_calc.calculate_reliability_score(candidate_skills, cv_paragraphs)
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        details_path = project_root / self.config['output']['files']['reliability_details']
        r_details.to_pickle(details_path)
        self.logger.info(f"R skoru detayları kaydedildi: {details_path}")
        
        score_path = project_root / self.config['output']['files']['reliability_score']
        with open(score_path, 'w') as f:
            f.write(str(r_score))
        self.logger.info(f"R skoru kaydedildi: {score_path}")
        
        self.logger.info("IP4 TAMAMLANDI!")
        return r_score
    
    def run_ip5_final_ranking(self, df_projects: pd.DataFrame, r_score: float):
        """IP5: Nihai adil sıralama"""
        self.logger.info("=" * 80)
        self.logger.info("IP5: ADİL SIRALAMA (MOO) HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        ranking_calc = FinalRankingCalculator(self.config, self.logger)
        df_final = ranking_calc.calculate_ranking(df_projects, r_score)
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['final_ranking']
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Nihai sıralama kaydedildi: {output_path}")
        
        # Top 10'u göster
        self.logger.info("\n🏆 TOP 10 PROJE:")
        for i, row in df_final.head(10).iterrows():
            proje_id = row['link'].split('/')[-1].replace('_en', '')
            self.logger.info(
                f"{i+1}. [Adil Skor: {row['Adil_Skor']:.2f}] "
                f"[M: {row['M_Skoru']:.1f}, S: {row['S_Skoru']:.1f}, R: {r_score:.1f}] "
                f"{proje_id}"
            )
        
        self.logger.info("IP5 TAMAMLANDI!")
        return df_final
    
    def run_full_pipeline(self):
        """Tüm pipeline'ı çalıştırır"""
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
            self.logger.info("✅ TÜM PIPELINE BAŞARIYLA TAMAMLANDI!")
            self.logger.info("=" * 80)
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Pipeline hatası: {e}", exc_info=True)
            raise


def main():
    """Ana fonksiyon"""
    pipeline = ProjectPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

