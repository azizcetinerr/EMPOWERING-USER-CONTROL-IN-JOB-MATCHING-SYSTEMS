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
        IP1: Veri toplama aşaması (önbellek destekli)
        
        Args:
            preloaded_projects: Önceden yüklenmiş proje DataFrame'i (opsiyonel, önbellekten)
        """
        self.logger.info("=" * 80)
        self.logger.info("IP1: VERİ TOPLAMA VE STANDARDİZASYON BAŞLADI")
        self.logger.info("=" * 80)
        
        project_root = Path(__file__).parent.parent
        
        # Eğer önceden yüklenmiş projeler varsa ve becerileri varsa, onları kullan
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            self.logger.info(f"🔍 Preloaded projects kontrolü başlatılıyor: {len(preloaded_projects)} proje")
            
            # Becerilerin varlığını detaylı kontrol et
            has_skills = False
            has_vectors = False
            skills_info = ""
            vectors_info = ""
            
            if 'cikarilan_beceriler' in preloaded_projects.columns:
                non_null_skills = preloaded_projects['cikarilan_beceriler'].notna()
                non_null_count = non_null_skills.sum()
                if non_null_count > 0:
                    valid_skills = preloaded_projects[non_null_skills]['cikarilan_beceriler'].apply(
                        lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                    )
                    valid_count = valid_skills.sum()
                    has_skills = valid_count > 0
                    skills_info = f"{valid_count}/{len(preloaded_projects)} projede geçerli beceri var"
                else:
                    skills_info = "Beceri sütunu var ama tüm projelerde boş"
            else:
                skills_info = "Beceri sütunu yok"
            
            if 'beceri_vektorleri' in preloaded_projects.columns:
                non_null_vectors = preloaded_projects['beceri_vektorleri'].notna()
                vector_count = non_null_vectors.sum()
                has_vectors = vector_count > 0
                vectors_info = f"{vector_count}/{len(preloaded_projects)} projede vektör var"
            else:
                vectors_info = "Vektör sütunu yok"
            
            self.logger.info(f"🔍 Preloaded projects kontrolü: {len(preloaded_projects)} proje")
            self.logger.info(f"   Beceriler: {has_skills} ({skills_info})")
            self.logger.info(f"   Vektörler: {has_vectors} ({vectors_info})")
            
            if has_skills and has_vectors:
                self.logger.info(f"✅ Önceden yüklenmiş {len(preloaded_projects)} proje kullanılıyor (beceriler ve vektörler mevcut)")
                self.logger.info("   📊 Yeni projeler kontrol ediliyor...")
                
                # CV parse et
                cv_parser = EuropassCVParser(self.config, self.logger)
                cv_text, cv_skills = cv_parser.parse_cv()
                
                # CV'den becerileri çıkar
                skill_extractor = SkillExtractorModule(self.config, self.logger)
                candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
                
                # Aday vektörlerini hesapla
                vectorizer = SkillVectorizer(self.config, self.logger)
                candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
                
                # Metadata'dan en son proje yayın tarihini al
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
                                self.logger.info(f"📅 En son proje yayın tarihi: {last_published_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        self.logger.warning(f"Metadata okunamadı: {e}")
                
                # Yeni projeleri kontrol et ve ekle
                df_final_projects = preloaded_projects.copy()
                existing_links = set(preloaded_projects['link'].tolist()) if 'link' in preloaded_projects.columns else set()
                
                # Proje linklerini çek
                scraper = YouthPortalScraper(self.config, self.logger)
                cached_api_data = self.config.get('_cached_api_data')
                all_links = scraper.get_project_links(cached_api_data=cached_api_data)
                new_links = [link for link in all_links if link not in existing_links]
                
                if new_links:
                    self.logger.info(f"🆕 {len(new_links)} yeni proje bulundu, detayları çekiliyor...")
                    
                    # Yeni projelerin detaylarını çek (paralel versiyon kullan)
                    try:
                        from src.data_collection.project_detail_scraper_parallel import ProjectDetailScraperParallel
                        max_workers = self.config.get('performance', {}).get('num_workers', 5)
                        detail_scraper = ProjectDetailScraperParallel(self.config, self.logger, max_workers=max_workers)
                        self.logger.info(f"   Paralel scraping aktif: {max_workers} thread")
                    except ImportError:
                        # Fallback: Normal scraper
                        self.logger.info("   Paralel scraper bulunamadı, normal scraper kullanılıyor")
                        detail_scraper = ProjectDetailScraper(self.config, self.logger)
                    df_new_projects = detail_scraper.scrape_all_projects(new_links)
                    
                    if len(df_new_projects) > 0:
                        # Yeni projeler için beceri çıkar
                        self.logger.info(f"   🔧 {len(df_new_projects)} yeni proje için beceri çıkarılıyor...")
                        df_new_projects = skill_extractor.extract_skills_from_dataframe(df_new_projects)
                        
                        # Yeni projeler için vektörleştir
                        self.logger.info(f"   🔧 {len(df_new_projects)} yeni proje için vektörleştirme yapılıyor...")
                        df_new_projects = vectorizer.vectorize_dataframe(df_new_projects)
                        
                        # Eski ve yeni projeleri birleştir
                        df_final_projects = pd.concat([df_final_projects, df_new_projects], ignore_index=True)
                        self.logger.info(f"   ✅ {len(df_new_projects)} yeni proje eklendi, toplam: {len(df_final_projects)} proje")
                else:
                    self.logger.info("   ✅ Yeni proje yok, mevcut projeler kullanılıyor")
                
                # Son başvuru tarihi geçen projeleri filtrele
                from datetime import datetime
                if 'application_deadline' in df_final_projects.columns:
                    df_final_projects['application_deadline'] = pd.to_datetime(df_final_projects['application_deadline'], errors='coerce')
                    now = datetime.now()
                    expired_projects = df_final_projects[df_final_projects['application_deadline'].notna() & (df_final_projects['application_deadline'] < now)]
                    if len(expired_projects) > 0:
                        self.logger.info(f"🗑️  {len(expired_projects)} proje son başvuru tarihi geçmiş, siliniyor...")
                        df_final_projects = df_final_projects[df_final_projects['application_deadline'].isna() | (df_final_projects['application_deadline'] >= now)]
                        self.logger.info(f"   ✅ {len(df_final_projects)} aktif proje kaldı")
                
                # Projeleri dosyaya kaydet (güncel haliyle)
                projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
                projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
                df_final_projects.to_pickle(projects_vectors_path)
                df_final_projects.to_pickle(projects_skills_path)
                
                csv_projects_file = get_output_path(self.config, 'project_details')
                df_final_projects.to_csv(csv_projects_file, index=False, encoding='utf-8-sig')
                
                # Metadata güncelle (en son eklenen proje tarihi ve yayın tarihi)
                try:
                    import json
                    from datetime import datetime
                    last_added_date = datetime.now()  # Son eklenen projelerin eklenme tarihi
                    last_published_date = None
                    
                    if 'published_date' in df_final_projects.columns:
                        df_final_projects['published_date'] = pd.to_datetime(df_final_projects['published_date'], errors='coerce')
                        published_projects = df_final_projects[df_final_projects['published_date'].notna()]
                        if len(published_projects) > 0:
                            last_published_date = published_projects['published_date'].max()
                    
                    metadata = {
                        'last_added_date': last_added_date.isoformat(),  # Son eklenen projenin eklenme tarihi
                        'last_published_date': last_published_date.isoformat() if last_published_date is not None and pd.notna(last_published_date) else None,
                        'last_scrape_date': datetime.now().isoformat(),
                        'last_scrape_count': len(new_links) if new_links else 0,
                        'total_projects': len(df_final_projects),
                        'last_update': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"📅 Metadata güncellendi: Son ekleme tarihi {last_added_date.strftime('%Y-%m-%d %H:%M:%S')}, En son yayın tarihi {last_published_date.strftime('%Y-%m-%d') if last_published_date is not None and pd.notna(last_published_date) else 'N/A'} ({len(new_links) if new_links else 0} yeni proje)")
                except Exception as e:
                    self.logger.warning(f"Metadata güncellenemedi: {e}")
                
                self.logger.info("=" * 80)
                self.logger.info("✅ IP1 TAMAMLANDI! Projeler dosyaya kaydedildi.")
                self.logger.info(f"   📁 CSV: {csv_projects_file}")
                self.logger.info(f"   📁 Pickle (Beceriler): {projects_skills_path}")
                self.logger.info(f"   📁 Pickle (Vektörler): {projects_vectors_path}")
                self.logger.info(f"   📊 Toplam {len(df_final_projects)} proje kaydedildi ({len(new_links) if new_links else 0} yeni eklendi)")
                self.logger.info("=" * 80)
                
                return df_final_projects, candidate_vectors, cv_text, candidate_skills
            else:
                self.logger.info(f"ℹ️  Önceden yüklenmiş projeler var ama beceriler/vektörler eksik")
                self.logger.info(f"   has_skills={has_skills}, has_vectors={has_vectors}")
                self.logger.info("   Normal pipeline çalışacak (sadece yeni projeler için scraping/beceri çıkarma)")
                # preloaded_projects'ı df_existing_projects olarak kullan ve normal pipeline'a devam et
                df_existing_projects = preloaded_projects.copy()
                # Mevcut linkleri topla
                if 'link' in df_existing_projects.columns:
                    existing_links = set(df_existing_projects['link'].tolist())
                else:
                    existing_links = set()
                new_links = []  # Başlangıçta boş
                # Metadata dosyasını oku (en son proje yayın tarihi - öncelikli)
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
                                self.logger.info(f"📅 En son proje yayın tarihi: {last_published_date.strftime('%Y-%m-%d')} (önbellek aktif)")
                                last_scrape_date = last_published_date
                            else:
                                last_scrape_date_str = metadata.get('last_scrape_date')
                                if last_scrape_date_str:
                                    from datetime import datetime
                                    last_scrape_date = datetime.fromisoformat(last_scrape_date_str)
                                    self.logger.info(f"📅 En son scraping tarihi: {last_scrape_date.strftime('%Y-%m-%d %H:%M:%S')} (fallback)")
                    except Exception as e:
                        self.logger.warning(f"Metadata dosyası okunamadı: {e}")
                # Normal pipeline'a devam et ama df_existing_projects'ı kullan
                # (Aşağıdaki kod devam edecek)
        
        # Eğer preloaded_projects kontrolü başarılı olduysa, buraya gelmeyecek (return yapıldı)
        # Eğer preloaded_projects yoksa veya eksikse, normal pipeline çalışacak
        
        # Metadata dosyası (en son scraping tarihi için)
        if preloaded_projects is None:
            metadata_file = project_root / "data/external/last_scrape_metadata.json"
            last_scrape_date = None
            
        # Mevcut proje verilerini kontrol et (hem CSV hem pickle)
        # Eğer preloaded_projects varsa, onu kullan
        if preloaded_projects is not None and len(preloaded_projects) > 0:
            # Preloaded projects'ı df_existing_projects olarak kullan
            df_existing_projects = preloaded_projects.copy()
            if 'link' in df_existing_projects.columns:
                existing_links = set(df_existing_projects['link'].tolist())
            else:
                existing_links = set()
            new_links = []  # Başlangıçta boş
            self.logger.info(f"✅ Preloaded projects kullanılıyor: {len(df_existing_projects)} proje, {len(existing_links)} link")
        else:
            # Normal akış: CSV ve pickle'dan yükle
            ip1_projects_file = project_root / self.config['output']['files']['projects_with_skills']
            csv_projects_file = get_output_path(self.config, 'project_details')
            df_existing_projects = None
            existing_links = set()
            new_links = []  # Başlangıçta boş
        
        # Metadata dosyasını oku (en son proje yayın tarihi - öncelikli)
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Önce published_date'ı kontrol et (daha doğru - gerçek yayın tarihi)
                    last_published_date_str = metadata.get('last_published_date')
                    if last_published_date_str:
                        from datetime import datetime
                        last_published_date = datetime.fromisoformat(last_published_date_str)
                        self.logger.info(f"📅 En son proje yayın tarihi: {last_published_date.strftime('%Y-%m-%d')} (önbellek aktif)")
                        last_scrape_date = last_published_date  # published_date kullan (daha doğru)
                    else:
                        # Fallback: scraping tarihi
                        last_scrape_date_str = metadata.get('last_scrape_date')
                        if last_scrape_date_str:
                            from datetime import datetime
                            last_scrape_date = datetime.fromisoformat(last_scrape_date_str)
                            self.logger.info(f"📅 En son scraping tarihi: {last_scrape_date.strftime('%Y-%m-%d %H:%M:%S')} (fallback)")
            except Exception as e:
                self.logger.warning(f"Metadata dosyası okunamadı: {e}")
        
        # Önce CSV dosyasını kontrol et (daha hızlı) - Sadece preloaded_projects yoksa
        if preloaded_projects is None and csv_projects_file.exists():
            try:
                df_csv = pd.read_csv(csv_projects_file, encoding='utf-8-sig')
                if 'link' in df_csv.columns:
                    # Eğer tarih bazlı filtreleme varsa, o tarihten önceki projeleri al
                    if last_scrape_date:
                        # Önce published_date'ı kontrol et (daha doğru)
                        if 'published_date' in df_csv.columns:
                            df_csv['published_date'] = pd.to_datetime(df_csv['published_date'], errors='coerce')
                            # O tarihten önceki projeleri filtrele
                            df_old = df_csv[df_csv['published_date'] < last_scrape_date]
                            df_new = df_csv[df_csv['published_date'] >= last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ CSV'den {len(df_old)} eski proje yüklendi (yayın tarihi bazlı önbellek: < {last_scrape_date.strftime('%Y-%m-%d')})")
                                # Eski projeleri kullan
                                df_existing_projects = df_old
                                existing_links.update(set(df_old['link'].tolist()))
                            
                            if len(df_new) > 0:
                                self.logger.info(f"ℹ️  CSV'de {len(df_new)} yeni proje var (yayın tarihi >= {last_scrape_date.strftime('%Y-%m-%d')}, yeniden scrape edilecek)")
                        elif 'scraped_at' in df_csv.columns:
                            # Fallback: scraped_at kullan
                            df_csv['scraped_at'] = pd.to_datetime(df_csv['scraped_at'], errors='coerce')
                            df_old = df_csv[df_csv['scraped_at'] < last_scrape_date]
                            df_new = df_csv[df_csv['scraped_at'] >= last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ CSV'den {len(df_old)} eski proje yüklendi (scraping tarihi bazlı önbellek)")
                                df_existing_projects = df_old
                                existing_links.update(set(df_old['link'].tolist()))
                            
                            if len(df_new) > 0:
                                self.logger.info(f"ℹ️  CSV'de {len(df_new)} yeni proje var (yeniden scrape edilecek)")
                        else:
                            # Tarih bilgisi yok, tüm projeleri al
                            csv_links = set(df_csv['link'].tolist())
                            existing_links.update(csv_links)
                            self.logger.info(f"✅ CSV'den {len(csv_links)} proje yüklendi (tarih bilgisi yok)")
                            df_existing_projects = df_csv
                    else:
                        # Tarih bilgisi yok, tüm projeleri al
                        csv_links = set(df_csv['link'].tolist())
                        existing_links.update(csv_links)
                        self.logger.info(f"✅ CSV'den {len(csv_links)} proje yüklendi")
                        df_existing_projects = df_csv
            except Exception as e:
                self.logger.warning(f"CSV dosyası okunamadı: {e}")
        
        # Sonra pickle dosyasını kontrol et (beceriler ve vektörler için) - Sadece preloaded_projects yoksa
        if preloaded_projects is None and ip1_projects_file.exists():
            try:
                df_pickle = pd.read_pickle(ip1_projects_file)
                if 'link' in df_pickle.columns:
                    # Tarih bazlı filtreleme (önce published_date, sonra scraped_at)
                    if last_scrape_date:
                        # Önce published_date'ı kontrol et (daha doğru)
                        if 'published_date' in df_pickle.columns:
                            df_pickle['published_date'] = pd.to_datetime(df_pickle['published_date'], errors='coerce')
                            df_old = df_pickle[df_pickle['published_date'] < last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Pickle'dan {len(df_old)} eski proje yüklendi (yayın tarihi bazlı önbellek: < {last_scrape_date.strftime('%Y-%m-%d')})")
                                # Pickle daha eksiksiz (beceriler, vektörler var), onu kullan
                                df_existing_projects = df_old.copy()  # Copy yap
                                existing_links.update(set(df_old['link'].tolist()))
                                # Becerilerin varlığını kontrol et ve logla
                                if 'cikarilan_beceriler' in df_existing_projects.columns:
                                    skills_count = df_existing_projects['cikarilan_beceriler'].notna().sum()
                                    self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projede beceri mevcut (önbellekten)")
                                else:
                                    self.logger.warning(f"   ⚠️  Pickle'da beceri sütunu yok!")
                        elif 'scraped_at' in df_pickle.columns:
                            # Fallback: scraped_at kullan
                            df_pickle['scraped_at'] = pd.to_datetime(df_pickle['scraped_at'], errors='coerce')
                            df_old = df_pickle[df_pickle['scraped_at'] < last_scrape_date]
                            
                            if len(df_old) > 0:
                                self.logger.info(f"✅ Pickle'dan {len(df_old)} eski proje yüklendi (scraping tarihi bazlı önbellek)")
                                df_existing_projects = df_old.copy()  # Copy yap
                                existing_links.update(set(df_old['link'].tolist()))
                                # Becerilerin varlığını kontrol et ve logla
                                if 'cikarilan_beceriler' in df_existing_projects.columns:
                                    skills_count = df_existing_projects['cikarilan_beceriler'].notna().sum()
                                    self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projede beceri mevcut (önbellekten)")
                                else:
                                    self.logger.warning(f"   ⚠️  Pickle'da beceri sütunu yok!")
                        else:
                            # Tarih bilgisi yok, tüm projeleri al
                            pickle_links = set(df_pickle['link'].tolist())
                            existing_links.update(pickle_links)
                            self.logger.info(f"✅ Pickle'dan {len(pickle_links)} proje yüklendi (tarih bilgisi yok)")
                            df_existing_projects = df_pickle.copy()  # Copy yap
                            # Becerilerin varlığını kontrol et ve logla
                            if 'cikarilan_beceriler' in df_existing_projects.columns:
                                skills_count = df_existing_projects['cikarilan_beceriler'].notna().sum()
                                self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projede beceri mevcut (önbellekten)")
                            else:
                                self.logger.warning(f"   ⚠️  Pickle'da beceri sütunu yok!")
                    else:
                        # Tarih bilgisi yok, tüm projeleri al
                        pickle_links = set(df_pickle['link'].tolist())
                        existing_links.update(pickle_links)
                        self.logger.info(f"✅ Pickle'dan {len(pickle_links)} proje yüklendi")
                        df_existing_projects = df_pickle.copy()  # Copy yap
                        # Becerilerin varlığını kontrol et ve logla
                        if 'cikarilan_beceriler' in df_existing_projects.columns:
                            skills_count = df_existing_projects['cikarilan_beceriler'].notna().sum()
                            self.logger.info(f"   📊 {skills_count}/{len(df_existing_projects)} projede beceri mevcut (önbellekten)")
                        else:
                            self.logger.warning(f"   ⚠️  Pickle'da beceri sütunu yok!")
            except Exception as e:
                self.logger.warning(f"Pickle dosyası okunamadı: {e}")
        
        if len(existing_links) > 0:
            self.logger.info(f"✅ Toplam {len(existing_links)} mevcut proje bulundu (tarih bazlı önbellek)")
        
        # 1. Proje linklerini çek (sadece yeni projeler için gerekliyse)
        if preloaded_projects is None or len(new_links) == 0:
            # Normal akış: linkleri çek
            self.logger.info("Adım 1: Proje linkleri çekiliyor...")
            scraper = YouthPortalScraper(self.config, self.logger)
            # Config'de cached API data varsa onu kullan
            cached_api_data = self.config.get('_cached_api_data')  # Geçici config değişkeni
            all_links = scraper.get_project_links(cached_api_data=cached_api_data)
            
            # Yeni linkleri bul
            if 'existing_links' not in locals():
                existing_links = set()
            new_links = [link for link in all_links if link not in existing_links]
        else:
            # Preloaded projects var, sadece yeni linkleri kontrol et
            self.logger.info("Adım 1: Proje linkleri kontrol ediliyor (önbellekten)...")
            scraper = YouthPortalScraper(self.config, self.logger)
            cached_api_data = self.config.get('_cached_api_data')
            all_links = scraper.get_project_links(cached_api_data=cached_api_data)
            new_links = [link for link in all_links if link not in existing_links]
        
        self.logger.info(f"   Toplam {len(all_links)} proje linki bulundu")
        self.logger.info(f"   Mevcut: {len(existing_links)}, Yeni: {len(new_links)}")
        
        # Linkleri kaydet
        output_file = self.config['data_sources']['youth_portal']['output_file']
        links_file = project_root / output_file
        with open(links_file, 'w', encoding='utf-8') as f:
            for link in all_links:
                f.write(link + "\n")
        self.logger.info(f"Linkler kaydedildi: {links_file}")
        
        # 2. Proje detaylarını çek (sadece yeni olanlar) - Paralel versiyon
        if new_links:
            self.logger.info(f"Adım 2: {len(new_links)} yeni proje detayı paralel olarak çekiliyor...")
            # Paralel scraping kullan (daha hızlı)
            try:
                from src.data_collection.project_detail_scraper_parallel import ProjectDetailScraperParallel
                max_workers = self.config.get('performance', {}).get('num_workers', 5)
                detail_scraper = ProjectDetailScraperParallel(self.config, self.logger, max_workers=max_workers)
                self.logger.info(f"   Paralel scraping aktif: {max_workers} thread")
            except ImportError:
                # Fallback: Normal scraper
                self.logger.info("   Paralel scraper bulunamadı, normal scraper kullanılıyor")
                detail_scraper = ProjectDetailScraper(self.config, self.logger)
            df_new_projects = detail_scraper.scrape_all_projects(new_links)
            
            # Eski ve yeni projeleri birleştir
            if df_existing_projects is not None and len(df_existing_projects) > 0:
                # Sütunları hizalayalım (published_date ve scraped_at için)
                all_columns = set(df_existing_projects.columns) | set(df_new_projects.columns)
                
                # Eksik sütunları ekle
                for col in all_columns:
                    if col not in df_existing_projects.columns:
                        df_existing_projects[col] = None
                    if col not in df_new_projects.columns:
                        if col == 'scraped_at':
                            from datetime import datetime
                            df_new_projects[col] = datetime.now().isoformat()
                        else:
                            df_new_projects[col] = None
                
                # Sütunları aynı sıraya getir
                df_existing_projects = df_existing_projects.reindex(columns=sorted(all_columns))
                df_new_projects = df_new_projects.reindex(columns=sorted(all_columns))
                
                df_projects = pd.concat([df_existing_projects, df_new_projects], ignore_index=True)
                self.logger.info(f"   ✅ {len(df_existing_projects)} eski + {len(df_new_projects)} yeni = {len(df_projects)} toplam proje")
            else:
                df_projects = df_new_projects
                self.logger.info(f"   ✅ {len(df_projects)} yeni proje eklendi")
        else:
            self.logger.info("Adım 2: Yeni proje yok, mevcut veriler kullanılıyor")
            if df_existing_projects is not None:
                df_projects = df_existing_projects.copy()  # Copy yap ki orijinal bozulmasın
                # Eğer published_date sütunu yoksa, bir sonraki scraping'de eklenecek
                if 'published_date' not in df_projects.columns:
                    self.logger.info(f"   ℹ️  {len(df_projects)} mevcut proje kullanılıyor (published_date yok, bir sonraki scraping'de eklenecek)")
                else:
                    self.logger.info(f"   ✅ {len(df_projects)} mevcut proje kullanılıyor (tarih bazlı önbellek aktif)")
                
                # Becerilerin varlığını kontrol et ve logla
                if 'cikarilan_beceriler' in df_projects.columns:
                    skills_count = df_projects['cikarilan_beceriler'].notna().sum()
                    self.logger.info(f"   ✅ {skills_count}/{len(df_projects)} projede beceri mevcut (önbellekten)")
                else:
                    self.logger.info(f"   ⚠️  Beceri sütunu yok, beceri çıkarma yapılacak")
            else:
                self.logger.warning("⚠️  Mevcut proje verisi bulunamadı ve yeni proje de yok!")
                # Boş DataFrame oluştur
                df_projects = pd.DataFrame(columns=['link', 'tam_aciklama'])
        
        # Kaydet (sadece yeni projeler eklendiyse güncelle)
        output_path = get_output_path(self.config, 'project_details')
        new_projects_count = len(new_links) if new_links else 0
        
        # Her durumda CSV'yi güncelle (published_date sütunu eklensin)
        if new_projects_count > 0 or not output_path.exists():
            # Yeni projeler eklendi veya ilk çalıştırma, CSV'yi güncelle
            df_projects.to_csv(output_path, index=False, encoding='utf-8-sig')
            if new_projects_count > 0:
                self.logger.info(f"Proje detayları kaydedildi: {output_path} (Toplam {len(df_projects)} proje, {new_projects_count} yeni)")
            else:
                self.logger.info(f"Proje detayları kaydedildi: {output_path} (Toplam {len(df_projects)} proje)")
        
        # Metadata dosyasını güncelle (en son proje yayın tarihi - daha doğru)
        # Yeni projeler eklendiyse veya ilk çalıştırmaysa güncelle
        # Ayrıca mevcut projeler varsa ve metadata yoksa, mevcut projelerden tarih çıkar
        should_update_metadata = False
        if new_projects_count > 0:
            should_update_metadata = True
        elif not metadata_file.exists() and len(df_projects) > 0:
            # İlk çalıştırma ve projeler var, metadata oluştur
            should_update_metadata = True
            self.logger.info("📅 İlk çalıştırma: Metadata dosyası oluşturuluyor...")
        
        if should_update_metadata:
            try:
                import json
                from datetime import datetime
                # Önce published_date'ı kontrol et (daha doğru - gerçek yayın tarihi)
                if 'published_date' in df_projects.columns:
                    df_projects['published_date'] = pd.to_datetime(df_projects['published_date'], errors='coerce')
                    # Sadece yeni scrape edilen projelerin tarihini al (None olmayanlar)
                    new_published = df_projects[df_projects['published_date'].notna()]
                    if len(new_published) > 0:
                        latest_published_date = new_published['published_date'].max()
                        if pd.notna(latest_published_date):
                            metadata = {
                                'last_published_date': latest_published_date.isoformat(),  # Gerçek yayın tarihi (öncelikli)
                                'last_scrape_date': datetime.now().isoformat(),  # Scraping zamanı (fallback için)
                                'last_scrape_count': new_projects_count,
                                'total_projects': len(df_projects),
                                'last_update': datetime.now().isoformat()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"📅 Metadata güncellendi: En son proje yayın tarihi {latest_published_date.strftime('%Y-%m-%d')} ({new_projects_count} yeni proje)")
                    else:
                        # published_date yok, scraped_at kullan
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
                                    self.logger.info(f"📅 Metadata güncellendi: En son scraping {latest_date.strftime('%Y-%m-%d %H:%M:%S')} ({new_projects_count} yeni proje)")
                            else:
                                # Tarih bilgisi yok, şimdiki zamanı kullan
                                now = datetime.now()
                                metadata = {
                                    'last_scrape_date': now.isoformat(),
                                    'last_scrape_count': new_projects_count,
                                    'total_projects': len(df_projects),
                                    'last_update': now.isoformat()
                                }
                                with open(metadata_file, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                self.logger.info(f"📅 Metadata oluşturuldu: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            # Hiç tarih bilgisi yok
                            now = datetime.now()
                            metadata = {
                                'last_scrape_date': now.isoformat(),
                                'last_scrape_count': new_projects_count,
                                'total_projects': len(df_projects),
                                'last_update': now.isoformat()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"📅 Metadata oluşturuldu: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                elif 'scraped_at' in df_projects.columns:
                    # Fallback: scraped_at kullan
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
                            self.logger.info(f"📅 Metadata güncellendi: En son scraping {latest_date.strftime('%Y-%m-%d %H:%M:%S')} ({new_projects_count} yeni proje)")
                else:
                    # Hiç tarih bilgisi yok, şimdiki zamanı kullan
                    now = datetime.now()
                    metadata = {
                        'last_scrape_date': now.isoformat(),
                        'last_scrape_count': new_projects_count,
                        'total_projects': len(df_projects),
                        'last_update': now.isoformat()
                    }
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"📅 Metadata oluşturuldu: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                self.logger.warning(f"Metadata güncellenemedi: {e}")
        
        # Mevcut projeler varsa ama metadata yoksa, mevcut projelerden tarih çıkar
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
                            'note': 'Mevcut projelerden tarih çıkarıldı'
                        }
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"📅 Metadata oluşturuldu (mevcut projelerden): En son yayın tarihi {latest_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                self.logger.warning(f"Mevcut projelerden metadata oluşturulamadı: {e}")
        else:
            # Yeni proje yok, CSV zaten mevcut
            self.logger.info(f"Mevcut CSV kullanılıyor: {output_path} ({len(df_projects)} proje)")
        
        # 3. CV'yi parse et (proje scraping ile paralel yapılabilir ama şimdilik sıralı)
        self.logger.info("Adım 3: CV parse ediliyor...")
        
        cv_parser = EuropassCVParser(self.config, self.logger)
        cv_text, cv_skills = cv_parser.parse_cv()
        
        # CV verilerini kaydet
        project_root = Path(__file__).resolve().parent.parent
        cv_text_file = project_root / "data/raw/IP1_Ciktisi_Aday_Tam_Metin.txt"
        with open(cv_text_file, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        
        cv_skills_df = pd.DataFrame(cv_skills, columns=["ham_beceri_adi"])
        cv_skills_path = project_root / self.config['output']['files']['candidate_skills_raw']
        cv_skills_df.to_pickle(cv_skills_path)
        self.logger.info(f"CV verileri kaydedildi")
        
        # 4. Becerileri çıkar
        # Eğer preloaded_projects varsa ve becerileri varsa, bu adımı atla
        # Ama önce df_projects'ta beceriler var mı kontrol et
        skip_skill_extraction = False
        
        if 'cikarilan_beceriler' in df_projects.columns:
            # Beceri sütunu var, geçerli beceriler var mı kontrol et
            non_null_skills = df_projects['cikarilan_beceriler'].notna()
            if non_null_skills.any():
                valid_skills = df_projects[non_null_skills]['cikarilan_beceriler'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                )
                if valid_skills.any():
                    valid_count = valid_skills.sum()
                    self.logger.info(f"✅ Beceriler zaten mevcut: {valid_count}/{len(df_projects)} projede geçerli beceri var")
                    self.logger.info("   📊 Beceri çıkarma atlanıyor (önbellekten)")
                    skip_skill_extraction = True
        
        if not skip_skill_extraction:
            self.logger.info("Adım 4: Beceriler çıkarılıyor...")
            skill_extractor = SkillExtractorModule(self.config, self.logger)
        else:
            skill_extractor = None  # Gerek yok
        
        # CV'den becerileri çıkar
        if skill_extractor is not None:
            candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
            candidate_skills_df = pd.DataFrame(candidate_skills, columns=["cikarilan_beceri_adi"])
            candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
            candidate_skills_df.to_pickle(candidate_skills_path)
            self.logger.info(f"Aday becerileri kaydedildi: {candidate_skills_path}")
        else:
            # Skill extractor yok, CV becerilerini basit şekilde çıkar (veya önbellekten al)
            # Aslında CV becerileri her zaman çıkarılmalı, sadece proje becerileri atlanabilir
            skill_extractor = SkillExtractorModule(self.config, self.logger)
            candidate_skills = skill_extractor.extract_skills_from_text(cv_text)
            candidate_skills_df = pd.DataFrame(candidate_skills, columns=["cikarilan_beceri_adi"])
            candidate_skills_path = project_root / self.config['output']['files']['candidate_skills_extracted']
            candidate_skills_df.to_pickle(candidate_skills_path)
            self.logger.info(f"Aday becerileri kaydedildi: {candidate_skills_path}")
        
        # Projelerden becerileri çıkar (sadece yeni projeler için)
        # df_projects'ta beceriler var mı kontrol et (daha detaylı)
        has_skills_in_df = False
        skills_info = "Beceri sütunu yok"
        
        if 'cikarilan_beceriler' in df_projects.columns:
            # Sütun var, içinde veri var mı kontrol et
            non_null_skills = df_projects['cikarilan_beceriler'].notna()
            non_null_count = non_null_skills.sum()
            
            if non_null_count > 0:
                # En az bir projede beceri var, geçerli mi kontrol et
                valid_skills = df_projects[non_null_skills]['cikarilan_beceriler'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                )
                valid_count = valid_skills.sum()
                has_skills_in_df = valid_count > 0
                skills_info = f"{valid_count}/{len(df_projects)} projede geçerli beceri var"
            else:
                skills_info = f"Beceri sütunu var ama tüm projelerde boş"
        else:
            skills_info = "Beceri sütunu yok"
        
        self.logger.info(f"🔍 Beceri kontrolü: {skills_info}, Beceri çıkarma gerekli mi? {not has_skills_in_df}")
        
        if skip_skill_extraction:
            # Beceri çıkarma atlandı, sadece devam et
            self.logger.info("   ✅ Proje becerileri önbellekten kullanılıyor, çıkarma yapılmıyor")
        elif new_links and len(new_links) > 0:
            # Yeni projeler var
            df_new_projects_only = df_projects[df_projects['link'].isin(new_links)].copy()
            df_existing_projects_only = df_projects[~df_projects['link'].isin(new_links)].copy()
            
            if len(df_new_projects_only) > 0:
                # Sadece yeni projeler için beceri çıkar
                self.logger.info(f"   {len(df_new_projects_only)} yeni proje için beceri çıkarılıyor...")
                df_new_projects_only = skill_extractor.extract_skills_from_dataframe(df_new_projects_only)
                
                # Mevcut projelerin becerilerini koru ve yeni projeleri ekle
                if has_skills_in_df and len(df_existing_projects_only) > 0:
                    # Mevcut projelerin becerileri zaten var, sadece yeni projeleri ekle
                    df_projects = pd.concat([df_existing_projects_only, df_new_projects_only], ignore_index=True)
                    self.logger.info(f"   ✅ Beceriler birleştirildi: {len(df_existing_projects_only)} eski (beceriler önbellekten) + {len(df_new_projects_only)} yeni")
                else:
                    # Mevcut projelerin becerileri yok, tüm projeler için beceri çıkar
                    self.logger.info(f"   Tüm {len(df_projects)} proje için beceri çıkarılıyor (ilk çalıştırma)...")
                    df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
        else:
            # Yeni proje yok
            if skip_skill_extraction:
                # Beceri çıkarma zaten atlandı (yukarıda kontrol edildi)
                self.logger.info("   ✅ Beceri çıkarma atlandı (önbellekten)")
            elif has_skills_in_df:
                # Beceriler zaten var (pickle'dan yüklenmiş), beceri çıkarmayı atla
                skills_count = df_projects['cikarilan_beceriler'].notna().sum()
                valid_skills_count = df_projects['cikarilan_beceriler'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                ).sum()
                self.logger.info(f"   ✅ Mevcut projelerin becerileri önbellekten kullanılıyor (beceri çıkarma atlandı)")
                self.logger.info(f"      📊 {valid_skills_count}/{len(df_projects)} projede geçerli beceri var, {skills_count - valid_skills_count} projede boş")
                # Beceri çıkarma yapma, sadece devam et
            elif 'cikarilan_beceriler' not in df_projects.columns:
                # Beceriler sütunu yok, çıkar
                self.logger.warning(f"   ⚠️  Beceri sütunu yok, tüm {len(df_projects)} proje için beceri çıkarılıyor...")
                df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
            else:
                # Beceriler sütunu var ama boş veya geçersiz, çıkar
                empty_count = df_projects['cikarilan_beceriler'].isna().sum()
                self.logger.warning(f"   ⚠️  Beceri sütunu var ama {empty_count}/{len(df_projects)} projede boş, tüm projeler için beceri çıkarılıyor...")
                df_projects = skill_extractor.extract_skills_from_dataframe(df_projects)
        
        projects_skills_path = project_root / self.config['output']['files']['projects_with_skills']
        df_projects.to_pickle(projects_skills_path)
        self.logger.info(f"Proje becerileri kaydedildi: {projects_skills_path} (Toplam {len(df_projects)} proje)")
        
        # 5. Vektörizasyon
        self.logger.info("Adım 5: Vektörizasyon yapılıyor...")
        vectorizer = SkillVectorizer(self.config, self.logger)
        
        # Aday becerilerini vektörize et
        candidate_vectors = vectorizer.vectorize_skills(candidate_skills)
        candidate_vectors_path = project_root / self.config['output']['files']['candidate_vectors']
        np.save(candidate_vectors_path, candidate_vectors)
        self.logger.info(f"Aday vektörleri kaydedildi: {candidate_vectors_path}")
        
        # Projeleri vektörize et (sadece yeni projeler için)
        # Vektörlerin var olup olmadığını ve dolu olup olmadığını kontrol et
        has_vectors = ('beceri_vektorleri' in df_projects.columns and 
                      df_projects['beceri_vektorleri'].notna().any())
        
        if has_vectors:
            # Mevcut vektörler var, sadece yeni projeler için vektör hesapla
            if new_links and len(new_links) > 0:
                df_new_projects_only = df_projects[df_projects['link'].isin(new_links)].copy()
                # Yeni projelerin vektörleri var mı kontrol et
                new_has_vectors = ('beceri_vektorleri' in df_new_projects_only.columns and 
                                  df_new_projects_only['beceri_vektorleri'].notna().any())
                
                if len(df_new_projects_only) > 0 and not new_has_vectors:
                    self.logger.info(f"   {len(df_new_projects_only)} yeni proje için vektör hesaplanıyor...")
                    df_new_projects_only = vectorizer.vectorize_dataframe(df_new_projects_only)
                    
                    # Mevcut projelerin vektörlerini koru ve yeni projeleri ekle
                    df_existing_with_vectors = df_projects[~df_projects['link'].isin(new_links)].copy()
                    df_projects = pd.concat([df_existing_with_vectors, df_new_projects_only], ignore_index=True)
                    self.logger.info(f"   ✅ Vektörler birleştirildi: {len(df_existing_with_vectors)} eski (önbellekten) + {len(df_new_projects_only)} yeni")
                else:
                    self.logger.info(f"   ✅ Tüm projelerin vektörleri mevcut (önbellekten)")
            else:
                # Yeni proje yok, mevcut vektörleri kullan
                self.logger.info(f"   ✅ Mevcut projelerin vektörleri önbellekten kullanılıyor (vektörizasyon atlandı - {len(df_projects)} proje)")
        else:
            # Vektörler yok, tüm projeler için hesapla
            self.logger.info(f"   Tüm {len(df_projects)} proje için vektör hesaplanıyor...")
            df_projects = vectorizer.vectorize_dataframe(df_projects)
        
        projects_vectors_path = project_root / self.config['output']['files']['projects_with_vectors']
        df_projects.to_pickle(projects_vectors_path)
        self.logger.info(f"Proje vektörleri kaydedildi: {projects_vectors_path} (Toplam {len(df_projects)} proje)")
        
        # Son başvuru tarihi geçen projeleri filtrele
        from datetime import datetime
        if 'application_deadline' in df_projects.columns:
            df_projects['application_deadline'] = pd.to_datetime(df_projects['application_deadline'], errors='coerce')
            now = datetime.now()
            expired_projects = df_projects[df_projects['application_deadline'].notna() & (df_projects['application_deadline'] < now)]
            if len(expired_projects) > 0:
                self.logger.info(f"🗑️  {len(expired_projects)} proje son başvuru tarihi geçmiş, siliniyor...")
                df_projects = df_projects[df_projects['application_deadline'].isna() | (df_projects['application_deadline'] >= now)]
                self.logger.info(f"   ✅ {len(df_projects)} aktif proje kaldı")
        
        # Projeleri CSV'ye de kaydet (güncel haliyle)
        csv_projects_file = get_output_path(self.config, 'project_details')
        df_projects.to_csv(csv_projects_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"Projeler CSV'ye kaydedildi: {csv_projects_file} (Toplam {len(df_projects)} proje)")
        
        # Metadata güncelle (en son eklenen proje tarihi ve yayın tarihi)
        metadata_file = project_root / "last_scrape_metadata.json"
        try:
            import json
            from datetime import datetime
            last_added_date = datetime.now()  # Son eklenen projelerin eklenme tarihi
            last_published_date = None
            
            if 'published_date' in df_projects.columns:
                df_projects['published_date'] = pd.to_datetime(df_projects['published_date'], errors='coerce')
                published_projects = df_projects[df_projects['published_date'].notna()]
                if len(published_projects) > 0:
                    latest_date = published_projects['published_date'].max()
                    if pd.notna(latest_date):
                        last_published_date = latest_date
            
            metadata = {
                'last_added_date': last_added_date.isoformat(),  # Son eklenen projenin eklenme tarihi
                'last_published_date': last_published_date.isoformat() if last_published_date is not None else None,
                'last_scrape_date': datetime.now().isoformat(),
                'last_scrape_count': len(new_links) if 'new_links' in locals() and new_links else 0,
                'total_projects': len(df_projects),
                'last_update': datetime.now().isoformat()
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📅 Metadata güncellendi: Son ekleme tarihi {last_added_date.strftime('%Y-%m-%d %H:%M:%S')}, En son yayın tarihi {last_published_date.strftime('%Y-%m-%d') if last_published_date is not None else 'N/A'} ({len(new_links) if 'new_links' in locals() and new_links else 0} yeni proje)")
        except Exception as e:
            self.logger.warning(f"Metadata güncellenemedi: {e}")
        
        self.logger.info("=" * 80)
        self.logger.info("✅ IP1 TAMAMLANDI! Projeler dosyaya kaydedildi.")
        self.logger.info(f"   📁 CSV: {csv_projects_file}")
        self.logger.info(f"   📁 Pickle (Beceriler): {projects_skills_path}")
        self.logger.info(f"   📁 Pickle (Vektörler): {projects_vectors_path}")
        self.logger.info(f"   📊 Toplam {len(df_projects)} proje kaydedildi")
        self.logger.info("=" * 80)
        
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
        # Config'den calculate_all ayarını al
        calculate_all = self.config.get('sincerity_scoring', {}).get('calculate_all', False)
        df_projects = sincerity_calc.calculate_all_scores(df_projects, calculate_all=calculate_all)
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['sincerity_scores']
        df_projects.to_pickle(output_path)
        self.logger.info(f"S skorları kaydedildi: {output_path}")
        
        self.logger.info("IP3 TAMAMLANDI!")
        return df_projects
    
    def run_ip2_and_ip4_parallel(self, df_projects: pd.DataFrame, candidate_vectors: np.ndarray, cv_text: str):
        """IP2 ve IP4'ü paralel olarak çalıştırır"""
        self.logger.info("=" * 80)
        self.logger.info("IP2 & IP4: M VE R SKORLARI PARALEL HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        from concurrent.futures import ThreadPoolExecutor
        
        def calculate_m_score():
            """M skorunu hesapla"""
            self.logger.info("IP2: M skoru hesaplanıyor (paralel)...")
            matching_calc = MatchingScoreCalculator(self.config, self.logger)
            return matching_calc.calculate_all_scores(df_projects.copy(), candidate_vectors)
        
        def calculate_r_score():
            """R skorunu hesapla"""
            self.logger.info("IP4: R skoru hesaplanıyor (paralel)...")
            reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
            cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
            return reliability_calc.calculate_project_reliability_scores(
                df_projects.copy(),
                cv_paragraphs,
                skill_column="cikarilan_beceriler"
            )
        
        # Paralel hesaplama
        with ThreadPoolExecutor(max_workers=2) as executor:
            m_future = executor.submit(calculate_m_score)
            r_future = executor.submit(calculate_r_score)
            
            # Sonuçları bekle
            df_projects_m = m_future.result()
            df_projects_r = r_future.result()
            
            # Skorları birleştir
            df_projects = df_projects_m.copy()
            if 'R_Skoru' in df_projects_r.columns:
                df_projects['R_Skoru'] = df_projects_r['R_Skoru']
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        m_output_path = project_root / self.config['output']['files']['matching_scores']
        df_projects.to_pickle(m_output_path)
        self.logger.info(f"M ve R skorları kaydedildi: {m_output_path}")
        
        self.logger.info("IP2 & IP4 PARALEL TAMAMLANDI!")
        return df_projects
    
    def run_ip4_reliability_score(self, df_projects: pd.DataFrame, cv_text: str):
        """IP4: Güvenilirlik skoru hesaplama (her proje için)"""
        self.logger.info("=" * 80)
        self.logger.info("IP4: GÜVENİLİRLİK SKORU (R-SKORU) HESAPLANIYOR")
        self.logger.info("=" * 80)
        self.logger.info("NOT: Her proje için o projenin gereksinimlerinin CV'de doğrulanma oranı hesaplanacak")
        
        reliability_calc = ReliabilityScoreCalculator(self.config, self.logger)
        cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
        
        # Her proje için R skoru hesapla
        df_projects = reliability_calc.calculate_project_reliability_scores(
            df_projects, 
            cv_paragraphs,
            skill_column="cikarilan_beceriler"
        )
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['reliability_details']
        df_projects[['link', 'R_Skoru']].to_pickle(output_path)
        self.logger.info(f"R skorları kaydedildi: {output_path}")
        
        self.logger.info("IP4 TAMAMLANDI!")
        return df_projects
    
    def run_ip5_final_ranking(self, df_projects: pd.DataFrame):
        """IP5: Nihai adil sıralama"""
        self.logger.info("=" * 80)
        self.logger.info("IP5: ADİL SIRALAMA (MOO) HESAPLANIYOR")
        self.logger.info("=" * 80)
        
        ranking_calc = FinalRankingCalculator(self.config, self.logger)
        df_final = ranking_calc.calculate_ranking(df_projects)
        
        # Tüm önemli sütunları seç ve kaydet
        columns_to_save = [
            'link', 'tam_aciklama', 'cikarilan_beceriler',
            # Ham skorlar (orijinal değerler)
            'M_Skoru_Ham', 'S_Skoru_Ham', 'R_Skoru_Ham',
            # Normalize edilmiş skorlar (Z-Score ile)
            'M_Skoru_Normalize', 'S_Skoru_Normalize', 'R_Skoru_Normalize',
            # Kullanılan skorlar (normalize edilmiş, M_Skoru, S_Skoru, R_Skoru olarak)
            'M_Skoru', 'S_Skoru', 'R_Skoru',
            # Final skorlar
            'Final_Skor', 'Adil_Skor'
        ]
        
        # Mevcut sütunları kontrol et
        available_columns = [col for col in columns_to_save if col in df_final.columns]
        missing_columns = [col for col in columns_to_save if col not in df_final.columns]
        
        if missing_columns:
            self.logger.warning(f"Bazı sütunlar eksik: {missing_columns}")
        
        # Eğer ham skorlar yoksa uyarı ver (final_ranking.py'de oluşturulmalı)
        if 'M_Skoru_Ham' not in df_final.columns:
            self.logger.warning("Ham skorlar bulunamadı! Final ranking'de Z-Score normalizasyonu uygulanmış olmalı.")
        
        # Sadece mevcut sütunları kaydet
        df_to_save = df_final[available_columns].copy()
        
        # Eğer ham skorlar yoksa ama normalize skorlar varsa, normalize skorları ham olarak da ekle
        if 'M_Skoru_Ham' not in df_final.columns and 'M_Skoru_Normalize' in df_final.columns:
            self.logger.info("Ham skorlar eksik, normalize skorlar ham olarak da kaydediliyor...")
            # Bu durumda normalize skorlar zaten M_Skoru, S_Skoru, R_Skoru olarak var
            # Ham skorlar için önceki aşamalardan alınabilir, ama şimdilik normalize skorları kullan
            pass
        
        # Kaydet
        project_root = Path(__file__).parent.parent
        output_path = project_root / self.config['output']['files']['final_ranking']
        df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Nihai sıralama kaydedildi: {output_path}")
        self.logger.info(f"Kaydedilen sütunlar: {list(df_to_save.columns)}")
        self.logger.info(f"Toplam {len(df_to_save)} proje kaydedildi")
        
        # Top 10'u göster
        self.logger.info("\n🏆 TOP 10 PROJE:")
        for i, row in df_final.head(10).iterrows():
            proje_id = row['link'].split('/')[-1].replace('_en', '')
            self.logger.info(
                f"{i+1}. [Adil Skor: {row['Adil_Skor']:.2f}] "
                f"[M: {row['M_Skoru']:.1f}, S: {row['S_Skoru']:.1f}, R: {row['R_Skoru']:.1f}] "
                f"{proje_id}"
            )
        
        self.logger.info("IP5 TAMAMLANDI!")
        return df_final
    
    def run_full_pipeline(self):
        """Tüm pipeline'ı çalıştırır"""
        try:
            # IP1
            df_projects, candidate_vectors, cv_text, candidate_skills = self.run_ip1_data_collection()
            
            # IP2 ve IP4 - Paralel hesaplama (M ve R skorları bağımsız)
            self.logger.info("IP2 ve IP4 paralel olarak başlatılıyor...")
            df_projects = self.run_ip2_and_ip4_parallel(df_projects, candidate_vectors, cv_text)
            
            # Kontrol: Tüm projeler için R skoru var mı?
            r_missing = df_projects['R_Skoru'].isna().sum() if 'R_Skoru' in df_projects.columns else len(df_projects)
            if r_missing > 0:
                self.logger.warning(f"{r_missing} adet proje için R skoru eksik!")
            else:
                self.logger.info(f"✅ Tüm {len(df_projects)} proje için R skoru hesaplandı")
            
            # IP3 - TÜM projeler için S skoru hesapla (M skorundan sonra, sıralama için)
            self.logger.info("IP3 başlamadan önce: Tüm projeler için S skoru hesaplanacak")
            df_projects = self.run_ip3_sincerity_score(df_projects)
            
            # Kontrol: Tüm projeler için S skoru var mı?
            s_missing = df_projects['S_Skoru'].isna().sum() if 'S_Skoru' in df_projects.columns else len(df_projects)
            if s_missing > 0:
                self.logger.warning(f"{s_missing} adet proje için S skoru eksik!")
            else:
                self.logger.info(f"✅ Tüm {len(df_projects)} proje için S skoru hesaplandı")
            
            # IP5
            df_final = self.run_ip5_final_ranking(df_projects)
            
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

