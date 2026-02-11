"""
Europass CV'den bilgileri çeken modül
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Tuple
import logging


class EuropassCVParser:
    """Europass CV parser sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.cv_url = config['data_sources']['europass_cv']['url']
        if not self.cv_url:
            raise ValueError("CV URL'i konfigürasyon dosyasında belirtilmemiş!")
        
        scraping_config = config.get('scraping', {}).get('selenium', {})
        self.headless = scraping_config.get('headless', True)
        self.wait_timeout = scraping_config.get('wait_timeout', 10)  # 15'ten 10'a düşürüldü
        self.page_load_timeout = scraping_config.get('page_load_timeout', 30)  # Sayfa yükleme timeout'u
        self.implicit_wait = scraping_config.get('implicit_wait', 5)  # Implicit wait
        
        self.driver = None
    
    def _setup_driver(self):
        """Selenium driver'ı kurar"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")  # Güvenlik için
        chrome_options.add_argument("--disable-dev-shm-usage")  # Bellek sorunları için
        chrome_options.add_argument("window-size=1920x1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Timeout'ları ayarla
        self.driver.set_page_load_timeout(self.page_load_timeout)  # Sayfa yükleme timeout'u
        self.driver.implicitly_wait(self.implicit_wait)  # Element bulma timeout'u
        
        self.logger.info(f"Selenium driver başlatıldı (page_load_timeout: {self.page_load_timeout}s, wait_timeout: {self.wait_timeout}s)")
    
    def parse_cv(self) -> Tuple[str, List[str]]:
        """
        CV'den tam metni ve beceri listesini çıkarır.
        
        Returns:
            (tam_metin, beceri_listesi) tuple'ı
        """
        if self.driver is None:
            self._setup_driver()
        
        try:
            self.logger.info(f"Europass CV linki çekiliyor: {self.cv_url}")
            
            # Sayfa yükleme işlemi (timeout ile korumalı)
            try:
                self.driver.get(self.cv_url)
                self.logger.info("Sayfa yüklendi, Skills bölümü bekleniyor...")
            except Exception as page_load_error:
                error_msg = str(page_load_error)
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    self.logger.error(f"❌ Sayfa yükleme timeout'u ({self.page_load_timeout}s) - Europass sunucusu yanıt vermiyor olabilir")
                    raise TimeoutError(f"Europass CV sayfası {self.page_load_timeout} saniye içinde yüklenemedi. Lütfen linki kontrol edin: {self.cv_url}")
                else:
                    self.logger.error(f"❌ Sayfa yükleme hatası: {error_msg}")
                    raise
            
            # Sayfanın yüklenmesini bekle (Skills başlığı var olana kadar)
            skills_header_xpath = "//h2[contains(text(), 'Skills')]"
            try:
                WebDriverWait(self.driver, self.wait_timeout).until(
                    EC.presence_of_element_located((By.XPATH, skills_header_xpath))
                )
                self.logger.info("✅ Europass profili yüklendi. Veri ayrıştırılıyor...")
            except Exception as wait_error:
                self.logger.error(f"❌ Skills bölümü {self.wait_timeout} saniye içinde bulunamadı. Sayfa yapısı değişmiş olabilir.")
                # Sayfa yüklendi ama Skills bölümü yok - yine de devam et, belki farklı bir yapı var
                self.logger.warning("⚠️  Skills bölümü bulunamadı, tüm sayfa içeriği işlenecek...")
            
            html_source = self.driver.page_source
            soup = BeautifulSoup(html_source, "html.parser")
            
            cv_text_list = []
            cv_beceri_listesi = []
            
            # Eğitim bölümü
            self._extract_section(soup, "Education and training", cv_text_list)
            
            # İş deneyimi
            self._extract_section(soup, "Work experience", cv_text_list)
            
            # Beceriler
            skills = self._extract_skills(soup)
            cv_text_list.extend(skills)
            cv_beceri_listesi.extend(skills)
            
            # Sertifikalar
            self._extract_section(soup, "Certificates", cv_text_list)
            
            # Tam metni birleştir
            cv_full_text = "\n".join(list(filter(None, cv_text_list)))
            
            self.logger.info(f"CV'den {len(cv_beceri_listesi)} adet ham beceri metni bulundu")
            
            return cv_full_text, cv_beceri_listesi
            
        except Exception as e:
            self.logger.error(f"CV parsing hatası: {e}", exc_info=True)
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Tarayıcı kapatıldı")
    
    def _extract_section(self, soup: BeautifulSoup, section_name: str, text_list: List[str]):
        """Belirli bir bölümü çıkarır"""
        header = soup.find("h2", string=lambda t: t and section_name in t)
        if header:
            current_tag = header.find_next_sibling()
            while current_tag and current_tag.name != 'h2':
                if current_tag.name in ['p', 'h4']:
                    text_list.append(current_tag.text.strip())
                current_tag = current_tag.find_next_sibling()
    
    def _extract_skills(self, soup: BeautifulSoup) -> List[str]:
        """Beceri bölümünü çıkarır"""
        skills = []
        skills_header = soup.find("h2", string=lambda t: t and "Skills" in t)
        if skills_header:
            skills_ul = skills_header.find_next("ul")
            if skills_ul:
                skill_items = skills_ul.find_all("li")
                for item in skill_items:
                    beceri_metni = item.text.strip()
                    if beceri_metni:
                        skills.append(beceri_metni)
        return skills

