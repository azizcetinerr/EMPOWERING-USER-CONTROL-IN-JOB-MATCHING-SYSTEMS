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
        
        self.cv_url = config.get('data_sources', {}).get('europass_cv', {}).get('url', '')
        # URL burada boş olabilir, parse_cv çağrıldığında kontrol edilecek
        if not self.cv_url:
            self.logger.info("CV Parser başlatıldı (URL henüz belirtilmedi, parse işleminden önce set edilmeli)")
        
        scraping_config = config.get('scraping', {}).get('selenium', {})
        self.headless = scraping_config.get('headless', True)
        self.wait_timeout = scraping_config.get('wait_timeout', 15)
        self.page_load_timeout = scraping_config.get('page_load_timeout', 300)
        self.implicit_wait = scraping_config.get('implicit_wait', 10)
        
        self.driver = None
    
    def _setup_driver(self):
        """Selenium driver'ı kurar"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("window-size=1920x1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Timeout ayarları (CV parsing için uzun metinler yüklenebilir)
        self.driver.set_page_load_timeout(self.page_load_timeout)
        self.driver.implicitly_wait(self.implicit_wait)
        
        self.logger.info(f"Selenium driver başlatıldı (page_load_timeout: {self.page_load_timeout}s, implicit_wait: {self.implicit_wait}s)")
    
    def parse_cv(self) -> Tuple[str, List[str]]:
        """
        CV'den tam metni ve beceri listesini çıkarır.
        
        Returns:
            (tam_metin, beceri_listesi) tuple'ı
        """

        if not self.cv_url:
            # Config'den tekrar kontrol et (dinamik update için)
            self.cv_url = self.config.get('data_sources', {}).get('europass_cv', {}).get('url', '')
            
        if not self.cv_url:
            raise ValueError("CV URL'i belirtilmemiş! Lütfen config dosyasında veya arayüzde CV linkini girin.")

        if self.driver is None:
            self._setup_driver()
        
        try:
            self.logger.info(f"Europass CV linki çekiliyor: {self.cv_url}")
            self.driver.get(self.cv_url)
            
            # Sayfanın yüklenmesini bekle
            skills_header_xpath = "//h2[contains(text(), 'Skills')]"
            try:
                WebDriverWait(self.driver, self.wait_timeout).until(
                    EC.presence_of_element_located((By.XPATH, skills_header_xpath))
                )
                self.logger.info("Europass profili yüklendi (Skills başlığı bulundu).")
            except Exception as e:
                self.logger.warning(f"Skills başlığı için zaman aşımı ({self.wait_timeout}s). Sayfa kaynağı analiz ediliyor... Hata: {e}")
                
                # Debugging: Ne yüklendi?
                try:
                    h2_list = self.driver.find_elements(By.TAG_NAME, "h2")
                    h2_texts = [h.text for h in h2_list]
                    self.logger.info(f"Mevcut H2 başlıkları: {h2_texts}")
                    
                    if not h2_texts:
                         # Belki H2 değildir, sayfa başlığına bakalım
                         self.logger.info(f"Sayfa Başlığı: {self.driver.title}")
                except Exception as debug_err:
                    self.logger.warning(f"Debug bilgisi çekilemedi: {debug_err}")

            self.logger.info("Veri ayrıştırılıyor...")
            
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

