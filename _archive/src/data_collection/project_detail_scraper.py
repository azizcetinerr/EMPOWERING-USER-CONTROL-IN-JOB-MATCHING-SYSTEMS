"""
Proje detay sayfalarından açıklamaları çeken modül
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging

from ..utils.data_cleaner import clean_project_descriptions


class ProjectDetailScraper:
    """Proje detay sayfalarını çeken sınıf"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        scraping_config = config.get('scraping', {}).get('requests', {})
        self.retry_count = scraping_config.get('retry_count', 3)
        self.retry_delay = scraping_config.get('retry_delay', 10)
        self.request_delay = scraping_config.get('request_delay', 1)
        self.timeout = scraping_config.get('timeout', 30)
    
    def extract_project_description(self, detail_url: str, retries: int = None) -> str:
        """
        URL'den proje açıklamasını çıkarır.
        
        Args:
            detail_url: Proje detay URL'i
            retries: Yeniden deneme sayısı (None ise config'den alır)
        
        Returns:
            Proje açıklama metni
        """
        if retries is None:
            retries = self.retry_count
        
        try:
            response = requests.get(detail_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            content_cards = soup.find_all("div", class_="card-content")
            full_description_text = ""
            
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header and "Description" in h6_header.text:
                    all_paragraphs = card.find_all("p")
                    for p in all_paragraphs:
                        full_description_text += p.text.strip() + "\n\n"
                    break
            
            return full_description_text.strip() if full_description_text else "AÇIKLAMA BULUNAMADI"
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retries > 0:
                self.logger.warning(f"429 Hatası! {self.retry_delay} saniye bekleniyor... (URL: {detail_url})")
                time.sleep(self.retry_delay)
                return self.extract_project_description(detail_url, retries - 1)
            else:
                return f"HATA: {e.response.status_code} Hatası"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Bağlantı hatası ({detail_url}): {e}")
            return "HATA: Bağlantı Hatası"
    
    def scrape_all_projects(self, links: List[str]) -> pd.DataFrame:
        """
        Tüm proje linklerinden detayları çeker.
        
        Args:
            links: Proje link listesi
        
        Returns:
            Proje detayları DataFrame
        """
        self.logger.info(f"{len(links)} adet proje detayı çekiliyor...")
        
        proje_verileri = []
        basarisiz_linkler = 0
        
        for i, link in enumerate(links):
            self.logger.info(f"İşleniyor... ({i+1}/{len(links)}) - {link}")
            
            full_description = self.extract_project_description(link)
            
            if "HATA" in full_description or "BULUNAMADI" in full_description:
                self.logger.warning(f"Bu ilan atlanıyor: {link} (Sebep: {full_description[:50]}...)")
                basarisiz_linkler += 1
            
            proje_verileri.append({
                "link": link,
                "tam_aciklama": full_description
            })
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # İlerleme raporu
            if (i + 1) % 50 == 0:
                self.logger.info(f"İlerleme: {i+1}/{len(links)} proje işlendi")
        
        df_projeler = pd.DataFrame(proje_verileri)
        
        # Veri temizleme
        df_projeler = clean_project_descriptions(df_projeler)
        
        self.logger.info(f"Toplam {len(df_projeler)} proje ilanı işlendi")
        if basarisiz_linkler > 0:
            self.logger.warning(f"{basarisiz_linkler} adet ilanın detay sayfası çekilemedi")
        
        return df_projeler

