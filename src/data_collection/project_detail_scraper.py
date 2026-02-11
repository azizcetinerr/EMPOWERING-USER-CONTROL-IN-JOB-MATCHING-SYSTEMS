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
    
    def extract_project_info(self, detail_url: str, retries: int = None) -> Dict:
        """
        URL'den proje bilgilerini çıkarır (açıklama + kurum bilgileri).
        
        Args:
            detail_url: Proje detay URL'i
            retries: Yeniden deneme sayısı (None ise config'den alır)
        
        Returns:
            Proje bilgileri sözlüğü
        """
        if retries is None:
            retries = self.retry_count
        
        try:
            response = requests.get(detail_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            content_cards = soup.find_all("div", class_="card-content")
            
            # Proje başlığını çıkar (h1 veya h2 içinde)
            project_title = None
            h1_title = soup.find("h1")
            if h1_title:
                project_title = h1_title.get_text(strip=True)
            else:
                h2_title = soup.find("h2")
                if h2_title:
                    project_title = h2_title.get_text(strip=True)
            
            # Eğer başlık bulunamazsa, dil kodlarını temizleyerek URL'den çıkar
            if not project_title or len(project_title) < 5:
                # URL'den proje ID'sini al ve title olarak kullan (son çare)
                import re
                match = re.search(r'/(\d+)_', detail_url)
                if match:
                    project_title = f"Project {match.group(1)}"
                else:
                    project_title = "Proje Başlığı Bulunamadı"
            
            # Açıklama
            full_description_text = ""
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header and "Description" in h6_header.text:
                    all_paragraphs = card.find_all("p")
                    for p in all_paragraphs:
                        text = p.text.strip()
                        # Dil kodlarını ve gereksiz metinleri temizle
                        # BGбългарскиCSčeština gibi dil kodlarını atla
                        if len(text) < 200 and any(chr.isalpha() and ord(chr) > 127 for chr in text[:10]):
                            # Muhtemelen dil kodu veya çok kısa metin, atla
                            continue
                        # Çok kısa veya anlamsız metinleri atla
                        if text and len(text) > 20:
                            full_description_text += text + "\n\n"
                    break
            
            # Kurum bilgilerini çek
            organization_name = None
            verified = False
            address = None
            website = None
            
            # Organization/Hosting organization bilgisini ara
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header:
                    header_text = h6_header.text.lower()
                    if "organization" in header_text or "hosting" in header_text:
                        # Kurum adını bul
                        org_elements = card.find_all(["p", "span", "div"])
                        for elem in org_elements:
                            text = elem.text.strip()
                            if text and len(text) > 3:
                                organization_name = text
                                break
                        break
            
            # Verified badge/icon ara
            verified_indicators = soup.find_all(string=lambda t: t and ("verified" in t.lower() or "certified" in t.lower()))
            if verified_indicators:
                verified = True
            
            # Address bilgisi ara
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header and ("address" in h6_header.text.lower() or "location" in h6_header.text.lower()):
                    address_elem = card.find(["p", "span"])
                    if address_elem:
                        address = address_elem.text.strip()
                    break
            
            # Website bilgisi ara
            website_links = soup.find_all("a", href=True)
            for link in website_links:
                href = link.get("href", "")
                if href and ("http" in href or "www" in href):
                    # Kurum web sitesi olabilir
                    if "youth.europa.eu" not in href and "europa.eu" not in href:
                        website = href
                        break
            
            # Proje yayın tarihini çek (created div'inden)
            published_date = None
            created_divs = soup.find_all("div", class_=lambda x: x and "created" in str(x).lower())
            for div in created_divs:
                span = div.find("span")
                if span:
                    date_text = span.get_text(strip=True)
                    # "Today", "1 day ago", "2 days ago" gibi ifadeleri parse et
                    published_date = self._parse_relative_date(date_text)
                    if published_date:
                        break
            
            # Son başvuru tarihini çek (deadline, application deadline, expires vb.)
            application_deadline = None
            # Önce "deadline", "application deadline", "expires", "closing date" gibi başlıkları ara
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header:
                    header_text = h6_header.text.lower()
                    if any(keyword in header_text for keyword in ["deadline", "expires", "closing", "application"]):
                        # Tarih bilgisini bul
                        date_elements = card.find_all(["p", "span", "div"])
                        for elem in date_elements:
                            text = elem.get_text(strip=True)
                            # Tarih formatlarını kontrol et (YYYY-MM-DD, DD/MM/YYYY, vb.)
                            parsed_date = self._parse_date_string(text)
                            if parsed_date:
                                application_deadline = parsed_date
                                break
                        if application_deadline:
                            break
            
            # Eğer card'larda bulunamadıysa, genel arama yap
            if not application_deadline:
                # "Deadline:", "Expires:", "Closes:" gibi etiketleri ara
                deadline_keywords = ["deadline", "expires", "closes", "application deadline", "closing date"]
                all_text = soup.get_text()
                for keyword in deadline_keywords:
                    if keyword.lower() in all_text.lower():
                        # Keyword'den sonraki tarih bilgisini bul
                        keyword_index = all_text.lower().find(keyword.lower())
                        if keyword_index >= 0:
                            # Keyword'den sonraki 50 karakter içinde tarih ara
                            text_after_keyword = all_text[keyword_index:keyword_index+100]
                            parsed_date = self._parse_date_string(text_after_keyword)
                            if parsed_date:
                                application_deadline = parsed_date
                        break
            
            return {
                "proje_adi": project_title,
                "tam_aciklama": full_description_text.strip() if full_description_text else "AÇIKLAMA BULUNAMADI",
                "organization_name": organization_name,
                "verified": verified,
                "address": address,
                "website": website,
                "published_date": published_date.isoformat() if published_date else None,
                "application_deadline": application_deadline.isoformat() if application_deadline else None,
                "scraped_at": None  # Bu normal scraper'da scraping zamanı eklenmeyecek, sadece published_date kullanılacak
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retries > 0:
                self.logger.warning(f"429 Hatası! {self.retry_delay} saniye bekleniyor... (URL: {detail_url})")
                time.sleep(self.retry_delay)
                return self.extract_project_info(detail_url, retries - 1)
            else:
                return {
                    "proje_adi": f"Proje {detail_url.split('/')[-1].split('_')[0]}",
                    "tam_aciklama": f"HATA: {e.response.status_code} Hatası",
                    "organization_name": None,
                    "verified": False,
                    "address": None,
                    "website": None,
                    "published_date": None,
                    "application_deadline": None,
                    "scraped_at": None
                }
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Bağlantı hatası ({detail_url}): {e}")
            return {
                "proje_adi": f"Proje {detail_url.split('/')[-1].split('_')[0]}",
                "tam_aciklama": "HATA: Bağlantı Hatası",
                "organization_name": None,
                "verified": False,
                "address": None,
                "website": None,
                "published_date": None,
                "application_deadline": None,
                "scraped_at": None
            }
    
    def _parse_relative_date(self, date_text: str):
        """
        "Today", "1 day ago", "2 days ago" gibi ifadeleri gerçek tarihe çevirir.
        
        Args:
            date_text: Tarih metni (örn: "Today", "1 day ago", "2 days ago")
        
        Returns:
            datetime objesi veya None
        """
        from datetime import datetime, timedelta
        import re
        
        date_text = date_text.strip().lower()
        
        if "today" in date_text:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # "X day(s) ago" formatını parse et
        match = re.search(r'(\d+)\s*(day|days)\s*ago', date_text)
        if match:
            days_ago = int(match.group(1))
            return (datetime.now() - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # "X week(s) ago" formatını parse et
        match = re.search(r'(\d+)\s*(week|weeks)\s*ago', date_text)
        if match:
            weeks_ago = int(match.group(1))
            return (datetime.now() - timedelta(weeks=weeks_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # "X month(s) ago" formatını parse et
        match = re.search(r'(\d+)\s*(month|months)\s*ago', date_text)
        if match:
            months_ago = int(match.group(1))
            # Yaklaşık olarak 30 gün = 1 ay
            return (datetime.now() - timedelta(days=months_ago * 30)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        return None
    
    def _parse_date_string(self, date_text: str):
        """
        Farklı formatlardaki tarih metinlerini parse eder.
        
        Args:
            date_text: Tarih metni (örn: "2024-12-31", "31/12/2024", "December 31, 2024")
        
        Returns:
            datetime objesi veya None
        """
        from datetime import datetime
        import re
        
        date_text = date_text.strip()
        
        # YYYY-MM-DD formatı
        try:
            return datetime.strptime(date_text[:10], "%Y-%m-%d")
        except:
            pass
        
        # DD/MM/YYYY veya MM/DD/YYYY formatı
        date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
        match = re.search(date_pattern, date_text)
        if match:
            try:
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))
            except:
                pass
        
        # "December 31, 2024" formatı
        try:
            return datetime.strptime(date_text[:20], "%B %d, %Y")
        except:
            pass
        
        try:
            return datetime.strptime(date_text[:20], "%d %B %Y")
        except:
            pass
        
        return None
    
    def extract_project_description(self, detail_url: str, retries: int = None) -> str:
        """
        URL'den proje açıklamasını çıkarır (geriye uyumluluk için).
        
        Args:
            detail_url: Proje detay URL'i
            retries: Yeniden deneme sayısı (None ise config'den alır)
        
        Returns:
            Proje açıklama metni
        """
        info = self.extract_project_info(detail_url, retries)
        return info.get("tam_aciklama", "AÇIKLAMA BULUNAMADI")
    
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
            
            project_info = self.extract_project_info(link)
            
            if "HATA" in project_info.get("tam_aciklama", "") or "BULUNAMADI" in project_info.get("tam_aciklama", ""):
                self.logger.warning(f"Bu ilan atlanıyor: {link} (Sebep: {project_info.get('tam_aciklama', '')[:50]}...)")
                basarisiz_linkler += 1
            
            from datetime import datetime
            proje_verileri.append({
                "link": link,
                "proje_adi": project_info.get("proje_adi", f"Proje {link.split('/')[-1].split('_')[0]}"),
                "tam_aciklama": project_info.get("tam_aciklama", ""),
                "organization_name": project_info.get("organization_name"),
                "verified": project_info.get("verified", False),
                "address": project_info.get("address"),
                "website": project_info.get("website"),
                "published_date": project_info.get("published_date"),  # Gerçek yayın tarihi
                "application_deadline": project_info.get("application_deadline"),  # Son başvuru tarihi
                "scraped_at": datetime.now().isoformat()  # Scraping zamanı
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

