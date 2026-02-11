"""
Paralel proje detay sayfalarından açıklamaları çeken modül (Hızlı versiyon)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from ..utils.data_cleaner import clean_project_descriptions


class ProjectDetailScraperParallel:
    """Paralel proje detay sayfalarını çeken sınıf (Hızlı versiyon)"""
    
    def __init__(self, config: dict, logger: logging.Logger = None, max_workers: int = 4):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
            max_workers: Paralel çalışacak thread sayısı (varsayılan: 4, optimize edildi)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.max_workers = max_workers
        
        scraping_config = config.get('scraping', {}).get('requests', {})
        self.retry_count = scraping_config.get('retry_count', 5)
        self.retry_delay = scraping_config.get('retry_delay', 10)
        self.request_delay = scraping_config.get('request_delay', 1.0)  # Adaptive olacak
        self.base_delay = 1.0  # Başlangıç delay
        self.timeout = scraping_config.get('timeout', 30)
        self.max_retry_delay = 30  # Maksimum bekleme süresi (429 hatası için)
        
        # Adaptive rate limiting
        self.consecutive_successes = 0
        self.consecutive_429_errors = 0
        self.current_delay = self.base_delay
    
    def extract_project_info(self, detail_url: str, retries: int = None) -> Dict:
        """
        Proje detay sayfasından bilgileri çıkarır (rate limiting ile).
        
        Args:
            detail_url: Proje detay sayfası URL'i
            retries: Tekrar deneme sayısı (None ise config'den alınır)
        
        Returns:
            Proje bilgileri sözlüğü
        """
        if retries is None:
            retries = self.retry_count
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(
                    detail_url,
                    timeout=self.timeout,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive'
                    }
                )
                
                # 429 hatası kontrolü
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 10))
                    wait_time = min(retry_after, self.max_retry_delay)
                    self.logger.warning(f"Rate limit (429)! {wait_time} saniye bekleniyor...")
                    self.consecutive_429_errors += 1
                    self.consecutive_successes = 0
                    # Delay'i artır
                    self.current_delay = min(self.current_delay * 1.5, 5.0)
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                # Başarılı istek - delay'i azalt (adaptive)
                self.consecutive_successes += 1
                self.consecutive_429_errors = 0
                if self.consecutive_successes > 10:
                    # 10 başarılı istekten sonra delay'i azalt
                    self.current_delay = max(self.current_delay * 0.9, self.base_delay)
                    self.consecutive_successes = 0
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Tam açıklama
                description_elements = soup.find_all(['p', 'div'], class_=lambda x: x and ('description' in str(x).lower() or 'content' in str(x).lower()))
                full_description = ' '.join([elem.get_text(strip=True) for elem in description_elements if elem.get_text(strip=True)])
                
                if not full_description:
                    # Alternatif: tüm paragraf etiketlerini al
                    paragraphs = soup.find_all('p')
                    full_description = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                
                # Organizasyon bilgileri
                org_name = None
                verified = False
                address = None
                website = None
                
                # Organizasyon adı
                org_elements = soup.find_all(['span', 'div', 'h3'], class_=lambda x: x and 'organization' in str(x).lower())
                if org_elements:
                    org_name = org_elements[0].get_text(strip=True)
                
                # Verified badge
                verified_elements = soup.find_all(['span', 'div'], class_=lambda x: x and 'verified' in str(x).lower())
                verified = len(verified_elements) > 0
                
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
                content_cards = soup.find_all("div", class_="card-content")
                for card in content_cards:
                    h6_header = card.find("h6")
                    if h6_header:
                        header_text = h6_header.text.lower()
                        if any(keyword in header_text for keyword in ["deadline", "expires", "closing", "application"]):
                            date_elements = card.find_all(["p", "span", "div"])
                            for elem in date_elements:
                                text = elem.get_text(strip=True)
                                parsed_date = self._parse_date_string(text)
                                if parsed_date:
                                    application_deadline = parsed_date
                                    break
                            if application_deadline:
                                break
                
                if not application_deadline:
                    # Genel arama
                    all_text = soup.get_text()
                    deadline_keywords = ["deadline", "expires", "closes", "application deadline", "closing date"]
                    for keyword in deadline_keywords:
                        if keyword.lower() in all_text.lower():
                            keyword_index = all_text.lower().find(keyword.lower())
                            if keyword_index >= 0:
                                text_after_keyword = all_text[keyword_index:keyword_index+100]
                                parsed_date = self._parse_date_string(text_after_keyword)
                                if parsed_date:
                                    application_deadline = parsed_date
                                    break
                
                return {
                    "tam_aciklama": full_description if full_description else "AÇIKLAMA BULUNAMADI",
                    "organization_name": org_name,
                    "verified": verified,
                    "address": address,
                    "website": website,
                    "published_date": published_date.isoformat() if published_date else None,
                    "application_deadline": application_deadline.isoformat() if application_deadline else None
                }
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limiting - exponential backoff
                    wait_time = min(self.request_delay * (2 ** attempt), self.max_retry_delay)
                    self.logger.warning(f"Rate limit (429) - {wait_time} saniye bekleniyor...")
                    time.sleep(wait_time)
                    if attempt < retries:
                        continue
                elif attempt < retries:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.warning(f"Proje detayı çekilemedi: {detail_url} - {str(e)}")
                    return {
                        "tam_aciklama": f"HATA: {str(e)}",
                        "organization_name": None,
                        "verified": False,
                        "address": None,
                        "website": None,
                        "published_date": None,
                        "application_deadline": None
                    }
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    wait_time = self.request_delay * (attempt + 1)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.warning(f"Proje detayı çekilemedi: {detail_url} - {str(e)}")
                    return {
                        "tam_aciklama": f"HATA: {str(e)}",
                        "organization_name": None,
                        "verified": False,
                        "address": None,
                        "website": None,
                        "published_date": None,
                        "application_deadline": None
                    }
            except Exception as e:
                self.logger.error(f"Beklenmeyen hata: {detail_url} - {str(e)}")
                return {
                    "tam_aciklama": f"HATA: {str(e)}",
                    "organization_name": None,
                    "verified": False,
                    "address": None,
                    "website": None,
                    "published_date": None,
                    "application_deadline": None
                }
        
        return {
            "tam_aciklama": "BULUNAMADI",
            "organization_name": None,
            "verified": False,
            "address": None,
            "website": None,
            "published_date": None,
            "application_deadline": None
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
    
    def _scrape_single_project(self, link: str) -> Dict:
        """Tek bir projeyi scrape eder (paralel işleme için)"""
        project_info = self.extract_project_info(link)
        
        # Adaptive delay - başarılı isteklerden sonra delay'i kullan
        time.sleep(self.current_delay)
        
        from datetime import datetime
        return {
            "link": link,
            "tam_aciklama": project_info.get("tam_aciklama", ""),
            "organization_name": project_info.get("organization_name"),
            "verified": project_info.get("verified", False),
            "address": project_info.get("address"),
            "website": project_info.get("website"),
            "published_date": project_info.get("published_date"),  # Gerçek yayın tarihi
            "application_deadline": project_info.get("application_deadline"),  # Son başvuru tarihi
            "scraped_at": datetime.now().isoformat()  # Scraping zamanı
        }
    
    def scrape_all_projects(self, links: List[str]) -> pd.DataFrame:
        """
        Tüm proje linklerinden detayları paralel olarak çeker.
        
        Args:
            links: Proje link listesi
        
        Returns:
            Proje detayları DataFrame
        """
        self.logger.info(f"{len(links)} adet proje detayı paralel olarak çekiliyor... (Max {self.max_workers} thread)")
        
        proje_verileri = []
        basarisiz_linkler = 0
        
        # Paralel scraping (adaptive rate limiting ile)
        self.logger.info(f"Paralel scraping başlatılıyor: {len(links)} proje, {self.max_workers} thread")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tüm görevleri başlat
            future_to_link = {executor.submit(self._scrape_single_project, link): link for link in links}
            
            # Sonuçları topla
            completed = 0
            total_429_errors = 0
            
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                completed += 1
                
                try:
                    project_data = future.result()
                    
                    if "HATA" in project_data.get("tam_aciklama", "") or "BULUNAMADI" in project_data.get("tam_aciklama", ""):
                        if "429" in project_data.get("tam_aciklama", ""):
                            total_429_errors += 1
                            # Çok fazla 429 hatası varsa, thread'leri yavaşlat
                            if total_429_errors >= 5 and total_429_errors % 5 == 0:
                                self.logger.warning(f"Çok fazla rate limit hatası ({total_429_errors})! Delay artırılıyor...")
                                self.current_delay = min(self.current_delay * 1.2, 5.0)
                        
                        self.logger.warning(f"Bu ilan atlanıyor: {link}")
                        basarisiz_linkler += 1
                    
                    proje_verileri.append(project_data)
                    
                    # İlerleme raporu (daha sık - her 10'da bir veya tamamlanınca)
                    elapsed = time.time() - start_time
                    if completed % 10 == 0 or completed == len(links):
                        remaining = len(links) - completed
                        avg_time = elapsed / completed if completed > 0 else 0
                        estimated_remaining = avg_time * remaining / self.max_workers
                        self.logger.info(
                            f"İlerleme: {completed}/{len(links)} proje işlendi "
                            f"({completed*100//len(links)}%) | "
                            f"Tahmini kalan süre: {int(estimated_remaining//60)} dakika {int(estimated_remaining%60)} saniye | "
                            f"429 hataları: {total_429_errors} | "
                            f"Paralel: {self.max_workers} thread"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Proje işleme hatası: {link} - {str(e)}")
                    basarisiz_linkler += 1
                    from datetime import datetime
                    proje_verileri.append({
                        "link": link,
                        "tam_aciklama": f"HATA: {str(e)}",
                        "organization_name": None,
                        "verified": False,
                        "address": None,
                        "website": None,
                        "published_date": None,
                        "scraped_at": datetime.now().isoformat()  # Hata durumunda da tarih ekle
                    })
        
        total_time = time.time() - start_time
        self.logger.info(f"Scraping tamamlandı: {total_time/60:.1f} dakika")
        
        df_projeler = pd.DataFrame(proje_verileri)
        
        # Veri temizleme
        df_projeler = clean_project_descriptions(df_projeler)
        
        self.logger.info(f"Toplam {len(df_projeler)} proje ilanı işlendi (Paralel: {self.max_workers} thread)")
        if basarisiz_linkler > 0:
            self.logger.warning(f"{basarisiz_linkler} adet ilanın detay sayfası çekilemedi")
        
        return df_projeler
