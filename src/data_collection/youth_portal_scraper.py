"""
Youth Portal'dan proje ilanlarını çeken modül
API endpoint kullanarak çok daha hızlı veri toplama
"""

import requests
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time

from ..utils.data_cleaner import remove_duplicate_links


class YouthPortalScraper:
    """Youth Portal API scraper sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # API ayarları
        self.base_url = config['data_sources']['youth_portal']['base_url']
        self.use_api = config['data_sources']['youth_portal'].get('use_api', True)
        
        # Requests ayarları
        scraping_config = config.get('scraping', {}).get('requests', {})
        self.retry_count = scraping_config.get('retry_count', 5)
        self.retry_delay = scraping_config.get('retry_delay', 10)
        self.request_delay = scraping_config.get('request_delay', 1)
        self.timeout = scraping_config.get('timeout', 30)
        
        # API endpoint - bugünün tarihini kullanarak aktif projeleri al
        today = datetime.now().strftime('%Y-%m-%d')
        api_base = config['data_sources']['youth_portal'].get('api_url', 'https://youth.europa.eu/api/rest/eyp/v1/search_en')
        self.api_url = (
            f"{api_base}"
            f"?type=Opportunity"
            f"&size=1000"  # Maksimum proje sayısı (API limiti)
            f"&from=0"
            f"&filters[status]=open"
            f"&filters[date_end][operator]=>="
            f"&filters[date_end][value]={today}"
            f"&filters[date_end][type]=must"
            f"&filters[funding_programme][id][0]=5"
            f"&filters[funding_programme][id][1]=4"
            f"&filters[funding_programme][id][2]=3"
            f"&filters[funding_programme][id][3]=2"
            f"&filters[funding_programme][id][4]=1"
            f"&filters[funding_programme][id][5]=8"
            f"&filters[funding_programme][id][6]=6"
            f"&filters[funding_programme][id][7]=7"
            f"&filters[date_application_end][operator]=>="
            f"&filters[date_application_end][value]={today}"
            f"&filters[date_application_end][type]=must"
            f"&filters[date_application_end][group]=deadline"
            f"&filters[has_no_deadline][value]=true"
            f"&filters[has_no_deadline][type]=must"
            f"&filters[has_no_deadline][group]=deadline"
            f"&fields[0]=opid"
            f"&fields[1]=title"
            f"&fields[2]=logo"
            f"&fields[3]=geocode.lat"
            f"&fields[4]=geocode.lon"
            f"&fields[5]=town"
            f"&fields[6]=country"
            f"&fields[7]=has_no_deadline"
            f"&fields[8]=duration"
            f"&fields[9]=date_start"
            f"&fields[10]=date_end"
            f"&fields[11]=date_application_end"
            f"&fields[12]=is_esc_related"
            f"&fields[13]=created"
        )
    
    def get_project_links(self, cached_api_data: Optional[Dict] = None) -> List[str]:
        """
        API'den tüm proje linklerini çeker (veya cache'den).
        
        Args:
            cached_api_data: Eğer verilirse, API isteği yapmak yerine bu veriyi kullanır
        
        Returns:
            Proje link listesi
        """
        try:
            # Cache'lenmiş veri varsa onu kullan
            if cached_api_data is not None:
                self.logger.info("📦 Cache'lenmiş API verileri kullanılıyor (API isteği atlanıyor)")
                response = cached_api_data
            else:
                self.logger.info(f"🚀 API'den proje listesi alınıyor: {self.api_url}")
                # API'den veri çek
                response = self._fetch_api_data()
            
            if not response:
                self.logger.error("API'den veri alınamadı")
                return []
            
            # Proje linklerini oluştur
            links = self._extract_links_from_api(response)
            
            # Duplicate'leri kaldır
            links = remove_duplicate_links(links)
            
            self.logger.info(f"✅ Toplam {len(links)} adet benzersiz proje linki bulundu")
            return links
            
        except Exception as e:
            self.logger.error(f"Link toplama hatası: {e}", exc_info=True)
            raise
    
    def _fetch_api_data(self) -> Optional[Dict]:
        """API'den veri çeker (retry mekanizması ile)"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        for attempt in range(self.retry_count):
            try:
                self.logger.debug(f"API isteği gönderiliyor (deneme {attempt + 1}/{self.retry_count})...")
                response = requests.get(
                    self.api_url,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    total = data.get('hits', {}).get('total', {}).get('value', 0)
                    self.logger.info(f"✅ API'den {total} proje bulundu")
                    return data
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.warning(f"⚠️  Rate limit (429). {wait_time} saniye bekleniyor...")
                    time.sleep(wait_time)
                else:
                    self.logger.warning(f"⚠️  API hatası: {response.status_code}. {self.retry_delay} saniye bekleniyor...")
                    time.sleep(self.retry_delay)
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"⚠️  Timeout hatası (deneme {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.warning(f"⚠️  API isteği hatası: {e} (deneme {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        self.logger.error("❌ API'den veri alınamadı (tüm denemeler başarısız)")
        return None
    
    def _extract_links_from_api(self, api_data: Dict) -> List[str]:
        """API yanıtından proje linklerini çıkarır"""
        links = []
        
        hits = api_data.get('hits', {}).get('hits', [])
        self.logger.info(f"📋 {len(hits)} proje işleniyor...")
        
        for hit in hits:
            source = hit.get('_source', {})
            opid = source.get('opid')
            
            if opid:
                # Proje URL'sini oluştur: https://youth.europa.eu/solidarity/opportunity/{opid}_en
                link = f"{self.base_url}/solidarity/opportunity/{opid}_en"
                links.append(link)
        
        return links
    
    def get_project_data_from_api(self) -> Optional[Dict]:
        """
        API'den ham proje verilerini döndürür (sadece link değil, tüm metadata)
        
        Returns:
            API yanıtı dict veya None
        """
        return self._fetch_api_data()

