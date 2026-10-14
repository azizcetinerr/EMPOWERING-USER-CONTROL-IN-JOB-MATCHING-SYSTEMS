"""
Module that collects project listings from the Youth Portal
Much faster data collection through the API endpoint
"""

import requests
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time

from ..utils.data_cleaner import remove_duplicate_links


class YouthPortalScraper:
    """Youth Portal API scraper class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # API settings
        self.base_url = config['data_sources']['youth_portal']['base_url']
        self.use_api = config['data_sources']['youth_portal'].get('use_api', True)
        
        # Requests settings
        scraping_config = config.get('scraping', {}).get('requests', {})
        self.retry_count = scraping_config.get('retry_count', 5)
        self.retry_delay = scraping_config.get('retry_delay', 10)
        self.request_delay = scraping_config.get('request_delay', 1)
        self.timeout = scraping_config.get('timeout', 30)
        
        # API endpoint - fetch active projects using today's date
        today = datetime.now().strftime('%Y-%m-%d')
        api_base = config['data_sources']['youth_portal'].get('api_url', 'https://youth.europa.eu/api/rest/eyp/v1/search_en')
        self.api_url = (
            f"{api_base}"
            f"?type=Opportunity"
            f"&size=1000"  # Maximum number of projects (API limit)
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
        Fetches all project links from the API (or from the cache).
        
        Args:
            cached_api_data: If given, this data is used instead of sending an API request
        
        Returns:
            List of project links
        """
        try:
            # Use cached data if available
            if cached_api_data is not None:
                self.logger.info("📦 Using cached API data (skipping API request)")
                response = cached_api_data
            else:
                self.logger.info(f"🚀 Fetching project list from the API: {self.api_url}")
                # Fetch data from the API
                response = self._fetch_api_data()
            
            if not response:
                self.logger.error("Could not fetch data from the API")
                return []
            
            # Build the project links
            links = self._extract_links_from_api(response)
            
            # Remove duplicates
            links = remove_duplicate_links(links)
            
            self.logger.info(f"✅ Found {len(links)} unique project links in total")
            return links
            
        except Exception as e:
            self.logger.error(f"Link collection error: {e}", exc_info=True)
            raise
    
    def _fetch_api_data(self) -> Optional[Dict]:
        """Fetches data from the API (with a retry mechanism)"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        for attempt in range(self.retry_count):
            try:
                self.logger.debug(f"Sending API request (attempt {attempt + 1}/{self.retry_count})...")
                response = requests.get(
                    self.api_url,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    total = data.get('hits', {}).get('total', {}).get('value', 0)
                    self.logger.info(f"✅ Found {total} projects in the API")
                    return data
                elif response.status_code == 429:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.warning(f"⚠️  Rate limit (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.warning(f"⚠️  API error: {response.status_code}. Waiting {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"⚠️  Timeout error (attempt {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.warning(f"⚠️  API request error: {e} (attempt {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        self.logger.error("❌ Could not fetch data from the API (all attempts failed)")
        return None
    
    def _extract_links_from_api(self, api_data: Dict) -> List[str]:
        """Extracts project links from the API response"""
        links = []
        
        hits = api_data.get('hits', {}).get('hits', [])
        self.logger.info(f"📋 Processing {len(hits)} projects...")
        
        for hit in hits:
            source = hit.get('_source', {})
            opid = source.get('opid')
            
            if opid:
                # Build the project URL: https://youth.europa.eu/solidarity/opportunity/{opid}_en
                link = f"{self.base_url}/solidarity/opportunity/{opid}_en"
                links.append(link)
        
        return links
    
    def get_project_data_from_api(self) -> Optional[Dict]:
        """
        Returns the raw project data from the API (all metadata, not only links)
        
        Returns:
            API response dict or None
        """
        return self._fetch_api_data()

