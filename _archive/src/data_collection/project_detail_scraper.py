"""
Module that fetches descriptions from project detail pages
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import logging

from ..utils.data_cleaner import clean_project_descriptions


class ProjectDetailScraper:
    """Class that fetches project detail pages"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
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
        Extracts the project description from a URL.
        
        Args:
            detail_url: Project detail URL
            retries: Number of retries (if None, read from the config)
        
        Returns:
            Project description text
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
            
            return full_description_text.strip() if full_description_text else "DESCRIPTION NOT FOUND"
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retries > 0:
                self.logger.warning(f"429 error! Waiting {self.retry_delay} seconds... (URL: {detail_url})")
                time.sleep(self.retry_delay)
                return self.extract_project_description(detail_url, retries - 1)
            else:
                return f"ERROR: {e.response.status_code} error"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error ({detail_url}): {e}")
            return "ERROR: Connection error"
    
    def scrape_all_projects(self, links: List[str]) -> pd.DataFrame:
        """
        Fetches details from all project links.
        
        Args:
            links: List of project links
        
        Returns:
            Project details DataFrame
        """
        self.logger.info(f"Fetching {len(links)} project details...")
        
        project_records = []
        failed_links = 0
        
        for i, link in enumerate(links):
            self.logger.info(f"Processing... ({i+1}/{len(links)}) - {link}")
            
            full_description = self.extract_project_description(link)
            
            if "ERROR" in full_description or "NOT FOUND" in full_description:
                self.logger.warning(f"Skipping this listing: {link} (Reason: {full_description[:50]}...)")
                failed_links += 1
            
            project_records.append({
                "link": link,
                "full_description": full_description
            })
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Progress report
            if (i + 1) % 50 == 0:
                self.logger.info(f"Progress: {i+1}/{len(links)} projects processed")
        
        df_projects = pd.DataFrame(project_records)
        
        # Data cleaning
        df_projects = clean_project_descriptions(df_projects)
        
        self.logger.info(f"{len(df_projects)} project listings processed in total")
        if failed_links > 0:
            self.logger.warning(f"Could not fetch the detail page of {failed_links} listings")
        
        return df_projects

