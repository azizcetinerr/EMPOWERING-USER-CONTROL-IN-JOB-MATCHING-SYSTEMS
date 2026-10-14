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
    
    def extract_project_info(self, detail_url: str, retries: int = None) -> Dict:
        """
        Extracts project information from a URL (description + organization information).
        
        Args:
            detail_url: Project detail URL
            retries: Number of retries (if None, read from the config)
        
        Returns:
            Project information dictionary
        """
        if retries is None:
            retries = self.retry_count
        
        try:
            response = requests.get(detail_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            content_cards = soup.find_all("div", class_="card-content")
            
            # Extract the project title (inside h1 or h2)
            project_title = None
            h1_title = soup.find("h1")
            if h1_title:
                project_title = h1_title.get_text(strip=True)
            else:
                h2_title = soup.find("h2")
                if h2_title:
                    project_title = h2_title.get_text(strip=True)
            
            # If no title is found, derive it from the URL after removing language codes
            if not project_title or len(project_title) < 5:
                # Take the project ID from the URL and use it as the title (last resort)
                import re
                match = re.search(r'/(\d+)_', detail_url)
                if match:
                    project_title = f"Project {match.group(1)}"
                else:
                    project_title = "Project title not found"
            
            # Description
            full_description_text = ""
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header and "Description" in h6_header.text:
                    all_paragraphs = card.find_all("p")
                    for p in all_paragraphs:
                        text = p.text.strip()
                        # Remove language codes and unnecessary text
                        # Skip language codes such as BGбългарскиCSčeština
                        if len(text) < 200 and any(chr.isalpha() and ord(chr) > 127 for chr in text[:10]):
                            # Probably a language code or a very short text, skip
                            continue
                        # Skip very short or meaningless texts
                        if text and len(text) > 20:
                            full_description_text += text + "\n\n"
                    break
            
            # Fetch the organization information
            organization_name = None
            verified = False
            address = None
            website = None
            
            # Look for the Organization/Hosting organization information
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header:
                    header_text = h6_header.text.lower()
                    if "organization" in header_text or "hosting" in header_text:
                        # Find the organization name
                        org_elements = card.find_all(["p", "span", "div"])
                        for elem in org_elements:
                            text = elem.text.strip()
                            if text and len(text) > 3:
                                organization_name = text
                                break
                        break
            
            # Look for a verified badge/icon
            verified_indicators = soup.find_all(string=lambda t: t and ("verified" in t.lower() or "certified" in t.lower()))
            if verified_indicators:
                verified = True
            
            # Look for the address information
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header and ("address" in h6_header.text.lower() or "location" in h6_header.text.lower()):
                    address_elem = card.find(["p", "span"])
                    if address_elem:
                        address = address_elem.text.strip()
                    break
            
            # Look for the website information
            website_links = soup.find_all("a", href=True)
            for link in website_links:
                href = link.get("href", "")
                if href and ("http" in href or "www" in href):
                    # Could be the organization's website
                    if "youth.europa.eu" not in href and "europa.eu" not in href:
                        website = href
                        break
            
            # Fetch the project publication date (from the created div)
            published_date = None
            created_divs = soup.find_all("div", class_=lambda x: x and "created" in str(x).lower())
            for div in created_divs:
                span = div.find("span")
                if span:
                    date_text = span.get_text(strip=True)
                    # Parse expressions such as "Today", "1 day ago", "2 days ago"
                    published_date = self._parse_relative_date(date_text)
                    if published_date:
                        break
            
            # Fetch the application deadline (deadline, application deadline, expires, etc.)
            application_deadline = None
            # First look for headings such as "deadline", "application deadline", "expires", "closing date"
            for card in content_cards:
                h6_header = card.find("h6")
                if h6_header:
                    header_text = h6_header.text.lower()
                    if any(keyword in header_text for keyword in ["deadline", "expires", "closing", "application"]):
                        # Find the date information
                        date_elements = card.find_all(["p", "span", "div"])
                        for elem in date_elements:
                            text = elem.get_text(strip=True)
                            # Check date formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
                            parsed_date = self._parse_date_string(text)
                            if parsed_date:
                                application_deadline = parsed_date
                                break
                        if application_deadline:
                            break
            
            # If not found in the cards, run a general search
            if not application_deadline:
                # Look for labels such as "Deadline:", "Expires:", "Closes:"
                deadline_keywords = ["deadline", "expires", "closes", "application deadline", "closing date"]
                all_text = soup.get_text()
                for keyword in deadline_keywords:
                    if keyword.lower() in all_text.lower():
                        # Find the date information after the keyword
                        keyword_index = all_text.lower().find(keyword.lower())
                        if keyword_index >= 0:
                            # Look for a date within 50 characters after the keyword
                            text_after_keyword = all_text[keyword_index:keyword_index+100]
                            parsed_date = self._parse_date_string(text_after_keyword)
                            if parsed_date:
                                application_deadline = parsed_date
                        break
            
            return {
                "project_name": project_title,
                "full_description": full_description_text.strip() if full_description_text else "DESCRIPTION NOT FOUND",
                "organization_name": organization_name,
                "verified": verified,
                "address": address,
                "website": website,
                "published_date": published_date.isoformat() if published_date else None,
                "application_deadline": application_deadline.isoformat() if application_deadline else None,
                "scraped_at": None  # This regular scraper does not add the scraping time; only published_date is used
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retries > 0:
                self.logger.warning(f"429 error! Waiting {self.retry_delay} seconds... (URL: {detail_url})")
                time.sleep(self.retry_delay)
                return self.extract_project_info(detail_url, retries - 1)
            else:
                return {
                    "project_name": f"Project {detail_url.split('/')[-1].split('_')[0]}",
                    "full_description": f"ERROR: {e.response.status_code} error",
                    "organization_name": None,
                    "verified": False,
                    "address": None,
                    "website": None,
                    "published_date": None,
                    "application_deadline": None,
                    "scraped_at": None
                }
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error ({detail_url}): {e}")
            return {
                "project_name": f"Project {detail_url.split('/')[-1].split('_')[0]}",
                "full_description": "ERROR: Connection error",
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
        Converts expressions such as "Today", "1 day ago", "2 days ago" into actual dates.
        
        Args:
            date_text: Date text (e.g. "Today", "1 day ago", "2 days ago")
        
        Returns:
            datetime object or None
        """
        from datetime import datetime, timedelta
        import re
        
        date_text = date_text.strip().lower()
        
        if "today" in date_text:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Parse the "X day(s) ago" format
        match = re.search(r'(\d+)\s*(day|days)\s*ago', date_text)
        if match:
            days_ago = int(match.group(1))
            return (datetime.now() - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Parse the "X week(s) ago" format
        match = re.search(r'(\d+)\s*(week|weeks)\s*ago', date_text)
        if match:
            weeks_ago = int(match.group(1))
            return (datetime.now() - timedelta(weeks=weeks_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Parse the "X month(s) ago" format
        match = re.search(r'(\d+)\s*(month|months)\s*ago', date_text)
        if match:
            months_ago = int(match.group(1))
            # Approximately 30 days = 1 month
            return (datetime.now() - timedelta(days=months_ago * 30)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        return None
    
    def _parse_date_string(self, date_text: str):
        """
        Parses date texts in different formats.
        
        Args:
            date_text: Date text (e.g. "2024-12-31", "31/12/2024", "December 31, 2024")
        
        Returns:
            datetime object or None
        """
        from datetime import datetime
        import re
        
        date_text = date_text.strip()
        
        # YYYY-MM-DD format
        try:
            return datetime.strptime(date_text[:10], "%Y-%m-%d")
        except:
            pass
        
        # DD/MM/YYYY or MM/DD/YYYY format
        date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
        match = re.search(date_pattern, date_text)
        if match:
            try:
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))
            except:
                pass
        
        # "December 31, 2024" format
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
        Extracts the project description from a URL (for backward compatibility).
        
        Args:
            detail_url: Project detail URL
            retries: Number of retries (if None, read from the config)
        
        Returns:
            Project description text
        """
        info = self.extract_project_info(detail_url, retries)
        return info.get("full_description", "DESCRIPTION NOT FOUND")
    
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
            
            project_info = self.extract_project_info(link)
            
            if "ERROR" in project_info.get("full_description", "") or "NOT FOUND" in project_info.get("full_description", ""):
                self.logger.warning(f"Skipping this listing: {link} (Reason: {project_info.get('full_description', '')[:50]}...)")
                failed_links += 1
            
            from datetime import datetime
            project_records.append({
                "link": link,
                "project_name": project_info.get("project_name", f"Project {link.split('/')[-1].split('_')[0]}"),
                "full_description": project_info.get("full_description", ""),
                "organization_name": project_info.get("organization_name"),
                "verified": project_info.get("verified", False),
                "address": project_info.get("address"),
                "website": project_info.get("website"),
                "published_date": project_info.get("published_date"),  # Actual publication date
                "application_deadline": project_info.get("application_deadline"),  # Application deadline
                "scraped_at": datetime.now().isoformat()  # Scraping time
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

