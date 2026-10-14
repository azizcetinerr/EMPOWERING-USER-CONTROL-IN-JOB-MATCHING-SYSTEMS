"""
Module that fetches descriptions from project detail pages in parallel (fast version)
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
    """Class that fetches project detail pages in parallel (fast version)"""
    
    def __init__(self, config: dict, logger: logging.Logger = None, max_workers: int = 4):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
            max_workers: Number of parallel threads (default: 4, optimized)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.max_workers = max_workers
        
        scraping_config = config.get('scraping', {}).get('requests', {})
        self.retry_count = scraping_config.get('retry_count', 5)
        self.retry_delay = scraping_config.get('retry_delay', 10)
        self.request_delay = scraping_config.get('request_delay', 1.0)  # Will be adaptive
        self.base_delay = 1.0  # Initial delay
        self.timeout = scraping_config.get('timeout', 30)
        self.max_retry_delay = 30  # Maximum wait time (for 429 errors)
        
        # Adaptive rate limiting
        self.consecutive_successes = 0
        self.consecutive_429_errors = 0
        self.current_delay = self.base_delay
    
    def extract_project_info(self, detail_url: str, retries: int = None) -> Dict:
        """
        Extracts information from a project detail page (with rate limiting).
        
        Args:
            detail_url: Project detail page URL
            retries: Number of retries (if None, read from the config)
        
        Returns:
            Project information dictionary
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
                
                # 429 error check
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 10))
                    wait_time = min(retry_after, self.max_retry_delay)
                    self.logger.warning(f"Rate limit (429)! Waiting {wait_time} seconds...")
                    self.consecutive_429_errors += 1
                    self.consecutive_successes = 0
                    # Increase the delay
                    self.current_delay = min(self.current_delay * 1.5, 5.0)
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                # Successful request - decrease the delay (adaptive)
                self.consecutive_successes += 1
                self.consecutive_429_errors = 0
                if self.consecutive_successes > 10:
                    # Decrease the delay after 10 successful requests
                    self.current_delay = max(self.current_delay * 0.9, self.base_delay)
                    self.consecutive_successes = 0
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Full description
                description_elements = soup.find_all(['p', 'div'], class_=lambda x: x and ('description' in str(x).lower() or 'content' in str(x).lower()))
                full_description = ' '.join([elem.get_text(strip=True) for elem in description_elements if elem.get_text(strip=True)])
                
                if not full_description:
                    # Alternative: take all paragraph tags
                    paragraphs = soup.find_all('p')
                    full_description = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                
                # Organization information
                org_name = None
                verified = False
                address = None
                website = None
                
                # Organization name
                org_elements = soup.find_all(['span', 'div', 'h3'], class_=lambda x: x and 'organization' in str(x).lower())
                if org_elements:
                    org_name = org_elements[0].get_text(strip=True)
                
                # Verified badge
                verified_elements = soup.find_all(['span', 'div'], class_=lambda x: x and 'verified' in str(x).lower())
                verified = len(verified_elements) > 0
                
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
                    # General search
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
                    "full_description": full_description if full_description else "DESCRIPTION NOT FOUND",
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
                    self.logger.warning(f"Rate limit (429) - waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    if attempt < retries:
                        continue
                elif attempt < retries:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.warning(f"Could not fetch project details: {detail_url} - {str(e)}")
                    return {
                        "full_description": f"ERROR: {str(e)}",
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
                    self.logger.warning(f"Could not fetch project details: {detail_url} - {str(e)}")
                    return {
                        "full_description": f"ERROR: {str(e)}",
                        "organization_name": None,
                        "verified": False,
                        "address": None,
                        "website": None,
                        "published_date": None,
                        "application_deadline": None
                    }
            except Exception as e:
                self.logger.error(f"Unexpected error: {detail_url} - {str(e)}")
                return {
                    "full_description": f"ERROR: {str(e)}",
                    "organization_name": None,
                    "verified": False,
                    "address": None,
                    "website": None,
                    "published_date": None,
                    "application_deadline": None
                }
        
        return {
            "full_description": "NOT FOUND",
            "organization_name": None,
            "verified": False,
            "address": None,
            "website": None,
            "published_date": None,
            "application_deadline": None
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
    
    def _scrape_single_project(self, link: str) -> Dict:
        """Scrapes a single project (for parallel processing)"""
        project_info = self.extract_project_info(link)
        
        # Adaptive delay - use the delay after successful requests
        time.sleep(self.current_delay)
        
        from datetime import datetime
        return {
            "link": link,
            "full_description": project_info.get("full_description", ""),
            "organization_name": project_info.get("organization_name"),
            "verified": project_info.get("verified", False),
            "address": project_info.get("address"),
            "website": project_info.get("website"),
            "published_date": project_info.get("published_date"),  # Actual publication date
            "application_deadline": project_info.get("application_deadline"),  # Application deadline
            "scraped_at": datetime.now().isoformat()  # Scraping time
        }
    
    def scrape_all_projects(self, links: List[str]) -> pd.DataFrame:
        """
        Fetches details from all project links in parallel.
        
        Args:
            links: List of project links
        
        Returns:
            Project details DataFrame
        """
        self.logger.info(f"Fetching {len(links)} project details in parallel... (max {self.max_workers} threads)")
        
        project_records = []
        failed_links = 0
        
        # Parallel scraping (with adaptive rate limiting)
        self.logger.info(f"Starting parallel scraping: {len(links)} projects, {self.max_workers} threads")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start all tasks
            future_to_link = {executor.submit(self._scrape_single_project, link): link for link in links}
            
            # Collect the results
            completed = 0
            total_429_errors = 0
            
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                completed += 1
                
                try:
                    project_data = future.result()
                    
                    if "ERROR" in project_data.get("full_description", "") or "NOT FOUND" in project_data.get("full_description", ""):
                        if "429" in project_data.get("full_description", ""):
                            total_429_errors += 1
                            # If there are too many 429 errors, slow the threads down
                            if total_429_errors >= 5 and total_429_errors % 5 == 0:
                                self.logger.warning(f"Too many rate limit errors ({total_429_errors})! Increasing the delay...")
                                self.current_delay = min(self.current_delay * 1.2, 5.0)
                        
                        self.logger.warning(f"Skipping this listing: {link}")
                        failed_links += 1
                    
                    project_records.append(project_data)
                    
                    # Progress report (more often - every 10 items or on completion)
                    elapsed = time.time() - start_time
                    if completed % 10 == 0 or completed == len(links):
                        remaining = len(links) - completed
                        avg_time = elapsed / completed if completed > 0 else 0
                        estimated_remaining = avg_time * remaining / self.max_workers
                        self.logger.info(
                            f"Progress: {completed}/{len(links)} projects processed "
                            f"({completed*100//len(links)}%) | "
                            f"Estimated time remaining: {int(estimated_remaining//60)} minutes {int(estimated_remaining%60)} seconds | "
                            f"429 errors: {total_429_errors} | "
                            f"Parallel: {self.max_workers} threads"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Project processing error: {link} - {str(e)}")
                    failed_links += 1
                    from datetime import datetime
                    project_records.append({
                        "link": link,
                        "full_description": f"ERROR: {str(e)}",
                        "organization_name": None,
                        "verified": False,
                        "address": None,
                        "website": None,
                        "published_date": None,
                        "scraped_at": datetime.now().isoformat()  # Add the date in the error case too
                    })
        
        total_time = time.time() - start_time
        self.logger.info(f"Scraping completed: {total_time/60:.1f} minutes")
        
        df_projects = pd.DataFrame(project_records)
        
        # Data cleaning
        df_projects = clean_project_descriptions(df_projects)
        
        self.logger.info(f"{len(df_projects)} project listings processed in total (parallel: {self.max_workers} threads)")
        if failed_links > 0:
            self.logger.warning(f"Could not fetch the detail page of {failed_links} listings")
        
        return df_projects
