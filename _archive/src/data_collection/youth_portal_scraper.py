"""
Module that collects project listings from the Youth Portal
"""

import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
from typing import List
import logging

from ..utils.data_cleaner import remove_duplicate_links


class YouthPortalScraper:
    """Youth Portal web scraping class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Selenium settings
        self.list_url = config['data_sources']['youth_portal']['list_url']
        self.base_url = config['data_sources']['youth_portal']['base_url']
        
        scraping_config = config.get('scraping', {}).get('selenium', {})
        self.headless = scraping_config.get('headless', True)
        self.wait_timeout = scraping_config.get('wait_timeout', 15)
        
        self.driver = None
    
    def _setup_driver(self):
        """Sets up the Selenium driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("window-size=1920x1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.logger.info("Selenium driver started")
    
    def get_project_links(self) -> List[str]:
        """
        Fetches all project links.
        
        Returns:
            List of project links
        """
        if self.driver is None:
            self._setup_driver()
        
        try:
            self.logger.info(f"Connecting to the main listing page: {self.list_url}")
            self.driver.get(self.list_url)
            
            # Find out the target number of projects
            target_project_count = self._get_target_project_count()
            self.logger.info(f"Target number of projects: {target_project_count}")
            
            # Load all listings
            self._load_all_projects(target_project_count)
            
            # Collect the links
            links = self._extract_links()
            
            # Remove duplicates
            links = remove_duplicate_links(links)
            
            self.logger.info(f"Found {len(links)} unique project links in total")
            return links
            
        except Exception as e:
            self.logger.error(f"Link collection error: {e}", exc_info=True)
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Browser closed")
    
    def _get_target_project_count(self) -> int:
        """Reads the target number of projects from the page"""
        try:
            target_label_xpath = "//h5/span[contains(text(), 'opportunities found')]"
            WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.XPATH, target_label_xpath))
            )
            target_text = self.driver.find_element(By.XPATH, target_label_xpath).text
            match = re.search(r'(\d+)', target_text)
            if match:
                return int(match.group(1))
            else:
                raise Exception("The target number text was found but the number could not be extracted")
        except Exception as e:
            self.logger.warning(f"Number of projects not found: {e}")
            return 0
    
    def _load_all_projects(self, target_count: int):
        """Clicks the 'Load more' button to load all projects"""
        load_more_xpath = "//button[contains(text(), 'Load more')]"
        
        while len(self.driver.find_elements(By.CLASS_NAME, "card-item")) < target_count:
            try:
                current_project_count = len(self.driver.find_elements(By.CLASS_NAME, "card-item"))
                
                if current_project_count >= target_count:
                    self.logger.info(f"All listings loaded (current: {current_project_count} / target: {target_count})")
                    break
                
                load_more_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, load_more_xpath))
                )
                self.driver.execute_script("arguments[0].click();", load_more_button)
                
                self.logger.debug(f"'Load more' clicked. Loaded: {current_project_count}/{target_count}")
                time.sleep(2)
                
            except (NoSuchElementException, TimeoutException):
                current_project_count = len(self.driver.find_elements(By.CLASS_NAME, "card-item"))
                self.logger.info(f"All listings loaded (the Load more button disappeared). Loaded: {current_project_count}")
                break
            except Exception as e:
                self.logger.warning(f"Error while clicking the button: {e}")
                break
    
    def _extract_links(self) -> List[str]:
        """Extracts all project links on the page"""
        html_source = self.driver.page_source
        soup = BeautifulSoup(html_source, "html.parser")
        listing_blocks = soup.find_all("div", class_="card-item")
        
        link_list = []
        for block in listing_blocks:
            link_element = block.find("div", class_="card-action")
            if link_element:
                link_tag = link_element.find("a")
                if link_tag and link_tag.has_attr('href'):
                    link = self.base_url + link_tag['href']
                    link_list.append(link)
        
        return link_list

