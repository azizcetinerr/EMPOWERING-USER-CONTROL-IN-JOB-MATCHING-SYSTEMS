"""
Module that extracts information from a Europass CV
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
    """Europass CV parser class"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.cv_url = config['data_sources']['europass_cv']['url']
        if not self.cv_url:
            raise ValueError("CV URL not specified in the configuration file!")
        
        scraping_config = config.get('scraping', {}).get('selenium', {})
        self.headless = scraping_config.get('headless', True)
        self.wait_timeout = scraping_config.get('wait_timeout', 10)  # Lowered from 15 to 10
        self.page_load_timeout = scraping_config.get('page_load_timeout', 30)  # Page load timeout
        self.implicit_wait = scraping_config.get('implicit_wait', 5)  # Implicit wait
        
        self.driver = None
    
    def _setup_driver(self):
        """Sets up the Selenium driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")  # For security
        chrome_options.add_argument("--disable-dev-shm-usage")  # For memory problems
        chrome_options.add_argument("window-size=1920x1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set the timeouts
        self.driver.set_page_load_timeout(self.page_load_timeout)  # Page load timeout
        self.driver.implicitly_wait(self.implicit_wait)  # Element lookup timeout
        
        self.logger.info(f"Selenium driver started (page_load_timeout: {self.page_load_timeout}s, wait_timeout: {self.wait_timeout}s)")
    
    def parse_cv(self) -> Tuple[str, List[str]]:
        """
        Extracts the full text and the skill list from the CV.
        
        Returns:
            (full_text, skill_list) tuple
        """
        if self.driver is None:
            self._setup_driver()
        
        try:
            self.logger.info(f"Fetching Europass CV link: {self.cv_url}")
            
            # Page loading (protected with a timeout)
            try:
                self.driver.get(self.cv_url)
                self.logger.info("Page loaded, waiting for the Skills section...")
            except Exception as page_load_error:
                error_msg = str(page_load_error)
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    self.logger.error(f"❌ Page load timeout ({self.page_load_timeout}s) - the Europass server may not be responding")
                    raise TimeoutError(f"The Europass CV page could not be loaded within {self.page_load_timeout} seconds. Please check the link: {self.cv_url}")
                else:
                    self.logger.error(f"❌ Page load error: {error_msg}")
                    raise
            
            # Wait for the page to load (until the Skills heading exists)
            skills_header_xpath = "//h2[contains(text(), 'Skills')]"
            try:
                WebDriverWait(self.driver, self.wait_timeout).until(
                    EC.presence_of_element_located((By.XPATH, skills_header_xpath))
                )
                self.logger.info("✅ Europass profile loaded. Parsing data...")
            except Exception as wait_error:
                self.logger.error(f"❌ Skills section not found within {self.wait_timeout} seconds. The page structure may have changed.")
                # The page loaded but there is no Skills section - continue anyway, there may be a different structure
                self.logger.warning("⚠️  Skills section not found, the whole page content will be processed...")
            
            html_source = self.driver.page_source
            soup = BeautifulSoup(html_source, "html.parser")
            
            cv_text_list = []
            cv_skill_list = []
            
            # Education section
            self._extract_section(soup, "Education and training", cv_text_list)
            
            # Work experience
            self._extract_section(soup, "Work experience", cv_text_list)
            
            # Skills
            skills = self._extract_skills(soup)
            cv_text_list.extend(skills)
            cv_skill_list.extend(skills)
            
            # Certificates
            self._extract_section(soup, "Certificates", cv_text_list)
            
            # Join the full text
            cv_full_text = "\n".join(list(filter(None, cv_text_list)))
            
            self.logger.info(f"Found {len(cv_skill_list)} raw skill texts in the CV")
            
            return cv_full_text, cv_skill_list
            
        except Exception as e:
            self.logger.error(f"CV parsing error: {e}", exc_info=True)
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Browser closed")
    
    def _extract_section(self, soup: BeautifulSoup, section_name: str, text_list: List[str]):
        """Extracts a specific section"""
        header = soup.find("h2", string=lambda t: t and section_name in t)
        if header:
            current_tag = header.find_next_sibling()
            while current_tag and current_tag.name != 'h2':
                if current_tag.name in ['p', 'h4']:
                    text_list.append(current_tag.text.strip())
                current_tag = current_tag.find_next_sibling()
    
    def _extract_skills(self, soup: BeautifulSoup) -> List[str]:
        """Extracts the skills section"""
        skills = []
        skills_header = soup.find("h2", string=lambda t: t and "Skills" in t)
        if skills_header:
            skills_ul = skills_header.find_next("ul")
            if skills_ul:
                skill_items = skills_ul.find_all("li")
                for item in skill_items:
                    skill_text = item.text.strip()
                    if skill_text:
                        skills.append(skill_text)
        return skills

