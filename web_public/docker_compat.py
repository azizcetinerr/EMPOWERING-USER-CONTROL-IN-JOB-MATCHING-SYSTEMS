"""
Server (Docker / Hugging Face Spaces) compatibility.

Adapts the Selenium settings of the CV parser to the container without touching the code under src/:
- the system chromedriver is used instead of downloading one with webdriver_manager
- Chrome is started with --no-sandbox and --disable-dev-shm-usage so that it can run inside Docker

Takes effect only if the CHROMEDRIVER_PATH environment variable is set; it has no effect locally (Mac).
"""

import os


def apply(logger=None):
    driver = os.environ.get("CHROMEDRIVER_PATH")
    if not driver:
        return False

    from selenium.webdriver.chrome.options import Options as _Options
    from src.data_collection import cv_parser

    browser = os.environ.get("CHROME_BIN")

    class ServerOptions(_Options):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--headless=new"):
                self.add_argument(arg)
            if browser:
                self.binary_location = browser

    class SystemDriver:
        def install(self):
            return driver

    cv_parser.Options = ServerOptions
    cv_parser.ChromeDriverManager = lambda *a, **k: SystemDriver()
    if logger:
        logger.info(f"🐳 Docker mode: chromedriver={driver}, chrome={browser or 'default'}")
    return True
