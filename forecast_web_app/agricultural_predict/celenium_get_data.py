from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import time





def get_data_from_web():
    options = webdriver.ChromeOptions()
    prefs = {
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": "false",
        "download.directory_upgrade": "true",
        "download.default_directory": "/tmp/downloads"
    }
    options.add_experimental_option("prefs", prefs)
    remote_webdriver = 'remote_chromedriver'
    with webdriver.Remote(f'http://localhost:4444/wd/hub', options=options) as driver:

        driver.get("https://agro.gov.vn/vn/nguonwmy.aspx")

        select_element = driver.find_element(By.ID, 'ctl00_maincontent_mathangnongsan')
        select = Select(select_element)
        if select:
            option_list = select.options
            select.select_by_visible_text('Gạo thường|Rice')

            time.sleep(5)

            driver.execute_script("""
            document.getElementById('ctl00_maincontent_tai_excel').removeAttribute("disabled");
            """)
            element = driver.find_element(By.ID, "ctl00_maincontent_tai_excel")
            element.click()

            time.sleep(5)

get_data_from_web()