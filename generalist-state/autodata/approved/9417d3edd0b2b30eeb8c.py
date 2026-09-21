import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


# set options for headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  

# set the path to the ChromeDriver executable
chrome_service = Service(r'C:\Users\...\chromedriver.exe') 

# create a new instance of the Chrome WebDriver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# set the initial page number and maximum number of pages to scrape
current_page = 1
max_pages = 8  
save_path = r'C:\Users\...\Selenium_control.txt'


# open the file in write mode to save the scraped data
with open(save_path, "w", encoding="utf-8") as file:
    while current_page <= max_pages:
        # Costruisci l'URL per la pagina corrente
        URL = f"https://forum.alfemminile.com/discussions/tagged/salute-mentale/p{current_page}"
        driver.get(URL)

        # increase timeout to wait for the page to load
        wait = WebDriverWait(driver, 5)

        try:
            comments_section = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/main/div[4]/div/div/div/div[2]/section")))
            
            
            # find all comment elements within the comments section
            comments = comments_section.find_elements(By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/main/div[4]/div/div/div/div[2]/section/div[2]/table")

            if comments:
                print(f"Pagina {current_page}")
                for comment in comments:
                    try:
                        title = comment.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/main/div[4]/div/div/div/div[2]/section/div[2]/table/tbody/tr[1]/td[1]").text
                        link = comment.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/main/div[4]/div/div/div/div[2]/section/div[2]/table/tbody/tr[1]/td[1]/div/a")
                        href = link.get_attribute("href")
                        
                        # open the link in a new tab
                        driver.execute_script("window.open(arguments[0]);", href)
                        driver.switch_to.window(driver.window_handles[1])
                        
                            
                        # wait for the full comment to load and extract its text
                        full_comment = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/main/div[4]/div/div/div/div[2]/section/section/div[1]/div/div/div[2]/div/div[1]"))).text

                    
                        file.write(f"{title}\n")
                        file.write(f"{full_comment}\n\n")
                        print(f"Title: {title}")
                        print(f"Content: {full_comment}")
                        print("-" * 80 + "\n")

                        # close the new tab and switch back to the original tab
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        
                    except Exception as e:
                        print(f"Si è verificato un errore durante l'estrazione dei dati del commento: {e}")

            else:
                print(f"Pagina {current_page}: Nessun commento trovato.")

            # go to the next page
            current_page += 1
            time.sleep(5)  

        except Exception as e:
            print(f"Si è verificato un errore durante l'attesa della sezione dei commenti: {e}")
            break

# close the WebDriver
driver.quit()
