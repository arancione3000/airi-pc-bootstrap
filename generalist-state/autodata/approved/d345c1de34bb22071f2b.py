import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def contains_exclusion_pattern(text, phrase, word1, word2, distance1, distance2): 
    phrase_pattern = re.compile(r'\bmi\b \bsono\b \bfidanzato\b', re.IGNORECASE)
    phrase_matches = list(phrase_pattern.finditer(text))

    if not phrase_matches:
        return False

    # Check if word1 and word2 are present in the surrounding text of the phrase
    word1_pattern = re.compile(r'\b' + re.escape(word1) + r'\b', re.IGNORECASE)
    word2_pattern = re.compile(r'\b' + re.escape(word2) + r'\b', re.IGNORECASE)

    for match in phrase_matches:
        # Get the surrounding text based on the specified distance
        surrounding_text = text[max(0, match.start() - distance1 * 10): match.end() + distance1 * 10]
        if word1_pattern.search(surrounding_text) and word2_pattern.search(surrounding_text):
            return True
    return False

def contains_exclusion_pattern1(text, phrase1, phrase2, max_distance):
    phrase1_pattern = re.compile(r'\b' + re.escape(phrase1) + r'\b', re.IGNORECASE)
    phrase2_pattern = re.compile(r'\b' + re.escape(phrase2) + r'\b', re.IGNORECASE)

    # find all occurrences of phrase1 and phrase2 in the text
    phrase1_matches = [match.start() for match in phrase1_pattern.finditer(text)]
    phrase2_matches = [match.start() for match in phrase2_pattern.finditer(text)]

    # Debug prints
    #print(f"Phrase1 '{phrase1}' matches: {phrase1_matches}")
    #print(f"Phrase2 '{phrase2}' matches: {phrase2_matches}")

    if not phrase1_matches or not phrase2_matches:
        return False

    # chack the distance between each occurrence of phrase1 and phrase2
    words = text.split()
    for p1_index in phrase1_matches:
        for p2_index in phrase2_matches:
            p1_word_index = len(text[:p1_index].split())
            p2_word_index = len(text[:p2_index].split())
            distance = abs(p1_word_index - p2_word_index)
            #print(f"Checking distance between '{phrase1}' at {p1_word_index} and '{phrase2}' at {p2_word_index}: {distance}")
            if distance <= max_distance:
                #print(f"Match found within distance: phrase1 at {p1_word_index}, phrase2 at {p2_word_index}, distance: {distance}")
                return True

    return False



def count_occurrences(patterns, text):
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def contains_past_reference(text):
    past_phrases = [
        r'\bsono stata\b(?:\W+\w+){0,8}\W*\banoressica\b',
        r'\bho sofferto\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bho avuto\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bsono stato\b(?:\W+\w+){0,8}\W*\banoressico\b',
        r'\bsofferto di\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bda bambina\b(?:\W+\w+){0,8}\W*\banoressica\b',
        r'\bda bambina\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bda bambino\b(?:\W+\w+){0,8}\W*\banoressico\b',
        r'\bda adolescente\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bda adolescente\b(?:\W+\w+){0,8}\W*\banoressica\b',
        r'\bda adolescente\b(?:\W+\w+){0,8}\W*\banoressico\b',
        r'\bpassato\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bpassato\b(?:\W+\w+){0,8}\W*\banoressica\b',
        r'\bpassato\b(?:\W+\w+){0,6}\W*\banoressico\b',
        r'\bdai \d+ ai \d+ anni\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\banno di\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\banno in\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\bnon sono mai caduta\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\blontana\b(?:\W+\w+){0,8}\W*\banoressia\b',
        r'\blontano\b(?:\W+\w+){0,8}\W*\banoressia\b'
    ]
    return any(re.search(phrase, text, re.IGNORECASE) for phrase in past_phrases)

def contains_present_reference(text):
    present_phrases = [
        r"\btutt'ora\b(?:\W+\w+){0,10}\W*\banoressia\b",
        r"\banoressia\b(?:\W+\w+){0,9}\W*\btutt'ora\b",
        r"\bancora\b(?:\W+\w+){0,10}\W*\banoressia\b",
        r"\banoressia\b(?:\W+\w+){0,9}\W*\bancora\b",
        r"\btutt'oggi\b(?:\W+\w+){0,10}\W*\banoressia\b",
        r"\banoressia\b(?:\W+\w+){0,9}\W*\btutt'oggi\b"
    ]
    return any(re.search(phrase, text, re.IGNORECASE) for phrase in present_phrases)

def extract_contexts(text, keywords, window=15):
    words = text.split()
    contexts = []

    for keyword in keywords:
        keyword_indices = [i for i, word in enumerate(words) if re.search(r'\b' + re.escape(keyword) + r'\b', word, re.IGNORECASE)]
        for index in keyword_indices:
            start = max(0, index - window)
            end = min(len(words), index + window + 1)
            context = ' '.join(words[start:end])
            contexts.append(context)

    return contexts

def is_relevant_comment(text):
    # Lists of present and past verbs
    present_verbs = [r'\bsono\b', r'\bho\b', r'\bvivo\b', r'\bsto\b', r'\bho\b', r'\bmangio\b', r'\bvomito\b', r'\bsento\b', r'\bpenso\b', r'\bsoffro\b', r'\bcerco\b', r'\bvoglio\b', r'\bposso\b', r'\bdevo\b', r'\bso\b', r'\briesco\b', r'\bvorrei\b', r'\bpotrei\b',r'\bdovrei\b', r'\briuscirei\b', r'\bavrei\b', r'\bsarei\b']
    past_verbs = [r'\bero\b', r'\bstavo\b', r'\bavevo\b', r'\bvivevo\b', r'\bmangiavo\b', r'\bvomitavo\b', r'\bsentivo\b', r'\bpensavo\b', r'\bcercavo\b', r'\bsoffrivo\b', r'\bvolevo\b', r'\bsapevo\b', r'\bpotevo\b', r'\bdovevo\b', r'\briuscivo\b']

    # Lists of first person and other person verbs
    first_person_verbs = present_verbs
    other_person_verbs = [r'\bè\b', r'\bha\b', r'\bmangia\b', r'\bvomita\b', r'\bsente\b', r'\bpensa\b', r'\bsoffre\b', r'\bviveva\b']

    # Lists of pronouns
    first_person_pronouns = [r'\bio\b', r'\bme\b', r'\bmio\b', r'\bmia\b', r'\bmiei\b', r'\bmie\b', r'\w+mi\b', r'\bmi\b']
    other_pronouns = [r'\blui\b', r'\blei\b', r'\bsuo\b', r'\bsua\b', r'\bsuoi\b', r'\bsue\b']

    # Count occurrences of present and past verbs
    present_count = count_occurrences(present_verbs, text)
    past_count = count_occurrences(past_verbs, text)

    # Count occurrences of first person and other pronouns
    first_person_count = count_occurrences(first_person_pronouns, text)
    other_pronouns_count = count_occurrences(other_pronouns, text)

    # Count occurrences of first person and other person verbs
    first_person_verbs_count = count_occurrences(first_person_verbs, text)
    other_person_verbs_count = count_occurrences(other_person_verbs, text)


    # Exclude comments with unwanted keywords
    unwanted_keywords = [r'\bbulimia\b', r'\bbinge\b', r'\bbingeeating\b', r'\bbulimica\b', r'\bbulimico\b', r'\babbuffata\b', r'\babbuffate\b', r'\bvomito\b', r'\bvomitare\b', r'\bmi sto abbuffando\b', r'\non ho mai sofferto di\b(?:\W+\w+){0,5}\W*\banoressia\b']
    if any(re.search(keyword, text, re.IGNORECASE) for keyword in unwanted_keywords):
      return False

    # Exclude comments with specific exclusion pattern
    if contains_exclusion_pattern(text, "mi sono fidanzato", "lei", "anoressia", 15, 7):
        return False

     # Exclude comments with specific exclusion pattern1
    if contains_exclusion_pattern1(text, "mia madre", "anoressia", 12):
        return False
    if contains_exclusion_pattern1(text, "mia madre", "alimentari", 12):
        return False
    if contains_exclusion_pattern1(text, "mia sorella", "anoressia", 12):
        return False
    if contains_exclusion_pattern1(text, "mia sorella", "alimentari", 12):
        return False
    if contains_exclusion_pattern1(text, "mia ragazza", "anoressia", 12):
        return False
    if contains_exclusion_pattern1(text, "mia ragazza", "alimentari", 12):
        return False
    if contains_exclusion_pattern1(text, "mia figlia", "anoressia", 12):
        return False
    if contains_exclusion_pattern1(text, "mia figlia", "alimentari", 12):
        return False

    # Check for past references
    past_references = contains_past_reference(text)
    present_references = contains_present_reference(text)

    if contains_past_reference(text) and not contains_present_reference(text):
        return False

  
    if present_count > past_count and first_person_count > other_pronouns_count and first_person_verbs_count > other_person_verbs_count:
       # Check context around the keywords "bulimia" and "malattia"
      keywords = ["anoressia", "malattia", 'disturbo alimentare', "disturbo dell'alimentazione"]
      contexts = extract_contexts(text, keywords)
      for context in contexts:
        context_first_person_verbs_count = count_occurrences(first_person_verbs, context)
        context_other_person_verbs_count = count_occurrences(other_person_verbs, context)


        if context_first_person_verbs_count > context_other_person_verbs_count:
          return True
        else:
          return False


    else:
     return False
    
# set up Chrome options for headless browsing
chrome_options = Options()
chrome_options.add_argument("--headless")  

# set up the Chrome driver service
chrome_service = Service(r'C:\Users\...\chromedriver-win64\chromedriver.exe')  

# initialize the Chrome driver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# set the starting page and maximum number of pages to scrape
current_page = 276
max_pages = 300  # Numero massimo di pagine
save_path = r'C:\Users\...\corpus_AN.txt'
#output_file = "corpus_AN.txt"

# open the output file for writing
with open(save_path, "w", encoding="utf-8") as file:
    while current_page <= max_pages:
        # Costruisci l'URL per la pagina corrente
        URL = f"https://forum.alfemminile.com/search?domain=places&query=anoressia&page={current_page}&scope=site&source=community"
        driver.get(URL)

        # increase the implicit wait time to allow for page loading
        wait = WebDriverWait(driver, 20)

        # wait for the comments section to load
        try:
            comments_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.pageBoxNoCompat.css-10oxk8k")))
            
            # Find all comment elements within the comments section
            comments = comments_section.find_elements(By.CSS_SELECTOR, "div.css-1gk1rmt-ListItem-styles-item")

            if comments:
                print(f"Pagina {current_page}")
                # Iterate through each comment and extract the title and content
                for comment in comments:
                    try:
                        title = comment.find_element(By.CSS_SELECTOR, "div.css-k17v61-ListItem-styles-titleContainer").text
                        link = comment.find_element(By.CSS_SELECTOR, "a.css-9iw9as-ListItem-styles-titleLink")
                        href = link.get_attribute("href")
                        
                        # open link in a new tab
                        driver.execute_script("window.open(arguments[0]);", href)
                        driver.switch_to.window(driver.window_handles[1])
                        
                        # check if the "First Page" button exists and click it if it does
                        try:
                            first_page_button = driver.find_element(By.CSS_SELECTOR, "a.Pager-p.p-1.FirstPage")
                            first_page_button.click()
                            
                            # wait for the page to load after clicking the button
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.Message.userContent")))
                        except Exception:
                            pass  # If the button doesn't exist, continue without clicking it

                        # wait for the full comment to load and extract its text
                        full_comment = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.Message.userContent"))).text

                        if ("anoressia" in full_comment.lower() or "anoressia" in title.lower()) and is_relevant_comment(full_comment):
                            # Write the title and full comment to the output file
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
            time.sleep(15)  # wait for the next page to load

        except Exception as e:
            print(f"Si è verificato un errore durante l'attesa della sezione dei commenti: {e}")
            break
    
# close the Chrome driver
driver.quit()
