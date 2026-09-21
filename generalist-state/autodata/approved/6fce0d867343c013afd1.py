import requests
from bs4 import BeautifulSoup
import re

def count_occurrences(patterns, text):
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def contains_past_reference(text):
    past_phrases = [
        r'\bstavo cadendo in\b(?:\W+\w+){0,5}\W*\bbulimia\b',
        r'\bsono caduta in\b(?:\W+\w+){0,5}\W*\bbulimia\b',
        r'\bsono caduto in\b(?:\W+\w+){0,5}\W*\bbulimia\b',
        r'\bsono stata\b(?:\W+\w+){0,8}\W*\bbulimica\b',
        r'\bho sofferto\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bho avuto\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bsono stato\b(?:\W+\w+){0,8}\W*\bbulimico\b',
        r'\bsofferto di\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bbambina\b(?:\W+\w+){0,8}\W*\bbulimica\b',
        r'\bbambina\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bbambino\b(?:\W+\w+){0,8}\W*\bbulimico\b',
        r'\badolescente\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\badolescente\b(?:\W+\w+){0,8}\W*\bbulimica\b',
        r'\badolescente\b(?:\W+\w+){0,8}\bbulimica\b',
        r'\badolescente\b(?:\W+\w+){0,8}\bbulimia\b',
        r'\bpassato\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bpassato\b(?:\W+\w+){0,8}\W*\bbulimica\b',
        r'\bpassato\b(?:\W+\w+){0,6}\W*\bbulimico\b',
        r'\bdai \d+ ai \d+ anni\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\banno di\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\banno in\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bnon sono mai caduta\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\blontana\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\blontano\b(?:\W+\w+){0,8}\W*\bbulimia\b',
        r'\bho avuto problemi\b(?:\W+\w+){0,11}\W*\bbulimia\b'
    ]
    return any(re.search(phrase, text, re.IGNORECASE) for phrase in past_phrases)

def contains_present_reference(text):
    present_phrases = [
        r"\btutt'ora\b(?:\W+\w+){0,10}\W*\bbulimia\b",
        r"\bbulimia\b(?:\W+\w+){0,9}\W*\btutt'ora\b",
        r"\bancora\b(?:\W+\w+){0,10}\W*\bbulimia\b",
        r"\bbulimia\b(?:\W+\w+){0,9}\W*\bancora\b",
        r"\btutt'oggi\b(?:\W+\w+){0,10}\W*\bbulimia\b",
        r"\bbulimia\b(?:\W+\w+){0,9}\W*\btutt'oggi\b",
        r"\bho iniziato\b(?:\W+\w+){0,5}\W*\bsempre più\babbuffata\b",
        r"\bho iniziato\b(?:\W+\w+){0,5}\W*\bsempre più\babbuffate\b"
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
    present_verbs = [r'\bsono\b', r'\bfaccio\b', r'\bassumo\b', r'\bsto\b', r'\bho\b', r'\bmangio\b', r'\bvomito\b', r'\bsento\b', r'\bpenso\b', r'\bsoffro\b', r'\bvoglio\b', r'\bcerco\b', r'\bso\b', r'\bposso\b', r'\bdevo\b', r'\briesco\b', r'\bvorrei\b', r'\bpotrei\b',r'\bdovrei\b', r'\briuscirei\b', r'\bavrei\b', r'\bsarei\b']
    past_verbs = [r'\bstavo\b', r'\bfacevo\b', r'\bassumevo\b', r'\bavevo\b', r'\bmangiavo\b', r'\bvomitavo\b', r'\bsentivo\b', r'\bpensavo\b', r'\bsoffrivo\b', r'\bvolevo\b', r'\bcercavo\b', r'\bsapevo\b', r'\bpotevo\b', r'\bdovevo\b', r'\briuscivo\b']

    # Lists of first person and other person verbs
    first_person_verbs = present_verbs
    other_person_verbs = [r'\bè\b', r'\bha\b', r'\bmangia\b', r'\bvomita\b', r'\bsente\b', r'\bpensa\b', r'\bsoffre\b', r'\bvogliono\b']

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
    unwanted_keywords = [r'\bnon ho mai sofferto di\b(?:\W+\w+){0,2}\W*\bbulimia\b', r'\bmia moglie\b', r'\bnon è mai sfociata in\b(?:\W+\w+){0,2}\W*\bbulimia\b', r'\bmi ha fatto guarire\b(?:\W+\w+){0,2}\W*\bbulimia\b', r'\bsono guarita\b(?:\W+\w+){0,2}\W*\bbulimia\b']
    if any(re.search(keyword, text, re.IGNORECASE) for keyword in unwanted_keywords):
        return False

    # Check for past references
    if contains_past_reference(text) and not contains_present_reference(text):
        return False

    # Verify general conditions
    if present_count > past_count and first_person_count > other_pronouns_count and first_person_verbs_count > other_person_verbs_count and len(text.split()) > 135:
      return True
    elif present_count > past_count and first_person_count > other_pronouns_count and first_person_verbs_count > other_person_verbs_count and len(text.split()) <= 135:
            # Check context around the keywords "bulimia" and "malattia"
            keywords = ["bulimia", "malattia", 'abbuffarmi', "abbuffata", "abbuffate"]
            contexts = extract_contexts(text, keywords)
            for context in contexts:
                context_first_person_verbs_count = count_occurrences(first_person_verbs, context)
                context_other_person_verbs_count = count_occurrences(other_person_verbs, context)


                if context_first_person_verbs_count > context_other_person_verbs_count:
                  return True

    return False

current_page = 1
proceed = True
output_file = "commenti_bulimia_psicologiitalia.txt"

with open(output_file, "w", encoding="utf-8") as file:
    while proceed:
        URL = f"https://www.psicologi-italia.it/ricerca-generica.html?t=domande&q=bulimia&p={current_page}"
        page = requests.get(URL)
        soup = BeautifulSoup(page.text, "html.parser")

        if current_page == 6:
            proceed = False
        else:
            page_questions = soup.find_all("div", class_="col-12 mb-4")

            for question in page_questions:
                url_link = question.find("a").attrs["href"]
                page_link = requests.get(url_link)
                soup_link = BeautifulSoup(page_link.text, "html.parser")

                title = question.find("h3", class_="title").text.strip()
                text = soup_link.find("div", class_="main-cnt").text.strip()

                if is_relevant_comment(text):
                    #file.write(title + "\n")
                    #file.write(text + "\n\n")  # Aggiungi una riga bianca tra i commenti

                    print(title)
                    print(text)
                    print("\n\n")

        current_page += 1

