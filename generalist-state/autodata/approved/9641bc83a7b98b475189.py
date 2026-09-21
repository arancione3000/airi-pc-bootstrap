import spacy
import textdescriptives as td
import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy.stats import hypergeom


# Spacy model for Italian
nlp = spacy.load("it_core_news_lg")

# add textdescriptives to the pipeline
nlp.add_pipe("textdescriptives/descriptive_stats")

#Upload input file
file_path = '/Users/.../corpus_BN.xlsx' 
df = pd.read_excel(file_path)

# Extract metrics (assuming 'Titolo' and 'Testo' are the column names)
def analyze_text(title, text):
    doc = nlp(text)  
    metrics = doc._.descriptive_stats  
    metrics['Titolo'] = title  # add title to the text
    return metrics


#  Iterate through the dataframe and analyze each text
results = []
for index, row in df.iterrows():
    title = row['Titolo']  
    text = row['Testo']    
    result = analyze_text(title, text)
    results.append(result)

# create dataframe with results
results_df = pd.DataFrame(results)

#  Visualize 
print(results_df.head())  


# ----------------------------
# HD-D FUNCTION
# ----------------------------
def hdd(text, sample_size=42):
    """
    Compute HD-D (Hypergeometric Distribution D)
    Default sample size = 42 (standard in literature)
    """

    # Basic tokenization (whitespace + lowercase)
    tokens = text.lower().split()
    N = len(tokens)

    # If text is too short, return NaN
    if N == 0:
        return np.nan

    # If text shorter than sample size, adjust sample size
    sample = min(sample_size, N)

    freqs = Counter(tokens)

    hdd_value = 0.0

    for freq in freqs.values():
        # Probability the type appears at least once in sample
        prob = 1 - hypergeom.pmf(0, N, freq, sample)
        hdd_value += prob / sample

    return hdd_value


# ----------------------------
# PROCESS ONE CORPUS
# ----------------------------
def process_corpus(file_path, corpus_label):
    df = pd.read_excel(file_path)

    # Merge Titolo + Testo
    df["merged_text"] = (
        df["Titolo"].fillna("").astype(str) + " " +
        df["Testo"].fillna("").astype(str)
    )

    # Compute token length
    df["tokens"] = df["merged_text"].apply(lambda x: len(x.split()))

    # Compute HD-D
    df["HD-D"] = df["merged_text"].apply(hdd)

    # Add corpus label
    df["corpus"] = corpus_label

    return df[["corpus", "Titolo", "Testo", "tokens", "HD-D"]]


# ----------------------------
# LOAD AND PROCESS ALL CORPORA
# ----------------------------
corpus_anoressia = process_corpus("/Users/.../corpus_anoressia.xlsx", "anoressia")
corpus_bulimia = process_corpus("/Users/.../corpus_bulimia.xlsx", "bulimia")
corpus_controllo = process_corpus("/Users/.../corpus_controllo.xlsx", "controllo")

# Merge into one dataframe
final_df = pd.concat(
    [corpus_anoressia, corpus_bulimia, corpus_controllo],
    ignore_index=True
)




