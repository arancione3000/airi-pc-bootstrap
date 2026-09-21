import pandas as pd
import re
from transformers import pipeline
from transformers import AutoTokenizer
from collections import Counter

# Load file and return a DataFrame
def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df

# clean text
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

def split_text(text, tokenizer, max_length=500):
    tokens = tokenizer(text, truncation=False, padding=False, return_tensors='pt')['input_ids'][0]
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def analyze_emotions(texts, emotion_pipeline):
    all_scores = []
    
    for text in texts:
        result = emotion_pipeline(text)[0]  # list of dicts
        all_scores.append(result)
    
    # Aggregate probabilities
    avg_scores = {}
    for label_dict in all_scores:
        for item in label_dict:
            label = item['label']
            score = item['score']
            avg_scores[label] = avg_scores.get(label, 0) + score
    
    # Compute mean
    for label in avg_scores:
        avg_scores[label] /= len(all_scores)
    
    # Select label with highest mean probability
    final_label = max(avg_scores, key=avg_scores.get)
    
    return final_label


# apply model
def analyze_file(file_path, output_file_path):
    # Caricamento del file Excel
    df = load_excel(file_path)

    # model
    emotion_model_name = "MilaNLProc/feel-it-italian-emotion"
    tokenizer_emotion = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_pipeline = pipeline("text-classification", model=emotion_model_name, tokenizer=tokenizer_emotion, return_all_scores=True)

    emotion_results = []


    for index, row in df.iterrows():
        title = row['Titolo']
        text = row['Testo']
        combined_text = clean_text(f"{title} {text}")

        # divide the text into chunks if it's too long for the model
        text_chunks = split_text(combined_text, tokenizer_emotion)

        # analyze emotions for each chunk 
        emotions = analyze_emotions(text_chunks, emotion_pipeline)

        if len(emotions) > 0:
            emotion = emotions[0]

        emotion_results.append(emotion)


    df['Emotion'] = emotion_results

    # save results
    df.to_excel(output_file_path, index=False)
    
# main
input_file = '/Users/.../input.xlsx'
output_file = '/Users/.../output.xlsx'
analyze_file(input_file, output_file)
