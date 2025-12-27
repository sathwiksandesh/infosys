import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# LOAD DATASET
# -------------------------------
DATASET_PATH = "Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
df = pd.read_excel(DATASET_PATH)
df = df[['utterance', 'intent']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_utterance'] = df['utterance'].apply(clean_text)

X = df['clean_utterance']
y = df['intent']

# -------------------------------
# TRAIN MODEL
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# -------------------------------
# SAVE MODEL & VECTORIZER
# -------------------------------
with open("nlu_intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")



