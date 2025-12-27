import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ----------------------------------------
st.set_page_config(page_title="BotTrainer NLU", layout="centered")

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Bitext_Sample_Customer_Service_Training_Dataset.xlsx")
    df = df[['utterance', 'intent']]
    return df

df = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_utterance"] = df["utterance"].apply(clean_text)

# ----------------------------------------
# TRAIN MODEL (NO PICKLE)
# ----------------------------------------
@st.cache_resource
def train_model(texts, labels):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return model, vectorizer

model, vectorizer = train_model(
    df["clean_utterance"],
    df["intent"]
)

# ----------------------------------------
# UI
# ----------------------------------------
st.title("🤖 BotTrainer – NLU Model Trainer & Evaluator")
st.write("Cloud-safe NLU intent prediction system")

# ----------------------------------------
# PREDICTION
# ----------------------------------------
st.subheader("🔹 Intent Prediction")

user_input = st.text_area(
    "Enter a customer message:",
    placeholder="e.g., I want to cancel my order"
)

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        st.success(f"✅ Predicted Intent: **{pred[0]}**")

# ----------------------------------------
# EVALUATION
# ----------------------------------------
st.subheader("🔹 Model Evaluation")

if st.button("Run Evaluation"):
    X_vec = vectorizer.transform(df["clean_utterance"])
    y_pred = model.predict(X_vec)

    acc = accuracy_score(df["intent"], y_pred)
    report = classification_report(df["intent"], y_pred)

    st.write(f"### 📊 Accuracy: `{acc:.4f}`")
    st.text(report)

    st.success("Evaluation completed successfully!")
