import streamlit as st
import pandas as pd
import re
import os

# ----------------------------------------
# SAFE joblib import (NO CRASH)
# ----------------------------------------
try:
    import joblib
except ModuleNotFoundError:
    st.error("❌ joblib is not installed. Please check requirements.txt")
    st.stop()

from sklearn.metrics import classification_report, accuracy_score

# ----------------------------------------
# PAGE CONFIG (MUST BE FIRST Streamlit cmd)
# ----------------------------------------
st.set_page_config(page_title="BotTrainer NLU", layout="centered")

# ----------------------------------------
# FILE PATHS (RELATIVE – STREAMLIT SAFE)
# ----------------------------------------
DATASET_PATH = "Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
MODEL_PATH = "nlu_intent_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# ----------------------------------------
# FILE EXISTENCE CHECK
# ----------------------------------------
missing_files = []
for file in [DATASET_PATH, MODEL_PATH, VECTORIZER_PATH]:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error("❌ Missing required files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------
df = pd.read_excel(DATASET_PATH)
df = df[['utterance', 'intent']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_utterance'] = df['utterance'].apply(clean_text)

# ----------------------------------------
# LOAD MODEL & VECTORIZER
# ----------------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------------------
# UI
# ----------------------------------------
st.title("🤖 BotTrainer – NLU Model Trainer & Evaluator")
st.write("Predict chatbot intents and evaluate model performance.")

# ----------------------------------------
# INTENT PREDICTION (TASK 5–6)
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
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)
        st.success(f"✅ Predicted Intent: **{prediction[0]}**")

# ----------------------------------------
# MODEL EVALUATION (TASK 7–8)
# ----------------------------------------
st.subheader("🔹 Model Evaluation on Full Dataset")

if st.button("Run Evaluation"):
    X = df['clean_utterance']
    y = df['intent']

    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    st.write(f"### 📊 Accuracy: `{accuracy:.4f}`")
    st.text("Classification Report:")
    st.text(report)

    with open("final_bottrainer_report.txt", "w") as f:
        f.write("BotTrainer – NLU Model Evaluation Report\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    st.success("📄 Final evaluation report generated successfully!")

