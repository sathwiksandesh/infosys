import streamlit as st
import pandas as pd
import re
import os
import pickle
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------------------
# PAGE CONFIG (FIRST STREAMLIT CALL)
# ----------------------------------------
st.set_page_config(page_title="BotTrainer NLU", layout="centered")

# ----------------------------------------
# FILE PATHS (RELATIVE ONLY)
# ----------------------------------------
DATASET_PATH = "Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
MODEL_PATH = "nlu_intent_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# ----------------------------------------
# CHECK FILES
# ----------------------------------------
for file in [DATASET_PATH, MODEL_PATH, VECTORIZER_PATH]:
    if not os.path.exists(file):
        st.error(f"❌ Required file missing: `{file}`")
        st.stop()

# ----------------------------------------
# LOAD DATA (SAFE)
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATASET_PATH)
    df = df[['utterance', 'intent']]
    return df

df = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_utterance'] = df['utterance'].apply(clean_text)

# ----------------------------------------
# LOAD MODEL (SAFE)
# ----------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------------
# UI
# ----------------------------------------
st.title("🤖 BotTrainer – NLU Model Trainer & Evaluator")
st.write("Predict chatbot intents and evaluate model performance.")

# ----------------------------------------
# INTENT PREDICTION
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
# MODEL EVALUATION
# ----------------------------------------
st.subheader("🔹 Model Evaluation")

if st.button("Run Evaluation"):
    X_vec = vectorizer.transform(df['clean_utterance'])
    y_pred = model.predict(X_vec)

    acc = accuracy_score(df['intent'], y_pred)
    report = classification_report(df['intent'], y_pred)

    st.write(f"### 📊 Accuracy: `{acc:.4f}`")
    st.text(report)

    with open("final_bottrainer_report.txt", "w") as f:
        f.write("BotTrainer – NLU Model Evaluation Report\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    st.success("📄 Final evaluation report generated successfully!")




