import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics import classification_report, accuracy_score
DATASET_PATH = "C:\infosys\Bitext_Sample_Customer_Service_Training_Dataset\Training\Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
df = pd.read_excel(DATASET_PATH)
df = df[['utterance', 'intent']]
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['clean_utterance'] = df['utterance'].apply(clean_text)
model = joblib.load("nlu_intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
st.set_page_config(page_title="BotTrainer NLU", layout="centered")
st.title("🤖 BotTrainer – NLU Model Trainer & Evaluator")
st.write("This application predicts chatbot intents and evaluates model performance.")
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
