import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "C:\infosys\Bitext_Sample_Customer_Service_Training_Dataset\Training\Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
df = pd.read_excel(file_path)
print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample Data:\n", df.head())
df = df[['utterance', 'intent']]
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_utterance'] = df['utterance'].apply(clean_text)
print("\nMissing Values:\n", df.isnull().sum())
plt.figure(figsize=(12,6))
sns.countplot(
    y=df['intent'],
    order=df['intent'].value_counts().index
)
plt.title("Intent Distribution")
plt.xlabel("Count")
plt.ylabel("Intent")
plt.show()
df[['clean_utterance', 'intent']].to_csv(
    "cleaned_chatbot_data.csv",
    index=False
)

print("✅ Cleaned dataset saved successfully.")
