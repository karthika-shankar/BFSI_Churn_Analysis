import os
import warnings
import platform

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMBA_NUM_THREADS"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

torch.set_num_threads(6)
device = torch.device("cpu")

try:
    df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\analaysis\Indian_Banking_Churn_Dataset.csv")
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    exit()

try:
    df["Account_Open_Date"] = pd.to_datetime(df["Account_Open_Date"], format="%Y-%m-%d", errors='coerce')
    df["Last_Transaction_Date"] = pd.to_datetime(df["Last_Transaction_Date"], format="%Y-%m-%d", errors='coerce')
except Exception as e:
    print(f"Date conversion error: {e}")

numeric_cols = ["Age", "Balance", "EstimatedSalary", "Monthly_Income", "CreditScore", "EMI_Amount"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

categorical_cols = [
    "Gender", "Geography", "Native_Place", "Marital_Status",
    "Account_Type", "Loan_Status", "Branch_Code", "Customer_Review"
]
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        mode_value = df[col].mode()
        if len(mode_value) > 0:
            df[col] = df[col].fillna(mode_value.iloc[0])

forward_fill_cols = ["Tenure", "HasCrCard", "IsActiveMember"]
existing_cols = [col for col in forward_fill_cols if col in df.columns]
if existing_cols:
    df[existing_cols] = df[existing_cols].ffill()

if "CustomerID" in df.columns:
    df = df[df["CustomerID"].notnull()]

df = df.drop_duplicates()


print("Loading sentiment analysis model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").to(device)
model.eval()
print("Model loaded successfully!")

def classify_sentiment(text):
    try:
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ("Neutral", 0.0)
        text = emoji.demojize(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            labels = ['Negative', 'Neutral', 'Positive']
            sentiment_idx = torch.argmax(scores).item()
            return labels[sentiment_idx], scores[sentiment_idx].item()
    except Exception as e:
        return ("Neutral", 0.0)

if "Customer_Review" in df.columns:
    print("Starting sentiment analysis...")

    sentiments = []
    scores = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {executor.submit(classify_sentiment, review): i for i, review in enumerate(df["Customer_Review"])}
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Sentiment Analysis"):
            i = future_to_idx[future]
            try:
                sentiment, score = future.result()
                sentiments.append((i, sentiment, score))
            except Exception:
                sentiments.append((i, "Neutral", 0.0))

    # Reordering results
    sentiments_sorted = sorted(sentiments, key=lambda x: x[0])
    df["Review_Sentiment"] = [s[1] for s in sentiments_sorted]
    df["Sentiment_Score"] = [s[2] for s in sentiments_sorted]

    print("Sentiment analysis completed!")
    print(df[["Customer_Review", "Review_Sentiment", "Sentiment_Score"]].head())


output_dir = r"C:\Users\priya\OneDrive\Desktop\analysis"
output_file = os.path.join(output_dir, "sentiment_optimized.csv")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(output_file, index=False)
print(f"Saved output CSV to: {output_file}")
