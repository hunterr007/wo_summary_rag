import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

# --- Configs ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "data/workorders.csv"
TARGET_ASSET = "HVAC-321"
TOP_K = 10
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") 
API_URL = os.getenv("GEMINI_API_URL")

# --- Load Work Order Data ---
df = pd.read_csv(CSV_PATH)
df['text'] = df.apply(lambda x: f"WO:{x['wonum']} | Desc:{x['description']} | Details:{x['longdescription']} | Failure:{x['failurecode']} | Hours:{x['laborhrs']}", axis=1)

# --- Embed Work Orders ---
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# --- Create FAISS Index ---
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# --- Retrieve Top-K Relevant WOs ---
target_df = df[df['assetnum'] == TARGET_ASSET].sort_values("wonum")
if target_df.empty:
    raise ValueError(f"No WOs found for asset: {TARGET_ASSET}")

latest_wos = target_df.tail(TOP_K).copy()

# Precompute stats
failure_counts = latest_wos['failurecode'].value_counts().to_dict()
average_hours_by_failure = latest_wos.groupby('failurecode')['laborhrs'].mean().round(2).to_dict()

# Build structured context
table_rows = "\n".join(
    f"{row['wonum']} | {row['failurecode']} | {row['laborhrs']}" for _, row in latest_wos.iterrows()
)

# --- Prompt Construction ---
prompt = f"""
You are a maintenance assistant. Summarize the last {TOP_K} work orders for asset {TARGET_ASSET}.

Work Orders Table:
WO Number | Failure Code | Hours
----------|--------------|------
{table_rows}

Failures:
{failure_counts}

Average Time per Failure:
{average_hours_by_failure}

Summarize any common failure codes, repeated issues, and general time taken. Be precise. Do not hallucinate data not shown above.
"""
#print(prompt)
# --- Gemini API Call ---
params = {"key": API_KEY}
headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {
            "parts": [{"text": prompt}]
        }
    ]
}

response = requests.post(API_URL, params=params, headers=headers, json=data)

# --- Output ---
if response.status_code == 200:
    result = response.json()
    parts = result['candidates'][0]['content']['parts']
    summary = parts[0]['text'] if parts else "[No summary returned]"
    print(f"\n🛠️ Summary for {TARGET_ASSET}:\n\n{summary}")
else:
    print(f"Error: {response.status_code} - {response.text}")
