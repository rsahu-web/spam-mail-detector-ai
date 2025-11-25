import pandas as pd
import joblib
from flask import Flask, render_template, request
from pathlib import Path
import re
import numpy as np

# Add these imports and logging setup near the top (after other imports)
import csv
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.csv"
if not LOG_FILE.exists():
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text", "label", "prob"])

app = Flask(__name__)

# ==== Your original feature extractor classes stay unchanged ====

SUSPICIOUS_WORDS = {
    'verify','account','password','click','login','urgent','suspend','update',
    'bank','confirm','ssn','social','security','payment','invoice','limited',
    'fraud','identity','authenticate','billing','wire','transfer','prize','winner'
}

url_regex = re.compile(r'https?://\S+|www\.\S+|\bbit\.ly/\S+|\bt\.co/\S+')

class SimpleMetaFeatures:
    """Must match EXACT class used during training"""
    def fit(self, X, y=None):
        return self

    def _extract(self, text):
        if not isinstance(text, str):
            text = str(text)
        urls = url_regex.findall(text)
        num_urls = len(urls)
        num_digits = sum(c.isdigit() for c in text)
        text_len = len(text)
        num_exclaim = text.count('!')
        words = re.findall(r'\w+', text.lower())
        suspicious = int(any(w in SUSPICIOUS_WORDS for w in words))
        shortlink = int(bool(re.search(r'\b(bit\.ly|tinyurl|t\.co)\b', text.lower())))
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        return [num_urls, num_digits, text_len, num_exclaim, suspicious, shortlink, uppercase_ratio]

    def transform(self, X):
        arr_texts = pd.Series(X).to_numpy().astype(str)
        out = [self._extract(t) for t in arr_texts]
        return np.array(out)

def make_meta(X_df):
    texts = X_df['text'].astype(str).to_numpy()
    meta = SimpleMetaFeatures().transform(texts)
    return meta

# ==== Pipeline loading logic unchanged ====
PIPELINE_PATH = "pipeline.joblib"

pipeline = None
if Path(PIPELINE_PATH).exists() and Path(PIPELINE_PATH).stat().st_size > 0:
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print("Loaded improved model:", PIPELINE_PATH)
    except Exception as e:
        print("Error loading pipeline.joblib:", e)
        pipeline = None
else:
    print("pipeline.joblib missing or empty. Run: python train_improved.py")

# ==== Routes ==== 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # If model not loaded, show a simple message
    if pipeline is None:
        return render_template(
            "after.html",
            data="MODEL NOT READY",
            prob="N/A",
            prob_percent=0,
            suspicious=[],
            urls=[],
            original_text=""
        )

    # Get text from form (limited to 5000 chars for safety)
    text = request.form.get('email', '')[:5000]

    # If user submitted empty text, just treat as legit
    if not text.strip():
        return render_template(
            "after.html",
            data="LEGIT",
            prob="0.00%",
            prob_percent=0,
            suspicious=[],
            urls=[],
            original_text=text
        )

    # --- Model prediction ---
    import pandas as pd, re
    X = pd.DataFrame({'text': [text]})
    proba = float(pipeline.predict_proba(X)[0][1])   # probability of phishing
    prob_percent = int(round(proba * 100))
    label = "PHISH" if proba > 0.5 else "LEGIT"

    # --- Simple explainability: URLs + suspicious words ---
    url_regex = re.compile(r'https?://\S+|www\.\S+|\bbit\.ly/\S+|\bt\.co/\S+')
    urls = url_regex.findall(text)

    word_tokens = re.findall(r'\w+', text.lower())
    SUSPICIOUS_UI = {
        'verify','account','password','click','login','urgent','suspend','update',
        'bank','confirm','invoice','payment','prize','winner','limited','wire','transfer'
    }
    suspicious_tokens = sorted({w for w in word_tokens if w in SUSPICIOUS_UI})

    # Log the prediction
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), text.replace('\n','\\n')[:3000], label, f"{proba:.4f}"])

    # Render result page
    return render_template(
        "after.html",
        data=label,
        prob=f"{proba*100:.2f}%",
        prob_percent=prob_percent,
        suspicious=suspicious_tokens,
        urls=urls,
        original_text=text
    )

# ==== Run server ====
if __name__ == '__main__':
    app.run(debug=True)
