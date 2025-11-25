# train_improved.py (fixed - replace your current file with this)
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------------------------
# 1) Load dataset (tailored to your CSV)
# -------------------------
DATA_FILE = "mail_data.csv"
if not Path(DATA_FILE).exists():
    raise SystemExit(f"Data file {DATA_FILE} not found in project root. Place mail_data.csv next to this script.")

df = pd.read_csv(DATA_FILE, encoding='utf-8', on_bad_lines='skip')
if 'Message' not in df.columns or 'Category' not in df.columns:
    print("Available columns:", df.columns.tolist())
    raise SystemExit("Expected columns 'Message' and 'Category' not found.")

df = df[['Message', 'Category']].rename(columns={'Message': 'text', 'Category': 'label'})

def map_label(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if 'spam' in x:
            return 1
        if 'ham' in x or 'legit' in x or 'normal' in x:
            return 0
    try:
        return int(x)
    except:
        return 0

df['label'] = df['label'].apply(map_label).astype(int)
df = df.dropna(subset=['text']).reset_index(drop=True)
print("Loaded dataset:", df.shape)

# -------------------------
# 2) Feature engineering
# -------------------------
SUSPICIOUS_WORDS = {
    'verify','account','password','click','login','urgent','suspend','update',
    'bank','confirm','ssn','social','security','payment','invoice','limited',
    'fraud','identity','authenticate','billing','wire','transfer','prize','winner'
}

url_regex = re.compile(r'https?://\S+|www\.\S+|\bbit\.ly/\S+|\bt\.co/\S+')

class SimpleMetaFeatures(BaseEstimator, TransformerMixin):
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
        # X may be an array-like or pandas Series. Convert to 1D numpy array safely:
        arr_texts = pd.Series(X).to_numpy().astype(str)
        out = [self._extract(t) for t in arr_texts]
        return np.array(out)

# Named top-level function to create meta features (pickleable)
def make_meta(X_df):
    # expects DataFrame-like with column 'text'
    texts = X_df['text'].astype(str).to_numpy()
    meta = SimpleMetaFeatures().transform(texts)
    return meta

# -------------------------
# 3) Build pipeline
# -------------------------
tfidf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_df=0.9, min_df=2, max_features=15000)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(transformers=[
    ('tfidf', tfidf, 'text'),
    # use the named function here (no lambda)
    ('meta', FunctionTransformer(make_meta, validate=False), ['text'])
], sparse_threshold=0)

pipeline = Pipeline([
    ('pre', preprocessor),
    # Scale numeric columns; keep with_mean=False to handle sparse matrix shapes safely
    ('scale', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000))
])

# -------------------------
# 4) Train / evaluate
# -------------------------
X = df[['text']]
y = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training samples:", X_train.shape[0], " Test samples:", X_test.shape[0])

param_grid = {'clf__C': [0.1, 1, 5]}
search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

best = search.best_estimator_
print("Best params:", search.best_params_)

y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else None

print("Classification report:")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=4))
if y_proba is not None:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 5) Save pipeline
# -------------------------
OUT = "pipeline.joblib"
joblib.dump(best, OUT)
print("Saved pipeline to", OUT)
print("You can now run 'python app.py' and the web app will use this pipeline (pipeline.joblib).")
