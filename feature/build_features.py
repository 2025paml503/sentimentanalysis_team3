"""
# feature/build_features.py
=====================================
Week 1 (Module M2) -- The Feature Engineering Pipeline

Both training script and the serving API import clean_text() from here.
This is the whole point: if training lowercases the text but the API does not,
the model receives a vector it was never trained on and returns a confidently
wrong answer with no error. That is training-serving skew, and sharing this one
function prevents it.
Reviews Sentiments analyzer – Centralised Feature Engineering Pipeline
This is the SINGLE source of truth for all feature transformations.
Training and serving both use this — no duplication, no skew.


What runs where:

clean_text() -> shared by training AND serving(this file)
TF-IDF vectorizer -> FIT in training (on the train split only, to
                     leakage), then saved as n artifact and LOADED by
                     serving, So the fitted vocabulary is also shared
                     --no skew.

Outputs the offline feature store: feature_store/feature_store.db

Run:
    python feature/build_features.py
"""

 
import pandas as pd
import sqlite3
import os
import re
from pathlib import Path

# Get the project root directory (parent of this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH    = PROJECT_ROOT / 'data' / 'raw' / 'Amazon_Reviews_3500records.csv'
FEATURE_STORE_DB = PROJECT_ROOT / 'feature_store' / 'feature_store.db'
TABLE_NAME       = 'review_features'

# -- Shared text cleaning (used by training and serving) ---
# A "pure" function: same input always gives the same output
# which makes features reproducible

HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTER = re.compile(r"[^a-z\s]")
EXTRA_SPACE = re.compile(r"\s+")

def clean_text(text):
    """
    Normalize one review string

    Steps, in the exact order ( order matters -- changing
    it changes every feature in the store):
    1. handle missing / non-string input
    2. lower case
    3. string Html tag
    4. drop everything that is not a letter or a space
    5. collapse repeated whitespace and trim
    """

    if text is None or not isinstance(text, str):
        return ""

    text = text.lower()
    text = HTML_TAG.sub(" ", text)
    text = NON_LETTER.sub(" ", text)
    text = EXTRA_SPACE.sub(" ", text).strip()
    return text

def combine_title_and_text(title, review_text):
    """
    The title carries a lot of sentiment, so we model title + body
    as one document rather than throwing the title away.
    """
    title = title if isinstance(title, str) else ""
    review_text = review_text if isinstance(review_text, str) else ""
    return (title + " " + review_text).strip()

def prepare_document(title, review_text):
    """ The full text transform for one review. Training and serving both call this"""
    return clean_text(combine_title_and_text(title, review_text))


# --- Simple numeric text features ----------------------
# These are not fed to the model(the model uses TF-IDF). We compute and
# store them becuase they are the cheapest, most interpretable drift
#signals avaialable in
def count_words(clean_doc):
    return len(clean_doc.split())

def count_chars(clean_doc):
    return len(clean_doc)


# ─── Pipeline Steps ─────────────────────────

def load_raw(path):
    return pd.read_csv(path, keep_default_na=False)
 
def build_feature_set(df):
    """Compose all transformations in order."""
    out = pd.DataFrame()
    out["review_id"] = df["review_id"]

    # the shared text transform, applied row by row
    out["clean_doc"] = [
        prepare_document(title, review_text)
        for title, review_text in zip(df["title"], df["review_text"])
    ]

    out["n_words"] = [count_words(d) for d in out["clean_doc"]]
    out["n_chars"] = [count_chars(d) for d in out["clean_doc"]]
    out["sentiment"] = df["sentiment"]

    # A review that cleans down to nothing (e.g. "!!!!" or "12345" cannot
    # be vectorized into anything meaningful, so we drop it here rather
    # than let it become a silent all-zero row in the training matrix
    before = len(out)
    out = out[out["n_words"] > 0].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} reviews(s) that cleaned to empty text")

    return out
  
  
# ─── Persist to Feature Store ────────────────────────────────
def save_to_feature_store(df):
    os.makedirs(PROJECT_ROOT / 'feature_store', exist_ok=True)
    conn = sqlite3.connect(FEATURE_STORE_DB)
    # if_exists='replace': idempotent – safe to re-run
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print(f'✅  Feature store updated: {len(df)} rows, {len(df.columns)} columns')
    print(f'   Columns: {list(df.columns)}')

def load_from_feature_store():
    conn = sqlite3.connect(FEATURE_STORE_DB)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df
  
  
if __name__ == '__main__':
    print('Loading raw data...')
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at: {RAW_DATA_PATH}")
    
    raw_df = load_raw(RAW_DATA_PATH)
    print(f'✅  Loaded {len(raw_df)} records from {RAW_DATA_PATH.name}')
 
    print('Building feature set...')
    features_df = build_feature_set(raw_df)
    print(f'✅  Built feature set with {len(features_df)} records')
 
    print('Persisting to feature store...')
    save_to_feature_store(features_df)
    print(f'✅  Feature store persisted to {FEATURE_STORE_DB}')
