# features/build_features.py
# Reviews Sentiments analyzer – Centralised Feature Engineering Pipeline
# This is the SINGLE source of truth for all feature transformations.
# Training and serving both use this — no duplication, no skew.
 
import pandas as pd
import sqlite3
import os
 
RAW_DATA_PATH    = '../data/raw/amazon_reviews.txt'
FEATURE_STORE_DB = '../feature_store/feature_store.db'
TABLE_NAME       = 'review_features'
 
# ─── Feature Engineering Functions ──────────────────────────
# Each function is pure: same input always produces same output.
# This is what makes features reproducible.
 
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
 
def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['customer_id'])
 
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode contract_type and payment_method.
    drop_first=True removes one dummy per group to avoid multicollinearity.
    The resulting column names are deterministic – this is what prevents skew.
    """
    return pd.get_dummies(
        df,
        columns=['contract_type', 'payment_method'],
        drop_first=True
    )
 
def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Compose all transformations in order."""
    df = drop_identifier(df)
    df = encode_categoricals(df)
    return df
 
 
# ─── Persist to Feature Store ────────────────────────────────
def save_to_feature_store(df: pd.DataFrame):
    os.makedirs('feature_store', exist_ok=True)
    conn = sqlite3.connect(FEATURE_STORE_DB)
    # if_exists='replace': idempotent – safe to re-run
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print(f'✅  Feature store updated: {len(df)} rows, {len(df.columns)} columns')
    print(f'   Columns: {list(df.columns)}')
 
 
if __name__ == '__main__':
    print('Loading raw data...')
    raw_df = load_raw(RAW_DATA_PATH)
 
    print('Building feature set...')
    features_df = build_feature_set(raw_df)
 
    print('Persisting to feature store...')
    save_to_feature_store(features_df)

