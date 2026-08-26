import json
from pathlib import Path

MODEL_PATH = Path('model_store/sentiment_model.pkl')
VECTORIZER_PATH = Path('model_store/tfidf_vectorizer.pkl')
META_PATH = Path('model_store/model_meta.pkl')

def load_artifacts():

    import joblib # imported here so the module is cheap to import in tests

    for path in (MODEL_PATH, VECTORIZER_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File not found at {path}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    return model, vectorizer, meta
