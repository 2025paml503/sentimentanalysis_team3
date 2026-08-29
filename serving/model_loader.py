import json
from pathlib import Path

# New canonical artifact is a serialized sklearn Pipeline that contains
# preprocessing (tfidf) and classifier steps. For backward compatibility,
# fall back to the older separate model + vectorizer files if pipeline isn't present.
PIPELINE_PATH = Path('model_store/sentiment_pipeline.pkl')
MODEL_PATH = Path('model_store/sentiment_model.pkl')
VECTORIZER_PATH = Path('model_store/tfidf_vectorizer.pkl')
# Meta might be saved as meta.json or model_meta.json depending on the script/version
META_PATHS = [Path('model_store/meta.json'), Path('model_store/model_meta.json')]


def load_artifacts():
    """Load serving artifacts.

    Returns a tuple: (model_classifier, vectorizer, meta)
    - model_classifier: an object with predict / predict_proba (the fitted classifier)
    - vectorizer: the fitted Vectorizer (has vocabulary_ and transform)
    - meta: dictionary with model metadata

    If a pipeline file exists, extract the named steps 'tfidf' and 'clf' and
    return them to preserve the existing inference code path.
    """
    import joblib  # imported here so the module is cheap to import in tests

    # load meta
    meta = None
    for p in META_PATHS:
        if p.exists():
            with open(p) as f:
                meta = json.load(f)
            break

    if meta is None:
        raise FileNotFoundError(f"None of the expected meta files were found: {META_PATHS}")

    # If pipeline exists, load and decompose
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        # expect standard step names from training pipeline: 'tfidf' and 'clf'
        try:
            vectorizer = pipeline.named_steps['tfidf']
            model = pipeline.named_steps['clf']
        except Exception as e:
            raise RuntimeError(f"Loaded pipeline but could not find expected steps: {e}")
        return model, vectorizer, meta

    # Fallback: expect older separate artifacts
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [str(p) for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
        raise FileNotFoundError(f"Missing model artifacts: {missing}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer, meta
