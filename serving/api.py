"""
@File: api.py
@Description:

Endpoints:
    GET /health liveness + which model version is loaded
    GET /model-info full lineage of the deployment artifact
    POST /predict one review -> sentiment
    POST /predict-batch up to 100 reviews in one call

Run from project root(not from inside serving/):
    uvicorn serving.api:app --port 8000
    open http://127.0.0.1:8000/docs

"""

import time
from typing import Optional
from unittest import result

from fastapi import FastAPI, Header, HTTPException
from scripts.regsetup import description

# This import is what prevents training-serving skew
from feature.build_features import prepare_document
from monitoring.prediction_logger import init_db, log_predictions
from serving.inference_schema import (BatchSentimentRequest,
                                      BatchSentimentResponse,
                                      SentimentRequest,
                                      SentimentResponse)
from serving.model_loader import load_artifacts

app = FastAPI(
    title="Amazon review sentiments service",
    description="Amazon review sentiments service",
    version="1.0",
)

model, vectorizer, meta = load_artifacts()
MODEL_VERSION = meta["model_version"]
VOCABULARY = vectorizer.vocabulary_

init_db()

def compute_oov_rate(clean_doc):
    """
    Share the words in this review that are not in the model's vocabulary

    This is single most useful drift signal for a text model. When new
    slang or a new product category appears in production traffic. oov_rate
    climbs long before anyone notices the predictions are wrong.
    """
    words = clean_doc.split()
    if not words:
        return 0.0
    unknown = sum(1 for w in words if w not in VOCABULARY)
    return unknown / len(words)

def predict_one(review_text, title):
    """
    Care inference path. Returns a SentimentResponse.

    Steps, in the same order as training:
        1. clean the text with the SHARED transform
        2. vectorize with the FITTED vocabulary ( transform, never fit)
        3. predict a probability
        4. derive the label, confidence and oov_rate
    """
    start = time.perf_counter()

    # 1. shared cleaning
    clean_doc = prepare_document(title, review_text)

    # Edge case: text that is legal JSON and non-blank, but cleans down to
    # nothing -- "12345", "!!!???", an emoji-only review. There is no
    # honest prediction to make here, so refuse instead of guessing
    if not clean_doc:
        raise HTTPException(
            status_code=422,
            detail="review_text contains no usable words after cleaning",
        )

    # 2. vectorize using the vocabulary fitted during training
    x = vectorizer.transform([clean_doc])

    # 3. predict
    prob = float(model.predict_proba(x)[0][1])

    # 4. derive the response fields
    sentiment = "positive" if prob > 0.5 else "negative"
    confidence = abs(prob - 0.5)*2
    oov_rate = compute_oov_rate(clean_doc)
    latency_ms = (time.perf_counter() - start) * 1000

    return SentimentResponse(
        sentiment=sentiment,
        positive_probability=round(prob, 4),
        confidence=round(confidence, 4),
        oov_rate=round(oov_rate, 4),
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 2),
    ), clean_doc

# ---- Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "vocab_size": len(VOCABULARY),
    }

# --- Model lineage
@app.get("/model-info")
def model_info():
    return meta

@app.post("/predict", response_model=SentimentResponse)
def predict(
        data: SentimentRequest,
        x_traffic_batch: Optional[str] = Header(
            default=None,
            description="Optional operational tag",
        ),
):
    response, clean_doc = predict_one(data.review_text, data.title)

    try:
        log_predictions(
            review_text=data.review_text,
            title=data.title,
            clean_doc=clean_doc,
            result=response.model_dump(),
            traffic_batch=x_traffic_batch,
        )
    except Exception as e:
        print(f"WARNING: prediction logging failed: {e}")

    return response

@app.post("/predict-batch", response_model=BatchSentimentResponse)
def predict_batch(
        data: BatchSentimentRequest,
        x_traffic_batch: Optional[str] = Header(default=None),
):
    """
    Batching amortizes the per-request overhead. Capped at 100 reviews by
    the schema so that one caller cannot block the service
    """
    start = time.perf_counter()
    results =[]

    for review in data.reviews:
        response, clean_doc = predict_one(review.review_text, review.title)
        results.append(response)
        try:
            log_predictions(
                review_text=review.review_text,
                title=review.title,
                clean_doc=clean_doc,
                result=response.model_dump(),
                traffic_batch=x_traffic_batch,
            )
        except Exception as e:
            print(f"WARNING: prediction logging failed: {e}")

    total_ms = (time.perf_counter() - start) * 1000

    return BatchSentimentResponse(
        results=results,
        count=len(results),
        total_latency_ms=round(total_ms, 2),
    )

















