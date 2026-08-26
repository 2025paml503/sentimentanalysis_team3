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
from fastapi import FastAPI, Header, HTTPException

# This import is what prevents training-serving skew
from feature.build_features import prepare_document
from monitoring.prediction_logger import init_db, log_predictions


















