"""
training/train_with_mlflow.py
================================

Trains several sentiment models, tracks every one of them in MLflow, then picks a
winner and freezes it into model_store/ as the deployment artifact.

Why MLflow: a .pkl file on its own tell you nothing. Six months from now
"which parameters produced this, and on what data?" has to have an answer.
Every parameter goes into log_params, every metric into log_metrics, every
file worth keeping into log_artifact.

Reproducibility rules followed here:
    * random_state=42 everywhere that randomness exists
    * features read from the immutable features store, never the raw CSV
    * TF-IDF is fitted on the TRAIN SPLIT ONLY ( fitting on all rows would leak
    test vocabulary into training and inflate the scores.)
    * the fitted vectorizer is saved as an artifact, so serving reuses the exact same
    vocabulary

Run:
    python training/train_with_mlflow.py
    mlflow ui # then open http://localhost:5000
"""

import json
import os
import sqlite3
import sys
from unittest import result

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score
                            , recall_score, roc_auc_score)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#from feature.build_features import FEATURE_STORE_DB

sys.path.insert(0, os.getcwd())

EXPERIMENT_NAME = "amazon_review_sentiment"

TRACKING_URI = "sqlite:///mlflow.db"

FEATURE_STORE_DB = "feature_store/feature_store.db"
TABLE_NAME = "review_features"
MODEL_DIR = "../model_store"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ---------- The experiment grid -------------
# Each dict below becomes one tracked MLflow run. Change a value, re-run,
# and MLflow keeps both versions side by side -- no more model_v2_FINAL.pkl
EXPERIMENTS = [
    {
        "run_name": "run1_logreg_baseline",
        "model_type": "logistic_regression",
        "C": 1.0,
        "ngram_max": 1,
        "max_features": 5000,
        "min_df": 2,
    },
    {
        "run_name": "run2_logreg_strong_reg",
        "model_type": "logistic_regression",
        "C": 0.1,
        "ngram_max": 1,
        "max_features": 5000,
        "min_df": 2,
    },
    {
        "run_name": "run3_logreg_baseline",
        "model_type": "logistic_regression",
        "C": 1.0,
        "ngram_max": 2,
        "max_features": 20000,
        "min_df": 2,
    },
    {
        "run_name": "run4_naive_bayes",
        "model_type": "multinomial_nb",
        "alpha": 1.0,
        "ngram_max": 2,
        "max_features": 20000,
        "min_df": 2,
    },
]

def load_features():
    """ Read the versioned feature snapshot. Never the raw CSV"""
    conn = sqlite3.connect(FEATURE_STORE_DB)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

def log_sklearn_model(model, name):
    try:
        mlflow.sklearn.log_model(model, name=name)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name)

def run_one_experiment(config, X_train_text,X_test_text, y_train, y_test):
    """ Train + evaluate + track one configuration. Returns a result dict."""
    print(f"\n---{config['run_name']}---")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config)
        mlflow.log_param("random_state", RANDOM_SEED)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.set_tag("dataset", "amazon_review_sentiment")
        mlflow.set_tag("feature_source", "feature_store/review_features")

        # TF-IDF : fit on TRAIN ONLY
        vectorizer = TfidfVectorizer(
            ngram_range=(1, config["ngram_max"]),
            max_features=config["max_features"],
            min_df=config["min_df"],
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        vocab_size = len(vectorizer.vocabulary_)
        mlflow.log_metric("vocab_size", vocab_size)
        print(f" TF-IDF vocabulary: {vocab_size} terms")

        # --- Train ---
        if config["model_type"] == "logistic_regression":
            model = LogisticRegression(
                C=config["C"],
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
        elif config["model_type"] == "multinomial_nb":
           # print("DEBUG config:", config)
            model = MultinomialNB(alpha=config["alpha"])
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        model.fit(X_train, y_train)

        #-- Evaluate ------------------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print(f" accuracy: {metrics['accuracy']:.4f}")
        print(f" roc_auc : {metrics['roc_auc']:.4f}")
        print(f" f1_score: {metrics['f1_score']:.4f}")

        # ---- Log artifacts ----------
        log_sklearn_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"run_id: {run_id}")

        result = dict(config)
        result.update(metrics)
        result["run_id"] = run_id
        result["vocab_size"] = vocab_size
        result["_model"] = model
        result["_vectorizer"] = vectorizer
        return result

def save_best_artifacts(best):
        """
        Freeze the winning run as the immutable deployment unit.

        Three files, because a text model needs all three to serve  correctly:
            sentiment_model.pkl -- the classifier
            tfidf_vectorizer.pkl -- the vectorizer
            model_meta.json -- the lineage : which run produced this
        """
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(best["_model"], f"{MODEL_DIR}/sentiment_model.pkl")
        joblib.dump(best["_vectorizer"], f"{MODEL_DIR}/tfidf_vectorizer.pkl")

        meta = {
            "run_id": best["run_id"],
            "run_name": best["run_name"],
            "model_type": best["model_type"],
            "experiment_name": EXPERIMENT_NAME,
            "vocab_size": best["vocab_size"],
            "ngram_max": best["ngram_max"],
            "max_features": best["max_features"],
            "metrics": {
                "accuracy": round(best["accuracy"], 3),
                "roc_auc": round(best["roc_auc"], 3),
                "f1_score": round(best["f1_score"], 3),
                "precision": round(best["precision"], 3),
                "recall": round(best["recall"], 3),
            },
            "random_state": RANDOM_SEED,
            "trained_on": "data/raw/Amazon_Reviews_3500records.csv",
        }
        with open(f"{MODEL_DIR}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n Frozen deployment artifacts in {MODEL_DIR}/:")
        print(" sentiment_model.pkl -- the classifier")
        print(" tfidf_vectorizer.pkl -- the vectorizer")
        print(" model_meta.json -- the lineage : which run produced this")

if __name__ == "__main__":
        print(" MLFlow tracked Experiments")

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow tracking backend: {TRACKING_URI}")

        df=load_features()
        print(f"Loaded {len(df)} records")

        X_text = df["clean_doc"]
        y = df["sentiment"]

        #stratify keeps the positive/negative ratio identical in both splits
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        print(f" train: {len(y_train)} rows, test: {len(y_test)} rows")

        results = []
        for config in EXPERIMENTS:
            results.append(
                run_one_experiment(config, X_train_text, X_test_text, y_train, y_test)
            )

        # ---- Comparison table
        print("MODEL COMPARISON")
        table = pd.DataFrame(
            [
                {
                    "run_name":r[ "run_name"],
                    "model": r["model_type"],
                    "vocab": r["vocab_size"],
                    "ngram_max": r["ngram_max"],
                    "accuracy": round(r["accuracy"], 3),
                    "roc_auc": round(r["roc_auc"], 3),
                    "f1_score": round(r["f1_score"], 3),
                }
                for r in results
            ]
        )
        print(table.to_string(index=False))

        # -- Select the best run
        # ROC-AUC is the primary criterion : it is threshold-independent and
        # does not flatter a model just because the classes happen to be
        # balanced. Accuracy is reported as a sanity check

        best = max(results, key=lambda r: r["roc_auc"])
        print(f"\n Best run by ROC-AUC: {best['run_name']} "
              f"(roc_auc={best['roc_auc']:.4f}, run_id={best['run_id']})")

        save_best_artifacts(best)

        table.to_csv(f"{MODEL_DIR}/experiment_comparison.csv", index=False)
        print(f" experiment_comparison.csv saved")

        print(f"\n To compare run in the MLflow UI:")
        print(f" mlflow ui -- backend-store-ui {TRACKING_URI}")
        print(" then open http://localhost:5000/")




































