"""
monitoring/analyze_drift.py
===========================
Drift analysis and retraining trigger

Compares production traffic ( prediction_logs/predictions.db) against the
training baseline (feature_store/feature_store.db) and prints a drift
report plus an evidence-based retraining recommendation.

Signals compared, and why each one earns its place:

    n_words/n_chars     Are production reviews getting shorter ? Short text
                        carries less signal and TF-IDF degrades on it.
    oov_rate            THE text drift signal. The share of words the model
                        has never seen. Language evolves; vocabulary fixed at
                        training time does not.
    confidence          When the model is unsure more often, it is being
                        asked questions it was not trained to answer.
    predicted class mix A sudden swing in the positive/negative split without
                        a business reason is a red flag.


The threshold used below (a normalized shift of more than 1 standard
deviation) is deliberately simple and explianable -- an on-call
engineer can verify it by hand.

Run:
    python monitoring/analyze_drift.py

"""

import os
import sqlite3
import sys

import joblib
import pandas as pd

FEATURE_STORE_DB = "feature_store/feature_store.db"
PREDICTION_DB = "prediction_logs/predictions.db"
VECTORIZER_PATH = "model_store/tfidf_vectorizer.pkl"

DRIFT_SIGMA_THRESHOLD = 1.0     # shift beyond 1 sigma is flagged
OOV_ABSOLUTE_THRESHOLD = 0.35   # > 35% unseen words is serious on its own
CONFIDENCE_FLOOR  = 0.50        # mean confidence below this is worrying

def load_training_baseline():
    """ Training-side statistics, computed from the immutable feature store """
    conn = sqlite3.connect(FEATURE_STORE_DB)
    df = pd.read_sql("select * from review_features", conn)
    conn.close()

    # Baseline OOV: score the training documents against the deployed
    # vocabulary. This is near zero by construction, which is exactly the
    # point -- it is the floor the production traffic is compared against.
    vectorizer = joblib.load(VECTORIZER_PATH)
    vocab = vectorizer.vocabulary_

    oov_rates =[]
    for doc in df["clean_doc"]:
        words = doc.split()
        if words:
            oov_rates.append(sum(1 for w in words if w not in vocab) / len(words))
    df["oov_rate"] = oov_rates

    return df

def load_production_logs():
    conn = sqlite3.connect(PREDICTION_DB)
    df = pd.read_sql("select * from predictions", conn)
    conn.close()
    return df

def compare_numeric(feature, train_df, prod_df):
    """
    Normalized shift: how many training standard deviations has the
    production mean moved? Dividing by sigma makes features with different
    units comparable on one scale.
    """
    t_mean = train_df[feature].mean()
    t_std = train_df[feature].std()
    p_mean = prod_df[feature].mean()
    p_std = prod_df[feature].std()

    shift = abs(p_mean - t_mean) / (t_std + 1e-9)
    drifted = shift > DRIFT_SIGMA_THRESHOLD
    flag = 'DRIFTED' if drifted else "stable "

    print(f"\n Feature      : {feature}")
    print(f" Training mean  : {t_mean} std : {t_std}")
    print(f" Production mean  : {p_mean} std : {p_std}")
    print(f" Normalized shift : {shift} sigma [{flag}]")

    return {"feature": feature, "train_mean": t_mean, "prod_mean": p_mean, "shift_sigma": shift, "drifted": drifted}

if __name__ == "__main__":

    for path in (FEATURE_STORE_DB, PREDICTION_DB, VECTORIZER_PATH):
        if not os.path.exists(path):
            print(f"{path} does not exist")
            print("Run pipeline first")
            sys.exit(1)

    train_df = load_training_baseline()
    prod_df = load_production_logs()

    if len(prod_df) == 0:
        print("No production logs found")
        print("Run: python monitoring/simulate_traffic.py")
        sys.exit(1)

    print(f"Training samples: {len(train_df)}")
    print(f"Production logs: {len(prod_df)}")

    #------------------ Numerical Drift ----------
    print("Numerical Drift")
    findings = [
        compare_numeric("n_words", train_df, prod_df),
        compare_numeric("n_chars", train_df, prod_df),
        compare_numeric("oov_rate", train_df, prod_df),
    ]

    # --- Prediction side signals ---------
    print("Prediction Distribution")
    mix = prod_df["sentiment"].value_counts(normalize=True) * 100
    for label, pct in mix.items():
        print(f" predicted {label}: {pct}%")

    mean_conf = prod_df["confidence"].mean()
    mean_prob = prod_df["positive_probability"].mean()

    print(f"\nMean confidence: {mean_conf}")
    print(f"\nMean probability: {mean_prob}")

    # Latency is reported as percentiles, not a mean. A single cold-start
    # request ( first call after startup, ~35 ms while caches warm) drags the
    # mean above the p95, which tell you nothing useful about typical
    # request cost. Percentiles are what a latency budget is written against.
    lat = prod_df["latency_ms"]
    print(f" median latency (ms)    : {lat.median()}")
    print(f" p95 latency (ms)    : {lat.quantile(0.95)}")
    print(f" max latency (ms)    : {lat.max()} (cold start)")

    # --- Batch-by-batch view (the actual way) ---
    if prod_df["traffic_batch"].notna().any():
        print("\n" + "=" * 64)
        print("Traffic Batch Statistics")
        print("\n" + "=" * 64)
        by_batch = prod_df.groupby("traffic_batch").agg(
            requests=("id", "count"),
            mean_words=("n_words", "mean"),
            mean_oov=("oov_rate", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_p_positive=("positive_probability", "mean"),
        )
        print(by_batch.to_string())

    # ---------- Retraining trigger evaluation -----------
    print("\n" + "=" * 64)
    print("Retraining trigger evaluation")
    print("\n" + "=" * 64)

    oov_finding = next(f for f in findings if f["feature"] == "oov_rate")
    words_finding = next (f for f in findings if f["feature"] == "n_words")

    # Each row is one piece of evidence. No single row triggers retraining
    # on its own -- that it the point of framework

    evidence = [
        (
            f"oov_rate_shifted > {DRIFT_SIGMA_THRESHOLD} sigma",
            oov_finding["drifted"],
            "Model vocabulary no longer covers production laungauges"
        ),
        (
            f"absolute oov_rate > {OOV_ABSOLUTE_THRESHOLD}",
            prod_df["oov_rate"].mean() > OOV_ABSOLUTE_THRESHOLD,
            " large share of review is invisible to model"
        ),
        (
            f"review length shifted > {DRIFT_SIGMA_THRESHOLD} sigma",
            words_finding["drifted"],
            "Input format have changed upstream"
        ),
        (
            f"mean confidence < {CONFIDENCE_FLOOR}",
            mean_conf < CONFIDENCE_FLOOR,
            "Model is guessing more often than it is deciding"
        ),
        (
            "predicted class mix beyond 80/20",
            mix.max() > 80,
            "Scores are collapsing toward one class"
        ),
    ]

    triggered =0
    for name, observed, implication in evidence:
        mark = "YES" if observed else "NO"
        print(f" [{mark}] {name}")
        if observed:
            print(f" --> {implication}")
            triggered += 1

    print("\n" + "=" * 64)
    print(f"Trigger Evaluation {triggered} of {len(evidence)}")

    # The recommendation is a recommendation. A human owns the decision --
    # retraining costs compute and carries the risk

    if triggered == 0:
        decision = "Do no Retrain - no material drift detected"
    elif triggered == 1:
        decision = "Monitor - one isolated signal is not sufficient evidence"

    elif triggered == 2:
        decision = "Collect more data"

    else:
        decision = "Retrain as multiple drift"

    print(f"Recommendation: {decision}")
    print("\n" + "=" * 64)

    print("\n Ground truth caveat: every signal above is computed from")
    print(" INPUTS and OUTPUTS only. None of them prove model is wrong")

    if triggered >= 3:
        print("\n If retraining is approved, pipeline is:")
        print(" 1. python data setup")
        print(" 2 .python validation/data_validation.py")
        print(" 3 .python features/build_features.py")
        print(" 4 .python training/train_with_mlflow.py")
        print(" 5. compare against deployed run in mlfow ui")
        print(" 6. document the promotion decision in README, then restart")





















