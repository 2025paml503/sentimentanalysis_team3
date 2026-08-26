import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("prediction_logs/predictions.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            title           TEXT,
            review_text     TEXT    NOT NULL,
            clean_doc       TEXT    NOT NULL,
            n_words         INTEGER NOT NULL,
            n_chars         INTEGER NOT NULL,
            oov_rate        REAL    NOT NULL,
            sentiment       TEXT    NOT NULL,
            positive_probability REAL NOT NULL,
            confidence      REAL    NOT NULL,
            latency_ms      REAL    NOT NULL,
            model_version   TEXT    NOT NULL,
            traffic_batch   TEXT 
        )
    """)
    conn.commit()
    conn.close()

def log_predictions(review_text, title, clean_doc, result, traffic_batch=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
         (timestamp, title, review_text, clean_doc, n_words, n_chars, 
           oov_rate, sentiment, positive_probability, confidence,
            latency_ms, model_version, traffic_batch)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc),
        title,
        review_text,
        clean_doc,
        len(clean_doc.split()),
        len(clean_doc),
        result["oov_rate"],
        result["sentiment"],
        result["positive_probability"],
        result["confidence"],
        result["latency_ms"],
        result["model_version"],
        traffic_batch
    ))
    conn.commit()
    conn.close()