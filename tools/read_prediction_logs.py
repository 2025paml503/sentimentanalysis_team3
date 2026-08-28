"""
tools/read_prediction_logs.py
===============================
Quick inspection of the prediction log

prediction_logs/predictions.db is a SQLite database with one table,
'predictions' (see monitoring/prediction_logger.py for the schema). This
script is just a convenient reader -- it does not modify anything.

Run:
    python tools/read_prediction_logs.py                # last 10 rows
    python tools/read_prediction_logs.py --limit 25     # last 25 rows
    python tools/read_prediction_logs.py --batch drifted # filter by traffic_batch
    python tools/read_prediction_logs.py --summary      # counts + average only

"""

import argparse
import sqlite3

import pandas as pd

DB_PATH = "prediction_logs/predictions.db"

def load_predictions(batch=None):
    conn = sqlite3.connect(DB_PATH)
    if batch:
        df = pd.read_sql(
            "SELECT * FROM predictions WHERE traffic_batch = ? ORDER BY id",
            conn, params=(batch,),
        )
    else:
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id", conn)
    conn.close()
    return df

def print_summary(df):
    print(f"Total rows: {len(df)}")
    if df.empty:
        return

    print("\nBy traffic_batch:")
    by_batch = df.groupby(df["traffic_batch"].fillna("(untagged)")).agg(
        requests=("id", "count"),
        mean_words=("n_words", "mean"),
        mean_oov=("oov_rate", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_p_positive=("positive_probability", "mean"),
    )

    print(by_batch.to_string())

    print("\nSentiment mix:")
    print(df["sentiment"].value_counts().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10,
                        help="how many most-recent rows to print (default: 10)")
    parser.add_argument("--batch", default=None,
                        help="filter by one traffic_batch value, e.g. normal / drifted")
    parser.add_argument("--summary", action="store_true",
                        help="print only the aggregate summary, no row dump")
    args = parser.parse_args()

    df = load_predictions(batch=args.batch)

    if args.summary:
        print_summary(df)
    else:
        cols = ["id", "timestamp", "title", "sentiment", "positive_probability",
                "oov_rate", "confidence", "latency_ms", "traffic_batch"]
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 300)

        print(df[cols].tail(args.limit).to_string(index=False))
        print(f"\n(showing last {min(args.limit, len(df))} of {len(df)} rows"
              f"--use --summary for aggregates, --batch to filter)")

















