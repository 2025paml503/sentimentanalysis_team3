from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "amazon_reviews.txt"
OUTPUT_PATH = Path(__file__).resolve().parent / "eda_summary.txt"
PLOT_PATH = Path(__file__).resolve().parent / "sentiment_distribution.png"


def load_reviews(path: Path) -> pd.DataFrame:
    """Load the review dataset from a tab-separated text file."""
    df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"], encoding="utf-8")
    return df


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add text-derived features for EDA."""
    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df["char_length"] = df["review"].str.len()
    df["word_count"] = df["review"].str.split().str.len()
    df["avg_word_length"] = df["review"].str.split().apply(lambda words: sum(len(w) for w in words) / len(words) if words else 0)
    return df


def get_top_words(series: pd.Series, n: int = 20) -> list[tuple[str, int]]:
    """Return the n most frequent words after normalizing text."""
    text = " ".join(series.astype(str).tolist()).lower()
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    word_counts = Counter(word for word in words if len(word) > 2)
    return word_counts.most_common(n)


def check_duplicates(df: pd.DataFrame) -> int:
    """Count exact duplicate rows (same review and label)."""
    return int(df.duplicated(subset=["review", "label"]).sum())


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing values per column."""
    return df.isnull().sum()


def check_non_english_chars(df: pd.DataFrame) -> int:
    """Return the number of reviews containing non-ASCII / non-English characters."""
    return int(df["review"].apply(lambda text: bool(re.search(r"[^\x00-\x7F]", text))).sum())


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the distribution of positive and negative sentiment labels."""
    counts = df["label"].value_counts().sort_index()
    labels = ["Negative", "Positive"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#d95f02", "#1b9e77"])
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    for bar, value in zip(ax.patches, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_report(df: pd.DataFrame) -> None:
    print("=== Exploratory Data Analysis Report ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nDuplicate rows:")
    print(f"{check_duplicates(df)}")

    print("\nMissing values:")
    print(check_missing_values(df).to_string())

    print("\nNon-English / non-ASCII characters in review text:")
    print(f"{check_non_english_chars(df)} rows")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    print("\nText length summary:")
    print(df["char_length"].describe().to_string())

    print("\nWord count summary:")
    print(df["word_count"].describe().to_string())

    print("\nTop 20 words:")
    for word, count in get_top_words(df["review"], n=20):
        print(f"{word}: {count}")


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Write a simple text summary file for the EDA."""
    lines: list[str] = []
    lines.append("=== Exploratory Data Analysis Report ===")
    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append("")
    lines.append(f"Duplicate rows: {check_duplicates(df)}")
    lines.append("")
    lines.append("Missing values:")
    lines.append(check_missing_values(df).to_string())
    lines.append("")
    lines.append(f"Non-English / non-ASCII review rows: {check_non_english_chars(df)}")
    lines.append("")
    lines.append("Label distribution:")
    lines.append(df["label"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Text length summary:")
    lines.append(df["char_length"].describe().to_string())
    lines.append("")
    lines.append("Top words:")
    for word, count in get_top_words(df["review"], n=20):
        lines.append(f"{word}: {count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary saved to: {output_path}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = load_reviews(DATA_PATH)
    df = summarize_dataset(df)
    plot_sentiment_distribution(df, PLOT_PATH)
    print_report(df)
    save_summary(df, OUTPUT_PATH)
    print(f"\nSentiment distribution chart saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
