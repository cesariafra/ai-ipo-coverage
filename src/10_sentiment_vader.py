#!/usr/bin/env python3
"""
Local sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Calculates general sentiment on successfully scraped articles in
data/scraped_articles_full.csv and saves results to data/vader_sentiment_results.csv.

Columns:
  - url
  - vader_compound
  - vader_label ("Positive" | "Negative" | "Neutral")

Usage:
    python src/10_sentiment_vader.py
    python src/10_sentiment_vader.py --limit 100
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("ERROR: pandas is required. Run: pip install pandas")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    raise SystemExit(
        "ERROR: vaderSentiment is required.\n"
        "Run: pip install vaderSentiment"
    )

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "scraped_articles_full.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "vader_sentiment_results.csv"

MAX_TEXT_LEN = 10_000   # characters to save memory and processing time
BATCH_SAVE_EVERY = 50   # flush to disk every N articles

OUTPUT_COLUMNS = ["url", "vader_compound", "vader_label"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_compound(compound: float) -> str:
    """Classify compound score into Positive, Negative, or Neutral."""
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def load_already_done(path: Path) -> set[str]:
    """Return the set of URLs already present in the output CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["url"])
        return set(df["url"].dropna().astype(str).tolist())
    except Exception as exc:
        print(f"[Warning] Could not read existing output for resume: {exc}")
        return set()


def save_batch(rows: list[dict], path: Path, write_header: bool) -> None:
    """Append a batch of records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_vader_analysis(limit: int | None = None) -> None:
    print("=" * 70)
    print("LOCAL SENTIMENT ANALYSIS — VADER")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    # 1. Load input dataset
    print(f"Loading input data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Total rows in dataset: {len(df):,}")

    # 2. Filter to SUCCESS with non-empty article text
    df = df[
        (df["category"] == "SUCCESS")
        & df["article_text"].notna()
        & (df["article_text"].str.strip() != "")
    ].copy()
    print(f"Eligible rows (SUCCESS + non-empty text): {len(df):,}")

    # 3. Resume capability: skip already processed URLs
    already_done = load_already_done(OUTPUT_CSV)
    if already_done:
        print(f"Resuming: {len(already_done):,} URLs already processed — skipping.")
        df = df[~df["url"].astype(str).isin(already_done)]

    if limit is not None:
        df = df.head(limit)

    total = len(df)
    if total == 0:
        print("No new articles to analyze. Exiting.")
        print_final_summary()
        return

    print(f"Articles to process this run: {total:,}")
    print("=" * 70 + "\n")

    # 4. Initialize VADER analyzer
    analyzer = SentimentIntensityAnalyzer()

    # 5. Determine whether header is needed
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    pending_batch: list[dict] = []
    run_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
    start_time = time.time()

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        url = str(row["url"])
        text = str(row["article_text"]).strip()
        truncated_text = text[:MAX_TEXT_LEN]

        # Calculate polarity scores
        scores = analyzer.polarity_scores(truncated_text)
        compound = round(scores["compound"], 4)
        label = classify_compound(compound)
        run_distribution[label] += 1

        pending_batch.append({
            "url": url,
            "vader_compound": compound,
            "vader_label": label,
        })

        # Periodic log & batch save
        if len(pending_batch) >= BATCH_SAVE_EVERY:
            save_batch(pending_batch, OUTPUT_CSV, write_header)
            write_header = False
            pending_batch = []
            progress_pct = (idx / total) * 100
            print(f"[{idx:>{len(str(total))}}/{total}] ({progress_pct:5.1f}%) Flushed to {OUTPUT_CSV.name}")

    # Flush remaining
    if pending_batch:
        save_batch(pending_batch, OUTPUT_CSV, write_header)
        write_header = False

    elapsed = time.time() - start_time
    print(f"\nCompleted processing {total:,} articles in {elapsed:.2f}s ({total / max(elapsed, 0.001):.1f} articles/s).")

    # 6. Print distribution summary
    print_final_summary(run_distribution)


def print_final_summary(run_counts: dict[str, int] | None = None) -> None:
    """Print overall distribution summary from the output CSV."""
    print("\n" + "=" * 70)
    print("VADER SENTIMENT DISTRIBUTION SUMMARY")
    print("=" * 70)

    if run_counts:
        run_total = sum(run_counts.values())
        print(f"This run ({run_total:,} articles):")
        for label in ["Positive", "Neutral", "Negative"]:
            count = run_counts.get(label, 0)
            pct = (count / run_total * 100) if run_total > 0 else 0.0
            print(f"  - {label:<8}: {count:>6,} ({pct:5.1f}%)")
        print("-" * 70)

    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        try:
            full_df = pd.read_csv(OUTPUT_CSV)
            total_saved = len(full_df)
            counts = full_df["vader_label"].value_counts().to_dict()

            print(f"Overall dataset ({total_saved:,} total processed in {OUTPUT_CSV.name}):")
            for label in ["Positive", "Neutral", "Negative"]:
                count = counts.get(label, 0)
                pct = (count / total_saved * 100) if total_saved > 0 else 0.0
                print(f"  - {label:<8}: {count:>6,} ({pct:5.1f}%)")
        except Exception as exc:
            print(f"[Warning] Could not calculate overall distribution: {exc}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local VADER sentiment analysis on scraped articles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all pending).",
    )
    args = parser.parse_args()
    run_vader_analysis(limit=args.limit)


if __name__ == "__main__":
    main()

